"""Entrenamiento del Two-Tower, selección de checkpoint por validación y exportación de artefactos."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..evaluation.metrics import per_user_metrics
from ..retrieval.faiss_index import FlatIPIndex, exact_search_excluding_seen
from .two_tower import ItemFeatures, TwoTower, contrastive_loss

log = logging.getLogger(__name__)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def new_model_version(lake, tag: str | None = None) -> str:
    version = datetime.now().strftime("%Y%m%d_%H%M")
    if tag:
        version += f"_{tag}"
    while lake.artifacts(version).exists():          # evita pisar una versión existente
        version = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
        time.sleep(1)
    return version


def validate(model: TwoTower, md, evalset, ks=(10, 20), n: int = 20) -> dict:
    """Recall/NDCG sobre un subconjunto de validación con ranking exacto sobre el universo completo."""
    universe = md.universe
    item_emb = model.encode_items(universe)
    user_emb = model.encode_users(evalset.user_idx, evalset.hist)
    recs, _ = exact_search_excluding_seen(user_emb, item_emb, universe, evalset.seen, max(n, max(ks)),
                                          normalize=False)
    df = per_user_metrics(recs, evalset.targets, ks)
    return {f"{m}@{k}": float(df[f"{m}@{k}"].mean()) for m in ("recall", "ndcg") for k in ks}


def train_two_tower(cfg, md, run, history_path: Path | None = None) -> tuple[TwoTower, dict, list]:
    tcfg, mcfg = cfg.train, cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    torch.manual_seed(tcfg.random_seed)
    np.random.seed(tcfg.random_seed)
    torch.set_num_threads(int(tcfg.num_threads))
    rng = np.random.default_rng(tcfg.random_seed)

    feats = ItemFeatures.from_arrays(md.features)
    n_users = len(md.users) + 1
    model = TwoTower(feats, n_users, mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    run.metric("model_parameters", n_params)

    ex = md.train_examples(int(tcfg.min_history_len))
    n = len(ex["item_idx"])
    has_hist = np.nonzero(ex["hist_len"] > 0)[0]
    no_hist = np.nonzero(ex["hist_len"] == 0)[0]      # solo existen si min_history_len = 0
    run.metrics(train_examples_with_history=len(has_hist), train_examples_without_history=len(no_hist))
    user_t = torch.from_numpy(ex["user_idx"])
    item_t = torch.from_numpy(ex["item_idx"])
    hist_t = torch.from_numpy(ex["hist"].astype(np.int64))

    log_q = None
    if tcfg.logq_correction:
        counts = md.items["train_pos_count"].to_numpy(np.float64)
        q = (counts + 1.0) / (counts.sum() + len(counts))
        log_q = torch.from_numpy(np.concatenate([[0.0], np.log(q)]).astype(np.float32))

    val_all = md.eval_set("validation")
    n_val = int(tcfg.val_users) or len(val_all)
    sub = rng.choice(len(val_all), size=min(n_val, len(val_all)), replace=False)
    val_sub = val_all.subset(np.sort(sub))

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    metric_key = tcfg.early_stopping_metric
    best = {"metric": -1.0, "epoch": 0, "state": None}
    history, bad_epochs = [], 0
    bs = int(tcfg.batch_size)
    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_idx = has_hist
        if len(no_hist) and tcfg.empty_history_frac > 0:
            extra = rng.choice(no_hist, size=int(len(no_hist) * tcfg.empty_history_frac), replace=False)
            epoch_idx = np.concatenate([has_hist, extra])
        perm = epoch_idx[rng.permutation(len(epoch_idx))]
        losses, accs = [], []
        for start in range(0, len(perm) - bs + 1, bs):          # se descarta el último batch incompleto
            idx = torch.from_numpy(perm[start:start + bs])
            z_u, z_i = model.forward_train(user_t[idx], item_t[idx], hist_t[idx])
            loss, info = contrastive_loss(z_u, z_i, item_t[idx], hist_t[idx], user_t[idx],
                                          tcfg.temperature, log_q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            accs.append(info["in_batch_acc"])
        val = validate(model, md, val_sub)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_in_batch_acc": float(np.mean(accs)),
               **{f"val_{k}": v for k, v in val.items()}, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log.info("epoch %02d | loss %.4f | acc %.3f | val recall@20 %.4f ndcg@20 %.4f | %.0fs", epoch,
                 row["train_loss"], row["train_in_batch_acc"], val["recall@20"], val["ndcg@20"], row["seconds"])
        if history_path is not None:
            pd.DataFrame(history).to_csv(history_path, index=False)
        current = val[metric_key]
        if current > best["metric"] + 1e-6:
            best = {"metric": current, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg.patience:
                log.info("Early stopping en epoch %d (mejor: %d)", epoch, best["epoch"])
                break
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"best_epoch": best["epoch"], "best_metric": best["metric"], "metric": metric_key,
                   "epochs_run": len(history)}, history


def export_artifacts(cfg, lake, md, model: TwoTower, model_version: str, summary: dict, history: list) -> dict:
    """model.pt + embeddings + FAISS + mappings + config.json (todo del MISMO checkpoint)."""
    out = lake.artifacts(model_version)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": model.mcfg, "n_users": model.n_users,
                "sizes": model.sizes}, out / "model.pt")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    # mappings de IDs versionados junto al modelo
    maps = out / "id_mappings"
    maps.mkdir(exist_ok=True)
    for name in ("items.parquet", "users.parquet", "transformers.joblib"):
        shutil.copy2(md.dir / "mappings" / name, maps / name)

    # embeddings de productos del universo + índice FAISS (IndexFlatIP, vectores L2-normalizados)
    universe = md.universe
    item_emb = model.encode_items(universe)
    emb_dir = out / "item_embeddings"
    emb_dir.mkdir(exist_ok=True)
    np.save(emb_dir / "item_embeddings.npy", item_emb)
    np.save(emb_dir / "item_ids.npy", universe)
    index = FlatIPIndex.build(item_emb, universe, normalize=model.cosine)
    index_info = index.save(out)

    feature_hashes = {n: sha1_file(md.dir / "features" / f"{n}.npy") for n in
                      ("id_idx", "text", "product_type", "audience", "store", "price", "has_price")}
    config = {
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": md.manifest["dataset_version"],
        "protocol_digest": md.manifest["protocol_digest"],
        "cutoffs": md.manifest["cutoffs"],
        "relevance_threshold": md.threshold,
        "hyperparameters": {"model": cfg.model.to_dict(), "train": cfg.train.to_dict(),
                            "retrieval": cfg.retrieval.to_dict()},
        "random_seed": cfg.train.random_seed,
        "training": summary,
        "n_universe_items": int(len(universe)),
        "embedding_dim": int(item_emb.shape[1]),
        "faiss": index_info,
        "checkpoint_sha1": sha1_file(out / "model.pt"),
        "mappings_sha1": {n: sha1_file(maps / n) for n in ("items.parquet", "users.parquet",
                                                          "transformers.joblib")},
        "feature_sha1": feature_hashes,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy2(out / "config.json", out / "model_config.json")
    return config


def load_model(lake, md, model_version: str) -> tuple[TwoTower, dict]:
    out = lake.artifacts(model_version)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["protocol_digest"] != md.manifest["protocol_digest"]:
        raise RuntimeError("El modelo fue entrenado con otra versión del protocolo/features.")
    if sha1_file(out / "model.pt") != config["checkpoint_sha1"]:
        raise RuntimeError("El checkpoint no coincide con config.json (archivo modificado).")
    ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    feats = ItemFeatures.from_arrays(md.features)
    model = TwoTower(feats, ckpt["n_users"], ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config



