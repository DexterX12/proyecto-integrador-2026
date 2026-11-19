"""Evaluación comparable: mismos usuarios, universo, filtro de vistos y relevancia para todos los modelos."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..models.baselines import ItemKNN, PopularityModel
from ..models.train import load_model
from ..retrieval.faiss_index import FlatIPIndex
from .metrics import bootstrap_ci, evaluate, per_user_metrics

log = logging.getLogger(__name__)


def check_index_consistency(model, index: FlatIPIndex, md, sample: int = 3000, seed: int = 0) -> dict:
    """Verifica que el índice FAISS guardado corresponde al checkpoint cargado y al universo actual."""
    same_ids = bool(np.array_equal(index.item_ids, md.universe))
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(index.item_ids), size=min(sample, len(index.item_ids)), replace=False))
    stored = np.stack([index.index.reconstruct(int(r)) for r in rows])
    fresh = model.encode_items(index.item_ids[rows])
    max_diff = float(np.abs(stored - fresh).max())
    return {"item_ids_match_universe": same_ids, "ntotal": int(index.index.ntotal),
            "max_abs_diff_sampled": max_diff, "ok": same_ids and max_diff < 1e-4}


def _details(df: pd.DataFrame, split: str, model: str, ks) -> pd.DataFrame:
    keep = ["user_id", "n_hist_pos", "has_user_id_embedding", "n_relevant"] + [
        f"{m}@{k}" for k in ks for m in ("recall", "precision", "ndcg", "map", "hit")]
    out = df[keep].copy()
    out.insert(0, "model", model)
    out.insert(0, "split", split)
    return out


def evaluate_all(cfg, lake, md, model_version: str | None, splits: list[str], run) -> dict:
    ks = list(cfg.retrieval.eval_ks)
    n_rec = max(cfg.retrieval.top_n, max(ks))
    items = md.items
    feats = md.features
    item_pop = np.concatenate([[0], items["train_pos_count"].to_numpy()]).astype(np.float64)
    content = feats["text"]
    universe = md.universe

    # ---- modelos
    pop = PopularityModel(cfg.baselines.popularity_min_reviews).fit(items)
    train_pos = md.interactions
    train_pos = train_pos[(train_pos["split"] == 0) & (train_pos["rating"] >= md.threshold)]
    knn = ItemKNN(cfg.baselines.itemknn_neighbors, cfg.baselines.itemknn_shrink,
                  cfg.baselines.itemknn_max_history).fit(train_pos, md.n_items, pop, feats["in_universe"])
    models = {"popularity": lambda ev: pop.recommend(ev.seen, n_rec),
              "item_knn": lambda ev: knn.recommend(ev.hist, ev.seen, n_rec)}
    tt_model, index, consistency = None, None, None
    if model_version:
        tt_model, config = load_model(lake, md, model_version)
        index = FlatIPIndex.load(lake.artifacts(model_version))
        consistency = check_index_consistency(tt_model, index, md)
        run.metric("faiss_consistency", consistency)
        assert consistency["ok"], f"Índice FAISS inconsistente con el checkpoint: {consistency}"
        models["two_tower"] = lambda ev: index.search_excluding_seen(
            tt_model.encode_users(ev.user_idx, ev.hist), ev.seen, n_rec)[0]

    results: dict = {"splits": {}, "cold_start": {}}
    details = []
    for split in splits:
        ev = md.eval_set(split)
        results["splits"][split] = {}
        for name, fn in models.items():
            recs = fn(ev)
            res, per_user = evaluate(recs, ev, ks, n_candidates=len(universe), content_vecs=content,
                                     item_pop=item_pop)
            lo, hi = bootstrap_ci(per_user["recall@20"].to_numpy())
            nlo, nhi = bootstrap_ci(per_user["ndcg@20"].to_numpy())
            res["ci95"] = {"recall@20": [lo, hi], "ndcg@20": [nlo, nhi]}
            results["splits"][split][name] = res
            details.append(_details(per_user, split, name, ks))
            log.info("[%s] %-10s recall@20=%.4f ndcg@20=%.4f (n=%d)", split, name,
                     res["overall"]["recall@20"], res["overall"]["ndcg@20"], res["n_users"])

    # ---- cold start de ítems (solo test): candidatos = train + validation, ítems nuevos sin ID entrenado
    if "test" in splits and tt_model is not None:
        ev = md.eval_set("test", extended=True)
        ext = md.universe_extended
        cold_mask = ~feats["in_universe"][np.concatenate(ev.targets)]
        n_cold_targets = int(cold_mask.sum())
        ext_index = FlatIPIndex.build(tt_model.encode_items(ext), ext, normalize=tt_model.cosine)
        ext_pop = pop
        recs_by_model = {
            "two_tower": ext_index.search_excluding_seen(tt_model.encode_users(ev.user_idx, ev.hist),
                                                          ev.seen, n_rec)[0],
            "two_tower_content_only_items": None,
            "popularity": ext_pop.recommend(ev.seen, n_rec),
            "item_knn": knn.recommend(ev.hist, ev.seen, n_rec),
        }
        # variante sin embedding de ID en la torre de producto: ¿cuánto aporta solo el contenido?
        emb_noid = tt_model.encode_items(ext, use_id=False)
        recs_by_model["two_tower_content_only_items"] = FlatIPIndex.build(emb_noid, ext, normalize=tt_model.cosine).search_excluding_seen(
            tt_model.encode_users(ev.user_idx, ev.hist), ev.seen, n_rec)[0]
        cold_ids = np.nonzero(~feats["in_universe"])[0]
        for name, recs in recs_by_model.items():
            res, per_user = evaluate(recs, ev, ks, n_candidates=len(ext))
            top20 = recs[:, :20]
            cold_hits, cold_total, cold_in_lists = 0, 0, int(np.isin(top20, cold_ids).sum())
            for r, t in zip(top20, ev.targets):
                ct = t[~feats["in_universe"][t]]
                cold_total += len(ct)
                cold_hits += int(np.isin(ct, r).sum())
            res["cold_items"] = {
                "cold_targets": cold_total, "cold_hits@20": cold_hits,
                "cold_recall@20": cold_hits / max(cold_total, 1),
                "share_of_top20_slots_cold_items": cold_in_lists / top20.size,
            }
            results["cold_start"][name] = res
            log.info("[test+cold] %-28s recall@20=%.4f cold_recall@20=%.4f", name,
                     res["overall"]["recall@20"], res["cold_items"]["cold_recall@20"])
        results["cold_start_meta"] = {"extended_universe_items": int(len(ext)),
                                      "cold_items_added": int(len(ext) - len(universe)),
                                      "cold_targets": n_cold_targets}

    results["meta"] = {
        "model_version": model_version, "dataset_version": md.manifest["dataset_version"],
        "protocol_digest": md.manifest["protocol_digest"], "cutoffs": md.manifest["cutoffs"],
        "relevance_threshold": md.threshold, "ks": ks, "top_n_generated": n_rec,
        "candidate_universe_items": int(len(universe)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_sets": {s: md.manifest["eval"][s] for s in ("validation", "test") if s in splits},
        "faiss_consistency": consistency,
        "baselines": {"popularity_min_reviews": cfg.baselines.popularity_min_reviews,
                      "itemknn_neighbors": cfg.baselines.itemknn_neighbors,
                      "itemknn_users_backfilled_last_split": int(getattr(knn, "n_backfilled", 0))},
    }
    results["_details"] = pd.concat(details, ignore_index=True)
    return results


def comparison_table(results: dict, split: str, ks=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    for model, res in results["splits"][split].items():
        row = {"model": model, "n_users": res["n_users"]}
        for k in ks:
            row[f"recall@{k}"] = res["overall"][f"recall@{k}"]
            row[f"ndcg@{k}"] = res["overall"][f"ndcg@{k}"]
        row["precision@10"] = res["overall"]["precision@10"]
        row["map@10"] = res["overall"]["map@10"]
        row["coverage@20"] = res["beyond_accuracy"]["@20"].get("coverage")
        row["diversity@20"] = res["beyond_accuracy"]["@20"].get("diversity")
        row["novelty@20"] = res["beyond_accuracy"]["@20"].get("novelty")
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(lake, model_version: str, results: dict) -> None:
    out = lake.evaluation(model_version)
    out.mkdir(parents=True, exist_ok=True)
    details = results.pop("_details")
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    details.to_parquet(out / "evaluation_details.parquet", index=False)
    tables = [comparison_table(results, s).assign(split=s) for s in results["splits"]]
    pd.concat(tables, ignore_index=True).to_csv(out / "comparison_table.csv", index=False)

