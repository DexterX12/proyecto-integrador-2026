"""Generación de recomendaciones Top-N precalculadas con el índice FAISS del modelo."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..data.features import build_context_frame, load_context_matrix, to_ms
from ..retrieval.faiss_index import FlatIPIndex

log = logging.getLogger(__name__)


def select_rec_users(cfg, md) -> np.ndarray:
    """Usuarios con recomendaciones: >= min_history_for_recs positivos antes del corte de test
    más todos los usuarios de evaluación (para poder contrastar con lo que realmente compraron)."""
    inter = md.interactions
    test_start = md.manifest["cutoffs"]["test_start_ms"]
    past = inter[(inter["ts"] < test_start) & (inter["rating"] >= md.threshold)]
    n_pos = past.groupby("user_id").size()
    base = n_pos[n_pos >= cfg.protocol.min_history_for_recs].index.to_numpy(dtype=object)
    eval_users = set(md.eval_set("test", extended=True).user_ids.tolist())
    return np.array(sorted(set(base.tolist()) | eval_users), dtype=object)


def generate_recommendations(cfg, lake, md, model, config, model_version: str, run) -> dict:
    top_n = int(cfg.retrieval.top_n)
    H = md.H
    test_start = md.manifest["cutoffs"]["test_start_ms"]
    users = select_rec_users(cfg, md)
    run.metric("users_selected", len(users))

    user_map = dict(zip(md.users["user_id"], md.users["user_idx"]))
    ctx = build_context_frame(md.interactions, test_start, H, md.threshold, user_map, users=users)
    ctx = ctx.sort_values("user_id").reset_index(drop=True)
    run.metric("users_with_context", len(ctx))

    hist = load_context_matrix(ctx, H)
    user_emb = model.encode_users(ctx["user_idx"].to_numpy(), hist)
    index = FlatIPIndex.load(lake.artifacts(model_version))
    ids, scores = index.search_excluding_seen(user_emb, list(ctx["seen"]), top_n)

    n = len(ctx)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    asin_of = md.asin_of
    table = pa.table({
        "user_id": pa.array(np.repeat(ctx["user_id"].to_numpy(dtype=object), top_n).tolist(), pa.string()),
        "rank": pa.array(np.tile(np.arange(1, top_n + 1, dtype=np.int16), n)),
        "item_idx": pa.array(ids.ravel().astype(np.int32)),
        "parent_asin": pa.array(asin_of[ids.ravel()].tolist(), pa.string()),
        "score": pa.array(scores.ravel().astype(np.float32)),
    })
    meta = {"model_version": model_version, "generated_at": generated_at,
            "as_of": md.manifest["cutoffs"]["test_start"], "dataset_version": md.manifest["dataset_version"]}
    table = table.replace_schema_metadata({k: str(v) for k, v in meta.items()})
    out = lake.recommendations(model_version)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "recommendations.parquet", compression="zstd", row_group_size=250_000)

    # usuarios con recomendaciones + ground truth de la ventana de test (para el dashboard)
    tw = md.interactions
    tw = tw[(tw["ts"] >= test_start) & (tw["rating"] >= md.threshold) & tw["user_id"].isin(ctx["user_id"])]
    n_test = tw.groupby("user_id").size()
    users_df = pd.DataFrame({
        "user_id": ctx["user_id"], "n_hist_pos": ctx["n_hist_pos"], "n_seen": ctx["n_seen"],
        "has_user_id_embedding": ctx["user_idx"] > 0,
    })
    users_df["n_test_positives"] = users_df["user_id"].map(n_test).fillna(0).astype(int)
    users_df.to_parquet(out / "recommendation_users.parquet", index=False)

    np.save(lake.artifacts(model_version) / "user_embeddings.npy", user_emb.astype(np.float32))
    ctx[["user_id"]].to_parquet(lake.artifacts(model_version) / "user_embedding_ids.parquet", index=False)

    # ---- validaciones de la salida
    viol = 0
    for u in range(n):
        viol += int(np.isin(ids[u], ctx["seen"].iloc[u]).sum())
    validation = {
        **meta, "top_n": top_n, "users": n, "rows": int(n * top_n),
        "recommended_seen_items_violations": viol,
        "users_with_less_than_top_n": int(((ids == 0).any(1)).sum()),
        "all_items_in_universe": bool(md.features["in_universe"][ids.ravel()].all()),
        "duplicates_within_list": int(sum(len(set(r)) != len(r) for r in ids)),
        "scores_sorted_desc": bool((np.diff(scores, axis=1) <= 1e-6).all()),
        "distinct_items_recommended": int(len(np.unique(ids))),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    run.metric("recommendation_validation", validation)

    # Fallback para usuarios sin historial: popularidad global (cold start de usuario)
    from ..models.baselines import PopularityModel
    pop = PopularityModel(cfg.baselines.popularity_min_reviews).fit(md.items)
    fb = pop.ranking[:top_n]
    pd.DataFrame({"rank": np.arange(1, len(fb) + 1), "item_idx": fb, "parent_asin": asin_of[fb]}).to_parquet(
        out / "popularity_fallback.parquet", index=False)
    return validation


def compute_projection(lake, md, model_version: str, n_sample: int = 6000, seed: int = 42) -> int:
    """Proyección UMAP 2D de embeddings de producto (muestra estratificada por tipo). Exploratoria."""
    import umap

    emb_dir = lake.artifacts(model_version) / "item_embeddings"
    emb = np.load(emb_dir / "item_embeddings.npy", mmap_mode="r")
    ids = np.load(emb_dir / "item_ids.npy")
    items = md.items.set_index("item_idx")
    ptype = md.features["product_type"][ids]
    rng = np.random.default_rng(seed)
    # muestra ponderada hacia ítems con más interacciones para que la nube sea representativa
    w = items.loc[ids, "train_count"].to_numpy(dtype=np.float64) + 1.0
    sel = np.sort(rng.choice(len(ids), size=min(n_sample, len(ids)), replace=False, p=w / w.sum()))
    xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(
        np.asarray(emb[sel]))
    from ..data.taxonomy import PRODUCT_TYPE_VOCAB
    vocab = ["pad"] + PRODUCT_TYPE_VOCAB
    df = pd.DataFrame({"item_idx": ids[sel], "parent_asin": md.asin_of[ids[sel]],
                       "x": xy[:, 0], "y": xy[:, 1],
                       "product_type": [vocab[t] for t in ptype[sel]]})
    df.to_parquet(lake.artifacts(model_version) / "umap_projection.parquet", index=False)
    return len(df)
