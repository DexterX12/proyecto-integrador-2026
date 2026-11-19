"""Métricas de ranking Top-K con protocolo común para todos los modelos.

Relevancia binaria: un ítem es relevante si el usuario lo calificó con rating >= umbral en la ventana
de evaluación. Los ítems vistos antes del corte ya fueron excluidos del ranking por el recomendador.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = {
    "hist_1": lambda n: n == 1,
    "hist_2_4": lambda n: (n >= 2) & (n <= 4),
    "hist_5plus": lambda n: n >= 5,
}


def _hit_matrix(recs: np.ndarray, targets: list) -> np.ndarray:
    hits = np.zeros(recs.shape, dtype=bool)
    for u in range(len(recs)):
        hits[u] = np.isin(recs[u], targets[u])
    return hits


def per_user_metrics(recs: np.ndarray, targets: list, ks=(5, 10, 20)) -> pd.DataFrame:
    """Métricas por usuario. `recs` [n, >=max(ks)] con item_idx ordenados por score (0 = vacío)."""
    n = len(recs)
    max_k = max(ks)
    assert recs.shape[1] >= max_k, "recs debe tener al menos max(ks) columnas"
    hits = _hit_matrix(recs[:, :max_k], targets).astype(np.float64)
    n_rel = np.array([len(t) for t in targets], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    cum_hits = np.cumsum(hits, axis=1)
    prec_at = cum_hits / np.arange(1, max_k + 1)
    out = {}
    for k in ks:
        h = hits[:, :k]
        n_hit = h.sum(1)
        dcg = (h * discounts[:k]).sum(1)
        ideal_len = np.minimum(n_rel, k).astype(int)
        idcg = np.array([discounts[:m].sum() for m in ideal_len])
        ap = (prec_at[:, :k] * h).sum(1) / np.maximum(np.minimum(n_rel, k), 1)
        out[f"recall@{k}"] = n_hit / n_rel
        out[f"precision@{k}"] = n_hit / k
        out[f"ndcg@{k}"] = np.where(idcg > 0, dcg / idcg, 0.0)
        out[f"map@{k}"] = ap
        out[f"hit@{k}"] = (n_hit > 0).astype(np.float64)
    df = pd.DataFrame(out)
    df["n_relevant"] = n_rel
    assert len(df) == n
    return df


def intra_list_diversity(recs: np.ndarray, item_vecs: np.ndarray, k: int) -> np.ndarray:
    """1 - similitud coseno media entre pares de la lista, con vectores de contenido (independientes
    del modelo evaluado)."""
    v = item_vecs[recs[:, :k]]
    norm = np.linalg.norm(v, axis=2, keepdims=True)
    v = v / np.maximum(norm, 1e-9)
    sim = np.einsum("nid,njd->nij", v, v)
    kk = recs[:, :k].shape[1]
    off = (sim.sum(axis=(1, 2)) - np.trace(sim, axis1=1, axis2=2)) / (kk * (kk - 1))
    return 1.0 - off


def novelty(recs: np.ndarray, item_pop: np.ndarray, k: int) -> np.ndarray:
    """Autoinformación media -log2 p(i) de los ítems recomendados (mayor = menos populares)."""
    p = (item_pop + 1.0) / (item_pop.sum() + len(item_pop))
    return (-np.log2(p[recs[:, :k]])).mean(1)


def evaluate(recs: np.ndarray, evalset, ks=(5, 10, 20), n_candidates: int | None = None,
             content_vecs: np.ndarray | None = None, item_pop: np.ndarray | None = None
             ) -> tuple[dict, pd.DataFrame]:
    """Devuelve (métricas agregadas con segmentos, DataFrame por usuario)."""
    per_user = per_user_metrics(recs, evalset.targets, ks)
    per_user.insert(0, "user_id", evalset.user_ids)
    per_user["n_hist_pos"] = evalset.n_hist_pos
    per_user["has_user_id_embedding"] = evalset.user_idx > 0
    metric_cols = [c for c in per_user.columns if "@" in c]

    result = {"n_users": int(len(per_user)), "n_relevant_total": int(per_user["n_relevant"].sum()),
              "overall": {c: float(per_user[c].mean()) for c in metric_cols}, "segments": {}, "beyond_accuracy": {}}
    for name, fn in SEGMENTS.items():
        mask = fn(per_user["n_hist_pos"].to_numpy())
        if mask.any():
            result["segments"][name] = {"n_users": int(mask.sum()),
                                        **{c: float(per_user.loc[mask, c].mean()) for c in metric_cols
                                           if c.split("@")[0] in ("recall", "ndcg")}}
    for flag, label in ((True, "user_id_known"), (False, "user_id_unknown")):
        mask = (per_user["has_user_id_embedding"] == flag).to_numpy()
        if mask.any():
            result["segments"][label] = {"n_users": int(mask.sum()),
                                         **{c: float(per_user.loc[mask, c].mean()) for c in metric_cols
                                            if c.split("@")[0] in ("recall", "ndcg")}}
    for k in ks:
        ba = {}
        if n_candidates:
            ba["coverage"] = float(len(np.unique(recs[:, :k])) / n_candidates)
        if content_vecs is not None:
            div = intra_list_diversity(recs, content_vecs, k)
            per_user[f"diversity@{k}"] = div
            ba["diversity"] = float(div.mean())
        if item_pop is not None:
            nov = novelty(recs, item_pop, k)
            per_user[f"novelty@{k}"] = nov
            ba["novelty"] = float(nov.mean())
        result["beyond_accuracy"][f"@{k}"] = ba
    return result, per_user


def bootstrap_ci(values: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """IC 95 % de la media por bootstrap sobre usuarios."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
