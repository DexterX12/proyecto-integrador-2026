"""Modelos de referencia ajustados SOLO con interacciones de train.

Al inferir, todos los modelos reciben el mismo contexto (historial y vistos anteriores al corte).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger(__name__)


def _filter_seen(candidates: np.ndarray, seen: np.ndarray, n: int) -> np.ndarray:
    keep = candidates[~np.isin(candidates, seen)]
    return keep[:n]


class PopularityModel:
    """Ranking global: positivos en train (desc) y rating medio (desc) entre ítems con >= min_reviews."""

    def __init__(self, min_reviews: int = 5):
        self.min_reviews = min_reviews
        self.ranking: np.ndarray | None = None

    def fit(self, items: pd.DataFrame) -> "PopularityModel":
        df = items[items["in_universe"]].copy()
        df["eligible"] = df["train_count"] >= self.min_reviews
        df = df.sort_values(["eligible", "train_pos_count", "train_mean_rating", "item_idx"],
                            ascending=[False, False, False, True], kind="stable")
        self.ranking = df["item_idx"].to_numpy(np.int64)
        self.n_eligible = int(df["eligible"].sum())
        return self

    def recommend(self, seen: list, n: int) -> np.ndarray:
        out = np.zeros((len(seen), n), dtype=np.int32)
        for u, s in enumerate(seen):
            top = self.ranking[: n + len(s)]
            rec = _filter_seen(top, np.asarray(s), n)
            out[u, :len(rec)] = rec
        return out


class ItemKNN:
    """Filtrado colaborativo item-based con coseno sobre positivos de train (con shrink)
    y relleno por popularidad cuando un usuario no alcanza N candidatos con score > 0."""

    def __init__(self, neighbors: int = 100, shrink: float = 10.0, max_history: int = 20):
        self.neighbors = neighbors
        self.shrink = shrink
        self.max_history = max_history

    def fit(self, train_pos: pd.DataFrame, n_items: int, popularity: PopularityModel,
            universe_mask: np.ndarray) -> "ItemKNN":
        self.pop = popularity
        self.n_items = n_items
        self.universe_mask = universe_mask
        u = pd.factorize(train_pos["user_id"])[0]
        i = train_pos["item_idx"].to_numpy()
        counts = np.bincount(u)
        keep = counts[u] >= 2          # usuarios con 1 positivo no aportan co-ocurrencias
        u, i = u[keep], i[keep]
        u = pd.factorize(u)[0]
        x = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(u.max() + 1, n_items + 1))
        x.data[:] = 1.0
        co = (x.T @ x).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        norm = np.sqrt(np.asarray(x.sum(0)).ravel()).astype(np.float32)
        coo = co.tocoo()
        sim = coo.data / (norm[coo.row] * norm[coo.col] + self.shrink)
        s = sp.csr_matrix((sim.astype(np.float32), (coo.row, coo.col)), shape=co.shape)
        self.sim = self._prune(s)
        log.info("ItemKNN: %s ítems con vecinos, nnz=%s", f"{(np.diff(self.sim.indptr) > 0).sum():,}",
                 f"{self.sim.nnz:,}")
        return self

    def _prune(self, s: sp.csr_matrix) -> sp.csr_matrix:
        k = self.neighbors
        lens = np.diff(s.indptr)
        big = np.nonzero(lens > k)[0]
        data, indices, indptr = s.data.copy(), s.indices, s.indptr
        mask = np.ones(len(data), dtype=bool)
        for r in big:
            a, b = indptr[r], indptr[r + 1]
            drop = np.argpartition(-data[a:b], k)[k:]
            mask[a + drop] = False
        rows = np.repeat(np.arange(s.shape[0]), lens)
        return sp.csr_matrix((data[mask], (rows[mask], indices[mask])), shape=s.shape)

    def recommend(self, hist: np.ndarray, seen: list, n: int) -> np.ndarray:
        h = hist[:, -self.max_history:]
        rows = np.repeat(np.arange(len(h)), h.shape[1])
        vals = (h.ravel() > 0).astype(np.float32)
        hm = sp.csr_matrix((vals, (rows, h.ravel())), shape=(len(h), self.n_items + 1))
        hm.eliminate_zeros()
        scores = (hm @ self.sim).tocsr()
        out = np.zeros((len(h), n), dtype=np.int32)
        self.n_backfilled = 0
        for u in range(len(h)):
            a, b = scores.indptr[u], scores.indptr[u + 1]
            cand, sc = scores.indices[a:b], scores.data[a:b]
            ok = self.universe_mask[cand] & ~np.isin(cand, seen[u])
            cand, sc = cand[ok], sc[ok]
            top = cand[np.argsort(-sc, kind="stable")[:n]]
            if len(top) < n:
                self.n_backfilled += 1
                extra = self.pop.ranking[: n + len(seen[u]) + len(top)]
                extra = extra[~np.isin(extra, seen[u]) & ~np.isin(extra, top)]
                top = np.concatenate([top, extra[: n - len(top)]])
            out[u, :len(top)] = top
        return out
