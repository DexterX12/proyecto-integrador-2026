"""Recuperación Top-N. FAISS IndexFlatIP (exacto) y un recuperador exacto en PyTorch de referencia.

Ambos excluyen los ítems ya vistos por el usuario en el contexto. El índice FAISS no filtra por
consulta, así que se sobre-recupera `n + (vistos del usuario dentro del universo)` y se filtra.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _seen_rows(seen: list, row_of_item: np.ndarray) -> list[np.ndarray]:
    out = []
    for s in seen:
        r = row_of_item[np.asarray(s, dtype=np.int64)]
        out.append(np.sort(r[r >= 0]))
    return out


class FlatIPIndex:
    """Envoltorio del índice con el mapeo fila -> item_idx del universo."""

    def __init__(self, index: faiss.Index, item_ids: np.ndarray, normalize: bool = True):
        assert index.ntotal == len(item_ids)
        self.index = index
        self.normalize = normalize
        self.item_ids = np.asarray(item_ids, dtype=np.int32)
        self.row_of_item = np.full(int(self.item_ids.max()) + 2, -1, dtype=np.int64)
        self.row_of_item[self.item_ids] = np.arange(len(self.item_ids))

    @classmethod
    def build(cls, item_emb: np.ndarray, item_ids: np.ndarray, normalize: bool = True) -> "FlatIPIndex":
        emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        emb = l2_normalize(emb) if normalize else emb      # normalize=False => producto punto puro
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(index, item_ids, normalize)

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        np.save(directory / "faiss_item_ids.npy", self.item_ids)
        (directory / "faiss_meta.json").write_text(json.dumps({"normalize": self.normalize}), encoding="utf-8")
        return {"normalize": self.normalize, "ntotal": int(self.index.ntotal), "dim": int(self.index.d),
                "index_sha1": hashlib.sha1((directory / "faiss.index").read_bytes()).hexdigest()}

    @classmethod
    def load(cls, directory: Path) -> "FlatIPIndex":
        meta = json.loads((directory / "faiss_meta.json").read_text(encoding="utf-8"))
        return cls(faiss.read_index(str(directory / "faiss.index")), np.load(directory / "faiss_item_ids.npy"),
                   meta["normalize"])

    def reconstruct_all(self) -> np.ndarray:
        return self.index.reconstruct_n(0, self.index.ntotal)

    def search_excluding_seen(self, user_emb: np.ndarray, seen: list, n: int, batch: int = 4096
                              ) -> tuple[np.ndarray, np.ndarray]:
        """Top-n por usuario sin ítems vistos. Devuelve (item_idx [u, n], scores [u, n])."""
        q = np.ascontiguousarray(user_emb, dtype=np.float32)
        q = l2_normalize(q) if self.normalize else q
        seen_rows = _seen_rows([np.where(np.asarray(s) < len(self.row_of_item), s, 0) for s in seen],
                               self.row_of_item)
        n_users = len(q)
        out_ids = np.zeros((n_users, n), dtype=np.int32)
        out_sc = np.full((n_users, n), -np.inf, dtype=np.float32)
        order = np.argsort([len(s) for s in seen_rows], kind="stable")
        for start in range(0, n_users, batch):
            sel = order[start:start + batch]
            k = min(n + max(len(seen_rows[u]) for u in sel), self.index.ntotal)
            scores, rows = self.index.search(q[sel], k)
            for j, u in enumerate(sel):
                keep = ~np.isin(rows[j], seen_rows[u])
                r, s = rows[j][keep][:n], scores[j][keep][:n]
                out_ids[u, :len(r)] = self.item_ids[r]
                out_sc[u, :len(r)] = s
        return out_ids, out_sc


def exact_search_excluding_seen(user_emb: np.ndarray, item_emb: np.ndarray, item_ids: np.ndarray,
                                seen: list, n: int, row_of_item: np.ndarray | None = None,
                                chunk: int = 256, normalize: bool = True
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Referencia exacta con matmul de PyTorch (usada en validación por época y para auditar FAISS)."""
    import torch  # import perezoso: la imagen del dashboard no necesita PyTorch

    item_ids = np.asarray(item_ids, dtype=np.int32)
    if row_of_item is None:
        row_of_item = np.full(int(item_ids.max()) + 2, -1, dtype=np.int64)
        row_of_item[item_ids] = np.arange(len(item_ids))
    u = torch.from_numpy(l2_normalize(user_emb) if normalize else user_emb.astype(np.float32))
    v = torch.from_numpy(l2_normalize(item_emb) if normalize else item_emb.astype(np.float32))
    seen_rows = _seen_rows([np.where(np.asarray(s) < len(row_of_item), s, 0) for s in seen], row_of_item)
    out_ids = np.zeros((len(u), n), dtype=np.int32)
    out_sc = np.zeros((len(u), n), dtype=np.float32)
    for start in range(0, len(u), chunk):
        sc = u[start:start + chunk] @ v.T
        rr = np.concatenate([np.full(len(seen_rows[start + j]), j) for j in range(len(sc))])
        cc = np.concatenate([seen_rows[start + j] for j in range(len(sc))])
        if len(rr):
            sc[torch.from_numpy(rr), torch.from_numpy(cc)] = -torch.inf
        top_sc, top_rows = torch.topk(sc, n, dim=1)
        out_ids[start:start + chunk] = item_ids[top_rows.numpy()]
        out_sc[start:start + chunk] = top_sc.numpy()
    return out_ids, out_sc


def index_fingerprint(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


