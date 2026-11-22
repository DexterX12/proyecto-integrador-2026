"""Acceso de solo lectura a los datos de modelado (salida de features_job)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .features import load_context_matrix

FEATURE_FILES = ["product_type", "audience", "store", "price", "has_price", "text", "id_idx", "log_pop",
                 "in_universe", "in_universe_extended"]


@dataclass
class EvalSet:
    """Usuarios de evaluación con su contexto (historial / vistos) y los ítems relevantes."""
    split: str
    user_ids: np.ndarray
    user_idx: np.ndarray        # [n] índice de embedding de ID (0 = OOV)
    hist: np.ndarray            # [n, H] ítems positivos previos (0 = padding)
    n_hist_pos: np.ndarray
    seen: list                  # lista de arrays con TODOS los ítems vistos antes del corte
    targets: list               # lista de arrays con ítems relevantes en la ventana

    def __len__(self) -> int:
        return len(self.user_ids)

    def subset(self, idx: np.ndarray) -> "EvalSet":
        return EvalSet(self.split, self.user_ids[idx], self.user_idx[idx], self.hist[idx],
                       self.n_hist_pos[idx], [self.seen[i] for i in idx],
                       [self.targets[i] for i in idx] if self.targets else [])


class ModelData:
    def __init__(self, lake, cfg, check_digest: bool = True):
        self.dir = lake.model_data
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"No existe {self.dir}. Ejecute primero: python -m jobs.features_job")
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        expected = cfg.digest("protocol", "features")
        if check_digest and self.manifest["protocol_digest"] != expected:
            raise RuntimeError(
                "La configuración (protocol/features) cambió desde que se generó el dataset "
                f"({self.manifest['protocol_digest']} != {expected}). Regenere con jobs.features_job "
                "o use otro protocol.dataset_version.")
        self.H = int(self.manifest["history_len"])
        self.threshold = float(self.manifest["relevance_threshold"])

    @cached_property
    def items(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "mappings" / "items.parquet")

    @cached_property
    def users(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "mappings" / "users.parquet")

    @cached_property
    def features(self) -> dict[str, np.ndarray]:
        return {n: np.load(self.dir / "features" / f"{n}.npy") for n in FEATURE_FILES}

    @property
    def n_items(self) -> int:
        return int(self.manifest["catalog_items"])

    @cached_property
    def universe(self) -> np.ndarray:
        """item_idx de los candidatos (ítems con interacciones en train), ordenados."""
        return np.nonzero(self.features["in_universe"])[0].astype(np.int32)

    @cached_property
    def universe_extended(self) -> np.ndarray:
        return np.nonzero(self.features["in_universe_extended"])[0].astype(np.int32)

    @cached_property
    def asin_of(self) -> np.ndarray:
        """parent_asin por item_idx (posición 0 = vacío)."""
        return np.concatenate([[""], self.items["parent_asin"].to_numpy(dtype=object)])

    @cached_property
    def interactions(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "interactions_split.parquet")

    def train_examples(self, min_history_len: int = 1) -> dict[str, np.ndarray]:
        df = pd.read_parquet(self.dir / "train" / "examples.parquet")
        df = df[df["hist_len"] >= min_history_len]
        hist = np.stack(df["history"].to_numpy()).astype(np.int32)
        return {"user_idx": df["user_idx"].to_numpy(np.int64), "item_idx": df["item_idx"].to_numpy(np.int64),
                "hist": hist, "ts": df["ts"].to_numpy(), "hist_len": df["hist_len"].to_numpy()}

    def eval_set(self, split: str, extended: bool = False) -> EvalSet:
        df = pd.read_parquet(self.dir / split / "eval.parquet")
        col = "targets_ext" if extended else "targets"
        df = df[df[col].map(len) > 0].reset_index(drop=True)
        return EvalSet(
            split=split + ("+ext" if extended else ""),
            user_ids=df["user_id"].to_numpy(dtype=object),
            user_idx=df["user_idx"].to_numpy(np.int64),
            hist=load_context_matrix(df, self.H),
            n_hist_pos=df["n_hist_pos"].to_numpy(np.int64),
            seen=[np.asarray(s, dtype=np.int64) for s in df["seen"]],
            targets=[np.asarray(t, dtype=np.int64) for t in df[col]],
        )

