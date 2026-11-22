"""Protocolo experimental y feature engineering (todo ajustado SOLO con train).

Reglas anti-leakage implementadas aquí:
  1. Partición temporal global por `timestamp` (train < val_start <= validation < test_start <= test).
  2. Mappings (ítems, usuarios, vocabularios), TF-IDF, SVD y normalizadores se ajustan con train.
  3. Historial de un ejemplo de entrenamiento = solo positivos con timestamp ESTRICTAMENTE anterior.
  4. Contexto de evaluación = interacciones anteriores al inicio de la ventana (train para validation,
     train+validation para test); los ítems de ese contexto se excluyen del ranking.
  5. Universo candidato = ítems con interacciones en train (documentado; los demás se cuentan).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .taxonomy import AUDIENCE_VOCAB, PRODUCT_TYPE_VOCAB

log = logging.getLogger(__name__)

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "validation", SPLIT_TEST: "test"}


def to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ----------------------------------------------------------------------------- k-core (diagnóstico)
def kcore_mask(users: np.ndarray, items: np.ndarray, k: int) -> np.ndarray:
    mask = np.ones(len(users), dtype=bool)
    nu, ni = users.max() + 1, items.max() + 1
    while True:
        uc = np.bincount(users[mask], minlength=nu)
        ic = np.bincount(items[mask], minlength=ni)
        new = mask & (uc[users] >= k) & (ic[items] >= k)
        if new.sum() == mask.sum():
            return new
        mask = new


def kcore_analysis(inter: pd.DataFrame, threshold: float, ks=(2, 3, 4, 5)) -> dict:
    """Cuánto del dataset sobrevive a un k-core. Justifica no aplicarlo como filtro principal."""
    u = pd.factorize(inter["user_id"])[0]
    i = inter["item_idx"].to_numpy()
    out = {"total": {"users": int(u.max() + 1), "items": int(len(np.unique(i))), "interactions": len(inter)}}
    for label, sel in (("all_ratings", np.ones(len(inter), bool)),
                       ("positives_only", (inter["rating"] >= threshold).to_numpy())):
        for k in ks:
            m = kcore_mask(u[sel], pd.factorize(i[sel])[0], k)
            kept_u, kept_i = u[sel][m], i[sel][m]
            out[f"{label}_k{k}"] = {
                "users": int(len(np.unique(kept_u))), "items": int(len(np.unique(kept_i))),
                "interactions": int(m.sum()),
                "pct_interactions_kept": round(100 * m.sum() / len(inter), 4),
            }
    return out


# ----------------------------------------------------------------------------- historiales
def _hist_matrix(users: np.ndarray, items: np.ndarray, ts: np.ndarray, H: int) -> np.ndarray:
    """Para filas ordenadas por (usuario, ts): ítems positivos previos (ts estrictamente menor)."""
    n = len(users)
    start = np.r_[True, users[1:] != users[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    rank = np.arange(n) - first
    hist = np.zeros((n, H), dtype=np.int32)
    for k in range(1, H + 1):
        idx = np.nonzero(rank >= k)[0]
        ok = ts[idx - k] < ts[idx]
        idx = idx[ok]
        hist[idx, H - k] = items[idx - k]
    return hist


def build_train_examples(inter: pd.DataFrame, user_map: dict, H: int, threshold: float,
                         include_empty: bool) -> pd.DataFrame:
    tr = inter[(inter["split"] == SPLIT_TRAIN) & (inter["rating"] >= threshold)]
    tr = tr.sort_values(["user_id", "ts", "item_idx"], kind="stable")
    codes = pd.factorize(tr["user_id"])[0]
    items = tr["item_idx"].to_numpy(dtype=np.int32)
    ts = tr["ts"].to_numpy()
    hist = _hist_matrix(codes, items, ts, H)
    hist_len = (hist > 0).sum(1).astype(np.int16)
    keep = np.ones(len(tr), bool) if include_empty else hist_len > 0
    user_idx = tr["user_id"].map(user_map).fillna(0).astype(np.int32).to_numpy()
    df = pd.DataFrame({
        "user_idx": user_idx[keep], "item_idx": items[keep], "ts": ts[keep],
        "hist_len": hist_len[keep], "history": list(hist[keep]),
    })
    return df


def build_context_frame(inter: pd.DataFrame, cutoff_ms: int, H: int, threshold: float,
                        user_map: dict, users: np.ndarray | None = None,
                        targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Contexto por usuario a la fecha `cutoff_ms`: historial positivo (últimos H) + ítems vistos.

    Solo usuarios con >= 1 positivo previo (sin historial no hay señal personalizada: cold start).
    """
    past = inter[inter["ts"] < cutoff_ms]
    if users is not None:
        past = past[past["user_id"].isin(users)]
    seen = past.groupby("user_id")["item_idx"].agg(lambda s: np.sort(s.to_numpy(np.int32)))
    pos = past[past["rating"] >= threshold].sort_values(["user_id", "ts", "item_idx"], kind="stable")
    n_pos = pos.groupby("user_id").size().rename("n_hist_pos")
    hist = pos.groupby("user_id").tail(H).groupby("user_id")["item_idx"].agg(
        lambda s: s.to_numpy(np.int32))
    frame = pd.concat([hist.rename("history"), n_pos], axis=1, join="inner")
    frame["seen"] = seen.reindex(frame.index)
    frame["n_seen"] = frame["seen"].map(len)
    frame.index.name = "user_id"
    frame = frame.reset_index()
    frame["user_idx"] = frame["user_id"].map(user_map).fillna(0).astype(np.int32)
    if targets is not None:
        tg = targets.groupby("user_id")["item_idx"].agg(lambda s: np.sort(s.to_numpy(np.int32)))
        frame = frame[frame["user_id"].isin(tg.index)].copy()
        frame["targets"] = tg.reindex(frame["user_id"]).to_numpy()
    return frame


def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def load_context_matrix(df: pd.DataFrame, H: int) -> np.ndarray:
    """list<int> -> matriz [n, H] rellena con 0 a la izquierda (orden cronológico)."""
    out = np.zeros((len(df), H), dtype=np.int32)
    for r, h in enumerate(df["history"].to_numpy()):
        h = np.asarray(h, dtype=np.int32)[-H:]
        if len(h):
            out[r, H - len(h):] = h
    return out


# ----------------------------------------------------------------------------- features de producto
def _text_column(products: pd.DataFrame, fields: list[str]) -> pd.Series:
    parts = [products[f].fillna("") for f in fields]
    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p
    return text.str.strip()


def build_item_features(products: pd.DataFrame, in_universe: np.ndarray, cfg) -> tuple[dict, dict]:
    """Devuelve (arrays indexados por item_idx con fila 0 = padding, transformadores ajustados)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    fcfg = cfg.features
    M = len(products)
    text = _text_column(products, list(fcfg.text_fields))

    tfidf = TfidfVectorizer(
        max_features=fcfg.tfidf_max_features, min_df=fcfg.tfidf_min_df,
        ngram_range=(1, fcfg.tfidf_ngram_max), sublinear_tf=True, strip_accents="unicode",
        stop_words="english", dtype=np.float32,
    )
    log.info("Ajustando TF-IDF sobre %s productos de train...", f"{in_universe.sum():,}")
    x_train = tfidf.fit_transform(text[in_universe])
    svd = TruncatedSVD(n_components=fcfg.svd_dim, random_state=fcfg.random_seed, n_iter=5)
    z_train = svd.fit_transform(x_train)
    z_mean, z_std = z_train.mean(0), z_train.std(0) + 1e-6
    log.info("Varianza explicada por SVD(%d): %.3f", fcfg.svd_dim, svd.explained_variance_ratio_.sum())
    z_all = svd.transform(tfidf.transform(text))
    z_all = np.clip((z_all - z_mean) / z_std, -5, 5).astype(np.float32)
    no_text = (text.str.len() == 0).to_numpy()
    z_all[no_text] = 0.0

    # categorías derivadas del título (vocabularios fijos; 0 = padding)
    ptype = products["product_type"].map({v: i + 1 for i, v in enumerate(PRODUCT_TYPE_VOCAB)}).fillna(
        len(PRODUCT_TYPE_VOCAB)).astype(np.uint8).to_numpy()
    aud = products["audience"].map({v: i + 1 for i, v in enumerate(AUDIENCE_VOCAB)}).fillna(
        len(AUDIENCE_VOCAB)).astype(np.uint8).to_numpy()

    # tienda: vocabulario con ítems de train; 1 = fuera de vocab, 2 = sin tienda
    store_norm = products["store"].str.lower().str.strip()
    counts = store_norm[in_universe].value_counts()
    counts = counts[counts >= fcfg.store_min_train_items].head(fcfg.store_vocab_size)
    store_vocab = {s: i + 3 for i, s in enumerate(counts.index)}
    store = store_norm.map(store_vocab).fillna(1).astype(np.int32)
    store[products["store"].isna()] = 2
    store = store.to_numpy()

    # precio: log1p + z-score (estadísticos de train) + indicador de faltante
    price = products["price"].to_numpy(dtype=np.float64)
    has_price = ~np.isnan(price)
    logp = np.log1p(np.where(has_price, price, 0.0))
    ref = logp[in_universe & has_price]
    p_mean, p_std = float(ref.mean()), float(ref.std() + 1e-6)
    price_z = np.where(has_price, np.clip((logp - p_mean) / p_std, -5, 5), 0.0).astype(np.float32)

    def pad(a):  # fila 0 = padding
        return np.concatenate([np.zeros((1,) + a.shape[1:], dtype=a.dtype), a])

    arrays = {
        "product_type": pad(ptype), "audience": pad(aud), "store": pad(store),
        "price": pad(price_z), "has_price": pad(has_price.astype(np.uint8)), "text": pad(z_all),
    }
    transformers = {
        "tfidf": tfidf, "svd": svd, "z_mean": z_mean, "z_std": z_std,
        "price_mean": p_mean, "price_std": p_std, "store_vocab": store_vocab,
        "text_fields": list(fcfg.text_fields),
        "product_type_vocab": PRODUCT_TYPE_VOCAB, "audience_vocab": AUDIENCE_VOCAB,
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
    }
    return arrays, transformers


def add_popularity_feature(arrays: dict, train_cnt: np.ndarray, in_universe: np.ndarray) -> None:
    """log(1 + nº de interacciones en train), estandarizado con ítems del universo. Solo usa train."""
    logc = np.log1p(train_cnt[1:].astype(np.float64))
    mu, sd = logc[in_universe].mean(), logc[in_universe].std() + 1e-6
    z = ((logc - mu) / sd).astype(np.float32)      # ítems sin train quedan en el mínimo (no en la media)
    arrays["log_pop"] = np.concatenate([[0.0], z]).astype(np.float32)


# ----------------------------------------------------------------------------- orquestación
def run_features(cfg, lake, run) -> dict:
    pcfg = cfg.protocol
    H = pcfg.history_len
    thr = pcfg.relevance_threshold
    val_start, test_start = to_ms(pcfg.val_start), to_ms(pcfg.test_start)

    inter = pd.read_parquet(lake.curated("interactions"), columns=["user_id", "parent_asin", "rating", "timestamp"])
    products = pd.read_parquet(lake.curated("products")).sort_values("parent_asin").reset_index(drop=True)
    run.metrics(interactions_loaded=len(inter), products_loaded=len(products))

    # item_idx estable: orden alfabético de parent_asin, 1..M (0 = padding/OOV)
    item_idx_map = pd.Series(np.arange(1, len(products) + 1, dtype=np.int32), index=products["parent_asin"])
    inter["item_idx"] = inter["parent_asin"].map(item_idx_map).astype(np.int32)
    assert inter["item_idx"].notna().all()
    inter = inter.rename(columns={"timestamp": "ts"}).drop(columns="parent_asin")
    inter["rating"] = inter["rating"].astype(np.float32)
    inter["split"] = np.where(inter["ts"] < val_start, SPLIT_TRAIN,
                              np.where(inter["ts"] < test_start, SPLIT_VAL, SPLIT_TEST)).astype(np.int8)
    inter = inter.sort_values(["user_id", "ts", "item_idx"], kind="stable").reset_index(drop=True)

    # --- diagnóstico k-core (informativo; justifica el protocolo)
    kc = kcore_analysis(inter, thr)
    lake.reports.mkdir(parents=True, exist_ok=True)
    (lake.reports / "kcore_analysis.json").write_text(json.dumps(kc, indent=2), encoding="utf-8")
    run.metric("kcore_3_positives_only_pct_kept", kc["positives_only_k3"]["pct_interactions_kept"])

    train = inter[inter["split"] == SPLIT_TRAIN]
    split_counts = {SPLIT_NAMES[s]: int((inter["split"] == s).sum()) for s in SPLIT_NAMES}
    run.metric("split_interactions", split_counts)

    # --- tabla de ítems (stats SOLO de train)
    M = len(products)
    train_cnt = np.bincount(train["item_idx"], minlength=M + 1)
    tp = train[train["rating"] >= thr]
    train_pos = np.bincount(tp["item_idx"], minlength=M + 1)
    rating_sum = np.bincount(train["item_idx"], weights=train["rating"], minlength=M + 1)
    in_universe = train_cnt[1:] >= pcfg.item_min_train_interactions
    has_id = train_cnt[1:] >= pcfg.item_id_min_train_interactions
    id_idx = np.zeros(M, dtype=np.int32)
    id_idx[has_id] = np.arange(1, has_id.sum() + 1, dtype=np.int32)
    items = pd.DataFrame({
        "item_idx": np.arange(1, M + 1, dtype=np.int32), "parent_asin": products["parent_asin"].to_numpy(),
        "id_idx": id_idx, "in_universe": in_universe,
        "train_count": train_cnt[1:].astype(np.int32), "train_pos_count": train_pos[1:].astype(np.int32),
        "train_mean_rating": np.where(train_cnt[1:] > 0, rating_sum[1:] / np.maximum(train_cnt[1:], 1), np.nan),
    })
    run.metrics(catalog_items=M, universe_items=int(in_universe.sum()), items_with_id_embedding=int(has_id.sum()))

    # --- usuarios con embedding de ID
    pos_per_user = tp.groupby("user_id").size()
    id_users = pos_per_user[pos_per_user >= pcfg.user_id_min_train_positives].index.sort_values()
    user_map = {u: i + 1 for i, u in enumerate(id_users)}
    users_df = pd.DataFrame({"user_idx": np.arange(1, len(id_users) + 1, dtype=np.int32), "user_id": id_users})
    run.metric("users_with_id_embedding", len(id_users))

    # --- features de producto (ajustadas con train)
    arrays, transformers = build_item_features(products, in_universe, cfg)
    add_popularity_feature(arrays, train_cnt, in_universe)
    arrays["id_idx"] = np.concatenate([[0], id_idx]).astype(np.int32)
    arrays["in_universe"] = np.concatenate([[False], in_universe])

    # --- ejemplos de entrenamiento
    train_ex = build_train_examples(inter, user_map, H, thr, include_empty=True)  # el trainer filtra por hist_len
    run.metrics(train_examples_all=len(train_ex), train_examples_with_history=int((train_ex["hist_len"] > 0).sum()))

    # --- conjuntos de evaluación (validation: contexto=train; test: contexto=train+validation)
    univ_full = arrays["in_universe"]
    eval_stats = {}
    # Universo extendido (solo test): ítems conocidos antes de la ventana = train + validation.
    # Permite medir cold start de ítems (ítems sin interacciones en train, con atributos de contenido).
    ext_universe = univ_full.copy()
    ext_universe[inter.loc[inter["split"] == SPLIT_VAL, "item_idx"].to_numpy()] = True
    for split, start, end in ((SPLIT_VAL, val_start, test_start), (SPLIT_TEST, test_start, None)):
        win = inter[(inter["ts"] >= start) & ((inter["ts"] < end) if end else True)]
        pos_win = win[win["rating"] >= thr]
        ext_mask = ext_universe if split == SPLIT_TEST else univ_full
        pos_in_univ = pos_win[univ_full[pos_win["item_idx"].to_numpy()]]
        pos_ext = pos_win[ext_mask[pos_win["item_idx"].to_numpy()]]
        ctx = build_context_frame(inter, start, H, thr, user_map, users=pos_ext["user_id"].unique(),
                                  targets=pos_ext)
        ctx = ctx.rename(columns={"targets": "targets_ext"})
        ctx["targets"] = [t[univ_full[t]] for t in ctx["targets_ext"]]   # protocolo principal
        n_main = ctx["targets"].map(len)
        users_with_targets = pos_in_univ["user_id"].nunique()
        eval_stats[SPLIT_NAMES[split]] = {
            "window_interactions": len(win), "window_positives": len(pos_win),
            "positives_outside_universe": int(len(pos_win) - len(pos_in_univ)),
            "positives_outside_extended_universe": int(len(pos_win) - len(pos_ext)),
            "users_with_targets_in_universe": int(users_with_targets),
            "eval_users": int((n_main > 0).sum()),
            "users_without_history_cold": int(users_with_targets - (n_main > 0).sum()),
            "eval_targets": int(n_main.sum()),
            "eval_users_extended": len(ctx), "eval_targets_extended": int(ctx["targets_ext"].map(len).sum()),
            "eval_users_with_user_id_embedding": int(((ctx["user_idx"] > 0) & (n_main > 0)).sum()),
        }
        save_frame(ctx, lake.model_data / SPLIT_NAMES[split] / "eval.parquet")
    arrays["in_universe_extended"] = ext_universe
    run.metric("eval_stats", eval_stats)

    # --- persistencia
    md = lake.model_data
    (md / "features").mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(md / "features" / f"{name}.npy", arr)
    (md / "mappings").mkdir(parents=True, exist_ok=True)
    items.to_parquet(md / "mappings" / "items.parquet", index=False)
    users_df.to_parquet(md / "mappings" / "users.parquet", index=False)
    joblib.dump(transformers, md / "mappings" / "transformers.joblib")
    save_frame(train_ex, md / "train" / "examples.parquet")
    inter_out = inter[["user_id", "item_idx", "rating", "ts", "split"]]
    save_frame(inter_out, md / "interactions_split.parquet")

    manifest = {
        "dataset_version": pcfg.dataset_version,
        "protocol_digest": cfg.digest("protocol", "features"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "relevance_threshold": thr, "history_len": H,
        "cutoffs": {"val_start": pcfg.val_start, "test_start": pcfg.test_start,
                    "val_start_ms": val_start, "test_start_ms": test_start},
        "split_interactions": split_counts,
        "catalog_items": M, "universe_items": int(in_universe.sum()),
        "users_total": int(inter["user_id"].nunique()), "users_with_id_embedding": len(id_users),
        "train_examples": len(train_ex), "train_examples_with_history": int((train_ex["hist_len"] > 0).sum()),
        "eval": eval_stats,
        "text_svd_explained_variance": transformers["svd_explained_variance"],
        "store_vocab_size": len(transformers["store_vocab"]),
        "price_stats": {"log_mean": transformers["price_mean"], "log_std": transformers["price_std"]},
        "train_max_ts": int(train["ts"].max()),
    }
    (md / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # invariantes de leakage verificadas en el momento de construir los datos
    assert manifest["train_max_ts"] < val_start
    assert (train_ex["ts"] < val_start).all()
    return manifest


