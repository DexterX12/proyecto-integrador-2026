"""POST /v1/recommend: Top-N a partir de un historial de productos, sin PyTorch.

Reproduce la torre de usuario con NumPy: media de los embeddings (normalizados) de los productos del
historial, concatenada con un vector de ID en cero (usuario sin ID), pasada por el MLP de 2 capas y
normalizada. La búsqueda es FAISS (IndexFlatIP) excluyendo los productos del historial.

El paquete (data/inference/model_version=<v>/) se descarga de S3 a /tmp en el arranque en frío.
Para pruebas locales: BUNDLE_DIR=<carpeta>.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

_STATE: dict = {}
MAX_HISTORY = 20
MAX_N = 50


def _download_bundle() -> str:
    local = os.environ.get("BUNDLE_DIR")
    if local:
        return local
    import boto3

    bucket, version = os.environ["DATA_BUCKET"], active_version()
    prefix = f"inference/model_version={version}/"
    dest = f"/tmp/bundle_{version}"
    os.makedirs(dest, exist_ok=True)
    s3 = boto3.client("s3")
    for name in ("faiss.index", "faiss_item_ids.npy", "faiss_meta.json", "user_mlp.npz", "asins.npy", "manifest.json"):
        target = os.path.join(dest, name)
        if not os.path.exists(target):
            s3.download_file(bucket, prefix + name, target)
    return dest


def active_version() -> str:
    if os.environ.get("MODEL_VERSION"):
        return os.environ["MODEL_VERSION"]
    import boto3

    return boto3.client("ssm").get_parameter(Name=os.environ["ACTIVE_VERSION_PARAM"])["Parameter"]["Value"]


def load_bundle(path: str) -> dict:
    import faiss

    index = faiss.read_index(os.path.join(path, "faiss.index"))
    item_ids = np.load(os.path.join(path, "faiss_item_ids.npy"))
    meta = json.load(open(os.path.join(path, "faiss_meta.json"), encoding="utf-8"))
    mlp = np.load(os.path.join(path, "user_mlp.npz"))
    asins = np.load(os.path.join(path, "asins.npy"))
    manifest = json.load(open(os.path.join(path, "manifest.json"), encoding="utf-8"))
    row_of_item = np.full(len(asins) + 2, -1, dtype=np.int64)       # tamaño catálogo + 2
    row_of_item[item_ids] = np.arange(len(item_ids))
    return {"index": index, "item_ids": item_ids, "normalize": bool(meta["normalize"]), "asins": asins,
            "row_of_item": row_of_item, "id_dim": int(mlp["id_dim"]), "cosine": bool(mlp["cosine"]),
            "w0": mlp["w0"], "b0": mlp["b0"], "w1": mlp["w1"], "b1": mlp["b1"], "manifest": manifest}


def _state() -> dict:
    if "bundle" not in _STATE:
        t0 = time.perf_counter()
        _STATE["bundle"] = load_bundle(_download_bundle())
        _STATE["load_s"] = time.perf_counter() - t0
    return _STATE["bundle"]


def history_rows(b: dict, history_asins: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Devuelve (filas FAISS del historial en el universo, item_idx del historial, asins desconocidos)."""
    q = np.array([a.encode() for a in history_asins], dtype="S10")
    pos = np.searchsorted(b["asins"], q)
    pos_c = np.minimum(pos, len(b["asins"]) - 1)
    known = (pos < len(b["asins"])) & (b["asins"][pos_c] == q)
    item_idx = pos[known] + 1
    unknown = [a for a, k in zip(history_asins, known) if not k]
    rows = b["row_of_item"][item_idx]
    return rows[rows >= 0], item_idx, unknown


def user_vector(b: dict, rows: np.ndarray) -> np.ndarray:
    h = np.stack([b["index"].reconstruct(int(r)) for r in rows]).mean(0)
    x = np.concatenate([np.zeros(b["id_dim"], dtype=np.float32), h]).astype(np.float32)
    x = np.maximum(b["w0"] @ x + b["b0"], 0.0)
    y = b["w1"] @ x + b["b1"]
    return (y / max(float(np.linalg.norm(y)), 1e-12)).astype(np.float32) if b["cosine"] else y.astype(np.float32)


def recommend(b: dict, history_asins: list[str], n: int) -> dict:
    history_asins = list(dict.fromkeys(history_asins))[-MAX_HISTORY:]
    rows, item_idx, unknown = history_rows(b, history_asins)
    if len(rows) == 0:
        return {"source": "no_usable_history", "items": [], "unknown_asins": unknown}
    vec = user_vector(b, rows)
    seen_rows = set(b["row_of_item"][item_idx][b["row_of_item"][item_idx] >= 0].tolist())
    scores, found = b["index"].search(vec[None], n + len(seen_rows))
    out = []
    for s, r in zip(scores[0], found[0]):
        if int(r) in seen_rows:
            continue
        out.append({"parent_asin": b["asins"][b["item_ids"][r] - 1].decode(), "score": round(float(s), 6)})
        if len(out) == n:
            break
    return {"source": "model", "items": out, "history_used": int(len(rows)), "unknown_asins": unknown}


def _resp(code: int, body: dict) -> dict:
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def handler(event, _context=None):
    try:
        body = json.loads(event.get("body") or "{}")
        history = body.get("history")
        if not isinstance(history, list) or not history or not all(isinstance(a, str) for a in history):
            return _resp(400, {"error": "'history' debe ser una lista no vacía de parent_asin"})
        n = max(1, min(int(body.get("n", 10)), MAX_N))
        b = _state()
        t0 = time.perf_counter()
        result = recommend(b, history[:200], n)
        result.update({"model_version": b["manifest"]["model_version"],
                       "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return _resp(200, result)
    except (ValueError, TypeError, json.JSONDecodeError):
        return _resp(400, {"error": "petición inválida: body JSON con 'history' (lista de parent_asin) y 'n' entero"})
