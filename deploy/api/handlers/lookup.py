"""GET /v1/recommendations/{user_id}?n=10 y GET /v1/health.

Lee el Top-50 precalculado de DynamoDB (una tabla por versión del modelo; la versión activa está en SSM).
Si el usuario no existe devuelve el fallback por popularidad (cold start de usuario).
"""
from __future__ import annotations

import json
import os
import time

MAX_N = 50
_CACHE: dict = {}


def _version() -> str:
    now = time.time()
    if _CACHE.get("v") and now - _CACHE["t"] < 60:                       # caché de 60 s
        return _CACHE["v"]
    import boto3

    _CACHE["v"] = boto3.client("ssm").get_parameter(Name=os.environ["ACTIVE_VERSION_PARAM"])["Parameter"]["Value"]
    _CACHE["t"] = now
    return _CACHE["v"]


def _tables(version: str):
    import boto3

    ddb = boto3.resource("dynamodb")
    prefix = os.environ["TABLE_PREFIX"]
    return ddb.Table(f"{prefix}-recs-{version}"), ddb.Table(f"{prefix}-products")


def _enrich(products, asins_scores: list):
    if not asins_scores:
        return []
    import boto3

    keys = [{"parent_asin": a} for a, _ in asins_scores]
    resp = boto3.resource("dynamodb").batch_get_item(RequestItems={products.name: {"Keys": keys}})
    info = {i["parent_asin"]: i for i in resp["Responses"].get(products.name, [])}
    return [{"rank": r + 1, "parent_asin": a, "score": s, "title": info.get(a, {}).get("title"),
             "image_url": info.get(a, {}).get("image_url"), "price": str(info.get(a, {}).get("price", ""))}
            for r, (a, s) in enumerate(asins_scores)]


def _resp(code: int, body: dict) -> dict:
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def handler(event, _context=None):
    route = event.get("rawPath", "")
    version = _version()
    if route.endswith("/health"):
        return _resp(200, {"status": "ok", "model_version": version})
    user = (event.get("pathParameters") or {}).get("user_id")
    if not user:
        return _resp(400, {"error": "falta user_id"})
    try:
        n = max(1, min(int((event.get("queryStringParameters") or {}).get("n", 10)), MAX_N))
    except ValueError:
        return _resp(400, {"error": "n debe ser un entero"})
    recs_table, products = _tables(version)
    item = recs_table.get_item(Key={"user_id": user}).get("Item")
    if not item:
        fb = products.get_item(Key={"parent_asin": "__popularity_fallback__"}).get("Item")
        pairs = json.loads(fb["recs"])[:n] if fb else []
        return _resp(200, {"source": "popularity_fallback", "model_version": version, "user_id": user,
                           "items": _enrich(products, pairs)})
    pairs = json.loads(item["recs"])[:n]
    return _resp(200, {"source": "precomputed", "model_version": version, "user_id": user,
                       "generated_at": item.get("generated_at"), "items": _enrich(products, pairs)})
