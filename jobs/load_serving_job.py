"""Job 7 (AWS): carga las recomendaciones de una versión en DynamoDB y la activa en SSM.

Qué escribe (el contrato que lee deploy/api/handlers/lookup.py):
  - Tabla `<prefix>-recs-<version>`  (clave user_id): un ítem por usuario con `recs` = JSON [[parent_asin, score], ...]
  - Tabla `<prefix>-products`        (clave parent_asin): título, imagen y precio SOLO de los productos recomendados,
                                     más el ítem `__popularity_fallback__` para usuarios sin recomendaciones.
  - Parámetro SSM `ACTIVE_VERSION_PARAM` = versión (al final y solo si la verificación por muestreo pasó).
Una tabla por versión permite el rollback repuntando el parámetro, sin recargar nada.

Uso:  python -m jobs.load_serving_job --model-version <v> [--no-activate]
Entorno: TABLE_PREFIX (p. ej. fashrec-dev), ACTIVE_VERSION_PARAM (p. ej. /fashrec-dev/active_version),
         FASHREC_DATA_ROOT (donde están recommendations/ y curated/products/), credenciales y región de AWS.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.config import add_common_args, load_config
from src.common.lake import Lake
from src.common.runlog import run_log

FALLBACK_KEY = "__popularity_fallback__"
CHUNK = 1000                      # ítems por tarea de escritura
SAMPLE_CHECK = 25                 # usuarios que se releen de DynamoDB antes de activar la versión


def user_blocks(user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inicio y fin de cada bloque contiguo de un mismo usuario (recommend_job escribe ordenado por usuario)."""
    change = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(user_ids)]])
    if len(set(user_ids[starts].tolist())) != len(starts):
        raise ValueError("recommendations.parquet no está agrupado por usuario")
    return starts, ends


def build_user_items(recs_path, version: str) -> tuple[list[dict], set[str], str | None]:
    pf = pq.ParquetFile(recs_path)
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}
    tbl = pf.read(columns=["user_id", "rank", "parent_asin", "score"])
    users = tbl.column("user_id").to_numpy(zero_copy_only=False)
    rank = tbl.column("rank").to_numpy()
    asin = tbl.column("parent_asin").to_numpy(zero_copy_only=False)
    score = tbl.column("score").to_numpy()
    starts, ends = user_blocks(users)
    items, asins = [], set()
    for s, e in zip(starts, ends):
        if e - s > 1 and not (np.diff(rank[s:e]) > 0).all():
            raise ValueError(f"ranks no ascendentes para el usuario {users[s]}")
        pairs = [[asin[i], round(float(score[i]), 6)] for i in range(s, e)]
        items.append({"user_id": users[s], "recs": json.dumps(pairs), "model_version": version,
                      "generated_at": meta.get("generated_at", "")})
        asins.update(asin[s:e].tolist())
    return items, asins, meta.get("generated_at")


def product_items(products_dir, asins: set[str]) -> list[dict]:
    df = pd.read_parquet(products_dir, columns=["parent_asin", "title", "image_url", "price"],
                         filters=[("parent_asin", "in", sorted(asins))])
    out = []
    for r in df.itertuples(index=False):
        item = {"parent_asin": r.parent_asin}
        if isinstance(r.title, str):
            item["title"] = r.title
        if isinstance(r.image_url, str):
            item["image_url"] = r.image_url
        if r.price is not None and not pd.isna(r.price):
            item["price"] = Decimal(str(round(float(r.price), 2)))
        out.append(item)
    return out


def ensure_table(ddb, name: str, key: str) -> bool:
    """Crea la tabla (on-demand) si no existe. Devuelve True si la creó."""
    try:
        ddb.meta.client.describe_table(TableName=name)
        return False
    except ddb.meta.client.exceptions.ResourceNotFoundException:
        pass
    ddb.create_table(TableName=name, BillingMode="PAY_PER_REQUEST",
                     AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
                     KeySchema=[{"AttributeName": key, "KeyType": "HASH"}])
    ddb.meta.client.get_waiter("table_exists").wait(TableName=name, WaiterConfig={"Delay": 2, "MaxAttempts": 60})
    return True


def _write_chunk(table_name: str, key: str, chunk: list[dict]) -> int:
    import boto3

    table = boto3.session.Session().resource("dynamodb").Table(table_name)   # los recursos de boto3 no son thread-safe
    with table.batch_writer(overwrite_by_pkeys=[key]) as bw:
        for item in chunk:
            bw.put_item(Item=item)
    return len(chunk)


def put_all(table_name: str, key: str, items: list[dict], workers: int) -> int:
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(lambda c: _write_chunk(table_name, key, c), chunks))


def verify_sample(ddb, table_name: str, items: list[dict], n: int = SAMPLE_CHECK) -> None:
    table = ddb.Table(table_name)
    for item in random.Random(0).sample(items, min(n, len(items))):
        got = table.get_item(Key={"user_id": item["user_id"]}, ConsistentRead=True).get("Item")
        if not got or got["recs"] != item["recs"]:
            raise RuntimeError(f"verificación fallida: {item['user_id']} no coincide en {table_name}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--table-prefix", default=os.environ.get("TABLE_PREFIX"))
    parser.add_argument("--active-version-param", default=os.environ.get("ACTIVE_VERSION_PARAM"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-activate", action="store_true", help="Carga la versión pero no cambia la activa")
    args = parser.parse_args(argv)
    if not args.table_prefix or (not args.no_activate and not args.active_version_param):
        parser.error("faltan --table-prefix y --active-version-param (o TABLE_PREFIX y ACTIVE_VERSION_PARAM)")
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    version = args.model_version
    recs_table = f"{args.table_prefix}-recs-{version}"
    products_table = f"{args.table_prefix}-products"

    import boto3

    with run_log("load_serving", lake, {"model_version": version, "table": recs_table}) as run:
        rec_dir = lake.recommendations(version)
        items, asins, generated_at = build_user_items(rec_dir / "recommendations.parquet", version)
        fb = pd.read_parquet(rec_dir / "popularity_fallback.parquet").sort_values("rank")
        fb_asins = fb["parent_asin"].tolist()
        asins.update(fb_asins)
        run.metrics(users=len(items), distinct_products=len(asins), generated_at=generated_at)

        ddb = boto3.resource("dynamodb")
        run.metric("recs_table_created", ensure_table(ddb, recs_table, "user_id"))
        run.metric("products_table_created", ensure_table(ddb, products_table, "parent_asin"))

        prods = product_items(lake.curated("products"), asins)
        prods.append({"parent_asin": FALLBACK_KEY, "recs": json.dumps([[a, None] for a in fb_asins])})
        run.metric("products_written", put_all(products_table, "parent_asin", prods, args.workers))
        run.metric("users_written", put_all(recs_table, "user_id", items, args.workers))
        verify_sample(ddb, recs_table, items)
        run.metric("sample_verified", min(SAMPLE_CHECK, len(items)))

        if args.no_activate:
            run.metric("activated", False)
        else:
            boto3.client("ssm").put_parameter(Name=args.active_version_param, Value=version, Type="String",
                                              Overwrite=True)
            run.metric("activated", True)
    print(f"Versión {version} cargada en {recs_table}" + ("" if args.no_activate else " y activada."))


if __name__ == "__main__":
    main()
