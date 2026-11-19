"""Acceso a datos para el dashboard: DuckDB sobre Parquet + JSON de artefactos.

La interfaz (Streamlit) solo importa este módulo; no contiene lógica de entrenamiento.
Cada consulta registra su tiempo, la versión de modelo usada y los errores en
`<data_root>/runs/serving_queries.jsonl`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..common.config import PROJECT_ROOT, load_config
from ..common.lake import Lake

log = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()


def _logged(fn):
    """Decorador: mide y registra cada consulta del repositorio."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            status, err = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._log_query(fn.__name__, (time.perf_counter() - t0) * 1000, status, err,
                            kwargs.get("model_version") or getattr(self, "_active_version", None))
    return wrapper


class Repository:
    def __init__(self, data_root: str | Path | None = None):
        cfg = load_config()
        if data_root is not None:
            os.environ["FASHREC_DATA_ROOT"] = str(data_root)
        self.cfg = cfg
        self.lake = Lake(cfg)
        self.root = self.lake.root
        self._local = threading.local()
        self.last_query_ms: dict[str, float] = {}
        self._active_version: str | None = None

    # ---- infraestructura ------------------------------------------------------------------------
    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if not hasattr(self._local, "con"):
            self._local.con = duckdb.connect()
        return self._local.con

    def _log_query(self, name: str, ms: float, status: str, err: str | None, version: str | None) -> None:
        self.last_query_ms[name] = ms
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query": name,
               "ms": round(ms, 1), "status": status, "model_version": version, "error": err}
        try:
            self.lake.runs.mkdir(parents=True, exist_ok=True)
            with _LOG_LOCK, open(self.lake.runs / "serving_queries.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:  # el log nunca debe tumbar la consulta
            pass

    @staticmethod
    def _sql_path(p: Path) -> str:
        return str(p).replace("\\", "/").replace("'", "''")

    def _read_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @property
    def md_dir(self) -> Path:
        return self.lake.model_data

    # ---- versiones y estado ---------------------------------------------------------------------
    def published(self) -> dict | None:
        return self._read_json(self.lake.published / "current.json")

    def publish_history(self) -> list:
        return self._read_json(self.lake.published / "history.json") or []

    def model_versions(self) -> list[str]:
        """Versiones con recomendaciones y evaluación completas (aptas para explorar)."""
        out = []
        for v in self.lake.list_model_versions():
            if (self.lake.recommendations(v) / "recommendations.parquet").exists() and \
               (self.lake.evaluation(v) / "metrics.json").exists():
                out.append(v)
        return out

    def default_version(self) -> str | None:
        pub = self.published()
        versions = self.model_versions()
        if pub and pub["model_version"] in versions:
            return pub["model_version"]
        return versions[-1] if versions else None

    def model_config(self, model_version: str) -> dict:
        return self._read_json(self.lake.artifacts(model_version) / "config.json") or {}

    def metrics(self, model_version: str) -> dict:
        return self._read_json(self.lake.evaluation(model_version) / "metrics.json") or {}

    def training_history(self, model_version: str) -> pd.DataFrame:
        p = self.lake.artifacts(model_version) / "training_history.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def recommendation_meta(self, model_version: str) -> dict:
        return self._read_json(self.lake.recommendations(model_version) / "validation.json") or {}

    def manifest(self) -> dict:
        return self._read_json(self.md_dir / "manifest.json") or {}

    def etl_report(self) -> dict:
        return self._read_json(self.lake.reports / "etl_quality.json") or {}

    def kcore_report(self) -> dict:
        return self._read_json(self.lake.reports / "kcore_analysis.json") or {}

    def runs(self) -> pd.DataFrame:
        rows = []
        for f in sorted(self.lake.runs.glob("*.json")):
            r = self._read_json(f)
            if r and not f.name.startswith("serving"):
                rows.append({"job": r["job"], "status": r["status"], "started_at": r["started_at"],
                             "duration_s": r.get("duration_s"), "file": f.name,
                             "error": r.get("error"), "metrics": r.get("metrics", {})})
        return pd.DataFrame(rows)

    def serving_log(self, n: int = 200) -> pd.DataFrame:
        p = self.lake.runs / "serving_queries.jsonl"
        if not p.exists():
            return pd.DataFrame()
        lines = p.read_text(encoding="utf-8").splitlines()[-n:]
        return pd.DataFrame([json.loads(x) for x in lines])

    # ---- tablas base ---------------------------------------------------------------------------
    def _products_sql(self) -> str:
        return f"read_parquet('{self._sql_path(self.lake.curated('products'))}/*.parquet')"

    def _items_sql(self) -> str:
        return f"read_parquet('{self._sql_path(self.md_dir / 'mappings' / 'items.parquet')}')"

    def _inter_sql(self) -> str:
        return f"read_parquet('{self._sql_path(self.md_dir / 'interactions_split.parquet')}')"

    def _recs_sql(self, v: str) -> str:
        return f"read_parquet('{self._sql_path(self.lake.recommendations(v) / 'recommendations.parquet')}')"

    # ---- overview ------------------------------------------------------------------------------
    @_logged
    def overview_stats(self) -> dict:
        row = self.con.execute(f"""
            select count(*) n, count(distinct user_id) users, count(distinct item_idx) items,
                   avg(rating) avg_rating, min(ts) tmin, max(ts) tmax
            from {self._inter_sql()}""").fetchone()
        return {"interactions": row[0], "users": row[1], "items": row[2], "avg_rating": row[3],
                "ts_min": row[4], "ts_max": row[5]}

    @_logged
    def rating_distribution(self) -> pd.DataFrame:
        return self.con.execute(f"select cast(rating as int) rating, count(*) n from {self._inter_sql()} "
                                "group by 1 order by 1").df()

    @_logged
    def reviews_per_month(self) -> pd.DataFrame:
        return self.con.execute(f"""
            select date_trunc('month', to_timestamp(ts/1000)) as "month", split, count(*) n
            from {self._inter_sql()} where ts >= 1262304000000 group by 1, 2 order by 1""").df()

    @_logged
    def interactions_per_entity(self) -> dict[str, pd.DataFrame]:
        u = self.con.execute(f"select c, count(*) n from (select count(*) c from {self._inter_sql()} "
                             "group by user_id) group by c order by c").df()
        i = self.con.execute(f"select c, count(*) n from (select count(*) c from {self._inter_sql()} "
                             "group by item_idx) group by c order by c").df()
        return {"users": u, "items": i}

    @_logged
    def category_distribution(self) -> pd.DataFrame:
        return self.con.execute(f"select product_type, audience, count(*) n from {self._products_sql()} "
                                "group by 1, 2").df()

    # ---- explorador de usuarios -------------------------------------------------------------------
    @_logged
    def list_users(self, model_version: str, search: str = "", only_with_targets: bool = False,
                   limit: int = 200) -> pd.DataFrame:
        p = self._sql_path(self.lake.recommendations(model_version) / "recommendation_users.parquet")
        where = ["1=1"]
        params: list = []
        if search:
            where.append("user_id ilike ?")
            params.append(f"%{search}%")
        if only_with_targets:
            where.append("n_test_positives > 0")
        return self.con.execute(
            f"select * from read_parquet('{p}') where {' and '.join(where)} "
            f"order by n_test_positives desc, n_hist_pos desc limit {int(limit)}", params).df()

    @_logged
    def user_history(self, user_id: str, model_version: str | None = None, limit: int = 15) -> pd.DataFrame:
        cut = self.manifest()["cutoffs"]["test_start_ms"]
        return self.con.execute(f"""
            select i.rating, to_timestamp(i.ts/1000) as date, it.parent_asin, p.title, p.product_type,
                   p.price, p.image_url, p.meta_average_rating as avg_rating
            from {self._inter_sql()} i
            join {self._items_sql()} it using (item_idx)
            left join {self._products_sql()} p on p.parent_asin = it.parent_asin
            where i.user_id = ? and i.ts < ?
            order by i.ts desc limit {int(limit)}""", [user_id, cut]).df()

    @_logged
    def user_future_positives(self, user_id: str, model_version: str | None = None) -> pd.DataFrame:
        cut = self.manifest()["cutoffs"]["test_start_ms"]
        thr = self.manifest()["relevance_threshold"]
        return self.con.execute(f"""
            select i.rating, to_timestamp(i.ts/1000) as date, it.parent_asin, p.title, p.image_url,
                   it.in_universe
            from {self._inter_sql()} i join {self._items_sql()} it using (item_idx)
            left join {self._products_sql()} p on p.parent_asin = it.parent_asin
            where i.user_id = ? and i.ts >= ? and i.rating >= ? order by i.ts""", [user_id, cut, thr]).df()

    @_logged
    def user_seen_asins(self, user_id: str) -> set[str]:
        cut = self.manifest()["cutoffs"]["test_start_ms"]
        df = self.con.execute(f"""select it.parent_asin from {self._inter_sql()} i
            join {self._items_sql()} it using (item_idx) where i.user_id = ? and i.ts < ?""",
                              [user_id, cut]).df()
        return set(df["parent_asin"])

    @_logged
    def recommendations(self, user_id: str, model_version: str, n: int = 10) -> pd.DataFrame:
        self._active_version = model_version
        return self.con.execute(f"""
            select r.rank, r.score, r.parent_asin, p.title, p.product_type, p.audience, p.store, p.price,
                   p.image_url, p.meta_average_rating as avg_rating, p.meta_rating_number as n_ratings
            from {self._recs_sql(model_version)} r
            left join {self._products_sql()} p on p.parent_asin = r.parent_asin
            where r.user_id = ? and r.rank <= ? order by r.rank""", [user_id, int(n)]).df()

    @_logged
    def popularity_fallback(self, model_version: str, n: int = 10) -> pd.DataFrame:
        p = self._sql_path(self.lake.recommendations(model_version) / "popularity_fallback.parquet")
        return self.con.execute(f"""
            select f.rank, p.parent_asin, p.title, p.product_type, p.audience, p.store, p.price, p.image_url,
                   p.meta_average_rating as avg_rating
            from read_parquet('{p}') f left join {self._products_sql()} p on p.parent_asin = f.parent_asin
            where f.rank <= ? order by f.rank""", [int(n)]).df()

    # ---- explorador de productos ---------------------------------------------------------------------
    @_logged
    def search_products(self, query: str, limit: int = 30) -> pd.DataFrame:
        q = query.strip()
        if not q:
            return pd.DataFrame()
        if len(q) == 10 and q.isalnum() and q.upper() == q:
            where, params = "parent_asin = ?", [q]
        else:
            where, params = "title ilike ?", [f"%{q}%"]
        return self.con.execute(f"""
            select p.parent_asin, p.title, p.product_type, p.audience, p.store, p.price, p.image_url,
                   p.meta_average_rating as avg_rating, it.train_count, it.in_universe
            from {self._products_sql()} p join {self._items_sql()} it using (parent_asin)
            where {where} order by it.train_count desc limit {int(limit)}""", params).df()

    @_logged
    def product(self, parent_asin: str) -> pd.Series | None:
        df = self.con.execute(f"""
            select p.*, it.item_idx, it.train_count, it.in_universe from {self._products_sql()} p
            join {self._items_sql()} it using (parent_asin) where p.parent_asin = ?""", [parent_asin]).df()
        return df.iloc[0] if len(df) else None

    @_logged
    def projection(self, model_version: str) -> pd.DataFrame:
        p = self.lake.artifacts(model_version) / "umap_projection.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        titles = self.con.execute(f"select parent_asin, title from {self._products_sql()} where parent_asin in "
                                  "(select unnest(?::varchar[]))", [df["parent_asin"].tolist()]).df()
        return df.merge(titles, on="parent_asin", how="left")

    def similar_items(self, model_version: str, item_idx: int, k: int = 10) -> pd.DataFrame:
        """Vecinos más cercanos con el índice FAISS del modelo (carga perezosa y en caché por proceso)."""
        from ..retrieval.faiss_index import FlatIPIndex
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            index = _index_cache(self.lake.artifacts(model_version))
            row = index.row_of_item[item_idx] if item_idx < len(index.row_of_item) else -1
            if row < 0:
                return pd.DataFrame()
            q = index.index.reconstruct(int(row))[None, :]
            scores, rows = index.index.search(q, k + 1)
            ids = index.item_ids[rows[0]]
            keep = ids != item_idx
            ids, scores = ids[keep][:k], scores[0][keep][:k]
            asins = self.con.execute(f"select item_idx, parent_asin from {self._items_sql()} "
                                     "where item_idx in (select unnest(?::int[]))", [ids.tolist()]).df()
            info = self.con.execute(f"""select parent_asin, title, product_type, price, image_url
                from {self._products_sql()} where parent_asin in (select unnest(?::varchar[]))""",
                                    [asins["parent_asin"].tolist()]).df()
            out = pd.DataFrame({"item_idx": ids, "similarity": scores}).merge(asins, on="item_idx").merge(
                info, on="parent_asin", how="left")
            return out.sort_values("similarity", ascending=False).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            status, err = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._log_query("similar_items", (time.perf_counter() - t0) * 1000, status, err, model_version)


_INDEX_CACHE: dict[str, object] = {}


def _index_cache(path: Path):
    from ..retrieval.faiss_index import FlatIPIndex
    key = str(path)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = FlatIPIndex.load(path)
    return _INDEX_CACHE[key]
