"""ETL batch en PySpark: JSONL RAW -> Parquet CURATED (reviews, products, interactions).

El código es portable a AWS Glue: solo cambia la sesión de Spark y las rutas (s3://...).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .spark_utils import write_parquet
from .taxonomy import AUDIENCES, PRODUCT_TYPES, OTHER_TYPE, UNKNOWN_AUDIENCE, spark_case_expr

log = logging.getLogger(__name__)

MIN_VALID_TS_MS = 946_684_800_000  # 2000-01-01
REVIEW_TEXT_MAX = 1000

REVIEW_SCHEMA = T.StructType([
    T.StructField("rating", T.DoubleType()),
    T.StructField("title", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("asin", T.StringType()),
    T.StructField("parent_asin", T.StringType()),
    T.StructField("user_id", T.StringType()),
    T.StructField("timestamp", T.LongType()),
    T.StructField("helpful_vote", T.LongType()),
    T.StructField("verified_purchase", T.BooleanType()),
])

# price se lee como string y se convierte con try_cast: en otras categorías de Amazon
# hay valores no numéricos y un cast directo rompería el job.
PRODUCT_SCHEMA = T.StructType([
    T.StructField("main_category", T.StringType()),
    T.StructField("title", T.StringType()),
    T.StructField("average_rating", T.DoubleType()),
    T.StructField("rating_number", T.LongType()),
    T.StructField("features", T.ArrayType(T.StringType())),
    T.StructField("description", T.ArrayType(T.StringType())),
    T.StructField("price", T.StringType()),
    T.StructField("images", T.ArrayType(T.StructType([
        T.StructField("thumb", T.StringType()),
        T.StructField("large", T.StringType()),
        T.StructField("variant", T.StringType()),
        T.StructField("hi_res", T.StringType()),
    ]))),
    T.StructField("store", T.StringType()),
    T.StructField("parent_asin", T.StringType()),
])


def _blank_to_null(col):
    trimmed = F.trim(F.regexp_replace(col, r"\s+", " "))
    return F.when(trimmed == "", F.lit(None)).otherwise(trimmed)


# ----------------------------------------------------------------------------- reseñas
def read_reviews(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.schema(REVIEW_SCHEMA).option("mode", "DROPMALFORMED").json(str(path))
    )


def normalize_reviews(raw: DataFrame, now_ms: int) -> tuple[DataFrame, DataFrame]:
    """Devuelve (reseñas válidas, reseñas con motivo de rechazo)."""
    flagged = raw.withColumn(
        "reject_reason",
        F.when(F.col("user_id").isNull() | (F.trim(F.col("user_id")) == ""), "user_id_nulo")
        .when(F.col("parent_asin").isNull() | (F.trim(F.col("parent_asin")) == ""), "parent_asin_nulo")
        .when(F.col("rating").isNull() | (F.col("rating") < 1) | (F.col("rating") > 5), "rating_invalido")
        .when(F.col("timestamp").isNull() | (F.col("timestamp") < MIN_VALID_TS_MS)
              | (F.col("timestamp") > now_ms), "timestamp_invalido"),
    )
    valid = (
        flagged.filter(F.col("reject_reason").isNull())
        .drop("reject_reason")
        .withColumn("asin", F.coalesce(F.col("asin"), F.col("parent_asin")))
        .withColumn("title", _blank_to_null(F.col("title")))
        .withColumn("text", F.substring(_blank_to_null(F.col("text")), 1, REVIEW_TEXT_MAX))
        .withColumn("helpful_vote", F.coalesce(F.col("helpful_vote"), F.lit(0)))
        .withColumn("verified_purchase", F.coalesce(F.col("verified_purchase"), F.lit(False)))
        .withColumn("review_date", F.to_date(F.timestamp_millis(F.col("timestamp"))))
        .dropDuplicates(["user_id", "parent_asin", "timestamp", "rating"])
    )
    return valid, flagged.filter(F.col("reject_reason").isNotNull())


# ----------------------------------------------------------------------------- productos
def read_products(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.schema(PRODUCT_SCHEMA).option("mode", "DROPMALFORMED").json(str(path))


def normalize_products(raw: DataFrame, max_desc_chars: int) -> DataFrame:
    main_img = F.expr("filter(images, x -> x.variant = 'MAIN' and x.large is not null)")
    price = F.expr("try_cast(price as double)")
    df = (
        raw.filter(F.col("parent_asin").isNotNull() & (F.trim(F.col("parent_asin")) != ""))
        .withColumn("title", _blank_to_null(F.col("title")))
        .withColumn("store", _blank_to_null(F.col("store")))
        .withColumn("price", F.when((price > 0) & (price < 100000), price))
        .withColumn("features_text", F.substring(_blank_to_null(F.concat_ws(" | ", F.col("features"))),
                                                 1, max_desc_chars))
        .withColumn("description_text", F.substring(_blank_to_null(F.concat_ws(" ", F.col("description"))),
                                                    1, max_desc_chars))
        .withColumn("image_url", F.coalesce(main_img[0]["large"], F.col("images")[0]["large"],
                                            F.col("images")[0]["hi_res"]))
        .withColumn("image_hi_res", F.coalesce(main_img[0]["hi_res"], F.col("images")[0]["hi_res"]))
        .withColumn("n_images", F.coalesce(F.size(F.col("images")), F.lit(0)))
        .withColumn("product_type", spark_case_expr(F.col("title"), PRODUCT_TYPES, OTHER_TYPE))
        .withColumn("audience", spark_case_expr(F.col("title"), AUDIENCES, UNKNOWN_AUDIENCE))
    )
    # Un parent_asin debe ser único: se conserva el registro con más reseñas.
    w = Window.partitionBy("parent_asin").orderBy(F.col("rating_number").desc_nulls_last())
    df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
    return df.select(
        "parent_asin", "title", "store", "price", "product_type", "audience",
        "features_text", "description_text", "image_url", "image_hi_res", "n_images",
        F.col("average_rating").alias("meta_average_rating"),
        F.col("rating_number").alias("meta_rating_number"),
    )


# ----------------------------------------------------------------------------- interacciones
def build_interactions(reviews: DataFrame) -> DataFrame:
    """Una fila por (user_id, parent_asin): la reseña más reciente, con el nº de reseñas."""
    w = Window.partitionBy("user_id", "parent_asin")
    ranked = (
        reviews.withColumn("n_reviews", F.count(F.lit(1)).over(w))
        .withColumn("_rn", F.row_number().over(w.orderBy(F.col("timestamp").desc(), F.col("rating").desc())))
        .filter(F.col("_rn") == 1)
    )
    return ranked.select("user_id", "parent_asin", "rating", "timestamp", "n_reviews", "verified_purchase")


# ----------------------------------------------------------------------------- destinos de escritura
class LocalSink:
    """Escribe CURATED en el data lake local (data/curated) y el reporte en data/reports."""

    def __init__(self, lake):
        self.lake = lake

    def write(self, df: DataFrame, table: str) -> None:
        write_parquet(df, self.lake.curated(table))

    def save_report(self, report: dict) -> str:
        self.lake.reports.mkdir(parents=True, exist_ok=True)
        path = self.lake.reports / "etl_quality.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return str(path)


# ----------------------------------------------------------------------------- orquestación
def run_etl(spark: SparkSession, cfg, lake, run, sink=None) -> dict:
    """`lake` solo debe exponer `raw(key)`; `sink` decide dónde se escribe CURATED (local por defecto)."""
    t0 = time.time()
    raw_reviews = lake.raw("raw_reviews")
    raw_meta = lake.raw("raw_metadata")
    now_ms = int(time.time() * 1000)

    # --- reseñas
    reviews_raw = read_reviews(spark, raw_reviews).cache()  # una sola lectura del JSONL
    valid, rejected = normalize_reviews(reviews_raw, now_ms)

    # --- productos
    products = normalize_products(read_products(spark, raw_meta), cfg.etl.max_description_chars).cache()
    n_products = products.count()
    run.metric("products_curated", n_products)

    # --- join reseñas <-> productos (inner: toda reseña curada tiene metadata)
    prod_keys = products.select("parent_asin")
    reviews_joined = valid.join(F.broadcast(prod_keys), "parent_asin", "inner").select(
        "user_id", "parent_asin", "asin", "rating", "timestamp", "review_date",
        "title", "text", "helpful_vote", "verified_purchase",
    ).cache()
    n_reviews = reviews_joined.count()
    run.metric("reviews_curated", n_reviews)

    interactions = build_interactions(reviews_joined).cache()
    n_inter = interactions.count()
    run.metric("interactions_curated", n_inter)

    # --- controles de calidad (todo se calcula sobre datos ya cacheados)
    raw_lines = spark.read.text(str(raw_reviews)).count()
    meta_lines = spark.read.text(str(raw_meta)).count()
    parsed_reviews = reviews_raw.count()
    reject_counts = {r["reject_reason"]: r["count"] for r in rejected.groupBy("reject_reason").count().collect()}
    n_rejected = sum(reject_counts.values())
    n_after_dedup = valid.count()
    n_orphans = n_after_dedup - n_reviews

    stats = interactions.agg(
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("parent_asin").alias("items"),
        F.avg("rating").alias("avg_rating"),
        F.min("timestamp").alias("ts_min"),
        F.max("timestamp").alias("ts_max"),
        F.sum((F.col("rating") >= cfg.protocol.relevance_threshold).cast("int")).alias("positives"),
        F.sum((F.col("n_reviews") > 1).cast("int")).alias("pairs_with_multiple_reviews"),
    ).first().asDict()

    asin_mismatch = reviews_joined.filter(F.col("asin") != F.col("parent_asin")).count()
    prod_nulls = products.agg(
        *[F.avg(F.col(c).isNull().cast("double")).alias(c) for c in
          ["title", "store", "price", "image_url", "features_text", "description_text"]]
    ).first().asDict()
    type_dist = {r["product_type"]: r["count"] for r in products.groupBy("product_type").count().collect()}
    aud_dist = {r["audience"]: r["count"] for r in products.groupBy("audience").count().collect()}
    reviews_per_user = interactions.groupBy("user_id").count()
    single = reviews_per_user.filter(F.col("count") == 1).count()

    report = {
        "raw": {"review_lines": raw_lines, "review_rows_parsed": parsed_reviews,
                "review_malformed_dropped": raw_lines - parsed_reviews, "metadata_lines": meta_lines},
        "reviews": {
            "rejected_by_reason": reject_counts, "rejected_total": n_rejected,
            "exact_duplicates_removed": parsed_reviews - n_rejected - n_after_dedup,
            "orphans_without_metadata": n_orphans,
            "curated": n_reviews,
            "asin_differs_from_parent_asin": asin_mismatch,
        },
        "interactions": {
            "curated": n_inter, "users": stats["users"], "items": stats["items"],
            "avg_rating": round(stats["avg_rating"], 4), "positives": stats["positives"],
            "user_item_pairs_with_multiple_reviews": stats["pairs_with_multiple_reviews"],
            "users_with_single_interaction": single,
            "sparsity": 1 - n_inter / (stats["users"] * stats["items"]),
            "timestamp_min": stats["ts_min"], "timestamp_max": stats["ts_max"],
        },
        "products": {
            "curated": n_products, "null_rate": {k: round(v, 4) for k, v in prod_nulls.items()},
            "product_type_distribution": dict(sorted(type_dist.items(), key=lambda kv: -kv[1])),
            "audience_distribution": dict(sorted(aud_dist.items(), key=lambda kv: -kv[1])),
        },
        "elapsed_prelim_s": round(time.time() - t0, 1),
    }
    # criterios de calidad que detienen el pipeline si fallan
    assert n_orphans == 0, f"{n_orphans} reseñas sin metadata"
    assert n_reviews > 0 and n_products > 0
    run.metric("rejected_reviews", n_rejected)
    run.metric("exact_duplicates_removed", report["reviews"]["exact_duplicates_removed"])

    # --- escritura CURATED
    sink = sink or LocalSink(lake)
    sink.write(products, "products")
    sink.write(reviews_joined, "reviews")
    sink.write(interactions, "interactions")
    run.metric("etl_quality_report", sink.save_report(report))
    return report
