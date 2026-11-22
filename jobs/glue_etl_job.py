"""Job de AWS Glue (PySpark 3.5, Glue 5.0): JSONL en S3 raw/ -> Parquet en S3 curated/.

Reutiliza `src.data.preprocessing.run_etl` (el mismo código del ETL local; la limpieza y los controles de
calidad no se duplican). Solo cambian el origen (s3://), el destino (escritura nativa distribuida de Spark,
`interactions` y `reviews` particionadas por `review_year`) y el registro de la ejecución (runs/ en S3).

Se publica con deploy/deploy.ps1:
    s3://<bucket>/glue-scripts/glue_etl_job.py     este archivo
    s3://<bucket>/glue-scripts/fashrec_src.zip     carpeta src/ (job argument --extra-py-files)
    s3://<bucket>/glue-scripts/default.yaml        configs/default.yaml (job argument --extra-files)
Argumentos del job (los fija deploy/pipeline/serverless.yml): --JOB_NAME y --DATA_BUCKET.

Salidas en S3:
    curated/{products,reviews,interactions}/   (reviews e interactions con particiones review_year=YYYY)
    reports/etl_quality.json, reports/raw_manifest.json, runs/etl_<fecha>.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RAW_REVIEWS_KEY = "raw/amazon-fashion/reviews/Amazon_Fashion.jsonl"
RAW_METADATA_KEY = "raw/amazon-fashion/metadata/meta_Amazon_Fashion.jsonl"
PARTITIONED = {"interactions", "reviews"}          # tablas con partición review_year (ver deploy/platform)


class S3Lake:
    """Solo lo que `run_etl` necesita de un Lake: las rutas de los archivos RAW."""

    def __init__(self, bucket: str):
        self.bucket = bucket

    def raw(self, key: str) -> str:
        return {"raw_reviews": f"s3://{self.bucket}/{RAW_REVIEWS_KEY}",
                "raw_metadata": f"s3://{self.bucket}/{RAW_METADATA_KEY}"}[key]


class S3Sink:
    """Destino de `run_etl` en S3: Parquet con zstd escrito por los ejecutores; el reporte va como JSON."""

    def __init__(self, bucket: str, base_uri: str | None = None):
        self.bucket = bucket
        self.base_uri = base_uri or f"s3://{bucket}"       # base_uri solo se cambia para probar sin S3

    def write(self, df, table: str) -> None:
        from pyspark.sql import functions as F

        uri = f"{self.base_uri}/curated/{table}"
        if table in PARTITIONED:
            # repartition por año: un archivo por partición en vez de cientos de archivos pequeños
            df = df.withColumn("review_year", F.year(F.timestamp_millis(F.col("timestamp")))).repartition("review_year")
            df.write.mode("overwrite").option("compression", "zstd").partitionBy("review_year").parquet(uri)
        else:
            df.coalesce(4).write.mode("overwrite").option("compression", "zstd").parquet(uri)

    def save_report(self, report: dict) -> str:
        return put_json(self.bucket, "reports/etl_quality.json", report)


def put_json(bucket: str, key: str, payload: dict) -> str:
    import boto3

    boto3.client("s3").put_object(Bucket=bucket, Key=key, ContentType="application/json",
                                  Body=json.dumps(payload, indent=2, default=str).encode("utf-8"))
    return f"s3://{bucket}/{key}"


def find_config(sc=None) -> Path:
    """Localiza default.yaml. `--extra-files` lo copia al directorio de trabajo del driver; como respaldo
    se consulta SparkFiles, que EXIGE un SparkContext ya creado (si no, lanza AssertionError)."""
    candidates = [Path("default.yaml"), Path.cwd() / "default.yaml"]
    if sc is not None:
        try:
            from pyspark import SparkFiles
            candidates.append(Path(SparkFiles.get("default.yaml")))
        except Exception:  # noqa: BLE001  (SparkFiles puede fallar según el modo de ejecución)
            pass
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"default.yaml no llegó al job (argumento --extra-files). Buscado en: {candidates}")


def main() -> None:
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    from src.common.config import load_config
    from src.common.runlog import Run, setup_logging
    from src.data.ingestion import build_raw_manifest_s3
    from src.data.preprocessing import run_etl

    args = getResolvedOptions(sys.argv, ["JOB_NAME", "DATA_BUCKET"])
    bucket = args["DATA_BUCKET"]
    setup_logging()

    sc = SparkContext.getOrCreate()             # SparkFiles.get() en find_config() necesita el contexto ya creado
    glue = GlueContext(sc)
    spark = glue.spark_session
    job = Job(glue)
    job.init(args["JOB_NAME"], args)
    cfg = load_config(find_config(sc))
    for key, value in {"spark.sql.session.timeZone": "UTC", "spark.sql.ansi.enabled": "false",
                       "spark.sql.shuffle.partitions": str(cfg.etl.shuffle_partitions)}.items():
        spark.conf.set(key, value)

    class S3Run(Run):                                # mismo formato que runs/*.json del pipeline local
        def _write(self):
            stamp = self.record["started_at"].replace(":", "").replace("-", "")
            return put_json(bucket, f"runs/{self.job}_{stamp}.json", self.record)

    run = S3Run("etl", None, {"dataset_version": cfg.protocol.dataset_version, "engine": "glue",
                              "job_run": args["JOB_NAME"]})
    t0 = time.perf_counter()
    try:
        lake = S3Lake(bucket)
        report = run_etl(spark, cfg, lake, run, sink=S3Sink(bucket))
        manifest = build_raw_manifest_s3(
            {"reviews": lake.raw("raw_reviews"), "metadata": lake.raw("raw_metadata")},
            {"reviews": report["raw"]["review_lines"], "metadata": report["raw"]["metadata_lines"]})
        put_json(bucket, "reports/raw_manifest.json", manifest)
        run.record["status"] = "success"
    except BaseException as exc:  # noqa: BLE001
        run.record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        run.record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run.record["duration_s"] = round(time.perf_counter() - t0, 2)
        run._write()
    job.commit()


if __name__ == "__main__":
    main()
