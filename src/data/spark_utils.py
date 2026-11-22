"""Utilidades de Spark: sesión local y escritura de Parquet portable.

En Linux/EC2/Glue se usa la escritura nativa distribuida de Spark. En Windows, Hadoop exige
`winutils.exe` para escribir al sistema de archivos local; para no depender de un binario
externo se usa `DataFrame.toArrow()` y PyArrow (las transformaciones siguen siendo Spark).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def get_spark(cfg, app_name: str = "fashrec-etl"):
    from pyspark.sql import SparkSession

    local_tmp = Path(os.environ.get("FASHREC_SPARK_TMP", Path.home() / ".fashrec_spark_tmp"))
    local_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master(cfg.etl.spark_master)
        .appName(app_name)
        .config("spark.driver.memory", cfg.etl.driver_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.etl.shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.local.dir", str(local_tmp))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "200000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _native_write_available() -> bool:
    return sys.platform != "win32" or bool(os.environ.get("HADOOP_HOME"))


def write_parquet(df, path: Path) -> int:
    """Escribe un DataFrame de Spark como directorio Parquet (reemplaza el existente)."""
    import pyarrow.parquet as pq  # perezoso: el job de AWS Glue no lo necesita

    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    if _native_write_available():
        df.write.mode("overwrite").option("compression", "zstd").parquet(str(path))
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in path.glob("*.parquet"))
    else:
        table = df.toArrow()
        pq.write_table(table, path / "part-00000.parquet", compression="zstd", row_group_size=500_000)
        n = table.num_rows
    log.info("Parquet escrito: %s (%s filas)", path, f"{n:,}")
    return n
