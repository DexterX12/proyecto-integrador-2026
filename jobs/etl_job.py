"""Job 1: ingesta + ETL (PySpark) -> data/curated/{reviews,products,interactions}.

Uso:  python -m jobs.etl_job [--config ...] [--set etl.spark_master=local[4]]
"""
from __future__ import annotations

import argparse

from src.common.config import add_common_args, load_config
from src.common.lake import Lake
from src.common.runlog import run_log
from src.data.ingestion import build_raw_manifest
from src.data.preprocessing import run_etl
from src.data.spark_utils import get_spark


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--recompute-manifest", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)

    with run_log("etl", lake, {"dataset_version": cfg.protocol.dataset_version}) as run:
        manifest = build_raw_manifest(
            {"reviews": lake.raw("raw_reviews"), "metadata": lake.raw("raw_metadata")},
            lake.reports / "raw_manifest.json", recompute=args.recompute_manifest,
        )
        run.metric("raw_lines", {k: v["lines"] for k, v in manifest["files"].items()})
        spark = get_spark(cfg)
        try:
            run_etl(spark, cfg, lake, run)
        finally:
            spark.stop()


if __name__ == "__main__":
    main()
