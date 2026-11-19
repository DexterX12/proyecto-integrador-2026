"""Job 4: evalúa baselines y Two-Tower con el mismo protocolo -> data/evaluation/model_version=<v>/

Durante el ajuste de hiperparámetros se usa solo validación:  --splits validation
La evaluación final en test debe ejecutarse una vez:        --splits validation test

Uso:  python -m jobs.evaluate_job --model-version 20261116_1614 --splits validation test
"""
from __future__ import annotations

import argparse
import json

from src.common.config import add_common_args, load_config
from src.common.lake import Lake
from src.common.runlog import run_log
from src.data.dataset import ModelData
from src.evaluation.evaluate import comparison_table, evaluate_all, save_results


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-version", default=None, help="Sin versión: solo baselines")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["validation", "test"])
    parser.add_argument("--no-save", action="store_true", help="Solo imprime la tabla (no escribe archivos)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    md = ModelData(lake, cfg)

    with run_log("evaluate", lake, {"model_version": args.model_version, "splits": args.splits}) as run:
        results = evaluate_all(cfg, lake, md, args.model_version, args.splits, run)
        for split in args.splits:
            print(f"\n=== {split} ===")
            print(comparison_table(results, split).round(4).to_string(index=False))
        if not args.no_save:
            version = args.model_version or "baselines_only"
            save_results(lake, version, results)
            run.metric("evaluation_dir", str(lake.evaluation(version)))
    return results


if __name__ == "__main__":
    main()
