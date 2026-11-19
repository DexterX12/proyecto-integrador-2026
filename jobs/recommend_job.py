"""Job 5: Top-N precalculado por usuario con FAISS -> data/recommendations/model_version=<v>/

Uso:  python -m jobs.recommend_job --model-version <v> [--skip-umap]
"""
from __future__ import annotations

import argparse

from src.common.config import add_common_args, load_config
from src.common.lake import Lake
from src.common.runlog import run_log
from src.data.dataset import ModelData
from src.models.inference import compute_projection, generate_recommendations
from src.models.train import load_model


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    md = ModelData(lake, cfg)

    with run_log("recommend", lake, {"model_version": args.model_version}) as run:
        model, config = load_model(lake, md, args.model_version)
        validation = generate_recommendations(cfg, lake, md, model, config, args.model_version, run)
        if not args.skip_umap:
            run.metric("umap_points", compute_projection(lake, md, args.model_version))
    return validation


if __name__ == "__main__":
    main()
