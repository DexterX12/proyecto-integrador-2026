"""Job 2: protocolo temporal + features -> data/model_data/dataset_version=<v>/

Uso:  python -m jobs.features_job
"""
from __future__ import annotations

import argparse

from src.common.config import add_common_args, load_config
from src.common.lake import Lake
from src.common.runlog import run_log
from src.data.features import run_features


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    with run_log("features", lake, {"dataset_version": cfg.protocol.dataset_version,
                                    "protocol_digest": cfg.digest("protocol", "features")}) as run:
        run_features(cfg, lake, run)


if __name__ == "__main__":
    main()
