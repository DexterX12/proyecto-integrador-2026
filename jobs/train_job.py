"""Job 3: entrena el Two-Tower, genera embeddings e índice FAISS -> data/artifacts/model_version=<v>/

Uso:  python -m jobs.train_job [--tag exp1] [--set train.epochs=10 model.embedding_dim=128]
"""
from __future__ import annotations

import argparse

from src.common.config import add_common_args, load_config
from src.common.lake import Lake
from src.common.runlog import run_log
from src.data.dataset import ModelData
from src.models.train import export_artifacts, new_model_version, train_two_tower


def run_tuning(cfg, lake, md, args) -> str:
    """Ajuste de hiperparámetros: solo usa validación; no genera artefactos versionados."""
    import json
    from src.common.lake import ensure
    label = args.tag or "run"
    with run_log(f"tune_{label}", lake, {"overrides": args.overrides}) as run:
        _, summary, history = train_two_tower(cfg, md, run)
        out = ensure(lake.reports / "sweeps")
        (out / f"{label}.json").write_text(json.dumps(
            {"label": label, "overrides": args.overrides, "summary": summary, "history": history}, indent=2))
    return label


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--tag", default=None, help="Sufijo opcional de la versión del modelo")
    parser.add_argument("--model-version", default=None,
                        help="Fija la versión (Step Functions la decide antes de lanzar las tareas); "
                             "sin este argumento se genera por fecha")
    parser.add_argument("--no-export", action="store_true",
                        help="Modo ajuste: entrena y guarda solo el resultado en reports/sweeps/<tag>.json")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    lake = Lake(cfg)
    md = ModelData(lake, cfg)
    if args.no_export:
        return run_tuning(cfg, lake, md, args)
    version = args.model_version or new_model_version(lake, args.tag)
    lake.artifacts(version).mkdir(parents=True, exist_ok=True)

    with run_log("train", lake, {"model_version": version, "overrides": args.overrides}) as run:
        run.metric("model_version", version)
        model, summary, history = train_two_tower(cfg, md, run, lake.artifacts(version) / "training_history.csv")
        run.metrics(**{f"best_{k}": v for k, v in summary.items()})
        config = export_artifacts(cfg, lake, md, model, version, summary, history)
        run.metric("artifacts", str(lake.artifacts(version)))
        run.metric("faiss_ntotal", config["faiss"]["ntotal"])
    print(f"MODEL_VERSION={version}")
    return version


if __name__ == "__main__":
    main()

