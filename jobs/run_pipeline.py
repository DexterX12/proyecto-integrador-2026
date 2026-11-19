"""Orquestación mínima:
ETL -> features -> entrenamiento -> evaluación -> recomendaciones -> publicación.

Cada etapa corre como un proceso independiente (equivalente a un job). Si una falla, el pipeline se
detiene y la versión publicada anterior permanece intacta.

Uso:
  python -m jobs.run_pipeline                       # todo, desde ETL
  python -m jobs.run_pipeline --from features       # reutiliza data/curated
  python -m jobs.run_pipeline --from evaluate --model-version <v>    # modelo ya entrenado
  python -m jobs.run_pipeline --set train.epochs=10
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

from src.common.config import PROJECT_ROOT

STAGES = ["etl", "features", "train", "evaluate", "recommend", "publish"]


def run(cmd: list[str]) -> str:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    out = []
    for line in proc.stdout:
        out.append(line)
        if "WARN" not in line and "Stage" not in line:
            print("   " + line.rstrip(), flush=True)
    proc.wait()
    print(f"   -> rc={proc.returncode} en {time.time() - t0:.0f}s", flush=True)
    if proc.returncode != 0:
        sys.exit(f"Etapa fallida: {' '.join(cmd)}")
    return "".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="etl", choices=STAGES)
    parser.add_argument("--to", dest="end", default="publish", choices=STAGES)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()
    py = sys.executable
    ov = ["--set", *args.overrides] if args.overrides else []
    version = args.model_version
    stages = STAGES[STAGES.index(args.start): STAGES.index(args.end) + 1]

    for stage in stages:
        if stage == "etl":
            run([py, "-m", "jobs.etl_job", *ov])
        elif stage == "features":
            run([py, "-m", "jobs.features_job", *ov])
        elif stage == "train":
            out = run([py, "-m", "jobs.train_job", *(["--tag", args.tag] if args.tag else []), *ov])
            version = re.search(r"MODEL_VERSION=(\S+)", out).group(1)
        elif stage in ("evaluate", "recommend", "publish"):
            if not version:
                sys.exit("Indique --model-version cuando no se ejecuta la etapa train.")
            if stage == "evaluate":
                run([py, "-m", "jobs.evaluate_job", "--model-version", version, "--splits", "validation", "test", *ov])
            elif stage == "recommend":
                run([py, "-m", "jobs.recommend_job", "--model-version", version, *ov])
            else:
                run([py, "-m", "jobs.publish_job", "--model-version", version, *ov])
    print(f"\nPipeline completo. Versión del modelo: {version}")


if __name__ == "__main__":
    main()

