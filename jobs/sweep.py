"""Barrido de hiperparámetros en paralelo (usa SOLO validación; test no se toca).

Uso:
  python -m jobs.sweep --parallel 3 --threads 4 --grid configs/sweep_round1.yaml

El YAML es un dict  etiqueta -> lista de overrides ("train.logq_correction=true", ...).
Resultados: data/reports/sweeps/<etiqueta>.json  y  data/reports/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.common.config import PROJECT_ROOT, load_config
from src.common.lake import Lake, ensure


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base", nargs="*", default=[], help="Overrides comunes a todas las corridas")
    args = parser.parse_args(argv)
    grid: dict[str, list[str]] = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    lake = Lake(load_config())
    out = ensure(lake.reports / "sweeps")

    pending = list(grid.items())
    running: list[tuple[str, subprocess.Popen]] = []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.parallel:
            label, ov = pending.pop(0)
            cmd = [sys.executable, "-m", "jobs.train_job", "--no-export", "--tag", label, "--set",
                   f"train.num_threads={args.threads}", *args.base, *ov]
            log = open(out / f"{label}.log", "w", encoding="utf-8")
            running.append((label, subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)))
            print(f"[{time.time() - t0:5.0f}s] lanzado {label}: {ov}", flush=True)
        time.sleep(5)
        for label, proc in list(running):
            if proc.poll() is not None:
                running.remove((label, proc))
                print(f"[{time.time() - t0:5.0f}s] terminado {label} (rc={proc.returncode})", flush=True)

    rows = []
    for label, ov in grid.items():
        f = out / f"{label}.json"
        if not f.exists():
            rows.append({"label": label, "status": "failed"})
            continue
        r = json.loads(f.read_text())
        best = max(r["history"], key=lambda h: h["val_ndcg@20"])
        rows.append({"label": label, "best_epoch": best["epoch"], "val_recall@20": best["val_recall@20"],
                     "val_ndcg@20": best["val_ndcg@20"], "epochs_run": len(r["history"]),
                     "final_train_loss": r["history"][-1]["train_loss"], "overrides": " ".join(ov)})
    df = pd.DataFrame(rows).sort_values("val_ndcg@20", ascending=False)
    df.to_csv(lake.reports / "sweep_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
