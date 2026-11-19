"""Registro de ejecuciones (observabilidad mínima): duración, estado y conteos por job.

Cada job escribe runs/<job>_<timestamp>.json. En AWS el mismo log iría a CloudWatch.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .lake import Lake, ensure


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


class Run:
    def __init__(self, job: str, lake: Lake, params: dict | None = None):
        self.job = job
        self.lake = lake
        self.record: dict = {
            "job": job,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": params or {},
            "metrics": {},
        }
        self._t0 = time.perf_counter()
        self.log = logging.getLogger(job)

    def metric(self, key: str, value) -> None:
        self.record["metrics"][key] = value
        self.log.info("%s = %s", key, value)

    def metrics(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.metric(k, v)

    def _write(self) -> Path:
        ensure(self.lake.runs)
        stamp = self.record["started_at"].replace(":", "").replace("-", "")
        path = self.lake.runs / f"{self.job}_{stamp}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.record, fh, indent=2, default=str)
        return path


@contextmanager
def run_log(job: str, lake: Lake, params: dict | None = None):
    setup_logging()
    run = Run(job, lake, params)
    run.log.info("Inicio del job '%s'", job)
    try:
        yield run
        run.record["status"] = "success"
    except BaseException as exc:
        run.record["status"] = "failed"
        run.record["error"] = f"{type(exc).__name__}: {exc}"
        run.record["traceback"] = traceback.format_exc()
        run.log.error("Job '%s' falló: %s", job, exc)
        raise
    finally:
        run.record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run.record["duration_s"] = round(time.perf_counter() - run._t0, 2)
        path = run._write()
        run.log.info("Job '%s' -> %s en %.1fs (log: %s)", job, run.record["status"],
                     run.record["duration_s"], path.name)
