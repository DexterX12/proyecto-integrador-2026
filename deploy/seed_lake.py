"""Sube a S3 los resultados YA calculados en local para tener evidencia en AWS sin ejecutar el pipeline completo.

Reproduce en el bucket el layout del data lake local (data/): curated/, model_data/, artifacts/, recommendations/,
evaluation/, published/, reports/, runs/ e inference/. Además:
  - re-particiona curated/interactions por review_year (el layout que espera la tabla de Athena y que escribe Glue)
  - copia recommendations.parquet a recommendations/model_version=<v>/recs/ (carpeta de la tabla de Athena)
  - genera el paquete de inferencia (jobs.export_inference_bundle) si todavía no existe
No sube los JSONL originales ni curated/reviews (no los usan ni la API ni el dashboard ni Athena).
Es idempotente: omite los objetos que ya existen con el mismo tamaño.

Uso:  python deploy/seed_lake.py --bucket <bucket> [--model-version <v>] [--data-root data] [--dry-run]
Después, jobs.load_serving_job carga DynamoDB y activa la versión (lo ejecuta deploy.ps1).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


class Uploader:
    def __init__(self, bucket: str, dry_run: bool, workers: int = 8):
        self.s3 = boto3.client("s3")
        self.bucket, self.dry_run, self.workers = bucket, dry_run, workers
        self.uploaded = self.skipped = self.bytes = 0

    def _existing(self, prefix: str) -> dict[str, int]:
        found: dict[str, int] = {}
        if self.dry_run:                       # la simulación no necesita credenciales
            return found
        try:
            for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    found[obj["Key"]] = obj["Size"]
        except self.s3.exceptions.NoSuchBucket:
            if not self.dry_run:
                raise
        return found

    def tree(self, local_dir: Path, prefix: str, include=lambda p: True) -> None:
        """Sube local_dir/** a s3://bucket/prefix/** (omite lo que ya está con el mismo tamaño)."""
        files = [p for p in sorted(local_dir.rglob("*")) if p.is_file() and include(p)]
        existing = self._existing(prefix.rstrip("/") + "/")
        todo = []
        for p in files:
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}"
            if existing.get(key) == p.stat().st_size:
                self.skipped += 1
            else:
                todo.append((p, key))
        size_mb = sum(p.stat().st_size for p, _ in todo) / 1e6
        print(f"  {prefix:<55} {len(todo):>4} archivos a subir ({size_mb:,.1f} MB), {len(files) - len(todo)} ya estaban")
        if self.dry_run:
            return
        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(lambda pk: self.s3.upload_file(str(pk[0]), self.bucket, pk[1]), todo))
        self.uploaded += len(todo)
        self.bytes += sum(p.stat().st_size for p, _ in todo)

    def file(self, local: Path, key: str) -> None:
        print(f"  {key}")
        if not self.dry_run:
            self.s3.upload_file(str(local), self.bucket, key)
            self.uploaded += 1


def partition_interactions(src_dir: Path, out_dir: Path) -> int:
    """curated/interactions (un archivo) -> out_dir/review_year=YYYY/part-0.parquet (mismo esquema que Glue)."""
    table = pa.concat_tables(pq.read_table(f) for f in sorted(src_dir.glob("*.parquet")))
    year = pc.year(table["timestamp"].cast(pa.timestamp("ms"))).cast(pa.int32())
    table = table.append_column("review_year", year)
    ds.write_dataset(
        table, out_dir, format="parquet",
        partitioning=ds.partitioning(pa.schema([("review_year", pa.int32())]), flavor="hive"),
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        basename_template="part-{i}.parquet", existing_data_behavior="overwrite_or_ignore",
    )
    return table.num_rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--model-version", help="Por defecto, la versión de data/published/current.json")
    ap.add_argument("--dry-run", action="store_true", help="Solo lista lo que subiría")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    data = Path(args.data_root)
    current = data / "published" / "current.json"
    version = args.model_version or (json.loads(current.read_text(encoding="utf-8"))["model_version"]
                                     if current.exists() else None)
    if not version:
        sys.exit("No hay versión publicada en data/published/current.json: indique --model-version")
    for need in (data / "curated" / "interactions", data / "curated" / "products", data / "model_data",
                 data / "artifacts" / f"model_version={version}", data / "recommendations" / f"model_version={version}",
                 data / "evaluation" / f"model_version={version}"):
        if not need.exists():
            sys.exit(f"Falta {need}: ejecute el pipeline local (python -m jobs.run_pipeline) antes de sembrar")

    print(f"Sembrando s3://{args.bucket} con la versión {version} ({'simulación' if args.dry_run else 'subida real'})")
    up = Uploader(args.bucket, args.dry_run, args.workers)

    with tempfile.TemporaryDirectory(prefix="fashrec_seed_") as tmp:
        parts = Path(tmp) / "interactions"
        n = partition_interactions(data / "curated" / "interactions", parts)
        print(f"  interactions re-particionadas por review_year: {n:,} filas, {len(list(parts.glob('review_year=*')))} particiones")
        up.tree(parts, "curated/interactions")
    up.tree(data / "curated" / "products", "curated/products", lambda p: p.suffix == ".parquet")
    up.tree(data / "model_data", "model_data")
    up.tree(data / "artifacts" / f"model_version={version}", f"artifacts/model_version={version}")
    rec = data / "recommendations" / f"model_version={version}"
    up.tree(rec, f"recommendations/model_version={version}")
    up.file(rec / "recommendations.parquet", f"recommendations/model_version={version}/recs/recommendations.parquet")
    up.tree(data / "evaluation" / f"model_version={version}", f"evaluation/model_version={version}")
    up.tree(data / "published", "published")
    up.tree(data / "reports", "reports")
    up.tree(data / "runs", "runs", lambda p: p.suffix == ".json")

    bundle = data / "inference" / f"model_version={version}"
    if not (bundle / "manifest.json").exists():
        print("  generando el paquete de inferencia (jobs.export_inference_bundle)...")
        if not args.dry_run:
            subprocess.run([sys.executable, "-m", "jobs.export_inference_bundle", "--model-version", version],
                           cwd=ROOT, check=True)
    if bundle.exists() or not args.dry_run:
        up.tree(bundle, f"inference/model_version={version}")

    print(f"\nListo: {up.uploaded} archivos subidos ({up.bytes / 1e6:,.1f} MB), {up.skipped} omitidos por ya existir."
          if not args.dry_run else "\nSimulación terminada: no se subió nada.")
    print(f"MODEL_VERSION={version}")


if __name__ == "__main__":
    main()
