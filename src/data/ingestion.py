"""Ingesta: verificación de los archivos RAW y manifiesto de integridad.

Los JSONL originales nunca se modifican. El manifiesto (tamaño + SHA-256 + nº de líneas) permite
comprobar después que siguen intactos y que corresponden a la misma versión del dataset.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _count_lines(path: Path, chunk: int = 16 * 1024 * 1024) -> int:
    n = 0
    last = b"\n"
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            n += block.count(b"\n")
            last = block[-1:]
    return n + (0 if last == b"\n" else 1)


def build_raw_manifest(files: dict[str, Path], out_path: Path, recompute: bool = False) -> dict:
    """Genera (o reutiliza) el manifiesto de los archivos RAW."""
    if out_path.exists() and not recompute:
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
        ok = all(
            name in manifest["files"]
            and manifest["files"][name]["size_bytes"] == path.stat().st_size
            for name, path in files.items()
        )
        if ok:
            log.info("Manifiesto RAW existente y consistente (tamaños coinciden).")
            return manifest
    entries = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo RAW '{name}': {path}")
        log.info("Calculando manifiesto de %s (%.1f MB)...", path.name, path.stat().st_size / 1e6)
        entries[name] = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "lines": _count_lines(path),
        }
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": entries}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_raw_manifest(files: dict[str, Path], manifest_path: Path) -> bool:
    """Comprueba que los RAW no cambiaron desde que se creó el manifiesto."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in files.items():
        if _sha256(path) != manifest["files"][name]["sha256"]:
            return False
    return True


# ----------------------------------------------------------------------------- RAW en S3
# En AWS los JSONL (2.5 GB) viven en S3 y no se descargan para verificarlos. El manifiesto guarda lo que
# S3 certifica de cada objeto (tamaño, ETag, VersionId y checksum si se subió con --checksum-algorithm);
# si alguien reemplaza el objeto después del ETL, el ETag/VersionId/tamaño dejan de coincidir.
# Un checksum SHA-256 de un objeto multiparte es "compuesto" (hash de hashes): NO es el SHA-256 del archivo.
def _split_s3_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def _s3_object_info(uri: str) -> dict:
    import boto3

    bucket, key = _split_s3_uri(uri)
    head = boto3.client("s3").head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    return {
        "uri": uri, "size_bytes": head["ContentLength"], "etag": head["ETag"].strip('"'),
        "version_id": head.get("VersionId"), "checksum_sha256": head.get("ChecksumSHA256"),
        "last_modified": head["LastModified"].isoformat(timespec="seconds"),
    }


def build_raw_manifest_s3(uris: dict[str, str], lines: dict[str, int]) -> dict:
    entries = {name: {**_s3_object_info(uri), "lines": lines.get(name)} for name, uri in uris.items()}
    return {"storage": "s3", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": entries}


def verify_raw_manifest_s3(manifest: dict) -> bool:
    """True si cada objeto RAW conserva el tamaño, ETag y VersionId registrados al momento del ETL."""
    for name, expected in manifest["files"].items():
        current = _s3_object_info(expected["uri"])
        for field in ("size_bytes", "etag", "version_id"):
            if current[field] != expected[field]:
                log.error("RAW %s cambió (%s): %s != %s", name, field, current[field], expected[field])
                return False
    return True
