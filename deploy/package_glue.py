"""Empaqueta src/ en deploy/build/fashrec_src.zip para el job de Glue (--extra-py-files).

Las rutas del zip usan "/" (Compress-Archive de Windows PowerShell 5.1 escribe "\\" y Linux no las lee).
Uso: python deploy/package_glue.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "build" / "fashrec_src.zip"


def main() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"{OUT} ({len(files)} archivos, {OUT.stat().st_size / 1024:.0f} KB)")
    return OUT


if __name__ == "__main__":
    main()
