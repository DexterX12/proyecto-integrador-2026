#!/usr/bin/env bash
# Construye la capa de dependencias (numpy + faiss-cpu para Linux x86_64, Python 3.13) en deploy/api/layer/python.
# Equivale a build_layer.ps1: no requiere Docker; usa ruedas precompiladas (--only-binary) y falla si no hay una compatible.
# Uso:  bash deploy/api/build_layer.sh        (FASHREC_PYTHON=<python con pip> para elegir el intérprete)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FASHREC_PYTHON:-python3}"
TARGET="$HERE/layer/python"

rm -rf "$HERE/layer"
mkdir -p "$TARGET"
"$PY" -m pip install numpy faiss-cpu \
  --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
  --python-version 3.13 --implementation cp --only-binary=:all: --target "$TARGET" \
  || { echo "No se pudieron obtener ruedas compatibles con Lambda (Linux x86_64, Python 3.13)" >&2; exit 1; }

# Reducir tamaño: sin metadatos, binarios de línea de comandos ni cachés
find "$TARGET" -maxdepth 1 -name '*.dist-info' -prune -exec rm -rf {} +
rm -rf "$TARGET/bin"
find "$TARGET" -type d -name '__pycache__' -prune -exec rm -rf {} +
echo "Capa lista: $(du -sm "$TARGET" | cut -f1) MB descomprimida (límite de Lambda: 250 MB entre código y capas)"
