#!/usr/bin/env bash
# Arranque del dashboard: baja de S3 la versión de modelo PUBLICADA (published/current.json) y lanza Streamlit.
# Solo trae lo que el dashboard consulta; no baja model.pt ni embeddings (≈ 300 MB que no se usan).
# Entorno obligatorio: DATA_BUCKET.  Opcional: MODEL_VERSION para forzar una versión distinta de la publicada.
set -euo pipefail

: "${DATA_BUCKET:?falta DATA_BUCKET}"
ROOT="${FASHREC_DATA_ROOT:-/data}"
S3="s3://${DATA_BUCKET}"
mkdir -p "$ROOT/published"

if [ -n "${MODEL_VERSION:-}" ]; then
  V="$MODEL_VERSION"
else
  if ! aws s3 cp "$S3/published/current.json" "$ROOT/published/current.json" --only-show-errors; then
    echo "[dashboard] no hay $S3/published/current.json: aún no se publicó ninguna versión." >&2
    echo "[dashboard] Ejecuta el pipeline (Step Functions) o 'deploy.ps1 -Seed' y reinicia el servicio." >&2
    exit 1
  fi
  V="$(python -c "import json,sys; print(json.load(open('$ROOT/published/current.json'))['model_version'])")"
fi
echo "[dashboard] versión del modelo: $V"

pull() { # pull <prefijo> [opciones de aws s3 sync...]
  local p="$1"; shift
  mkdir -p "$ROOT/$p"
  aws s3 sync "$S3/$p" "$ROOT/$p" --only-show-errors "$@"
}

pull published
pull runs
pull reports
pull curated/products
pull model_data --exclude "*" \
  --include "*/manifest.json" --include "*/interactions_split.parquet" --include "*/mappings/items.parquet"
pull artifacts --exclude "*" \
  --include "model_version=$V/config.json" --include "model_version=$V/training_history.csv" \
  --include "model_version=$V/umap_projection.parquet" --include "model_version=$V/faiss.index" \
  --include "model_version=$V/faiss_item_ids.npy" --include "model_version=$V/faiss_meta.json"
pull "recommendations/model_version=$V"
pull "evaluation/model_version=$V"
du -sh "$ROOT" | tail -1

exec streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
