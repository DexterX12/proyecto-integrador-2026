#!/usr/bin/env bash
# Ejecuta UNA etapa del pipeline dentro de Fargate.
#   1. baja de S3 a $FASHREC_DATA_ROOT (/data) solo los prefijos que la etapa lee
#   2. ejecuta el comando recibido (p. ej. python -m jobs.train_job --model-version X)
#   3. sube a S3 solo lo que la etapa escribe (siempre sube runs/, también si falló)
# El bucket S3 tiene el mismo layout que data/ en local, por lo que el código del pipeline no cambia.
#
# Entorno obligatorio: DATA_BUCKET, STAGE (features|train|evaluate|recommend|publish|export|load)
# MODEL_VERSION es obligatorio en todas las etapas salvo features.
set -euo pipefail

: "${DATA_BUCKET:?falta DATA_BUCKET}"
: "${STAGE:?falta STAGE}"
[ "$#" -gt 0 ] || { echo "[entrypoint] no se indicó el comando a ejecutar" >&2; exit 64; }

ROOT="${FASHREC_DATA_ROOT:-/data}"
S3="s3://${DATA_BUCKET}"
V="${MODEL_VERSION:-}"
if [ "$STAGE" != "features" ] && [ -z "$V" ]; then
  echo "[entrypoint] falta MODEL_VERSION para la etapa '$STAGE'" >&2; exit 64
fi

A="artifacts/model_version=$V"
R="recommendations/model_version=$V"
E="evaluation/model_version=$V"
case "$STAGE" in
  features)  PULL=(curated/interactions curated/products);                    PUSH=(model_data reports) ;;
  train)     PULL=(model_data);                                               PUSH=("$A") ;;
  evaluate)  PULL=(model_data "$A");                                          PUSH=("$E") ;;
  recommend) PULL=(model_data "$A");                                          PUSH=("$R" "$A") ;;
  publish)   PULL=(model_data "$A" "$R" "$E" reports published);              PUSH=(published) ;;
  export)    PULL=(model_data "$A");                                          PUSH=("inference/model_version=$V") ;;
  load)      PULL=("$R" curated/products);                                    PUSH=() ;;
  *) echo "[entrypoint] STAGE desconocido: $STAGE" >&2; exit 64 ;;
esac

mkdir -p "$ROOT"
for p in "${PULL[@]}"; do
  echo "[entrypoint] bajando $S3/$p"
  mkdir -p "$ROOT/$p"
  aws s3 sync "$S3/$p" "$ROOT/$p" --only-show-errors
done
df -h "$ROOT" | tail -1

echo "[entrypoint] etapa=$STAGE version=${V:-n/a} comando: $*"
set +e
"$@"
rc=$?
set -e

mkdir -p "$ROOT/runs"
aws s3 sync "$ROOT/runs" "$S3/runs" --only-show-errors || true      # el registro de la ejecución se conserva aunque falle
if [ "$rc" -ne 0 ]; then
  echo "[entrypoint] la etapa '$STAGE' falló con código $rc; no se sube nada más" >&2
  exit "$rc"
fi

if [ "$STAGE" = "recommend" ]; then
  # Athena: la tabla `recommendations` apunta a .../model_version=<v>/recs/, que solo debe contener este Parquet
  # (en la carpeta de la versión hay otros Parquet con otro esquema).
  mkdir -p "$ROOT/$R/recs"
  cp "$ROOT/$R/recommendations.parquet" "$ROOT/$R/recs/recommendations.parquet"
fi

for p in "${PUSH[@]}"; do
  echo "[entrypoint] subiendo $p"
  aws s3 sync "$ROOT/$p" "$S3/$p" --only-show-errors
done
echo "[entrypoint] etapa '$STAGE' terminada"
