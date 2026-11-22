#!/usr/bin/env bash
# Derriba todos los servicios FashRec de AWS con un solo comando, desde WSL (equivale a deploy/destroy.ps1).
#
# Uso:  bash deploy/destroy.sh [--dry-run] [--include-data] [--yes] [--stage X] [--region X] [--sls "cmd"] [--env-file RUTA]
#   --dry-run        solo muestra qué haría
#   --include-data   además vacía y borra el bucket de datos (por defecto se CONSERVA: DeletionPolicy Retain, ≈ $0.1/mes)
#   --yes            no pide confirmación
# Orden: detener lo que corre -> vaciar ECR -> serverless remove (dashboard, api, pipeline, platform)
#        -> borrar tablas DynamoDB por versión y el presupuesto -> (opcional) borrar el bucket
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$ROOT/deploy"
# shellcheck source=lib/common.sh
source "$DEPLOY/lib/common.sh"
usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

ENV_FILE="$ROOT/.env"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[i]}" = "--env-file" ]; then ENV_FILE="${args[i + 1]:-}"; fi
done
load_env "$ENV_FILE"

STAGE="${FASHREC_STAGE:-dev}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
SLS_CMD="${FASHREC_SLS:-osls}"
INCLUDE_DATA=0; ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:?falta valor}"; shift 2 ;;
    --region) REGION="${2:?falta valor}"; shift 2 ;;
    --sls) SLS_CMD="${2:?falta valor}"; shift 2 ;;
    --env-file) shift 2 ;;
    --include-data) INCLUDE_DATA=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "opción desconocida: $1 (usa --help)" ;;
  esac
done

PREFIX="fashrec-$STAGE"
export AWS_DEFAULT_REGION="$REGION" AWS_REGION="$REGION"
read -r -a SLS_ARR <<<"$SLS_CMD"

# --- 0. Identidad y confirmación ---------------------------------------------------------------
command -v aws >/dev/null 2>&1 || fail "No se encontró AWS CLI en WSL. Ejecuta: bash deploy/setup_wsl.sh"
ACCOUNT="$(q sts get-caller-identity --query Account --output text || true)"
[ -n "$ACCOUNT" ] || fail "No hay credenciales válidas de AWS (aws sts get-caller-identity falló). Nada se borró. Revisa .env."
BUCKET="fashrec-data-$ACCOUNT-$REGION-$STAGE"
info "Cuenta: $ACCOUNT | Región: $REGION | Stage: $STAGE"
info "Se derribarán los stacks: fashrec-dashboard-$STAGE, fashrec-api-$STAGE, fashrec-pipeline-$STAGE, fashrec-platform-$STAGE"
if [ "$INCLUDE_DATA" = 1 ]; then warn "Se BORRARÁ el bucket s3://$BUCKET con todo su contenido (incluidas versiones)."
else info "El bucket s3://$BUCKET se conserva (usa --include-data para borrarlo)."; fi
if [ "$ASSUME_YES" = 0 ] && ! dry; then
  read -r -p "Escribe 'borrar' para continuar: " answer
  [ "$answer" = "borrar" ] || { echo "Cancelado."; exit 0; }
fi

words() { tr '[:space:]' '\n' <<<"$1" | grep -v -e '^$' -e '^None$' || true; }

# --- 1. Detener lo que esté corriendo ---------------------------------------------------------
info $'\n[1/5] Deteniendo ejecuciones y tareas activas'
sfn="$(q stepfunctions list-state-machines --query "stateMachines[?name=='$PREFIX-pipeline'].stateMachineArn | [0]" --output text || true)"
if [ -n "$sfn" ] && [ "$sfn" != "None" ]; then
  for e in $(words "$(q stepfunctions list-executions --state-machine-arn "$sfn" --status-filter RUNNING --query 'executions[].executionArn' --output text || true)"); do
    act_soft stepfunctions stop-execution --execution-arn "$e"
  done
fi
for r in $(words "$(q glue get-job-runs --job-name "$PREFIX-etl" --query "JobRuns[?JobRunState=='RUNNING'||JobRunState=='STARTING'].Id" --output text || true)"); do
  act_soft glue batch-stop-job-run --job-name "$PREFIX-etl" --job-run-ids "$r"
done
if [ -n "$(words "$(q ecs describe-services --cluster "$PREFIX" --services "$PREFIX-dashboard" --query "services[?status=='ACTIVE'].serviceName" --output text || true)")" ]; then
  act_soft ecs update-service --cluster "$PREFIX" --service "$PREFIX-dashboard" --desired-count 0 --query service.serviceName --output text
fi
for t in $(words "$(q ecs list-tasks --cluster "$PREFIX" --query taskArns --output text || true)"); do
  act_soft ecs stop-task --cluster "$PREFIX" --task "$t"
done

# --- 2. Vaciar imágenes de ECR (un repositorio con imágenes no se puede borrar) ---------------
info $'\n[2/5] Vaciando repositorios de ECR'
for repo in "$PREFIX-pipeline" "$PREFIX-dashboard"; do
  for d in $(words "$(q ecr list-images --repository-name "$repo" --query 'imageIds[*].imageDigest' --output text || true)" | sort -u); do
    act_soft ecr batch-delete-image --repository-name "$repo" --image-ids "imageDigest=$d"
  done
done

# --- 3. serverless remove en orden inverso -----------------------------------------------------
info $'\n[3/5] Eliminando los 4 servicios (serverless remove)'
# serverless remove también resuelve el yml: las variables obligatorias deben existir aunque su valor ya no importe.
export FASHREC_VPC_ID="${FASHREC_VPC_ID:-unused-for-remove}" FASHREC_SUBNET_ID="${FASHREC_SUBNET_ID:-unused-for-remove}" \
       FASHREC_ALLOWED_CIDR="${FASHREC_ALLOWED_CIDR:-unused-for-remove}"
for svc in dashboard api pipeline platform; do
  stack="fashrec-$svc-$STAGE"
  status="$(q cloudformation describe-stacks --stack-name "$stack" --query 'Stacks[0].StackStatus' --output text || true)"
  if [ -z "$status" ]; then echo "  $stack no existe, se omite"; continue; fi
  echo "  removiendo $stack ($status)"
  if dry; then printf '  %s[dry-run] %s remove --stage %s --region %s  (en deploy/%s)%s\n' "$C_DIM" "$SLS_CMD" "$STAGE" "$REGION" "$svc" "$C_OFF"; continue; fi
  if ! ( cd "$DEPLOY/$svc" && "${SLS_ARR[@]}" remove --stage "$STAGE" --region "$REGION" ); then
    printf '%sFalló "serverless remove" de %s. Se detiene para no dejar dependencias rotas.%s\n' "$C_RED" "$svc" "$C_OFF" >&2
    printf '%sRevisa CloudFormation > Stacks > %s > Events, corrige y vuelve a ejecutar este script.%s\n' "$C_RED" "$stack" "$C_OFF" >&2
    exit 1
  fi
done

# --- 4. Lo que los stacks no poseen: tablas de recomendaciones por versión y presupuesto -------
info $'\n[4/5] Borrando tablas DynamoDB creadas por load_serving_job y el presupuesto'
for t in $(words "$(q dynamodb list-tables --query "TableNames[?starts_with(@, '$PREFIX-')]" --output text || true)"); do
  act_soft dynamodb delete-table --table-name "$t"
done
if q budgets describe-budget --account-id "$ACCOUNT" --budget-name "$PREFIX" >/dev/null; then
  act_soft budgets delete-budget --account-id "$ACCOUNT" --budget-name "$PREFIX"
fi

# --- 5. Bucket de datos (opcional) -----------------------------------------------------------------
info $'\n[5/5] Bucket de datos'
bucket_exists=0; q s3api head-bucket --bucket "$BUCKET" >/dev/null && bucket_exists=1
empty_versions() { # vacía versiones y marcadores de borrado por lotes de 1000 (el bucket está versionado)
  local kind payload tmp
  tmp="$(mktemp)"
  for kind in Versions DeleteMarkers; do
    while :; do
      payload="$(aws s3api list-object-versions --bucket "$BUCKET" --max-items 1000 \
                  --query "{Objects: ${kind}[].{Key:Key,VersionId:VersionId}, Quiet: \`true\`}" --output json 2>/dev/null || true)"
      if [ -z "$payload" ] || grep -q '"Objects": null' <<<"$payload"; then break; fi
      printf '%s' "$payload" >"$tmp"
      aws s3api delete-objects --bucket "$BUCKET" --delete "file://$tmp" >/dev/null || fail "no se pudo borrar un lote de $kind"
      echo "  lote de $kind borrado"
    done
  done
  rm -f "$tmp"
}
if [ "$INCLUDE_DATA" = 1 ]; then
  if [ "$bucket_exists" = 0 ]; then echo "  el bucket no existe"
  elif dry; then echo "  [dry-run] vaciar versiones y borrar s3://$BUCKET"
  else empty_versions; aws s3api delete-bucket --bucket "$BUCKET" && ok "bucket borrado"; fi
elif [ "$bucket_exists" = 1 ]; then warn "conservado: s3://$BUCKET (≈ \$0.1/mes). Para borrarlo: bash deploy/destroy.sh --include-data"; fi

# --- Verificación final ----------------------------------------------------------------------------
info $'\nVerificación: lo que queda en la cuenta'
left="$(q cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE ROLLBACK_COMPLETE DELETE_FAILED \
        --query "StackSummaries[?starts_with(StackName, 'fashrec-') && ends_with(StackName, '-$STAGE')].StackName" --output text || true)"
echo "  stacks fashrec-*-$STAGE: ${left:-ninguno}"
lt="$(q dynamodb list-tables --query "TableNames[?starts_with(@, '$PREFIX-')]" --output text || true)"
echo "  tablas DynamoDB: ${lt:-ninguna}"
if q s3api head-bucket --bucket "$BUCKET" >/dev/null; then echo "  bucket de datos: existe (sin costo relevante)"; else echo "  bucket de datos: no existe"; fi
if dry; then warn "Fue un --dry-run: no se modificó nada."; fi
