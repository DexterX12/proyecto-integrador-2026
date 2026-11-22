#!/usr/bin/env bash
# Despliega FashRec en AWS.
#
# Uso:  bash deploy/deploy.sh [opciones]
#   --dry-run            muestra el plan sin modificar AWS
#   --seed               sube el modelo ya entrenado en local, carga DynamoDB, prueba la API y consulta Athena
#   --upload-raw         sube los 2 JSONL originales (2.5 GB) a S3
#   --run-pipeline       inicia la máquina de estados (requiere --upload-raw la primera vez)
#   --wait-pipeline      espera a que termine la ejecución
#   --start-dashboard    enciende el dashboard e imprime su URL
#   --skip-images        no reconstruye las imágenes Docker
#   --stage X --region X --sls "cmd" --vpc-id X --subnet-id X --allowed-cidr X.X.X.X/32 --budget-email X
#   --env-file RUTA      usa otro archivo en lugar de .env
#   --yes                no pide confirmación
# Cada opción también se puede fijar en .env (FASHREC_SEED=1, ...). Las opciones de la línea de comandos tienen mayor peso.
#
# Pasos: 1 verificaciones | 2 platform | 3 código de Glue | 4 imágenes Docker -> ECR | 5 JSONL | 6 seed
#        7 api + pruebas | 8 pipeline y dashboard | 9 ejecutar pipeline | 10 encender dashboard
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$ROOT/deploy"
# shellcheck source=lib/common.sh
source "$DEPLOY/lib/common.sh"

usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

# --- archivo .env (se busca --env-file antes de leer el resto de opciones)
ENV_FILE="$ROOT/.env"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[i]}" = "--env-file" ]; then ENV_FILE="${args[i + 1]:-}"; fi
done
[ -f "$ENV_FILE" ] || warn "no existe $ENV_FILE: se usarán valores por defecto y el entorno (copia .env.example a .env)"
load_env "$ENV_FILE"

# --- valores por defecto (.env / entorno) y opciones de línea de comandos
STAGE="${FASHREC_STAGE:-dev}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
SLS_CMD="${FASHREC_SLS:-osls}"
VPC_ID="${FASHREC_VPC_ID:-}"; SUBNET_ID="${FASHREC_SUBNET_ID:-}"; ALLOWED_CIDR="${FASHREC_ALLOWED_CIDR:-}"
BUDGET_EMAIL="${FASHREC_BUDGET_EMAIL:-}"
flag() { if is_true "${!1:-}"; then echo 1; else echo 0; fi; }
SEED="$(flag FASHREC_SEED)"; UPLOAD_RAW="$(flag FASHREC_UPLOAD_RAW)"; RUN_PIPELINE="$(flag FASHREC_RUN_PIPELINE)"
WAIT_PIPELINE="$(flag FASHREC_WAIT_PIPELINE)"; START_DASHBOARD="$(flag FASHREC_START_DASHBOARD)"
SKIP_IMAGES="$(flag FASHREC_SKIP_IMAGES)"; ASSUME_YES=0
POLL="${FASHREC_POLL_SECONDS:-10}"

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:?falta valor}"; shift 2 ;;
    --region) REGION="${2:?falta valor}"; shift 2 ;;
    --sls) SLS_CMD="${2:?falta valor}"; shift 2 ;;
    --vpc-id) VPC_ID="${2:?falta valor}"; shift 2 ;;
    --subnet-id) SUBNET_ID="${2:?falta valor}"; shift 2 ;;
    --allowed-cidr) ALLOWED_CIDR="${2:?falta valor}"; shift 2 ;;
    --budget-email) BUDGET_EMAIL="${2:?falta valor}"; shift 2 ;;
    --env-file) shift 2 ;;
    --seed) SEED=1; shift ;;
    --upload-raw) UPLOAD_RAW=1; shift ;;
    --run-pipeline) RUN_PIPELINE=1; shift ;;
    --wait-pipeline) WAIT_PIPELINE=1; shift ;;
    --start-dashboard) START_DASHBOARD=1; shift ;;
    --skip-images) SKIP_IMAGES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "opción desconocida: $1 (usa --help)" ;;
  esac
done

PREFIX="fashrec-$STAGE"
TOTAL=10
export AWS_DEFAULT_REGION="$REGION" AWS_REGION="$REGION"
read -r -a SLS_ARR <<<"$SLS_CMD"
PY="$(python_bin)"
step() { printf '\n%s[%s/%s] %s%s\n' "$C_CYAN" "$1" "$TOTAL" "$2" "$C_OFF"; }

run_sls() { # run_sls <servicio> [verbo]
  local svc="$1" verb="${2:-deploy}"
  if dry; then printf '  %s[dry-run] %s %s --stage %s --region %s  (deploy/%s)%s\n' "$C_DIM" "$SLS_CMD" "$verb" "$STAGE" "$REGION" "$svc" "$C_OFF"; return 0; fi
  echo "  $SLS_CMD $verb --stage $STAGE --region $REGION  (deploy/$svc)"
  ( cd "$DEPLOY/$svc" && "${SLS_ARR[@]}" "$verb" --stage "$STAGE" --region "$REGION" ) \
    || fail "'$SLS_CMD $verb' falló en deploy/$svc. Revisa CloudFormation > Stacks > fashrec-$svc-$STAGE > Events."
}

# ----------------------------------------------------------------------------------------------- 1. Verificaciones
step 1 "Verificaciones previas"
command -v aws >/dev/null 2>&1 || fail "No se encontró AWS CLI en WSL. Ejecuta: bash deploy/setup_wsl.sh"
if ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/tmp/fashrec_sts.err)"; then
  ok "Cuenta $ACCOUNT ($(q sts get-caller-identity --query Arn --output text))"
else
  if dry; then warn "sin credenciales válidas (se continúa solo por --dry-run)"; ACCOUNT="000000000000"
  else
    fail "Sin credenciales válidas de AWS: $(head -c 300 /tmp/fashrec_sts.err)
       Configura AWS_PROFILE o AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY en .env (o ejecuta 'aws configure' / 'aws configure sso' en WSL)."
  fi
fi
BUCKET="fashrec-data-$ACCOUNT-$REGION-$STAGE"

if ! command -v "${SLS_ARR[0]}" >/dev/null 2>&1; then
  warn "No se encontró '${SLS_ARR[0]}'. Ejecuta: bash deploy/setup_wsl.sh"
  dry || fail "Serverless CLI no disponible."
fi
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || fail "No hay intérprete de Python ($PY). Ejecuta: bash deploy/setup_wsl.sh"
ok "Python: $PY"
if [ "$SKIP_IMAGES" = 0 ]; then
  if dv="$(docker version --format '{{.Server.Version}}' 2>/dev/null)" && [ -n "$dv" ]; then ok "Docker: $dv"
  elif dry; then warn "Docker no responde (se continúa solo por --dry-run)"
  else fail "Docker no responde en WSL. Comprueba 'docker info' (o usa --skip-images si las imágenes ya están en ECR)."; fi
fi

if [ -z "$VPC_ID" ]; then VPC_ID="$(q ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text || true)"; fi
if [ -z "$SUBNET_ID" ] && [ -n "$VPC_ID" ] && [ "$VPC_ID" != "None" ]; then
  SUBNET_ID="$(q ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" Name=map-public-ip-on-launch,Values=true \
                 --query 'Subnets[0].SubnetId' --output text || true)"
fi
if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ] || [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" = "None" ]; then
  if dry; then warn "no se pudo detectar VPC/subred (valores ficticios por --dry-run)"; VPC_ID="vpc-dryrun"; SUBNET_ID="subnet-dryrun"
  else fail "No hay VPC/subred pública por defecto. Define FASHREC_VPC_ID y FASHREC_SUBNET_ID en .env (subred pública con ruta a un Internet Gateway)."; fi
fi
if [ -z "$ALLOWED_CIDR" ]; then
  if myip="$(curl -fsS --max-time 15 https://checkip.amazonaws.com 2>/dev/null)"; then ALLOWED_CIDR="$(echo "$myip" | tr -d '[:space:]')/32"
  elif dry; then ALLOWED_CIDR="203.0.113.10/32"
  else fail "No pude detectar tu IP pública. Define FASHREC_ALLOWED_CIDR=x.x.x.x/32 en .env."; fi
fi
ok "VPC $VPC_ID | subred $SUBNET_ID | el dashboard solo aceptará $ALLOWED_CIDR"
quota="$(q service-quotas get-service-quota --service-code fargate --quota-code L-3032A538 --query Quota.Value --output text || true)"
if [[ "$quota" =~ ^[0-9] ]] && awk "BEGIN{exit !($quota < 16)}"; then
  warn "Tu cuota de Fargate es $quota vCPU y Train/Recommend piden 16. El pipeline fallaría al lanzar esas tareas: solicita un aumento (Service Quotas > Fargate On-Demand vCPU) antes de --run-pipeline."
fi

if [ "$ASSUME_YES" = 0 ] && ! dry; then
  printf '\n%sSe desplegará en la cuenta %s, región %s, stage "%s".%s\n' "$C_YELLOW" "$ACCOUNT" "$REGION" "$STAGE" "$C_OFF"
  read -r -p "Escribe 'si' para continuar: " answer
  [ "$answer" = "si" ] || { echo "Cancelado."; exit 0; }
fi
export FASHREC_VPC_ID="$VPC_ID" FASHREC_SUBNET_ID="$SUBNET_ID" FASHREC_ALLOWED_CIDR="$ALLOWED_CIDR"
export FASHREC_DASHBOARD_DESIRED=0          # el dashboard se enciende explícitamente (--start-dashboard)

if [ -n "$BUDGET_EMAIL" ]; then
  if q budgets describe-budget --account-id "$ACCOUNT" --budget-name "$PREFIX" >/dev/null; then ok "Presupuesto $PREFIX ya existe"
  else
    bj="$(mktemp)"; nj="$(mktemp)"
    printf '{"BudgetName":"%s","BudgetType":"COST","TimeUnit":"MONTHLY","BudgetLimit":{"Amount":"10","Unit":"USD"}}' "$PREFIX" >"$bj"
    printf '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"%s"}]}]' "$BUDGET_EMAIL" >"$nj"
    act_soft budgets create-budget --account-id "$ACCOUNT" --budget "file://$bj" --notifications-with-subscribers "file://$nj"
    ok "Presupuesto de 10 USD/mes con alerta al 80% a $BUDGET_EMAIL"
  fi
fi

# ----------------------------------------------------------------------------------------------- 2. platform
step 2 "Servicio platform (S3, Glue Catalog, Athena, DynamoDB, ECR, ECS, IAM, SNS, SSM)"
run_sls platform
if ! dry; then
  real_bucket="$(stack_output "fashrec-platform-$STAGE" BucketName)"
  [ -n "$real_bucket" ] || fail "No encuentro las salidas de fashrec-platform-$STAGE."
  BUCKET="$real_bucket"
fi
ok "Bucket: s3://$BUCKET"

# ----------------------------------------------------------------------------------------------- 3. Glue
step 3 "Código del job de Glue -> s3://$BUCKET/glue-scripts/"
act s3 cp "$ROOT/jobs/glue_etl_job.py" "s3://$BUCKET/glue-scripts/glue_etl_job.py" --only-show-errors
act s3 cp "$ROOT/configs/default.yaml" "s3://$BUCKET/glue-scripts/default.yaml" --only-show-errors
if dry; then echo "  [dry-run] $PY deploy/package_glue.py ; aws s3 cp deploy/build/fashrec_src.zip ..."
else
  "$PY" "$DEPLOY/package_glue.py" || fail "No se pudo empaquetar src/ para Glue."
  act s3 cp "$DEPLOY/build/fashrec_src.zip" "s3://$BUCKET/glue-scripts/fashrec_src.zip" --only-show-errors
fi

# ----------------------------------------------------------------------------------------------- 4. Imágenes
step 4 "Imágenes Docker (pipeline y dashboard) -> ECR"
if [ "$SKIP_IMAGES" = 1 ]; then warn "--skip-images: se usan las imágenes que ya estén en ECR"
else
  ecr="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
  if dry; then PIPE_URI="$ecr/$PREFIX-pipeline"; DASH_URI="$ecr/$PREFIX-dashboard"
  else PIPE_URI="$(stack_output "fashrec-platform-$STAGE" PipelineRepositoryUri)"; DASH_URI="$(stack_output "fashrec-platform-$STAGE" DashboardRepositoryUri)"; fi
  if dry; then echo "  [dry-run] aws ecr get-login-password | docker login $ecr ; docker build/push $PIPE_URI y $DASH_URI"
  else
    [ -n "$PIPE_URI" ] && [ -n "$DASH_URI" ] || fail "No encuentro los repositorios de ECR en las salidas de platform."
    echo "  login en ${PIPE_URI%%/*}"
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${PIPE_URI%%/*}" || fail "docker login a ECR falló."
    echo "  construyendo y subiendo imágenes (la primera vez tarda varios minutos: torch + dependencias)..."
    docker build --platform linux/amd64 -f "$DEPLOY/pipeline/Dockerfile" -t "$PIPE_URI:latest" "$ROOT" || fail "docker build del pipeline falló."
    docker push "$PIPE_URI:latest" || fail "docker push del pipeline falló."
    docker build --platform linux/amd64 -f "$DEPLOY/dashboard/Dockerfile" -t "$DASH_URI:latest" "$ROOT" || fail "docker build del dashboard falló."
    docker push "$DASH_URI:latest" || fail "docker push del dashboard falló."
    ok "Imágenes en ECR: $PIPE_URI:latest y $DASH_URI:latest"
  fi
fi

# ----------------------------------------------------------------------------------------------- 5. RAW
step 5 "Datos originales (JSONL, 2.5 GB)"
if [ "$UPLOAD_RAW" = 0 ]; then warn "omitido (usa --upload-raw para poder ejecutar el pipeline completo con Glue)"
else
  for pair in "Amazon_Fashion.jsonl|raw/amazon-fashion/reviews/Amazon_Fashion.jsonl" \
              "meta_Amazon_Fashion.jsonl|raw/amazon-fashion/metadata/meta_Amazon_Fashion.jsonl"; do
    file="$ROOT/${pair%%|*}"; key="${pair##*|}"
    [ -f "$file" ] || fail "No existe $file"
    size="$(stat -c %s "$file")"
    remote="$(q s3api head-object --bucket "$BUCKET" --key "$key" --query ContentLength --output text || true)"
    if [ "$remote" = "$size" ]; then ok "$key ya está en S3 ($((size / 1048576)) MB)"; continue; fi
    echo "  subiendo $key ($((size / 1048576)) MB)..."
    act s3 cp "$file" "s3://$BUCKET/$key" --checksum-algorithm SHA256 --only-show-errors
  done
fi

# ----------------------------------------------------------------------------------------------- 6. Seed
MODEL_VERSION=""
if [ -f "$ROOT/data/published/current.json" ]; then
  MODEL_VERSION="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['model_version'])" "$ROOT/data/published/current.json" 2>/dev/null || true)"
fi
step 6 "Resultados del modelo local -> S3 y DynamoDB (--seed)"
if [ "$SEED" = 0 ]; then warn "omitido (sin --seed, la API y el dashboard no tendrán datos hasta que corra el pipeline en AWS)"
elif [ -z "$MODEL_VERSION" ]; then fail "No hay data/published/current.json: ejecuta primero el pipeline local (python -m jobs.run_pipeline)."
elif dry; then echo "  [dry-run] $PY deploy/seed_lake.py --bucket $BUCKET --model-version $MODEL_VERSION ; $PY -m jobs.load_serving_job --model-version $MODEL_VERSION"
else
  "$PY" -c "import boto3, pyarrow, pandas, yaml" 2>/dev/null || fail "Faltan paquetes de Python en $PY. Ejecuta: bash deploy/setup_wsl.sh"
  ( cd "$ROOT" && "$PY" "$DEPLOY/seed_lake.py" --bucket "$BUCKET" --model-version "$MODEL_VERSION" ) \
    || fail "seed_lake.py falló (si pide torch/faiss para generar el paquete de inferencia: bash deploy/setup_wsl.sh sin --minimal)."
  echo "  cargando recomendaciones en DynamoDB y activando la versión en SSM..."
  ( cd "$ROOT" && TABLE_PREFIX="$PREFIX" ACTIVE_VERSION_PARAM="$(stack_output "fashrec-platform-$STAGE" ActiveVersionParam)" \
      "$PY" -m jobs.load_serving_job --model-version "$MODEL_VERSION" ) || fail "load_serving_job falló."
fi

# ----------------------------------------------------------------------------------------------- 7. API
step 7 "Servicio api (Lambda + API Gateway) y pruebas funcionales"
if [ ! -d "$DEPLOY/api/layer/python" ]; then
  if dry; then echo "  [dry-run] bash deploy/api/build_layer.sh"
  else
    echo "  construyendo la capa numpy + faiss (ruedas de Linux)..."
    FASHREC_PYTHON="$PY" bash "$DEPLOY/api/build_layer.sh" || fail "build_layer.sh falló."
  fi
else ok "capa ya construida (borra deploy/api/layer para reconstruirla)"; fi
run_sls api
if dry; then API_URL="https://<api-id>.execute-api.$REGION.amazonaws.com"; else API_URL="$(stack_output "fashrec-api-$STAGE" ApiUrl)"; fi
ok "API: $API_URL"
if [ "$SEED" = 1 ] && ! dry; then
  printf '\n  --- pruebas funcionales (la primera llamada a /v1/recommend incluye el arranque en frío) ---\n'
  ( cd "$ROOT" && "$PY" "$DEPLOY/smoke_test.py" --api-url "$API_URL" --model-version "$MODEL_VERSION" ) \
    || warn "alguna prueba de la API falló: revisa la salida y CloudWatch Logs de las Lambdas."
elif [ "$SEED" = 0 ]; then warn "pruebas omitidas: sin --seed no hay datos que consultar"; fi

# ----------------------------------------------------------------------------------------------- 8. pipeline + dashboard
step 8 "Servicios pipeline (Glue + Fargate + Step Functions) y dashboard"
run_sls pipeline
run_sls dashboard
if dry; then PIPELINE_ARN="arn:aws:states:$REGION:$ACCOUNT:stateMachine:$PREFIX-pipeline"; else PIPELINE_ARN="$(stack_output "fashrec-pipeline-$STAGE" PipelineArn)"; fi

# ----------------------------------------------------------------------------------------------- 9. Pipeline
step 9 "Ejecución del pipeline en AWS (--run-pipeline)"
EXEC_ARN=""
if [ "$RUN_PIPELINE" = 0 ]; then warn "omitido (usa --run-pipeline; requiere --upload-raw la primera vez)"
else
  for key in raw/amazon-fashion/reviews/Amazon_Fashion.jsonl raw/amazon-fashion/metadata/meta_Amazon_Fashion.jsonl; do
    if ! dry && ! q s3api head-object --bucket "$BUCKET" --key "$key" >/dev/null; then fail "Falta s3://$BUCKET/$key. Ejecuta con --upload-raw."; fi
  done
  VER="$(date -u +%Y%m%d_%H%M)_aws"
  if dry; then echo "  [dry-run] aws stepfunctions start-execution --state-machine-arn $PIPELINE_ARN --input {model_version:$VER}"
  else
    EXEC_ARN="$(aws stepfunctions start-execution --state-machine-arn "$PIPELINE_ARN" --input "{\"model_version\":\"$VER\"}" --query executionArn --output text)" \
      || fail "No se pudo iniciar la ejecución."
    ok "Ejecución iniciada: $EXEC_ARN"
    ok "Versión del modelo que se entrenará: $VER"
    if [ "$WAIT_PIPELINE" = 1 ]; then
      echo "  esperando (ETL en Glue + 6 tareas Fargate; en AWS puede tardar de 30 a 90 minutos)..."
      while :; do
        sleep "$((POLL * 6))"
        status="$(aws stepfunctions describe-execution --execution-arn "$EXEC_ARN" --query status --output text)"
        printf '  %s  %s\n' "$(date +%H:%M:%S)" "$status"
        [ "$status" = "RUNNING" ] || break
      done
      if [ "$status" != "SUCCEEDED" ]; then warn "La ejecución terminó en $status. Revisa Step Functions > Executions y los logs de CloudWatch (/fashrec/$STAGE/pipeline)."
      else ok "Pipeline completado: la API y el dashboard sirven la versión $VER"; fi
    else warn "Sigue en ejecución: mírala en Step Functions > State machines > $PREFIX-pipeline"; fi
  fi
fi

# ----------------------------------------------------------------------------------------------- 10. Dashboard
step 10 "Dashboard (--start-dashboard)"
DASH_URL=""
if [ "$START_DASHBOARD" = 0 ]; then warn "apagado (desiredCount 0). Para encenderlo: --start-dashboard, o ecs update-service ... --desired-count 1"
else
  if ! dry && ! q s3api head-object --bucket "$BUCKET" --key published/current.json >/dev/null; then
    warn "todavía no hay versión publicada en S3 (published/current.json): usa --seed o espera a que termine el pipeline. No se enciende."
  else
    act ecs update-service --cluster "$PREFIX" --service "$PREFIX-dashboard" --desired-count 1 --query service.serviceName --output text
    if ! dry; then
      echo "  esperando la tarea (descarga la imagen y ~1 GB de datos de S3; 3 a 6 minutos)..."
      ip=""
      for _ in $(seq 1 60); do
        sleep "$POLL"
        task="$(q ecs list-tasks --cluster "$PREFIX" --service-name "$PREFIX-dashboard" --desired-status RUNNING --query 'taskArns[0]' --output text || true)"
        { [ -z "$task" ] || [ "$task" = "None" ]; } && continue
        eni="$(q ecs describe-tasks --cluster "$PREFIX" --tasks "$task" \
               --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value | [0]" --output text || true)"
        { [ -z "$eni" ] || [ "$eni" = "None" ]; } && continue
        ip="$(q ec2 describe-network-interfaces --network-interface-ids "$eni" --query 'NetworkInterfaces[0].Association.PublicIp' --output text || true)"
        [ "$ip" = "None" ] && ip=""
        [ -n "$ip" ] && break
      done
      if [ -n "$ip" ]; then
        DASH_URL="http://$ip:8501"
        for _ in $(seq 1 40); do
          if curl -fsS --max-time 5 "$DASH_URL/_stcore/health" 2>/dev/null | grep -q ok; then break; fi
          sleep "$POLL"
        done
        ok "Dashboard: $DASH_URL   (solo accesible desde $ALLOWED_CIDR)"
      else warn "la tarea no obtuvo IP pública a tiempo: revisa ECS > $PREFIX > $PREFIX-dashboard > Tasks y CloudWatch Logs /fashrec/$STAGE/dashboard"; fi
    fi
  fi
fi

# ----------------------------------------------------------------------------------------------- Evidencia en Athena
athena() { # athena "<sql>"
  local sql="$1" qid state="" scanned result
  qid="$(aws athena start-query-execution --query-string "$sql" --work-group "$PREFIX" \
          --query-execution-context "Database=fashrec_$STAGE" --query QueryExecutionId --output text)"
  for _ in $(seq 1 60); do
    state="$(aws athena get-query-execution --query-execution-id "$qid" --query QueryExecution.Status.State --output text)"
    case "$state" in SUCCEEDED|FAILED|CANCELLED) break ;; esac
    sleep 2
  done
  scanned="$(aws athena get-query-execution --query-execution-id "$qid" --query QueryExecution.Statistics.DataScannedInBytes --output text)"
  result="$(q athena get-query-results --query-execution-id "$qid" --query 'ResultSet.Rows[1].Data[0].VarCharValue' --output text || echo -)"
  printf '  %-9s resultado=%-9s escaneado=%12s bytes   %s\n' "$state" "$result" "$scanned" "$sql"
}
if [ "$SEED" = 1 ] && ! dry; then
  info $'\nConsultas de Athena (evidencia de particionamiento: compara "escaneado")'
  athena "SELECT count(DISTINCT user_id) FROM interactions"
  athena "SELECT count(DISTINCT user_id) FROM interactions WHERE review_year = 2021"
  athena "SELECT count(DISTINCT parent_asin) FROM recommendations WHERE model_version = '$MODEL_VERSION'"
fi

# ----------------------------------------------------------------------------------------------- Resumen
C="https://console.aws.amazon.com"
printf '\n%s================ RESUMEN ================%s\n' "$C_GREEN" "$C_OFF"
echo "Cuenta $ACCOUNT | $REGION | stage $STAGE | bucket s3://$BUCKET"
[ -n "${API_URL:-}" ] && echo "API:       $API_URL/v1/health"
[ -n "$DASH_URL" ] && echo "Dashboard: $DASH_URL"
[ -n "$EXEC_ARN" ] && echo "Pipeline:  $EXEC_ARN"

if dry; then warn "Fue un --dry-run: no se modificó nada en AWS."; fi
