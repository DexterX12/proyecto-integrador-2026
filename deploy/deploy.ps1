<#
.SYNOPSIS
  Despliega FashRec en AWS (4 servicios del Serverless Framework + imágenes + datos) con un solo comando.
  Para tumbar todo después: deploy\destroy.ps1

.DESCRIPTION
  Pasos (cada uno es idempotente; se puede volver a ejecutar):
   1. Verificaciones: credenciales, herramientas, Docker en WSL, red (VPC/subred) y tu IP pública
   2. serverless deploy de `platform` (S3, Glue Catalog, Athena, DynamoDB, ECR, ECS, IAM, SNS, SSM)
   3. Sube el código del job de Glue (glue_etl_job.py, src.zip, default.yaml)
   4. Construye (Docker en WSL) y sube las imágenes del pipeline y del dashboard a ECR
   5. -UploadRaw : sube los 2 JSONL originales (2.5 GB) para poder ejecutar el pipeline completo
   6. -Seed      : sube a S3 el modelo ya entrenado en local y carga DynamoDB (evidencia sin esperar el pipeline)
   7. serverless deploy de `api` y pruebas funcionales (deploy\smoke_test.py)
   8. serverless deploy de `pipeline` (Glue + Fargate + Step Functions) y `dashboard`
   9. -RunPipeline : lanza la máquina de estados (ETL en Glue -> ... -> carga de DynamoDB)
  10. -StartDashboard : enciende el dashboard (Fargate) e imprime su URL
  Al final imprime las secciones de la consola donde tomar evidencia y el comando para tumbar todo.

  Costo: lo único que cobra por estar encendido es el dashboard (≈ $0.05/hora) cuando lo enciendes; el resto es
  por uso. Apágalo con:  aws ecs update-service --cluster fashrec-<stage> --service fashrec-<stage>-dashboard --desired-count 0

.EXAMPLE
  powershell -File deploy\deploy.ps1 -DryRun                                   # muestra el plan sin tocar AWS
  powershell -File deploy\deploy.ps1 -Seed -StartDashboard -BudgetEmail yo@correo.com
  powershell -File deploy\deploy.ps1 -Seed -UploadRaw -RunPipeline -WaitPipeline -StartDashboard
#>
[CmdletBinding()]
param(
    [string]$Stage = "dev",
    [string]$Region = "us-east-1",
    [string]$Sls = "serverless",              # o "npx osls@3" si usas la bifurcación abierta
    [string]$VpcId,                           # por defecto: la VPC por defecto de la cuenta
    [string]$SubnetId,                        # por defecto: una subred pública de esa VPC
    [string]$AllowedCidr,                     # por defecto: tu IP pública /32 (acceso al dashboard)
    [string]$BudgetEmail,                     # si se indica, crea un AWS Budget de 10 USD/mes con alertas
    [switch]$Seed,
    [switch]$UploadRaw,
    [switch]$RunPipeline,
    [switch]$WaitPipeline,
    [switch]$StartDashboard,
    [switch]$SkipImages,
    [switch]$DryRun,
    [switch]$Yes
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Deploy = $PSScriptRoot
$Build = Join-Path $Deploy "build"
$Prefix = "fashrec-$Stage"
$env:AWS_DEFAULT_REGION = $Region
$env:AWS_REGION = $Region
$TotalSteps = 10

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Ok($m) { Write-Host "  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  AVISO: $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }
function Step($n, $title) { Write-Host "`n[$n/$TotalSteps] $title" -ForegroundColor Cyan }

# Consulta de solo lectura (se ejecuta también en -DryRun). Devuelve $null si falla.
function Query { param([string[]]$a) $o = & aws @a 2>$null; if ($LASTEXITCODE -ne 0) { return $null }; return $o }
# Comando de AWS que modifica algo: en -DryRun solo se imprime.
function Act {
    param([string[]]$a, [switch]$Soft)
    if ($DryRun) { Write-Host "  [dry-run] aws $($a -join ' ')" -ForegroundColor DarkGray; return $true }
    & aws @a
    if ($LASTEXITCODE -ne 0) {
        if ($Soft) { Warn "falló 'aws $($a[0..2] -join ' ')' (código $LASTEXITCODE)"; return $false }
        Fail "falló 'aws $($a[0..2] -join ' ')' (código $LASTEXITCODE)"
    }
    return $true
}
function Stack-Output([string]$stack, [string]$key) {
    $v = Query @("cloudformation", "describe-stacks", "--stack-name", $stack, "--query", "Stacks[0].Outputs[?OutputKey=='$key'].OutputValue", "--output", "text")
    if ($v -and $v -ne "None") { return $v.Trim() } else { return $null }
}

$slsParts = @($Sls -split "\s+" | Where-Object { $_ })
function Invoke-Sls([string]$Service, [string]$Verb = "deploy") {
    $line = "$Sls $Verb --stage $Stage --region $Region  (deploy\$Service)"
    if ($DryRun) { Write-Host "  [dry-run] $line" -ForegroundColor DarkGray; return }
    Write-Host "  $line"
    $slsArgs = @($slsParts | Select-Object -Skip 1) + @($Verb, "--stage", $Stage, "--region", $Region)
    Push-Location (Join-Path $Deploy $Service)
    try { & $slsParts[0] @slsArgs }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { Fail "'$Sls $Verb' falló en deploy\$Service. Revisa CloudFormation > Stacks > fashrec-$Service-$Stage > Events." }
}

function Write-Utf8([string]$path, [string]$text) {
    [IO.File]::WriteAllText($path, $text.Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
}

# ------------------------------------------------------------------------------------------ 1. Verificaciones
Step 1 "Verificaciones previas"
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { Fail "No se encontró AWS CLI. Instálalo: https://aws.amazon.com/cli/" }
$identity = & aws sts get-caller-identity --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    $msg = ($identity | Out-String)
    if ($DryRun) { Warn "sin credenciales válidas (se continúa solo porque es -DryRun)"; $Account = "000000000000" }
    elseif ($msg -match "parse config") {
        Fail "El archivo $env:USERPROFILE\.aws\config está dañado (AWS CLI no puede leerlo). Renómbralo a config.bak y ejecuta 'aws configure' (o 'aws configure sso')."
    } else { Fail "Sin credenciales válidas de AWS. Ejecuta 'aws configure' (o 'aws configure sso' + 'aws sso login'). Detalle: $msg" }
} else {
    $who = ($identity | Out-String) | ConvertFrom-Json
    $Account = $who.Account
    Ok "Cuenta $Account ($($who.Arn))"
}
$Bucket = "fashrec-data-$Account-$Region-$Stage"

if (-not (Get-Command $slsParts[0] -ErrorAction SilentlyContinue)) {
    Warn "No se encontró '$($slsParts[0])'. Instala: npm i -g osls@3   (o 'npm i -g serverless' y 'serverless login'). Luego usa -Sls serverless."
    if (-not $DryRun) { Fail "Serverless CLI no disponible." }
}
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Py = if (Test-Path $VenvPy) { $VenvPy } else { "python" }
Ok "Python: $Py"

if (-not $SkipImages) {
    $dv = & wsl -e docker version --format "{{.Server.Version}}" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $dv) {
        if ($DryRun) { Warn "Docker no responde dentro de WSL (se continúa solo por -DryRun)" }
        else { Fail "Docker no responde dentro de WSL. Abre una terminal WSL y comprueba 'docker info' (o usa -SkipImages si las imágenes ya están en ECR)." }
    } else { Ok "Docker en WSL: $dv" }
}

if (-not $VpcId) { $VpcId = Query @("ec2", "describe-vpcs", "--filters", "Name=isDefault,Values=true", "--query", "Vpcs[0].VpcId", "--output", "text") }
if (-not $SubnetId -and $VpcId -and $VpcId -ne "None") {
    $SubnetId = Query @("ec2", "describe-subnets", "--filters", "Name=vpc-id,Values=$VpcId", "Name=map-public-ip-on-launch,Values=true", "--query", "Subnets[0].SubnetId", "--output", "text")
}
if (-not $VpcId -or $VpcId -eq "None" -or -not $SubnetId -or $SubnetId -eq "None") {
    if ($DryRun) { Warn "no se pudo detectar VPC/subred (se usan valores ficticios por -DryRun)"; $VpcId = "vpc-dryrun"; $SubnetId = "subnet-dryrun" }
    else { Fail "No hay VPC/subred pública por defecto. Pásalas: -VpcId vpc-xxxx -SubnetId subnet-xxxx (subred pública con ruta a un Internet Gateway)." }
}
if (-not $AllowedCidr) {
    try { $AllowedCidr = ((Invoke-RestMethod "https://checkip.amazonaws.com" -TimeoutSec 15).ToString().Trim()) + "/32" }
    catch { if ($DryRun) { $AllowedCidr = "203.0.113.10/32" } else { Fail "No pude detectar tu IP pública. Pásala: -AllowedCidr x.x.x.x/32" } }
}
Ok "VPC $VpcId | subred $SubnetId | el dashboard solo aceptará $AllowedCidr"
$quota = Query @("service-quotas", "get-service-quota", "--service-code", "fargate", "--quota-code", "L-3032A538", "--query", "Quota.Value", "--output", "text")
if ($quota -match '^\d' -and [double]$quota -lt 16) {
    Warn "Tu cuota de Fargate es $quota vCPU y las etapas Train y Recommend piden 16. El pipeline fallaría al lanzar esas tareas: solicita un aumento (Service Quotas > Fargate On-Demand vCPU) antes de usar -RunPipeline."
}

if (-not $Yes -and -not $DryRun) {
    Write-Host "`nSe desplegará en la cuenta $Account, región $Region, stage '$Stage'." -ForegroundColor Yellow
    if ((Read-Host "Escribe 'si' para continuar") -ne "si") { Write-Host "Cancelado."; exit 0 }
}
$env:FASHREC_VPC_ID = $VpcId
$env:FASHREC_SUBNET_ID = $SubnetId
$env:FASHREC_ALLOWED_CIDR = $AllowedCidr
$env:FASHREC_DASHBOARD_DESIRED = "0"          # el dashboard se enciende explícitamente (-StartDashboard)

# Presupuesto con alerta (opcional)
if ($BudgetEmail) {
    $budget = @{ BudgetName = "fashrec-$Stage"; BudgetType = "COST"; TimeUnit = "MONTHLY"; BudgetLimit = @{ Amount = "10"; Unit = "USD" } } | ConvertTo-Json -Compress
    $notif = @(@{ Notification = @{ NotificationType = "ACTUAL"; ComparisonOperator = "GREATER_THAN"; Threshold = 80; ThresholdType = "PERCENTAGE" };
                  Subscribers = @(@{ SubscriptionType = "EMAIL"; Address = $BudgetEmail }) }) | ConvertTo-Json -Depth 5 -Compress
    if ($notif -notmatch '^\[') { $notif = "[$notif]" }
    Write-Utf8 (Join-Path $env:TEMP "fashrec-budget.json") $budget
    Write-Utf8 (Join-Path $env:TEMP "fashrec-budget-notif.json") $notif
    $exists = Query @("budgets", "describe-budget", "--account-id", $Account, "--budget-name", "fashrec-$Stage")
    if (-not $exists) {
        [void](Act @("budgets", "create-budget", "--account-id", $Account, "--budget", "file://$($env:TEMP -replace '\\','/')/fashrec-budget.json",
                     "--notifications-with-subscribers", "file://$($env:TEMP -replace '\\','/')/fashrec-budget-notif.json") -Soft)
        Ok "Presupuesto de 10 USD/mes con alerta al 80% a $BudgetEmail"
    } else { Ok "Presupuesto fashrec-$Stage ya existe" }
}

# ------------------------------------------------------------------------------------------ 2. platform
Step 2 "Servicio platform (S3, Glue Catalog, Athena, DynamoDB, ECR, ECS, IAM, SNS, SSM)"
Invoke-Sls "platform"
if (-not $DryRun) {
    $realBucket = Stack-Output "fashrec-platform-$Stage" "BucketName"
    if (-not $realBucket) { Fail "No encuentro las salidas de fashrec-platform-$Stage." }
    $Bucket = $realBucket
}
Ok "Bucket: s3://$Bucket"

# ------------------------------------------------------------------------------------------ 3. Glue
Step 3 "Código del job de Glue -> s3://$Bucket/glue-scripts/"
[void](Act @("s3", "cp", (Join-Path $Root "jobs\glue_etl_job.py"), "s3://$Bucket/glue-scripts/glue_etl_job.py", "--only-show-errors"))
[void](Act @("s3", "cp", (Join-Path $Root "configs\default.yaml"), "s3://$Bucket/glue-scripts/default.yaml", "--only-show-errors"))
if ($DryRun) { Write-Host "  [dry-run] $Py deploy\package_glue.py ; aws s3 cp deploy\build\fashrec_src.zip ..." -ForegroundColor DarkGray }
else {
    & $Py (Join-Path $Deploy "package_glue.py"); if ($LASTEXITCODE -ne 0) { Fail "No se pudo empaquetar src/ para Glue." }
    [void](Act @("s3", "cp", (Join-Path $Build "fashrec_src.zip"), "s3://$Bucket/glue-scripts/fashrec_src.zip", "--only-show-errors"))
}

# ------------------------------------------------------------------------------------------ 4. Imágenes
Step 4 "Imágenes Docker (pipeline y dashboard) -> ECR"
if ($SkipImages) { Warn "-SkipImages: se usan las imágenes que ya estén en ECR" }
else {
    $pipeUri = if ($DryRun) { "$Account.dkr.ecr.$Region.amazonaws.com/$Prefix-pipeline" } else { Stack-Output "fashrec-platform-$Stage" "PipelineRepositoryUri" }
    $dashUri = if ($DryRun) { "$Account.dkr.ecr.$Region.amazonaws.com/$Prefix-dashboard" } else { Stack-Output "fashrec-platform-$Stage" "DashboardRepositoryUri" }
    $registry = ($pipeUri -split "/")[0]
    New-Item -ItemType Directory -Force $Build | Out-Null
    $wslRepo = if ($DryRun) { "/mnt/c/repo" } else { ((& wsl -e wslpath -a ($Root -replace '\\', '/')) | Out-String).Trim() }
    $script = @"
set -euo pipefail
cd '$wslRepo'
docker build --platform linux/amd64 -f deploy/pipeline/Dockerfile -t '${pipeUri}:latest' .
docker push '${pipeUri}:latest'
docker build --platform linux/amd64 -f deploy/dashboard/Dockerfile -t '${dashUri}:latest' .
docker push '${dashUri}:latest'
"@
    $scriptPath = Join-Path $Build "push_images.sh"
    Write-Utf8 $scriptPath $script
    if ($DryRun) { Write-Host "  [dry-run] aws ecr get-login-password | wsl docker login $registry ; wsl bash deploy\build\push_images.sh" -ForegroundColor DarkGray }
    else {
        Write-Host "  login en $registry"
        aws ecr get-login-password --region $Region | wsl -e docker login --username AWS --password-stdin $registry
        if ($LASTEXITCODE -ne 0) { Fail "docker login a ECR falló." }
        Write-Host "  construyendo y subiendo imágenes (la primera vez tarda varios minutos: torch + dependencias)..."
        $wslScript = ((& wsl -e wslpath -a ($scriptPath -replace '\\', '/')) | Out-String).Trim()
        & wsl -e bash $wslScript
        if ($LASTEXITCODE -ne 0) { Fail "La construcción o el push de las imágenes falló (revisa la salida de Docker arriba)." }
        Ok "Imágenes en ECR: $pipeUri:latest y $dashUri:latest"
    }
}

# ------------------------------------------------------------------------------------------ 5. RAW
Step 5 "Datos originales (JSONL, 2.5 GB)"
if (-not $UploadRaw) { Warn "omitido (usa -UploadRaw para poder ejecutar el pipeline completo con Glue)" }
else {
    $raws = @(
        @{ File = (Join-Path $Root "Amazon_Fashion.jsonl");      Key = "raw/amazon-fashion/reviews/Amazon_Fashion.jsonl" },
        @{ File = (Join-Path $Root "meta_Amazon_Fashion.jsonl"); Key = "raw/amazon-fashion/metadata/meta_Amazon_Fashion.jsonl" })
    foreach ($r in $raws) {
        if (-not (Test-Path $r.File)) { Fail "No existe $($r.File)" }
        $size = (Get-Item $r.File).Length
        $remote = Query @("s3api", "head-object", "--bucket", $Bucket, "--key", $r.Key, "--query", "ContentLength", "--output", "text")
        if ($remote -and [int64]$remote -eq $size) { Ok "$($r.Key) ya está en S3 ($([math]::Round($size/1MB)) MB)"; continue }
        Write-Host "  subiendo $($r.Key) ($([math]::Round($size/1MB)) MB)..."
        [void](Act @("s3", "cp", $r.File, "s3://$Bucket/$($r.Key)", "--checksum-algorithm", "SHA256", "--only-show-errors"))
    }
}

# ------------------------------------------------------------------------------------------ 6. Seed
$ModelVersion = $null
$currentJson = Join-Path $Root "data\published\current.json"
if (Test-Path $currentJson) { $ModelVersion = (Get-Content $currentJson -Raw | ConvertFrom-Json).model_version }
Step 6 "Resultados del modelo local -> S3 y DynamoDB (-Seed)"
if (-not $Seed) { Warn "omitido (sin -Seed, la API y el dashboard no tendrán datos hasta que corra el pipeline en AWS)" }
elseif (-not $ModelVersion) { Fail "No hay data\published\current.json: ejecuta primero el pipeline local (python -m jobs.run_pipeline)." }
else {
    if ($DryRun) { Write-Host "  [dry-run] $Py deploy\seed_lake.py --bucket $Bucket --model-version $ModelVersion" -ForegroundColor DarkGray }
    else {
        Push-Location $Root
        try {
            & $Py (Join-Path $Deploy "seed_lake.py") --bucket $Bucket --model-version $ModelVersion
            if ($LASTEXITCODE -ne 0) { Fail "seed_lake.py falló." }
            $env:TABLE_PREFIX = $Prefix
            $env:ACTIVE_VERSION_PARAM = Stack-Output "fashrec-platform-$Stage" "ActiveVersionParam"
            Write-Host "  cargando recomendaciones en DynamoDB y activando la versión en SSM..."
            & $Py -m jobs.load_serving_job --model-version $ModelVersion
            if ($LASTEXITCODE -ne 0) { Fail "load_serving_job falló." }
        } finally { Pop-Location }
    }
}

# ------------------------------------------------------------------------------------------ 7. API
Step 7 "Servicio api (Lambda + API Gateway) y pruebas funcionales"
$layerDir = Join-Path $Deploy "api\layer\python"
if (-not (Test-Path $layerDir)) {
    if ($DryRun) { Write-Host "  [dry-run] powershell -File deploy\api\build_layer.ps1" -ForegroundColor DarkGray }
    else {
        Write-Host "  construyendo la capa numpy + faiss (ruedas de Linux, sin Docker)..."
        $env:PATH = "$(Split-Path $Py);$env:PATH"
        Push-Location (Join-Path $Deploy "api")
        try { & powershell -NoProfile -File build_layer.ps1 } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { Fail "build_layer.ps1 falló." }
    }
} else { Ok "capa ya construida (borra deploy\api\layer para reconstruirla)" }
Invoke-Sls "api"
$ApiUrl = if ($DryRun) { "https://<api-id>.execute-api.$Region.amazonaws.com" } else { Stack-Output "fashrec-api-$Stage" "ApiUrl" }
Ok "API: $ApiUrl"
if ($Seed -and -not $DryRun) {
    Write-Host "`n  --- pruebas funcionales (la primera llamada a /v1/recommend incluye el arranque en frío) ---"
    Push-Location $Root
    try { & $Py (Join-Path $Deploy "smoke_test.py") --api-url $ApiUrl --model-version $ModelVersion } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { Warn "alguna prueba de la API falló: revisa la salida y CloudWatch Logs de las Lambdas." }
} elseif (-not $Seed) { Warn "pruebas omitidas: sin -Seed no hay datos que consultar" }

# ------------------------------------------------------------------------------------------ 8. pipeline + dashboard
Step 8 "Servicios pipeline (Glue + Fargate + Step Functions) y dashboard"
Invoke-Sls "pipeline"
Invoke-Sls "dashboard"
$PipelineArn = if ($DryRun) { "arn:aws:states:${Region}:${Account}:stateMachine:$Prefix-pipeline" } else { Stack-Output "fashrec-pipeline-$Stage" "PipelineArn" }

# ------------------------------------------------------------------------------------------ 9. Pipeline
Step 9 "Ejecución del pipeline en AWS (-RunPipeline)"
$ExecArn = $null
if (-not $RunPipeline) { Warn "omitido (usa -RunPipeline; requiere -UploadRaw la primera vez)" }
else {
    foreach ($k in @("raw/amazon-fashion/reviews/Amazon_Fashion.jsonl", "raw/amazon-fashion/metadata/meta_Amazon_Fashion.jsonl")) {
        if (-not $DryRun -and -not (Query @("s3api", "head-object", "--bucket", $Bucket, "--key", $k))) { Fail "Falta s3://$Bucket/$k. Ejecuta con -UploadRaw." }
    }
    $ver = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmm") + "_aws"
    $inputJson = Join-Path $env:TEMP "fashrec-sfn-input.json"
    Write-Utf8 $inputJson ('{"model_version":"' + $ver + '"}')
    if ($DryRun) { Write-Host "  [dry-run] aws stepfunctions start-execution --state-machine-arn $PipelineArn --input {model_version:$ver}" -ForegroundColor DarkGray }
    else {
        $ExecArn = (& aws stepfunctions start-execution --state-machine-arn $PipelineArn --input "file://$($inputJson -replace '\\','/')" --query executionArn --output text).Trim()
        if ($LASTEXITCODE -ne 0) { Fail "No se pudo iniciar la ejecución." }
        Ok "Ejecución iniciada: $ExecArn"
        Ok "Versión del modelo que se entrenará: $ver"
        if ($WaitPipeline) {
            Write-Host "  esperando (ETL en Glue + 6 tareas Fargate; en AWS puede tardar de 30 a 90 minutos)..."
            do {
                Start-Sleep 60
                $status = (& aws stepfunctions describe-execution --execution-arn $ExecArn --query status --output text).Trim()
                Write-Host ("  {0:HH:mm:ss}  {1}" -f (Get-Date), $status)
            } while ($status -eq "RUNNING")
            if ($status -ne "SUCCEEDED") { Warn "La ejecución terminó en $status. Revisa Step Functions > Executions y los logs de CloudWatch (/fashrec/$Stage/pipeline)." }
            else { Ok "Pipeline completado: la API y el dashboard sirven la versión $ver" }
        } else { Warn "Sigue en ejecución: mírala en Step Functions > State machines > $Prefix-pipeline" }
    }
}

# ------------------------------------------------------------------------------------------ 10. Dashboard
Step 10 "Dashboard (-StartDashboard)"
$DashUrl = $null
if (-not $StartDashboard) { Warn "apagado (desiredCount 0). Para encenderlo: ecs update-service ... --desired-count 1, o vuelve a correr con -StartDashboard" }
else {
    $hasData = Query @("s3api", "head-object", "--bucket", $Bucket, "--key", "published/current.json")
    if (-not $DryRun -and -not $hasData) { Warn "todavía no hay versión publicada en S3 (published/current.json): usa -Seed o espera a que termine el pipeline. No se enciende." }
    else {
        [void](Act @("ecs", "update-service", "--cluster", $Prefix, "--service", "$Prefix-dashboard", "--desired-count", "1", "--query", "service.serviceName", "--output", "text"))
        if (-not $DryRun) {
            Write-Host "  esperando la tarea (descarga la imagen y ~1 GB de datos de S3; 3 a 6 minutos)..."
            $ip = $null
            for ($i = 0; $i -lt 60 -and -not $ip; $i++) {
                Start-Sleep 15
                $task = Query @("ecs", "list-tasks", "--cluster", $Prefix, "--service-name", "$Prefix-dashboard", "--desired-status", "RUNNING", "--query", "taskArns[0]", "--output", "text")
                if (-not $task -or $task -eq "None") { continue }
                $eni = Query @("ecs", "describe-tasks", "--cluster", $Prefix, "--tasks", $task, "--query", "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value | [0]", "--output", "text")
                if ($eni -and $eni -ne "None") { $ip = Query @("ec2", "describe-network-interfaces", "--network-interface-ids", $eni, "--query", "NetworkInterfaces[0].Association.PublicIp", "--output", "text") }
                if ($ip -eq "None") { $ip = $null }
            }
            if ($ip) {
                $DashUrl = "http://${ip}:8501"
                for ($i = 0; $i -lt 40; $i++) {
                    try { if ((Invoke-WebRequest "$DashUrl/_stcore/health" -UseBasicParsing -TimeoutSec 5).Content -match "ok") { break } } catch { }
                    Start-Sleep 10
                }
                Ok "Dashboard: $DashUrl   (solo accesible desde $AllowedCidr)"
            } else { Warn "la tarea no obtuvo IP pública a tiempo: revisa ECS > $Prefix > $Prefix-dashboard > Tasks y CloudWatch Logs /fashrec/$Stage/dashboard" }
        }
    }
}

# ------------------------------------------------------------------------------------------ Evidencia en Athena
if ($Seed -and -not $DryRun) {
    Info "`nConsultas de Athena (evidencia de particionamiento: compara 'Data scanned')"
    function Invoke-Athena([string]$sql) {
        $qid = (& aws athena start-query-execution --query-string $sql --work-group $Prefix --query-execution-context "Database=fashrec_$Stage" --query QueryExecutionId --output text).Trim()
        for ($i = 0; $i -lt 60; $i++) {
            $q = (& aws athena get-query-execution --query-execution-id $qid --output json | ConvertFrom-Json).QueryExecution
            if ($q.Status.State -in @("SUCCEEDED", "FAILED", "CANCELLED")) { break }
            Start-Sleep 2
        }
        $res = if ($q.Status.State -eq "SUCCEEDED") { ((& aws athena get-query-results --query-execution-id $qid --output json | ConvertFrom-Json).ResultSet.Rows | Select-Object -Skip 1 | Select-Object -First 1).Data[0].VarCharValue } else { $q.Status.StateChangeReason }
        "  {0,-8} resultado={1,-10} escaneado={2,12:N0} bytes   {3}" -f $q.Status.State, $res, $q.Statistics.DataScannedInBytes, $sql
    }
    Invoke-Athena "SELECT count(DISTINCT user_id) FROM interactions"
    Invoke-Athena "SELECT count(DISTINCT user_id) FROM interactions WHERE review_year = 2021"
    Invoke-Athena "SELECT count(DISTINCT parent_asin) FROM recommendations WHERE model_version = '$ModelVersion'"
}

# ------------------------------------------------------------------------------------------ Resumen
$c = "https://console.aws.amazon.com"
Write-Host "`n================ RESUMEN ================" -ForegroundColor Green
Write-Host "Cuenta $Account | $Region | stage $Stage | bucket s3://$Bucket"
if ($ApiUrl) { Write-Host "API:       $ApiUrl/v1/health" }
if ($DashUrl) { Write-Host "Dashboard: $DashUrl" }
if ($ExecArn) { Write-Host "Pipeline:  $ExecArn" }
Write-Host @"

Dónde tomar evidencia en la consola de AWS (región $Region):
  CloudFormation  $c/cloudformation/home?region=$Region#/stacks         4 stacks fashrec-*-$Stage en CREATE_COMPLETE
  S3              $c/s3/buckets/${Bucket}?region=$Region                raw/, curated/ (review_year=...), model_data/, artifacts/, inference/
  Glue            $c/glue/home?region=$Region#/v2/data-catalog/databases   base fashrec_$Stage y tablas; Jobs: $Prefix-etl
  Athena          $c/athena/home?region=$Region#/query-editor           workgroup $Prefix; consultas con 'Data scanned'
  DynamoDB        $c/dynamodb/home?region=$Region#tables                 $Prefix-recs-<versión> (Explore items) y $Prefix-products
  Systems Manager $c/systems-manager/parameters?region=$Region           /$Prefix/active_version
  API Gateway     $c/apigateway/main/apis?region=$Region               API $Prefix (rutas y throttling)
  Lambda          $c/lambda/home?region=$Region#/functions               fashrec-api-$Stage-lookup y -recommend (Monitor, Logs)
  Step Functions  $c/states/home?region=$Region#/statemachines           $Prefix-pipeline (grafo de la ejecución)
  ECS             $c/ecs/v2/clusters/$Prefix/services?region=$Region     $Prefix-dashboard (tareas y logs); tareas del pipeline
  CloudWatch      $c/cloudwatch/home?region=$Region#logsV2:log-groups    /fashrec/$Stage/pipeline, /fashrec/$Stage/dashboard, /aws/lambda/...
  ECR             $c/ecr/repositories?region=$Region                     imágenes de pipeline y dashboard
  Billing         $c/billing/home#/budgets                               presupuesto y costos

TUMBAR TODO (evita sobrecostos):
  powershell -File deploy\destroy.ps1              # servicios; conserva el bucket (≈ `$0.1/mes)
  powershell -File deploy\destroy.ps1 -IncludeData # además borra el bucket y deja la cuenta sin nada de FashRec
Solo apagar el dashboard (lo único que cobra por hora):
  aws ecs update-service --cluster $Prefix --service $Prefix-dashboard --desired-count 0
"@
if ($DryRun) { Warn "Fue un -DryRun: no se modificó nada en AWS." }
