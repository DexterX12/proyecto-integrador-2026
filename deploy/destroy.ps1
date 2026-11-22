<#
.SYNOPSIS
  Derriba todos los servicios FashRec de AWS con un solo comando (dashboard, api, pipeline, platform).

.DESCRIPTION
  Orden: detener lo que esté corriendo -> serverless remove de dashboard, api, pipeline, platform
  -> borrar lo que los stacks no poseen (tablas DynamoDB por versión, imágenes de ECR)
  -> (opcional) vaciar y borrar el bucket de datos.

  El bucket de datos tiene DeletionPolicy: Retain: por defecto se CONSERVA (cuesta ~ $0.1 al mes con ~4 GB).
  Con -IncludeData se borra también, y ya no queda nada facturable de este proyecto.

.EXAMPLE
  powershell -File deploy\destroy.ps1 -DryRun                 # solo muestra qué haría
  powershell -File deploy\destroy.ps1                         # derriba servicios, conserva el bucket
  powershell -File deploy\destroy.ps1 -IncludeData            # derriba todo, incluido el bucket
  powershell -File deploy\destroy.ps1 -Sls "npx osls@3"       # si usan osls en lugar de serverless
#>
[CmdletBinding()]
param(
    [string]$Stage = "dev",
    [string]$Region = "us-east-1",
    [string]$Sls = "serverless",
    [switch]$IncludeData,
    [switch]$DryRun,
    [switch]$Yes
)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$prefix = "fashrec-$Stage"
$env:AWS_REGION = $Region
$env:AWS_DEFAULT_REGION = $Region

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Warn($m) { Write-Host $m -ForegroundColor Yellow }

# Consulta de solo lectura: se ejecuta siempre (también en -DryRun).
function Query { param([string[]]$a) $o = & aws @a 2>$null; if ($LASTEXITCODE -ne 0) { return $null }; return $o }
# Acción que modifica AWS: en -DryRun solo se imprime.
function Act {
    param([string[]]$a)
    if ($DryRun) { Write-Host "  [dry-run] aws $($a -join ' ')" -ForegroundColor DarkGray; return }
    & aws @a 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Warn "  aviso: falló 'aws $($a[0..2] -join ' ')' (código $LASTEXITCODE), se continúa" }
}

# --- 0. Identidad y confirmación ---------------------------------------------------------------
$account = Query @("sts", "get-caller-identity", "--query", "Account", "--output", "text")
if (-not $account) {
    Write-Host "No hay credenciales válidas de AWS (aws sts get-caller-identity falló). Nada se borró." -ForegroundColor Red
    exit 1
}
$bucket = "fashrec-data-$account-$Region-$Stage"
Info "Cuenta: $account | Región: $Region | Stage: $Stage"
Info "Se derribarán los stacks: fashrec-dashboard-$Stage, fashrec-api-$Stage, fashrec-pipeline-$Stage, fashrec-platform-$Stage"
if ($IncludeData) { Warn "Se BORRARÁ el bucket s3://$bucket con todo su contenido (incluidas versiones)." }
else { Info "El bucket s3://$bucket se conserva (usa -IncludeData para borrarlo)." }

if (-not $Yes -and -not $DryRun) {
    $answer = Read-Host "Escribe 'borrar' para continuar"
    if ($answer -ne "borrar") { Write-Host "Cancelado."; exit 0 }
}

# --- 1. Detener lo que esté corriendo (sigue cobrando y bloquea el borrado) ---------------------
Info "`n[1/5] Deteniendo ejecuciones y tareas activas"
$sfnArn = Query @("stepfunctions", "list-state-machines", "--query", "stateMachines[?name=='$prefix-pipeline'].stateMachineArn | [0]", "--output", "text")
if ($sfnArn -and $sfnArn -ne "None") {
    $execs = Query @("stepfunctions", "list-executions", "--state-machine-arn", $sfnArn, "--status-filter", "RUNNING", "--query", "executions[].executionArn", "--output", "text")
    foreach ($e in ($execs -split "\s+" | Where-Object { $_ })) { Act @("stepfunctions", "stop-execution", "--execution-arn", $e) }
}
$runs = Query @("glue", "get-job-runs", "--job-name", "$prefix-etl", "--query", "JobRuns[?JobRunState=='RUNNING'||JobRunState=='STARTING'].Id", "--output", "text")
foreach ($r in ($runs -split "\s+" | Where-Object { $_ })) { Act @("glue", "batch-stop-job-run", "--job-name", "$prefix-etl", "--job-run-ids", $r) }
$svc = Query @("ecs", "describe-services", "--cluster", $prefix, "--services", "$prefix-dashboard", "--query", "services[?status=='ACTIVE'].serviceName", "--output", "text")
if ($svc) { Act @("ecs", "update-service", "--cluster", $prefix, "--service", "$prefix-dashboard", "--desired-count", "0") }
$tasks = Query @("ecs", "list-tasks", "--cluster", $prefix, "--query", "taskArns", "--output", "text")
foreach ($t in ($tasks -split "\s+" | Where-Object { $_ })) { Act @("ecs", "stop-task", "--cluster", $prefix, "--task", $t) }

# --- 2. Vaciar imágenes de ECR (un repositorio con imágenes no se puede borrar) -----------------
Info "`n[2/5] Vaciando repositorios de ECR"
foreach ($repo in @("$prefix-pipeline", "$prefix-dashboard")) {
    $ids = Query @("ecr", "list-images", "--repository-name", $repo, "--query", "imageIds[*].imageDigest", "--output", "text")
    foreach ($d in ($ids -split "\s+" | Where-Object { $_ } | Select-Object -Unique)) {
        Act @("ecr", "batch-delete-image", "--repository-name", $repo, "--image-ids", "imageDigest=$d")
    }
}

# --- 3. serverless remove en orden inverso ------------------------------------------------------
Info "`n[3/5] Eliminando los 4 servicios (serverless remove)"
# serverless remove también resuelve el yml: las variables obligatorias deben existir aunque su valor ya no importe.
foreach ($v in @("FASHREC_VPC_ID", "FASHREC_SUBNET_ID", "FASHREC_ALLOWED_CIDR")) {
    if (-not (Get-Item "Env:$v" -ErrorAction SilentlyContinue)) { Set-Item "Env:$v" "unused-for-remove" }
}
$slsParts = $Sls -split "\s+" | Where-Object { $_ }
foreach ($svcName in @("dashboard", "api", "pipeline", "platform")) {
    $stack = "fashrec-$svcName-$Stage"
    $exists = Query @("cloudformation", "describe-stacks", "--stack-name", $stack, "--query", "Stacks[0].StackStatus", "--output", "text")
    if (-not $exists) { Write-Host "  $stack no existe, se omite"; continue }
    Write-Host "  removiendo $stack ($exists)"
    if ($DryRun) { Write-Host "  [dry-run] $Sls remove --stage $Stage --region $Region  (en deploy\$svcName)" -ForegroundColor DarkGray; continue }
    Push-Location (Join-Path $root $svcName)
    try { & $slsParts[0] @($slsParts[1..($slsParts.Count)] | Where-Object { $_ }) remove --stage $Stage --region $Region }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Falló 'serverless remove' de $svcName. Se detiene para no dejar dependencias rotas." -ForegroundColor Red
        Write-Host "Revisa el evento de error en CloudFormation > Stacks > $stack > Events, corrige y vuelve a ejecutar este script." -ForegroundColor Red
        exit 1
    }
}

# --- 4. Lo que los stacks no poseen: tablas de recomendaciones por versión ---------------------
Info "`n[4/5] Borrando tablas DynamoDB creadas por load_serving_job"
$tables = Query @("dynamodb", "list-tables", "--query", "TableNames[?starts_with(@, '$prefix-')]", "--output", "text")
foreach ($t in ($tables -split "\s+" | Where-Object { $_ })) { Act @("dynamodb", "delete-table", "--table-name", $t) }

# --- Presupuesto creado por deploy.ps1 -BudgetEmail (si existe) --------------------------------------
if (Query @("budgets", "describe-budget", "--account-id", $account, "--budget-name", $prefix)) {
    Info "`nBorrando el presupuesto $prefix"
    Act @("budgets", "delete-budget", "--account-id", $account, "--budget-name", $prefix)
}

# --- 5. Bucket de datos (opcional) ---------------------------------------------------------------
Info "`n[5/5] Bucket de datos"
$null = Query @("s3api", "head-bucket", "--bucket", $bucket)
$bucketExists = ($LASTEXITCODE -eq 0)
if ($IncludeData) {
    if ($DryRun) { Write-Host "  [dry-run] vaciar versiones y borrar s3://$bucket" -ForegroundColor DarkGray }
    else {
        $tmp = Join-Path $env:TEMP "fashrec-delete-batch.json"
        do {
            $json = (& aws s3api list-object-versions --bucket $bucket --max-items 1000 --output json 2>$null) -join "`n"
            $page = if ($json) { $json | ConvertFrom-Json } else { $null }
            $objs = @(); if ($page) { $objs += @($page.Versions); $objs += @($page.DeleteMarkers) }
            $objs = @($objs | Where-Object { $_ })
            if ($objs.Count -gt 0) {
                $payload = @{ Objects = @($objs | ForEach-Object { @{ Key = $_.Key; VersionId = $_.VersionId } }); Quiet = $true } | ConvertTo-Json -Depth 5 -Compress
                [IO.File]::WriteAllText($tmp, $payload, (New-Object Text.UTF8Encoding($false)))
                & aws s3api delete-objects --bucket $bucket --delete "file://$tmp" | Out-Null
                Write-Host "  borrados $($objs.Count) objetos/versiones"
            }
        } while ($objs.Count -gt 0)
        Remove-Item $tmp -ErrorAction SilentlyContinue
        & aws s3api delete-bucket --bucket $bucket 2>&1 | Out-Null
    }
} else {
    if ($bucketExists) { Warn "  conservado: s3://$bucket (≈ `$0.1/mes). Para borrarlo: deploy\destroy.ps1 -IncludeData" }
}

# --- Verificación final ----------------------------------------------------------------------------
Info "`nVerificación: lo que queda en la cuenta"
$left = Query @("cloudformation", "list-stacks", "--stack-status-filter", "CREATE_COMPLETE", "UPDATE_COMPLETE", "ROLLBACK_COMPLETE", "DELETE_FAILED",
                "--query", "StackSummaries[?starts_with(StackName, 'fashrec-') && ends_with(StackName, '-$Stage')].StackName", "--output", "text")
Write-Host ("  stacks fashrec-*-{0}: {1}" -f $Stage, $(if ($left) { $left } else { "ninguno" }))
$leftTables = Query @("dynamodb", "list-tables", "--query", "TableNames[?starts_with(@, '$prefix-')]", "--output", "text")
Write-Host ("  tablas DynamoDB: {0}" -f $(if ($leftTables) { $leftTables } else { "ninguna" }))
$null = Query @("s3api", "head-bucket", "--bucket", $bucket)
Write-Host ("  bucket de datos: {0}" -f $(if ($LASTEXITCODE -eq 0) { "existe (sin costo relevante)" } else { "no existe" }))
if ($DryRun) { Warn "`nFue un -DryRun: no se modificó nada." }
