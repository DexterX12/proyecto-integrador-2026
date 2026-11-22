# Construye la capa de dependencias (numpy + faiss-cpu para Linux x86_64, Python 3.13) en deploy/api/layer/python.
# No requiere Docker: usa ruedas precompiladas (--only-binary) y falla si no existe una compatible.
# Uso (desde deploy/api):  powershell -File build_layer.ps1
$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "layer\python"
if (Test-Path (Join-Path $PSScriptRoot "layer")) { Remove-Item -Recurse -Force (Join-Path $PSScriptRoot "layer") }
New-Item -ItemType Directory -Force $target | Out-Null
python -m pip install numpy faiss-cpu `
    --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 `
    --python-version 3.13 --implementation cp --only-binary=:all: --target $target
if ($LASTEXITCODE -ne 0) { throw "No se pudieron obtener ruedas compatibles con Lambda (Linux x86_64, Python 3.13)" }
# Reducir tamaño: sin metadatos, binarios de línea de comandos ni cachés
Get-ChildItem $target -Directory -Filter "*.dist-info" | Remove-Item -Recurse -Force
Get-ChildItem $target -Directory -Filter "bin" | Remove-Item -Recurse -Force
Get-ChildItem $target -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
$mb = (Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
"Capa lista: {0:N0} MB descomprimida (límite de Lambda: 250 MB entre código y capas)" -f $mb
