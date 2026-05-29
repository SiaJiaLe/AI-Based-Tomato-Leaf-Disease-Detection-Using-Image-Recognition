# Interactive shell in model_trainer (GPU). Run training scripts manually inside.
# Usage:
#   .\docker\scripts\train-shell.ps1           # shell only (image must exist)
#   .\docker\scripts\train-shell.ps1 -Build    # build image first, then shell
#
# Inside container (WORKDIR /app):
#   bash scripts/run_training.sh
#   python src/train.py
#   exit

param(
    [switch]$Build,
    [Parameter(ValueFromRemainingArguments = $true)]
    $Args
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Yellow
}

if ($Build) {
    & (Join-Path $PSScriptRoot "build-train-gpu.ps1")
}

Write-Host ""
Write-Host "Opening training container shell (GPU override)..." -ForegroundColor Cyan
Write-Host "  Working directory: /app  (resnet34_model/)" -ForegroundColor DarkGray
Write-Host "  Full pipeline:     bash scripts/run_training.sh" -ForegroundColor DarkGray
Write-Host "  Single script:     python src/train.py" -ForegroundColor DarkGray
Write-Host ""

docker compose `
  -f docker-compose.yml `
  -f docker-compose.training.yml `
  --profile training `
  run --rm -it model_trainer bash @Args
