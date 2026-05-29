# Build the model_trainer image with GPU compose override (no container start).
# Usage: .\docker\scripts\build-train-gpu.ps1

param([Parameter(ValueFromRemainingArguments = $true)]$Args)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

Write-Host "Building model_trainer (CUDA image + GPU compose override)..." -ForegroundColor Cyan
docker compose `
  -f docker-compose.yml `
  -f docker-compose.training.yml `
  build model_trainer @Args

Write-Host ""
Write-Host "Done. Next: .\docker\scripts\train-shell.ps1" -ForegroundColor Green
