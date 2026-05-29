param([Parameter(ValueFromRemainingArguments=$true)]$Args)
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
docker compose up --build @Args
