param([Parameter(ValueFromRemainingArguments=$true)]$Args)
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
docker compose -f docker-compose.yml -f docker-compose.training.yml --profile training up model_trainer --build @Args
