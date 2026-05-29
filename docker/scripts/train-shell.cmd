@echo off
cd /d "%~dp0..\.."
if not exist .env copy .env.example .env
echo Opening training container shell...
echo   Inside container: bash scripts/run_training.sh
docker compose -f docker-compose.yml -f docker-compose.training.yml --profile training run --rm -it model_trainer bash %*
