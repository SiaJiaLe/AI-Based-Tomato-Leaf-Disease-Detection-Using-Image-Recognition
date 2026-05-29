@echo off
cd /d "%~dp0..\.."
echo Building model_trainer (CUDA image + GPU compose override)...
docker compose -f docker-compose.yml -f docker-compose.training.yml build model_trainer %*
if errorlevel 1 exit /b 1
echo.
echo Done. Next: docker\scripts\train-shell.cmd
