# Docker setup

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine + Compose v2 (Linux)
- For **GPU training** locally: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### PowerShell: "running scripts is disabled"

Windows may block `*.ps1` files. Use either:

- **`.cmd` wrappers** (no policy change): `docker\scripts\build-train-gpu.cmd`
- **Bypass once**: `powershell -ExecutionPolicy Bypass -File .\docker\scripts\build-train-gpu.ps1`
- **Allow scripts for your user** (one time): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- **Run Docker directly** (see commands in each section below)

## Quick start (backend + frontend)

From the repository root:

```powershell
copy .env.example .env
docker compose up --build
```

| Service    | URL |
|------------|-----|
| Frontend   | http://localhost:5173 |
| API docs   | http://localhost:8000/docs |
| Health     | http://localhost:8000/api/v1/health |
| PostgreSQL | `localhost:5432` (user/db: see `.env`) |

Stop: `Ctrl+C` or `docker compose down`

Detached: `docker compose up --build -d`

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Backend + frontend + PostgreSQL (daily dev) |
| `docker-compose.training.yml` | GPU override for `model_trainer` |
| `backend/Dockerfile` | FastAPI + uvicorn (dev reload) |
| `frontend/Dockerfile` | Vue 3 + Vite dev server |
| `resnet34_model/Dockerfile` | PyTorch CUDA training image |
| `resnet34_model/Dockerfile.cpu` | CPU-only training (no GPU) |

Each service has a `.dockerignore` to keep builds fast.

## Training container

### Recommended workflow: build once, shell in, run `.sh`

**1. Build the training image (GPU override):**

```powershell
docker\scripts\build-train-gpu.cmd
```

Or: `docker compose -f docker-compose.yml -f docker-compose.training.yml build model_trainer`

**2. Open an interactive shell in the container:**

```powershell
docker\scripts\train-shell.cmd
```

Or: `docker compose -f docker-compose.yml -f docker-compose.training.yml --profile training run --rm -it model_trainer bash`

**3. Inside the container** (`WORKDIR` is `/app` = `resnet34_model/`):

```bash
bash scripts/run_training.sh
# or run steps individually:
# python scripts/prepare_dataset.py
# python src/train.py
```

Type `exit` to leave the container. Edit Python files on your PC — they are mounted live; no rebuild unless `requirements.txt` or `Dockerfile` changes.

Artifacts go to `resnet34_model/outputs/` on your machine (and the `model_artifacts` volume at `/app/outputs`).

### One-shot training (attached logs, no shell)

```powershell
.\docker\scripts\train-gpu.ps1
```

Same as `docker compose ... up model_trainer --build` — runs `python src/train.py` only (not the full `.sh` pipeline).

### With NVIDIA GPU (Linux / WSL2 with GPU support)

GPU settings come from `docker-compose.training.yml`. Requires NVIDIA Container Toolkit / Docker Desktop GPU support.

Artifacts are written under `resnet34_model/outputs/` and the `model_artifacts` volume.

### CPU only (smoke test / no GPU)

```powershell
docker build -f resnet34_model/Dockerfile.cpu -t tomato-trainer-cpu ./resnet34_model
docker run --rm -v ${PWD}/resnet34_model:/app -w /app tomato-trainer-cpu python -c "import torch; print(torch.__version__)"
```

Training will fail with `NotImplementedError` until `src/train.py` is implemented — that is expected in Phase 0.

## Useful commands

```powershell
# Rebuild one service
docker compose build backend
docker compose up backend

# Logs
docker compose logs -f backend

# Shell inside backend
docker compose exec backend bash

# Remove containers and volumes
docker compose down -v
```

## Troubleshooting

### `env file .env not found`

```powershell
copy .env.example .env
```

### Frontend cannot reach API

`VITE_API_URL` must be the URL **your browser** uses (`http://localhost:8000`), not `http://backend:8000`.

### PostCSS / `Unexpected token ''` on `package.json`

**Cause:** UTF-8 **BOM** on `package.json` (and often all frontend files) from Windows PowerShell / OneDrive. PostCSS tries to parse `package.json` and fails on the invisible `` byte.

**Fix applied in repo:** BOM stripped from `frontend/`, plus `postcss.config.cjs` and `.editorconfig`.

Restart frontend (bind mount picks up files immediately):

```powershell
docker compose restart frontend
```

If it still fails, reset `node_modules` volume and rebuild:

```powershell
docker compose down
docker volume rm tomato-disease_frontend_node_modules
docker compose up --build frontend
```

In Cursor/VS Code: save config files as **UTF-8** (not "UTF-8 with BOM").

### Backend health check fails / slow first start

First build installs PyTorch and can take several minutes. Increase `start_period` in `docker-compose.yml` if needed.

### GPU training on Windows

Docker Desktop GPU support requires WSL2 backend + NVIDIA drivers. Otherwise use **Sunway HPC** (`resnet34_model/scripts/hpc/`) for training.

### Port already in use

Change `BACKEND_PORT`, `FRONTEND_PORT`, or `POSTGRES_PORT` in `.env`.

### Database connection failed (`database_connected: false`)

- Ensure the `postgres` container is healthy: `docker compose ps`
- `DATABASE_URL` must use host `postgres` inside Docker (not `localhost`)
- After changing DB credentials, reset volume: `docker compose down -v` (deletes data)
