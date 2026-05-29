# AI-Based Tomato Leaf Disease Detection Using Image Recognition

**FYP — Sia Jia Le (22062566), Sunway University**

Monorepo for tomato leaf disease classification: Vue 3 frontend, FastAPI backend (DDD), and ResNet34 training module. The frontend and backend are **decoupled** and communicate only via REST API (`/api/v1/*`).

## Repository structure

```
├── backend/              # FastAPI REST API (Domain-Driven Design)
├── frontend/             # Vue 3 + Vite web UI
├── resnet34_model/       # PyTorch training, evaluation, export (+ HPC scripts)
├── docker/               # Docker documentation and helper scripts
├── docker-compose.yml          # includes PostgreSQL
├── docker-compose.training.yml
└── implementation_plan_v2.md
```

## Docker

See **[docker/README.md](docker/README.md)** for full setup, GPU training, and troubleshooting.

```powershell
copy .env.example .env
docker compose up --build
```

Helper scripts (PowerShell):

```powershell
docker\scripts\up.cmd                  # backend + frontend + postgres
docker\scripts\build-train-gpu.cmd     # build ML training image (GPU)
docker\scripts\train-shell.cmd         # shell → bash scripts/run_training.sh
```

(`.ps1` versions exist; use `.cmd` if PowerShell blocks scripts. See [docker/README.md](docker/README.md).)

| Service  | URL |
|----------|-----|
| Frontend | http://localhost:5173 |
| API docs | http://localhost:8000/docs |
| Health   | http://localhost:8000/api/v1/health |

## Training (GPU)

- **Sunway HPC (primary):** [resnet34_model/scripts/hpc/README.md](resnet34_model/scripts/hpc/README.md)
- **Local Docker (optional):**

```powershell
docker compose -f docker-compose.yml -f docker-compose.training.yml --profile training up model_trainer --build
```

## API endpoints (scaffold)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Service health |
| POST | `/api/v1/predict` | Upload image, predict disease |
| GET | `/api/v1/predictions` | List prediction history |
| GET | `/api/v1/predictions/{id}` | Get single prediction |

Full contract: [implementation_plan_v2.md](implementation_plan_v2.md).

## Status

**Phase 0 (current):** folder structure and placeholder files. Business logic, ML training, and UI workflows come in later phases.

## Documentation

- [implementation_plan.md](implementation_plan.md) — original plan
- [implementation_plan_v2.md](implementation_plan_v2.md) — amended plan (API-first, HPC, scaffold)
- [docker/README.md](docker/README.md) — Docker setup guide
