# AI-Based Tomato Leaf Disease Detection Using Image Recognition

**FYP — Sia Jia Le (22062566), Sunway University**

Monorepo for tomato leaf disease classification: Vue 3 frontend, FastAPI backend (DDD), and ResNet34 training module. The frontend and backend are **decoupled** and communicate only via REST API (`/api/v1/*`).

## Repository structure

```
├── backend/          # FastAPI REST API (Domain-Driven Design)
├── frontend/         # Vue 3 + Vite web UI
├── resnet34_model/   # PyTorch training, evaluation, export (+ HPC scripts)
├── docker-compose.yml
└── implementation_plan_v2.md
```

## Quick start (scaffold — local dev)

1. Copy environment file:
   ```bash
   cp .env.example .env
   ```

2. Start backend and frontend:
   ```bash
   docker compose up --build
   ```

3. Open:
   - Frontend: http://localhost:5173
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/api/v1/health

## Training (GPU)

- **Sunway HPC (primary):** see [resnet34_model/scripts/hpc/README.md](resnet34_model/scripts/hpc/README.md) — Docker/Singularity first, Python venv fallback.
- **Local (optional):** `docker compose --profile training up model_trainer` (requires NVIDIA GPU).

## API endpoints (scaffold)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Service health |
| POST | `/api/v1/predict` | Upload image, predict disease |
| GET | `/api/v1/predictions` | List prediction history |
| GET | `/api/v1/predictions/{id}` | Get single prediction |

Full contract: [implementation_plan_v2.md](implementation_plan_v2.md).

## Status

**Phase 0 (current):** folder structure and placeholder files only. Business logic, ML training, and UI workflows are implemented in later phases.

## Documentation

- [implementation_plan.md](implementation_plan.md) — original plan
- [implementation_plan_v2.md](implementation_plan_v2.md) — amended plan (API-first, HPC, scaffold)
