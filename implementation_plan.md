# Implementation Plan: AI-Based Tomato Leaf Disease Detection System
**FYP by Sia Jia Le (22062566) — Sunway University**

---

## Overview

This plan translates your Capstone Project 1 proposal into a fully Dockerized, production-ready development environment split across three top-level directories:

```
tomato-disease-detection/
├── backend/          # FastAPI (Python) — REST API + DDD business logic
├── frontend/         # Vue 3 — Web UI for image upload & prediction
└── resnet34_model/   # PyTorch training, evaluation, and model export
```

---

## 1. Top-Level Structure

```
tomato-disease-detection/
├── docker-compose.yml          # Orchestrates all three services
├── .env                        # Shared environment variables
├── .gitignore
├── README.md
│
├── backend/
├── frontend/
└── resnet34_model/
```

### docker-compose.yml — Service Overview

| Service        | Build Context      | Port  | Role                                  |
|----------------|--------------------|-------|---------------------------------------|
| `backend`      | `./backend`        | 8000  | FastAPI REST API                      |
| `frontend`     | `./frontend`       | 5173  | Vue 3 dev server (Vite)               |
| `model_trainer`| `./resnet34_model` | —     | One-off training/eval container       |

---

## 2. `resnet34_model/` — Deep Learning Module

This is a **standalone Python service** responsible for the full ML lifecycle: training, evaluation, and exporting the model for the backend to consume.

### Folder Structure

```
resnet34_model/
├── Dockerfile
├── requirements.txt
│
├── data/
│   ├── raw/                    # Downloaded PlantVillage dataset (gitignored)
│   │   ├── Tomato_Bacterial_Spot/
│   │   ├── Tomato_Early_Blight/
│   │   ├── Tomato_Late_Blight/
│   │   ├── Tomato_Leaf_Mold/
│   │   ├── Tomato_Septoria_Leaf_Spot/
│   │   ├── Tomato_Spider_Mites/
│   │   ├── Tomato_Target_Spot/
│   │   ├── Tomato_Yellow_Leaf_Curl_Virus/
│   │   ├── Tomato_Mosaic_Virus/
│   │   └── Tomato_Healthy/
│   └── processed/              # After split: train/val/test subfolders
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── config.py               # Hyperparameters, paths, constants
│   ├── dataset.py              # DataLoader, augmentation pipeline
│   ├── model.py                # ResNet34 initialization & head replacement
│   ├── train.py                # Two-stage training loop
│   ├── evaluate.py             # Metrics: accuracy, precision, recall, F1, confusion matrix
│   ├── export.py               # Save best checkpoint + class_labels.json
│   └── utils.py                # Helpers: seeding, logging, plotting
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── outputs/
│   ├── checkpoints/            # Epoch checkpoints saved during training
│   ├── best_model.pth          # Best checkpoint selected by val macro F1
│   ├── class_labels.json       # {0: "Bacterial_Spot", 1: "Early_Blight", ...}
│   └── evaluation_report/      # Confusion matrix, per-class metrics, plots
│
└── scripts/
    ├── prepare_dataset.py      # Download + split into train/val/test (70:15:15)
    └── run_training.sh         # Shell script to kick off full pipeline
```

### Key Design Decisions

**`src/config.py`**
```python
# All hyperparameters in one place — easy to tune
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10
STAGE_A_LR = 1e-3     # Feature extraction (frozen backbone)
STAGE_B_LR = 1e-4     # Fine-tuning (last residual block unfrozen)
STAGE_A_EPOCHS = 15
STAGE_B_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7
DATASET_SPLIT = (0.70, 0.15, 0.15)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

**`src/dataset.py` — Augmentation pipeline (training only)**
```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**`src/model.py` — Two-stage model setup**
- Stage A: freeze all backbone layers, train only `fc` (classifier head)
- Stage B: unfreeze `layer4` (last residual block group) + `fc`, use lower LR

**`src/train.py` — Checkpointing + early stopping**
- Save checkpoint at end of every epoch
- Track best `val_macro_f1`; restore best weights at end of training
- Log training/validation loss and metrics per epoch

**`src/export.py`**
- Copy best checkpoint to `outputs/best_model.pth`
- Write `outputs/class_labels.json` preserving exact index-to-classname mapping used during training

### Dockerfile (resnet34_model)

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/train.py"]
```

### requirements.txt

```
torch==2.2.0
torchvision==0.17.0
numpy
scikit-learn
matplotlib
seaborn
Pillow
tqdm
```

---

## 3. `backend/` — FastAPI + Domain-Driven Design

The backend exposes a REST API for the frontend to call. It follows **DDD (Domain-Driven Design)** with four layers: Domain → Application → Infrastructure → Interface.

### Folder Structure

```
backend/
├── Dockerfile
├── requirements.txt
├── main.py                     # FastAPI app entry point
│
├── domain/                     # LAYER 1: Pure business rules (no frameworks)
│   ├── entities/
│   │   ├── prediction.py       # Prediction entity: image_id, label, confidence, timestamp
│   │   └── disease.py          # Disease value object: name, severity_hint, treatment_tip
│   ├── repositories/
│   │   └── prediction_repository.py   # Abstract interface (port)
│   └── services/
│       └── disease_service.py  # Domain logic: map prediction to treatment advice
│
├── application/                # LAYER 2: Use cases (orchestrates domain)
│   ├── use_cases/
│   │   ├── predict_disease.py  # Use case: receive image → return prediction result
│   │   └── get_prediction_history.py
│   └── dtos/
│       ├── prediction_request.py   # Input DTO: validated image upload
│       └── prediction_response.py  # Output DTO: label, confidence, advice
│
├── infrastructure/             # LAYER 3: External concerns (ML model, DB, storage)
│   ├── ml/
│   │   ├── resnet34_inferencer.py  # Loads best_model.pth, runs inference
│   │   ├── preprocessor.py         # Replicates training-time preprocessing exactly
│   │   └── postprocessor.py        # Softmax → top-1 label + confidence
│   ├── persistence/
│   │   ├── sqlite_prediction_repo.py   # Concrete repo (implements domain interface)
│   │   └── database.py                 # SQLite setup via SQLAlchemy
│   └── storage/
│       └── image_store.py      # Saves uploaded images to /uploads
│
├── interface/                  # LAYER 4: HTTP controllers (FastAPI routers)
│   ├── routers/
│   │   ├── prediction_router.py    # POST /predict, GET /predictions
│   │   └── health_router.py        # GET /health
│   └── middleware/
│       └── cors.py             # CORS config for Vue frontend
│
├── shared/
│   ├── config.py               # Env vars: model path, upload dir, DB URL
│   └── exceptions.py           # Custom exception types
│
└── tests/
    ├── unit/
    │   ├── test_disease_service.py
    │   └── test_preprocessor.py
    └── integration/
        └── test_prediction_api.py
```

### DDD Layer Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│  INTERFACE LAYER (FastAPI Routers)                              │
│  Receives HTTP request → calls Application use case             │
├─────────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER (Use Cases + DTOs)                           │
│  Orchestrates: validates input → calls Domain → calls Infra     │
├─────────────────────────────────────────────────────────────────┤
│  DOMAIN LAYER (Entities, Value Objects, Abstract Repos)         │
│  Pure business rules. No framework imports. Always stable.      │
├─────────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE LAYER (ML Model, DB, File Storage)              │
│  Implements Domain interfaces. Framework-dependent.             │
└─────────────────────────────────────────────────────────────────┘
```

### Key API Endpoints

| Method | Endpoint         | Description                                 |
|--------|------------------|---------------------------------------------|
| POST   | `/predict`       | Upload a tomato leaf image, get prediction  |
| GET    | `/predictions`   | List past predictions (history)             |
| GET    | `/health`        | Backend + model loaded status               |

### `POST /predict` — Request/Response

**Request:** `multipart/form-data` with field `image` (JPEG/PNG)

**Response:**
```json
{
  "prediction_id": "uuid-...",
  "label": "Early_Blight",
  "confidence": 0.9423,
  "advice": "Apply copper-based fungicide. Remove infected leaves.",
  "timestamp": "2026-05-29T10:30:00Z"
}
```

### `infrastructure/ml/resnet34_inferencer.py` — Critical design

```python
# Loads model ONCE at startup (singleton pattern)
# Shares loaded model across all requests — no redundant reloading
class ResNet34Inferencer:
    def __init__(self, model_path: str, labels_path: str):
        self.model = self._load_model(model_path)
        self.labels = self._load_labels(labels_path)

    def predict(self, image: PIL.Image) -> tuple[str, float]:
        tensor = self.preprocessor.transform(image)
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        conf, idx = probs.max(1)
        return self.labels[idx.item()], conf.item()
```

### Dockerfile (backend)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### requirements.txt

```
fastapi==0.111.0
uvicorn[standard]
python-multipart
torch==2.2.0
torchvision==0.17.0
Pillow
SQLAlchemy
aiosqlite
pydantic-settings
```

---

## 4. `frontend/` — Vue 3 (Vite + Composition API)

A clean, agricultural-themed web UI for farmers/researchers to upload leaf images and receive disease predictions.

### Folder Structure

```
frontend/
├── Dockerfile
├── package.json
├── vite.config.js
├── index.html
│
├── public/
│   └── favicon.svg
│
└── src/
    ├── main.js                 # App entry, Vue + Router + Pinia setup
    ├── App.vue                 # Root component with layout
    │
    ├── assets/
    │   ├── styles/
    │   │   ├── main.css        # CSS variables, global reset
    │   │   └── theme.css       # Color palette (greens/earth tones)
    │   └── icons/              # SVG icons
    │
    ├── components/
    │   ├── layout/
    │   │   ├── AppHeader.vue   # Navigation bar
    │   │   └── AppFooter.vue
    │   ├── prediction/
    │   │   ├── ImageUploader.vue   # Drag-and-drop + file picker
    │   │   ├── ImagePreview.vue    # Shows selected image
    │   │   ├── PredictButton.vue   # Triggers API call
    │   │   └── ResultCard.vue      # Displays label, confidence, advice
    │   └── history/
    │       └── PredictionList.vue  # Table of past predictions
    │
    ├── views/
    │   ├── HomeView.vue        # Landing page
    │   ├── DetectView.vue      # Main prediction workflow
    │   └── HistoryView.vue     # Past predictions
    │
    ├── stores/
    │   └── predictionStore.js  # Pinia store: state, actions for API calls
    │
    ├── services/
    │   └── api.js              # Axios instance + API methods
    │
    └── router/
        └── index.js            # Vue Router routes
```

### Core Prediction Workflow (`DetectView.vue`)

```
User opens DetectView
    → Drops/selects image via ImageUploader
    → ImagePreview renders the image
    → Clicks "Detect Disease" button
    → predictionStore.predict(file) called
        → POST /predict to backend
        → Loading state shown
    → ResultCard appears with:
        ✓ Disease label
        ✓ Confidence percentage bar
        ✓ Treatment advice
        ✓ Severity indicator
```

### `stores/predictionStore.js` (Pinia)

```javascript
export const usePredictionStore = defineStore('prediction', {
  state: () => ({
    currentResult: null,
    history: [],
    loading: false,
    error: null,
  }),
  actions: {
    async predict(imageFile) {
      this.loading = true
      const formData = new FormData()
      formData.append('image', imageFile)
      const response = await api.post('/predict', formData)
      this.currentResult = response.data
      this.history.unshift(response.data)
      this.loading = false
    }
  }
})
```

### `services/api.js`

```javascript
import axios from 'axios'
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000,
})
export default api
```

### Dockerfile (frontend)

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
```

---

## 5. Docker Compose — Full Orchestration

```yaml
# docker-compose.yml
version: '3.9'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - model_artifacts:/app/model_artifacts  # Shared with model trainer
      - prediction_uploads:/app/uploads
    environment:
      - MODEL_PATH=/app/model_artifacts/best_model.pth
      - LABELS_PATH=/app/model_artifacts/class_labels.json
    depends_on:
      - model_trainer

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000

  model_trainer:
    build: ./resnet34_model
    volumes:
      - ./resnet34_model/data:/app/data
      - model_artifacts:/app/outputs  # Exports here; backend reads from here
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles:
      - training  # Only runs when explicitly invoked

volumes:
  model_artifacts:      # Shared between trainer and backend
  prediction_uploads:
```

**To run training:**
```bash
docker compose --profile training up model_trainer
```

**To run the full app (after training):**
```bash
docker compose up backend frontend
```

---

## 6. Data Flow Across All Three Services

```
                    ┌─────────────────────────────────────────┐
                    │         resnet34_model/                 │
                    │                                         │
                    │  1. Prepare dataset (70:15:15 split)    │
                    │  2. Stage A: Train classifier head      │
                    │  3. Stage B: Fine-tune layer4 + head    │
                    │  4. Export best_model.pth               │
                    │     + class_labels.json                 │
                    └──────────────┬──────────────────────────┘
                                   │ (Docker volume: model_artifacts)
                    ┌──────────────▼──────────────────────────┐
                    │              backend/                   │
                    │                                         │
                    │  FastAPI loads model at startup         │
                    │  POST /predict:                         │
                    │    Interface → Application → Domain     │
                    │    → Infrastructure (ML inferencer)     │
                    │    → Saves prediction to SQLite         │
                    │    → Returns JSON response              │
                    └──────────────┬──────────────────────────┘
                                   │ HTTP (REST API)
                    ┌──────────────▼──────────────────────────┐
                    │             frontend/                   │
                    │                                         │
                    │  Vue 3 UI:                              │
                    │    ImageUploader → Pinia store          │
                    │    → axios POST /predict                │
                    │    → ResultCard renders prediction      │
                    └─────────────────────────────────────────┘
```

---

## 7. Development Phases Mapped to Capstone Project 2 Timeline

| CP2 Phase        | Week | What to Build                                               |
|------------------|------|-------------------------------------------------------------|
| Phase 1          | 1    | Set up all 3 Docker containers; verify GPU access in trainer |
| Phase 2          | 2    | `prepare_dataset.py` — download PlantVillage, split 70:15:15|
| Phase 3          | 3    | `dataset.py` augmentation pipeline; verify preprocessing consistency |
| Phase 4          | 4    | `model.py` ResNet34 init; Stage A training loop in `train.py` |
| Phase 5          | 5–6  | Stage B fine-tuning; early stopping; checkpointing; export  |
| Phase 6          | 7    | `evaluate.py` — metrics, confusion matrix, baseline comparison |
| Phase 7          | 8    | Backend DDD structure; `resnet34_inferencer.py`; Vue UI     |
| Phase 8          | 9    | Integration testing; preprocessing consistency validation   |
| Phase 9          | 10   | Result analysis; misclassification discussion               |
| Phase 10         | 11–12| Final report + presentation slides                          |

---

## 8. Critical Implementation Notes

### Preprocessing Consistency (Most Important)
The **exact same** transforms must be applied in both `resnet34_model/src/dataset.py` and `backend/infrastructure/ml/preprocessor.py`. The safest approach is to define the normalization constants once in a shared config and import them in both places. Any mismatch here silently degrades prediction accuracy.

### Model Artifact Sharing
The Docker named volume `model_artifacts` is the bridge between the training container and the backend container. After training completes, `best_model.pth` and `class_labels.json` land in this volume, and the backend reads them on startup.

### DDD in Python — Simple Rule
- `domain/` — never import FastAPI, SQLAlchemy, or torch
- `application/` — never import FastAPI or SQLAlchemy directly
- `infrastructure/` — implements the abstract interfaces defined in `domain/repositories/`
- `interface/` — only imports from `application/` (never directly from `domain/` or `infrastructure/`)

### Vue Frontend ↔ Backend CORS
The backend must allow requests from `http://localhost:5173` during development. This is configured in `backend/interface/middleware/cors.py` and applied in `main.py`.

### Class Label Alignment
When `export.py` writes `class_labels.json`, it must capture the exact `dataset.class_to_idx` mapping from PyTorch's `ImageFolder`. This ensures index 0 in the model output always maps to the same disease name in both evaluation and the deployed API.

---

## 9. Technology Summary

| Layer             | Technology               | Why                                              |
|-------------------|--------------------------|--------------------------------------------------|
| ML Training       | PyTorch 2.2 + Torchvision| Official ResNet34 pretrained weights available   |
| Backend Framework | FastAPI                  | Async, fast, auto-generated OpenAPI docs         |
| Backend ORM       | SQLAlchemy + SQLite      | Lightweight, no extra DB container needed        |
| Frontend          | Vue 3 + Vite + Pinia     | Composition API, fast dev server, simple state   |
| HTTP Client       | Axios                    | Standard, Promise-based, easy interceptors       |
| Containerization  | Docker + Docker Compose  | Reproducible environment across machines         |
| GPU Support       | NVIDIA Container Toolkit | Enables CUDA in training container               |
