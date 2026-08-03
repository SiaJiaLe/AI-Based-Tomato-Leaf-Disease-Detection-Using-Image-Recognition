# Implementation Plan — Mobile Scanner Prototype (PWA)
## Tomato Leaf Disease Detection
**FYP by Sia Jia Le (22062566) — Sunway University**
**Supersedes all previous implementation_plan versions (full DDD system is being retired)**

---

## 1. Goal

Replace the decoupled FastAPI backend + Vue frontend + PostgreSQL system with a **single client-side Vue PWA** that runs EfficientNet-B0 inference **entirely in the phone browser** via `onnxruntime-web` (WASM). No server, no database, no network round-trip for prediction.

This directly satisfies two report requirements:

- **§1.1 Deployment Performance Measurement** — model size, inference latency (cold vs warm), and peak memory are all measurable on the target device from within the app.
- **§3.8 Data Security & Privacy** — inference runs on-device (image never leaves the phone), EXIF metadata is stripped, and captured images are never persisted.

The ML research (`experiments/`, `resnet34_model/`, baseline model folders, `MODEL_TRAINING_EVALUATION_REPORT.md`, `plan*.md`) is **untouched** — it is the thesis evidence.

---

## 2. What gets DELETED (app/system only)

| Target | Reason |
|---|---|
| `backend/` | FastAPI DDD server no longer used (inference moves on-device) |
| `frontend/` | Old server-coupled Vue app replaced by fresh PWA |
| `docker/` | Container setup for backend/frontend/postgres |
| `docker-compose.yml`, `docker-compose.training.yml` | Orchestration for the retired stack |

**Optional (ask before removing):** `backend.tar`, `frontend.tar` (old archives), `postgresql-18.4-2-windows-x64.exe` (DB installer).

**Explicitly KEPT:** everything ML/research — `experiments/`, `resnet34_model/`, `AlexNet/ KNN/ SVM/ VGG16/ ResNet50/ MobileNetV2/ EfficientNetB0/ RandomForest/`, `background_randomization.py`, `MODEL_TRAINING_EVALUATION_REPORT.md`, all `plan*.md`, `best_model.pth`.

---

## 3. New app structure (`mobile-scanner/`)

```
mobile-scanner/
  index.html
  package.json
  vite.config.js              # Vite + vite-plugin-pwa; COOP/COEP headers for memory API
  public/
    model/
      best_model.onnx         # converted from best_model.pth (Step A)
      class_labels.json       # 8 classes, generated from checkpoint class_to_idx
    disease_info.json         # static per-class name + short treatment tip (no DB)
  src/
    main.js
    App.vue
    services/
      inference.js            # ORT session, preprocessing parity, softmax, top-k
      preprocess.js           # Resize(256 shortest side) -> CenterCrop(224) -> normalize -> NCHW
      exif.js                 # canvas re-encode strips EXIF/GPS before use
      metrics.js              # model size, latency (cold/warm), peak memory
    components/
      CameraCapture.vue       # getUserMedia capture + file-upload fallback
      ResultCard.vue          # top prediction, confidence, treatment tip
      DiagnosticsPanel.vue    # the §1.1 measurements, shown on demand
```

Single-view app (no router needed) to keep it a "simple scanner." Diagnostics panel is a collapsible section.

---

## 4. Model conversion (Step A — needs Python + torch)

1. **Inspect** `best_model.pth` — print `config` (backbone), `num_classes`, and `class_to_idx`. Confirms it is EfficientNet-B0 (17 MB is consistent) and the 8-class ordering.
2. **Export to ONNX** — rebuild the architecture, load `state_dict`, `torch.onnx.export` with input `(1, 3, 224, 224)` float32, opset 17.
3. **Generate `class_labels.json`** straight from `checkpoint["class_to_idx"]` → guarantees label indices match the model's 8 output neurons (no hand-typing).
4. **Optional int8 dynamic quantization** (`onnxruntime.quantization`) to shrink the model file and speed up WASM — improves the §1.1 model-size metric. Offered as a variant; I'll produce both and we pick.
5. **Where it runs:** local machine if `torch` is installed; otherwise a short Colab snippet. Determined during Step A (I'll ask permission before running anything).

> If the checkpoint turns out to still be 10-class, I'll stop and flag it rather than silently shipping the wrong labels.

---

## 5. Preprocessing parity (critical)

Port `backend/infrastructure/ml/preprocessor.py` exactly into `src/services/preprocess.js`:

- Resize **shortest side** to 256 (aspect-preserving — NOT 256×256), bilinear
- Center-crop 224×224
- `/255`, then normalize with ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`
- HWC → CHW → add batch dim → Float32 `(1,3,224,224)`

Any mismatch silently degrades accuracy, so this is unit-verified against a known image if feasible.

---

## 6. Privacy implementation (§3.8)

- **On-device:** ORT WASM runs in-browser; zero `fetch`/XHR of the image. Verifiable in DevTools network tab.
- **EXIF/GPS stripped:** the captured/selected image is drawn to a `<canvas>` and read back as raw pixels; the canvas path carries no EXIF, so GPS/time never enter the pipeline.
- **No persistence:** image held only in memory; `URL.revokeObjectURL` after prediction; no `localStorage`/`IndexedDB`/cache of user images. The service worker caches only app shell + model, never user photos.

---

## 7. Deployment metrics (§1.1)

`DiagnosticsPanel.vue` shows:

- **Model size:** byte length of `best_model.onnx` (from fetch `Content-Length` / `arrayBuffer.byteLength`).
- **Inference latency:** run N=30 inferences; report **first (cold) inference separately** (includes session/model load), then **mean ± std of warm runs** via `performance.now()`.
- **Peak memory:** `performance.measureUserAgentSpecificMemory()` (requires cross-origin isolation via COOP/COEP headers, configured in `vite.config.js`); fallback to `performance.memory` (Chromium) with a documented limitation note.

---

## 8. Step-by-step execution order

1. Write this plan → **await approval** (you are here).
2. Delete app/system dirs/files (Step 2) — with per-item confirmation.
3. Convert model → `best_model.onnx` + 8-class `class_labels.json` (Step A) — ask permission for the Python command.
4. Scaffold `mobile-scanner/` (Vite + Vue 3 + vite-plugin-pwa).
5. Implement `preprocess.js` + `inference.js` (parity).
6. Camera/upload UI + `ResultCard` + static `disease_info.json`.
7. EXIF strip + no-persistence wiring.
8. `DiagnosticsPanel` metrics.
9. PWA manifest + service worker (installable, offline model cache).
10. Update `README.md` + `CLAUDE.md` to describe the new single-app architecture.
11. Local smoke test (`npm run dev`) + one sample prediction; then summary walkthrough.

---

## 9. Open questions for you

1. **Quantize the model?** Produce both FP32 and int8 ONNX and compare size/latency, or FP32 only? (Recommend: both, pick after measuring.)
2. **App folder name** — `mobile-scanner/` OK, or prefer `app/`?
3. **Delete the optional archives** (`backend.tar`, `frontend.tar`, the Postgres `.exe`)?
4. **Treatment tips** — keep a small static per-disease tip card (nice UX), or classification-only?
