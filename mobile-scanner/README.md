# Tomato Leaf Disease Scanner (Mobile PWA)

A single-page **Progressive Web App** that detects tomato leaf disease **entirely on the device**. The EfficientNet-B0 model (8 classes) runs in the phone browser via [onnxruntime-web](https://onnxruntime.ai/) (WASM) — there is no backend, no database, and no network round-trip for prediction.

This replaces the previous FastAPI + Vue + PostgreSQL system. The ML research pipeline (`experiments/`, `resnet34_model/`, baseline models, report) is unchanged.

## Why a PWA

The prototype targets a farmer with a mid-range phone in a field, so three deployment properties (report §1.1) are **measured on the device**, and privacy (report §3.8) is **enforced by architecture**:

| Requirement | How this app satisfies it |
|---|---|
| Model size | ONNX file size shown in the Performance panel (FP32 ~17 MB, INT8 ~4.7 MB) |
| Inference latency | Cold (1st) vs warm mean±std over 30 runs, via `performance.now()` |
| Peak memory | `performance.measureUserAgentSpecificMemory()` (needs cross-origin isolation) |
| On-device (no upload) | Inference is WASM in-browser; the image never leaves the device |
| EXIF/GPS stripped | Images are decoded to canvas/`ImageBitmap`; file metadata is discarded |
| No image retention | Image held in memory only; object URLs revoked; SW never caches user photos |

## Run locally

```powershell
cd mobile-scanner
npm install
npm run dev          # http://localhost:5173
```

To test on a phone, expose the dev server over HTTPS/LAN (camera + PWA install require a secure context) or use `npm run build && npm run preview`.

## Structure

```
public/
  model/best_model.onnx        FP32 model (from best_model.pth)
  model/best_model.int8.onnx   int8 quantized variant
  model/class_labels.json      8-class index -> name (from checkpoint)
  disease_info.json            per-class name + treatment tip
src/
  services/preprocess.js       resize256 -> centre-crop224 -> ImageNet norm (parity)
  services/inference.js        onnxruntime-web session + predict + softmax
  services/exif.js             metadata-stripping image decode
  services/metrics.js          size / latency / peak-memory measurement
  components/CameraCapture.vue  camera + file fallback
  components/ResultCard.vue     top disease, confidence, tip, alternatives
  components/DiagnosticsPanel.vue  the §1.1 measurements
scripts/
  convert_to_onnx.py           best_model.pth -> ONNX (FP32 + int8) + labels
```

## Regenerating the model

From the **repo root** (not `mobile-scanner/`):

```powershell
python mobile-scanner/scripts/convert_to_onnx.py
```

Reuses `experiments/common` read-only to rebuild the exact architecture, loads `best_model.pth`, and writes the ONNX files + `class_labels.json` into `public/model/`.

## Preprocessing parity (critical)

`src/services/preprocess.js` must mirror the training/eval transform exactly:
`Resize(shortest side → 256, bilinear) → CenterCrop(224) → /255 → Normalize(ImageNet)`.
Any change here silently degrades accuracy.
