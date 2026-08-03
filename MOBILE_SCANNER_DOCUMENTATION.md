# Mobile Scanner — Implementation Documentation

**FYP by Sia Jia Le (22062566) — Sunway University**
**Tomato Leaf Disease Detection — On-Device Prototype**

---

## 1. Overview

The prototype is a **Progressive Web App (PWA)** that detects tomato leaf disease **entirely on the user's device**. A tomato-leaf image is captured (camera) or selected (gallery), and an **EfficientNet-B0** convolutional neural network classifies it into one of **8 classes** — all inside the phone's web browser, with **no server, no database, and no network request for the prediction**.

This design is a deliberate response to two constraints in the project report:

| Report section | Requirement | How the prototype satisfies it |
|---|---|---|
| **§1.1 Deployment Performance Measurement** | Model size, inference latency, and peak memory must be *measured on the target device*, not assumed | A built-in performance panel measures all three from inside the running app |
| **§3.8 Data Security & Privacy** | The prototype must not leak a farmer's location-tagged field images | Inference runs on-device (nothing uploaded), EXIF/GPS metadata is stripped, and images are never stored |

The application is served as static files from an **nginx container** (Docker). The container only delivers HTML/JS/WASM/model files — the machine learning still executes in the visitor's browser.

The wider machine-learning research of the project (`experiments/`, `resnet34_model/`, baseline model folders, the training/evaluation report) is unchanged; the mobile scanner is a separate, self-contained deployment prototype that consumes the trained model.

### Technology stack

| Layer | Technology |
|---|---|
| UI framework | Vue 3 (Composition API, `<script setup>`) |
| Build tool | Vite 6 |
| In-browser inference | onnxruntime-web (WebAssembly backend) |
| PWA / offline | vite-plugin-pwa (Workbox service worker) |
| Model format | ONNX (FP32 and int8-quantized) |
| Deployment | Docker multi-stage build → nginx (Alpine) |
| Model conversion | Python + PyTorch + timm (`scripts/convert_to_onnx.py`) |

---

## 2. Folder structure

```
mobile-scanner/
├── Dockerfile                     Multi-stage build: Node builds the app, nginx serves it
├── nginx.conf                     Web-server config: MIME types + cross-origin isolation headers
├── .dockerignore                  Excludes node_modules/dist from the Docker build context
├── .gitignore                     Excludes build artefacts from git
├── package.json                   Dependencies and npm scripts
├── vite.config.js                 Build config: PWA, WASM copy, dev-server headers
├── index.html                     HTML entry point (mounts the Vue app)
├── README.md                      Short project readme / run instructions
│
├── public/                        Static assets copied verbatim into the build
│   ├── disease_info.json          Per-class display name, type, and treatment tip
│   └── model/
│       ├── best_model.onnx        EfficientNet-B0 in ONNX (FP32, ~17.4 MB)
│       ├── best_model.int8.onnx   int8-quantized variant (~4.7 MB)
│       └── class_labels.json      Class index → class name (8 classes)
│
├── scripts/
│   └── convert_to_onnx.py         Converts best_model.pth → ONNX (run at build time, not in browser)
│
└── src/                           Application source
    ├── main.js                    Bootstraps the Vue application
    ├── App.vue                    Root component: orchestrates the whole scan flow
    ├── assets/
    │   └── main.css               Global styles / design tokens
    ├── components/
    │   ├── CameraCapture.vue       Image input (camera + file fallback)
    │   ├── ResultCard.vue          Displays prediction, confidence, tip, alternatives
    │   └── DiagnosticsPanel.vue    The §1.1 on-device performance measurements
    └── services/
        ├── preprocess.js           Image → normalized tensor (parity with training)
        ├── inference.js            Loads ONNX model, runs prediction, softmax, labels
        ├── exif.js                 Metadata-stripping image decode (privacy)
        └── metrics.js              Size / latency / memory measurement helpers
```

A clean separation is used throughout:

- **`components/`** are presentational Vue files (what the user sees).
- **`services/`** are plain JavaScript modules containing all non-UI logic (preprocessing, inference, privacy, measurement). This keeps the machine-learning logic testable and independent of the UI, mirroring the layered discipline used in the rest of the project.

---

## 3. File-by-file description

### 3.1 Application shell

#### `index.html`
The single HTML page. It sets a mobile-friendly viewport (`viewport-fit=cover` for notched phones), a theme colour, the page title, and a single `<div id="app">` mount point. It loads `/src/main.js` as an ES module. Everything else is rendered by Vue.

#### `src/main.js`
The bootstrap file. It imports the global stylesheet and the root `App.vue` component, creates the Vue application, and mounts it onto `#app`. Deliberately minimal — no router is used because the app is a single screen.

#### `src/assets/main.css`
Global styling and **design tokens** (CSS custom properties): the green colour palette, card background, text/muted colours, border, shadow, and danger colour. It also constrains the layout to a phone-width column (`max-width: 480px`), styles the primary/secondary buttons and the reusable `.card` container, and respects the device safe-area insets. Centralising the tokens here means the whole app can be re-themed from one place.

#### `src/App.vue` — the orchestrator
The root component that ties every piece together. Its responsibilities:

- **Model loading.** On mount it loads `disease_info.json` and calls `loadModel()` (from `inference.js`) for the selected variant. It tracks `modelReady` / `modelError` so the UI can show a spinner or an error card.
- **Model-variant switch.** A dropdown lets the user choose **FP32** or **INT8**; changing it clears the current result and reloads the corresponding ONNX file.
- **Handling a captured image.** When `CameraCapture` emits an image, `onImage()` runs `predict()`, then draws the exact 224×224 crop the model saw into a preview thumbnail (via `canvas.toBlob` + object URL). **Privacy step:** it closes the source `ImageBitmap` immediately, so the raw image is not retained.
- **State reset.** "Scan another leaf" revokes the preview object URL (freeing memory and ensuring the image is not kept) and clears the result.
- **Layout.** Renders the header, then conditionally the capture UI *or* the result card, the model/performance controls, the optional diagnostics panel, and a persistent privacy footer.

### 3.2 Components

#### `src/components/CameraCapture.vue`
Handles all image input and guarantees metadata is stripped before anything else happens.

- **Live camera.** `startCamera()` requests the rear camera via `navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })` and streams it into a `<video>` element. `capture()` paints the current video frame onto a `<canvas>` and converts it to an `ImageBitmap`. Because a canvas holds only raw pixels, the emitted image inherently carries **no EXIF/GPS metadata**.
- **File fallback.** If the camera is unavailable or denied, "Choose photo" opens a file picker (`accept="image/*" capture="environment"`). The selected file is passed through `decodeClean()` (see `exif.js`) which decodes it to a metadata-free `ImageBitmap`.
- **Lifecycle safety.** The camera stream is stopped on capture, on cancel, and on component unmount (`onBeforeUnmount`), so the camera light never stays on.
- It emits a single `image` event carrying the clean `ImageBitmap`; it knows nothing about the model.

#### `src/components/ResultCard.vue`
Pure presentation of a finished prediction. Given the prediction result and the disease-info lookup, it shows:

- the **preview thumbnail** (the actual model input crop),
- the **top disease** name and its type (e.g. "Fungal (Alternaria)"),
- a **confidence bar** and percentage,
- a **treatment tip** drawn from `disease_info.json`,
- the **ranked alternatives** (2nd–4th most likely classes with their probabilities).

It contains no logic beyond formatting percentages — all numbers are computed upstream.

#### `src/components/DiagnosticsPanel.vue` — report §1.1
The on-device performance measurement tool. Pressing **Measure**:

1. builds a zero-input runner (`makeBenchmarkRunner`) so timing reflects pure model execution independent of any particular image,
2. calls `benchmarkLatency()` to record the **cold** (first) inference separately, then the **warm mean ± standard deviation** over 30 runs,
3. calls `measureMemory()` for **peak memory**,
4. reads the **model file size** of the currently loaded variant.

It renders a table of: model variant, model size (MB), cold inference (ms), warm inference (ms, mean ± std), peak memory (MB), and which memory-measurement method was used. This directly produces the three numbers required by §1.1, measured on whatever device is running the app.

### 3.3 Services (non-UI logic)

#### `src/services/preprocess.js` — preprocessing parity (critical)
Transforms an image into the exact tensor the network was trained on. **Any deviation here silently reduces accuracy**, so it reproduces the training/validation transform precisely (ported from the original backend `preprocessor.py`):

1. **Resize** the shortest side to **256 px** (aspect-preserving), using the browser's bilinear scaling — *not* a naïve 256×256 squash.
2. **Centre-crop** to **224×224**.
3. Divide pixel values by 255, then **normalize** with the ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
4. Reorder from interleaved RGBA into **CHW** channel order and add a batch dimension → a `Float32Array` shaped `[1, 3, 224, 224]`.

It also exports `softmax()`, used to turn raw logits into probabilities. It returns the preview canvas as well, so the UI can show exactly what the model saw.

#### `src/services/inference.js` — the inference engine
Owns the onnxruntime-web session and the prediction flow:

- **Local WASM.** Sets `ort.env.wasm.wasmPaths = '/ort/'` so the WebAssembly runtime is loaded from our own origin, never a third-party CDN — reinforcing the "nothing leaves the device" guarantee.
- **`loadModel(variant)`** fetches the chosen ONNX file (`fp32` or `int8`) as an `ArrayBuffer`, records its byte length (for the size metric), and creates an `InferenceSession` on the WASM execution provider. It also loads `class_labels.json`.
- **`predict(source)`** runs the full chain: `preprocess` → build an `ort.Tensor` → `session.run` → `softmax` → map each probability to its class name via the labels → sort descending. Returns the top result plus the ranked list and the preview canvas.
- **Helpers:** `makeBenchmarkRunner()` (zero-input closure for the latency test), `getModelBytes()`, `getLoadedVariant()`, and `prettyLabel()` which turns a raw label such as `Tomato___Septoria_leaf_spot` into a readable `Septoria Leaf Spot`.

#### `src/services/exif.js` — privacy (report §3.8)
A small module whose single job is to **strip metadata**. `decodeClean(file)` decodes a file/blob into an `ImageBitmap` using `createImageBitmap(..., { imageOrientation: 'from-image' })`. Decoding to a bitmap keeps only raw pixels — all EXIF fields (timestamps, camera model, and critically **GPS coordinates**) are discarded — while `imageOrientation: 'from-image'` still honours the EXIF rotation flag so the picture appears the right way up. Because every image path (camera and file) ends up as pixels only, location data never enters the application.

#### `src/services/metrics.js` — measurement helpers (report §1.1)
Pure functions used by the diagnostics panel:

- **`modelSizeMB(bytes)`** converts the loaded model's byte length to megabytes.
- **`benchmarkLatency(runOnce, runs)`** times one **cold** run (which includes one-time cost such as kernel compilation) separately, then computes the **mean and standard deviation** of `runs` warm runs — matching the report's requirement to report the first inference separately.
- **`measureMemory()`** prefers `performance.measureUserAgentSpecificMemory()` (accurate, but only available when the page is *cross-origin isolated* — see the Docker headers below); it falls back to the Chromium-only `performance.memory` heap size, and reports which method was used so the measurement is honest about its limitations.

### 3.4 Static data

#### `public/disease_info.json`
A static lookup keyed by the full class name (e.g. `Tomato___Early_blight`). Each entry holds a human-readable **name**, the disease **type** (fungal / bacterial / viral / pest), and a concise **treatment tip**. Keeping this as static JSON removes any need for a database — the previous system used a PostgreSQL `treatment_options` table; here the same advisory content ships as a small file.

#### `public/model/class_labels.json`
Maps model output index → class name for the 8 classes. It is **generated from the checkpoint itself** during conversion (never hand-typed), so the label order is guaranteed to match the model's 8 output neurons.

#### `public/model/best_model.onnx` and `best_model.int8.onnx`
The deployable model. `best_model.onnx` is the full-precision (FP32) export (~17.4 MB). `best_model.int8.onnx` is a dynamically quantized 8-bit version (~4.7 MB, ~27 % of the FP32 size) offered for the size/latency comparison in §1.1.

---

## 4. The scan pipeline (end-to-end data flow)

```
 User taps "Open camera" / "Choose photo"
        │
        ▼
 CameraCapture.vue ──► exif.decodeClean()  ►  ImageBitmap (pixels only, no GPS/EXIF)
        │  emit('image', bitmap)
        ▼
 App.vue: onImage(bitmap)
        │
        ▼
 inference.predict(bitmap)
        │
        ├─► preprocess.js : resize 256 → crop 224 → /255 → ImageNet-normalize → Float32[1,3,224,224]
        ├─► onnxruntime-web : session.run()  (WebAssembly, on-device)
        ├─► softmax(logits) → probabilities
        └─► map to class labels → sort
        │
        ▼
 App.vue: draw 224² preview, close() the bitmap (not stored)
        │
        ▼
 ResultCard.vue : disease + confidence + treatment tip + alternatives
```

No step in this chain contacts a server. The image lives only in memory and is released immediately after the prediction.

---

## 5. Model conversion — `scripts/convert_to_onnx.py`

The browser cannot run a PyTorch `.pth` checkpoint, so the model is converted to ONNX once, offline. This script is **not** part of the running app — it is a build-time tool.

What it does:

1. **Loads** `best_model.pth` (repository root) and reads its embedded `config`, `class_to_idx`, and `val_macro_f1`.
2. **Rebuilds the exact architecture** by importing the project's own builder (`experiments/common/backbones.py`) *read-only* — EfficientNet-B0 + CBAM attention + the strong classifier head — so the exported graph is structurally identical to the trained model. A single warm-up forward pass materializes the lazily-created CBAM channels, then the checkpoint weights are loaded (strict match).
3. **Writes `class_labels.json`** straight from the checkpoint's `class_to_idx`, guaranteeing label/index alignment.
4. **Exports FP32 ONNX** with a fixed input shape `[1, 3, 224, 224]`, opset 17.
5. **Exports an int8 dynamically-quantized ONNX** using `onnxruntime.quantization` for a smaller, faster variant.
6. **Self-checks parity** between the PyTorch and ONNX outputs where the environment allows.

The confirmed model is run `efficientnetb0_seedrep_bgrandreal_s43` (the background-randomized "real" recipe, seed 43), with **validation macro-F1 = 0.935**, over **8 classes** (`Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Yellow_Leaf_Curl_Virus, healthy`).

To regenerate the model, run from the **repository root**:

```powershell
python mobile-scanner/scripts/convert_to_onnx.py
```

---

## 6. Docker implementation

The prototype is packaged with Docker so it runs identically anywhere without a manual Node/nginx setup. Three files cooperate: `Dockerfile`, `nginx.conf`, and the repository-root `docker-compose.yml`.

### 6.1 `Dockerfile` — multi-stage build

A **two-stage build** keeps the final image tiny (it contains no Node.js, no source, and no `node_modules`).

```dockerfile
# --- build stage: compile the Vite PWA ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

# --- serve stage: static nginx ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

- **Stage 1 (`node:20-alpine`)** installs dependencies and runs `npm run build`. Vite compiles the Vue app, bundles the JavaScript, generates the PWA service worker, and — via `vite-plugin-static-copy` — copies the onnxruntime-web WASM binaries into `dist/ort/`. `package.json` is copied *before* the source so Docker can cache the `npm install` layer and skip re-installing when only source files change.
- **Stage 2 (`nginx:alpine`)** starts from a bare nginx image and copies **only** the compiled `dist/` output plus our nginx config. The result is a small static-file server. `daemon off;` keeps nginx in the foreground so the container stays alive.

### 6.2 `nginx.conf` — headers and MIME types

nginx serves the built files, but two adjustments are essential for this specific app:

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # Cross-origin isolation
    add_header Cross-Origin-Opener-Policy   "same-origin"  always;
    add_header Cross-Origin-Embedder-Policy "require-corp" always;
    add_header Cross-Origin-Resource-Policy "same-origin"  always;

    # Force correct MIME for ES-module WASM glue, and never cache it
    location ~ \.mjs$ {
        default_type text/javascript;
        add_header Cross-Origin-Opener-Policy   "same-origin"  always;
        add_header Cross-Origin-Embedder-Policy "require-corp" always;
        add_header Cross-Origin-Resource-Policy "same-origin"  always;
        add_header Cache-Control "no-store" always;
    }

    # SPA fallback
    location / { try_files $uri $uri/ /index.html; }
}
```

**(a) Cross-origin isolation.** `performance.measureUserAgentSpecificMemory()` (the §1.1 peak-memory metric) and onnxruntime-web's multi-threaded WASM only work when the page is *cross-origin isolated*. This requires the two headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on every response. They are applied at server scope. Because everything the app loads (model, WASM, scripts) is same-origin, `require-corp` is safe.

**(b) `.mjs` MIME type.** onnxruntime-web loads its WASM glue code as an ES module named `ort-wasm-simd-threaded.jsep.mjs`. The default nginx `mime.types` maps `.js` but **not** `.mjs`, so nginx served it as `application/octet-stream`; browsers enforce strict MIME checking on module scripts and refused to execute it (the "no available backend found" error seen during development). The `location ~ \.mjs$` block forces `text/javascript`. Note that adding an `add_header` inside a location makes nginx drop the inherited server-scope headers, so the three isolation headers are repeated there.

**(c) `no-store` on `.mjs`.** After the MIME fix, a second caching issue appeared: because the file bytes were unchanged, the browser revalidated with the same ETag, nginx returned `304 Not Modified`, and the browser **reused its cached copy together with the old, wrong content-type**. Marking these responses `Cache-Control: no-store` stops that stale-MIME reuse.

**(d) SPA fallback.** `try_files $uri $uri/ /index.html;` routes unknown paths back to `index.html` so a refresh on any URL still loads the app.

### 6.3 `docker-compose.yml` — orchestration

At the repository root, a single service builds and runs the container:

```yaml
services:
  scanner:
    build: ./mobile-scanner
    container_name: leaf-scanner
    ports:
      - "8080:80"        # http://localhost:8080
    restart: unless-stopped
```

- **`build: ./mobile-scanner`** builds the image from the Dockerfile above.
- **`ports: "8080:80"`** maps container port 80 to host port **8080** (port 5173 was already in use on the development machine). `localhost` is treated as a *secure context* by browsers, so the camera and PWA features work without HTTPS during local testing.
- **`restart: unless-stopped`** brings the container back after a reboot or crash.

### 6.4 `.dockerignore`

Excludes `node_modules`, `dist`, `dev-dist`, and `.git` from the build context. This makes the `COPY . .` step fast and prevents a bulky host `node_modules` from leaking into the image (dependencies are installed fresh inside the build stage instead).

### 6.5 Running it

```powershell
# from the repository root
docker compose up -d --build      # build image and start (first build ~1-4 min)
# open http://localhost:8080
docker compose down               # stop and remove the container
```

Because the container serves only static files while inference runs in the browser, the same image can be deployed to any static host or edge/CDN environment unchanged — the deployment footprint is just HTML, JavaScript, WASM, and the ONNX model.

---

## 7. How the implementation maps back to the report

| Report requirement | Implementing files |
|---|---|
| §1.1 Model size | `metrics.js` (`modelSizeMB`), `inference.js` (`getModelBytes`), `DiagnosticsPanel.vue` |
| §1.1 Inference latency (cold vs warm) | `metrics.js` (`benchmarkLatency`), `DiagnosticsPanel.vue` |
| §1.1 Peak memory | `metrics.js` (`measureMemory`) + cross-origin-isolation headers in `nginx.conf`/`vite.config.js` |
| §3.8 On-device inference (no upload) | `inference.js` (in-browser WASM, local `wasmPaths`) |
| §3.8 EXIF/GPS stripped | `exif.js`, `CameraCapture.vue` (canvas capture) |
| §3.8 No image retention | `App.vue` (`bitmap.close()`, `URL.revokeObjectURL`) |
| Preprocessing parity | `preprocess.js` (mirrors training transform) |
| Deployment | `Dockerfile`, `nginx.conf`, `docker-compose.yml` |
```
