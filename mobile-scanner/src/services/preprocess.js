// Preprocessing — MUST match the training/inference eval transform exactly.
// Ported from backend/infrastructure/ml/preprocessor.py:
//   Resize(shortest side -> 256, bilinear) -> CenterCrop(224) -> /255
//   -> Normalize(ImageNet mean/std) -> CHW -> batch dim -> Float32 (1,3,224,224)
// Any deviation here silently degrades accuracy.

const RESIZE_TO = 256
const CROP = 224
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

// Draw `source` (HTMLImageElement | ImageBitmap | HTMLCanvasElement) into a
// 224x224 canvas following resize-shortest-side-256 then centre-crop.
function toCropCanvas(source) {
  const w = source.naturalWidth ?? source.width
  const h = source.naturalHeight ?? source.height

  const ratio = RESIZE_TO / Math.min(w, h)
  const newW = Math.round(w * ratio)
  const newH = Math.round(h * ratio)

  // Resize onto an intermediate canvas (browser uses bilinear when smoothing on).
  const resized = document.createElement('canvas')
  resized.width = newW
  resized.height = newH
  const rctx = resized.getContext('2d', { willReadFrequently: true })
  rctx.imageSmoothingEnabled = true
  rctx.imageSmoothingQuality = 'high'
  rctx.drawImage(source, 0, 0, newW, newH)

  // Centre-crop 224x224.
  const left = Math.floor((newW - CROP) / 2)
  const top = Math.floor((newH - CROP) / 2)
  const crop = document.createElement('canvas')
  crop.width = CROP
  crop.height = CROP
  const cctx = crop.getContext('2d', { willReadFrequently: true })
  cctx.drawImage(resized, left, top, CROP, CROP, 0, 0, CROP, CROP)
  return crop
}

// Returns { data: Float32Array(1*3*224*224), dims: [1,3,224,224], previewCanvas }
export function preprocess(source) {
  const canvas = toCropCanvas(source)
  const { data } = canvas.getContext('2d', { willReadFrequently: true })
    .getImageData(0, 0, CROP, CROP) // RGBA, row-major

  const plane = CROP * CROP
  const out = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255
    const g = data[i * 4 + 1] / 255
    const b = data[i * 4 + 2] / 255
    // CHW layout with ImageNet normalization.
    out[i] = (r - MEAN[0]) / STD[0]
    out[plane + i] = (g - MEAN[1]) / STD[1]
    out[2 * plane + i] = (b - MEAN[2]) / STD[2]
  }
  return { data: out, dims: [1, 3, CROP, CROP], previewCanvas: canvas }
}

export function softmax(logits) {
  const max = Math.max(...logits)
  const exps = logits.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}
