// On-device inference service. Loads the ONNX model once and runs it entirely
// in the browser via onnxruntime-web (WASM). No network request carries the
// user's image — inference is fully local (report §3.8).

import * as ort from 'onnxruntime-web'
import { preprocess, softmax } from './preprocess.js'

// Serve WASM binaries from our own origin (copied to /ort by vite.config.js)
// instead of a CDN, so nothing third-party is contacted.
ort.env.wasm.wasmPaths = '/ort/'

const MODELS = {
  fp32: '/model/best_model.onnx',
  int8: '/model/best_model.int8.onnx',
}

let session = null
let inputName = null
let modelBytes = 0
let loadedVariant = null
let labels = null

export async function loadLabels() {
  if (labels) return labels
  const res = await fetch('/model/class_labels.json')
  labels = await res.json() // { "0": "Tomato___...", ... }
  return labels
}

// Load (or switch) the model variant. Returns { variant, modelBytes }.
export async function loadModel(variant = 'fp32') {
  if (session && loadedVariant === variant) return { variant, modelBytes }
  const url = MODELS[variant]
  const buf = await (await fetch(url)).arrayBuffer()
  modelBytes = buf.byteLength
  session = await ort.InferenceSession.create(new Uint8Array(buf), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  inputName = session.inputNames[0]
  loadedVariant = variant
  await loadLabels()
  return { variant, modelBytes }
}

// Run one forward pass on an already-preprocessed tensor spec.
async function runTensor({ data, dims }) {
  const feeds = { [inputName]: new ort.Tensor('float32', data, dims) }
  const out = await session.run(feeds)
  return out[session.outputNames[0]].data // Float32Array logits
}

// Full predict: preprocess image -> infer -> softmax -> sorted results.
// Returns { top, results: [{index, className, label, prob}], previewCanvas }
export async function predict(source) {
  const spec = preprocess(source)
  const logits = Array.from(await runTensor(spec))
  const probs = softmax(logits)
  const results = probs
    .map((prob, index) => ({
      index,
      className: labels[String(index)],
      label: prettyLabel(labels[String(index)]),
      prob,
    }))
    .sort((a, b) => b.prob - a.prob)
  return { top: results[0], results, previewCanvas: spec.previewCanvas }
}

// Expose a zero-input closure for the latency benchmark (same shape, no image
// dependency) so timing reflects pure model execution.
export function makeBenchmarkRunner() {
  const dims = [1, 3, 224, 224]
  const data = new Float32Array(3 * 224 * 224)
  return () => runTensor({ data, dims })
}

export function getModelBytes() {
  return modelBytes
}

export function getLoadedVariant() {
  return loadedVariant
}

// "Tomato___Septoria_leaf_spot" -> "Septoria Leaf Spot"
export function prettyLabel(raw) {
  if (!raw) return raw
  return raw
    .replace(/^Tomato___/, '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
