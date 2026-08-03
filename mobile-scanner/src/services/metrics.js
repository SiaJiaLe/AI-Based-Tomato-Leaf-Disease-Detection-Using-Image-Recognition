// Deployment performance measurement (report §1.1), taken on the target device
// from inside the running app: model size, inference latency (cold vs warm),
// and peak memory.

// Model size = byte length of the ONNX file actually loaded.
export function modelSizeMB(byteLength) {
  return byteLength / (1024 * 1024)
}

// Latency: the first (cold) inference is reported separately because it carries
// one-time cost (graph load, wasm warm-up, kernel compilation); the warm mean
// ± std over `runs` repetitions reflects steady-state device performance.
export async function benchmarkLatency(runOnce, runs = 30) {
  const t0 = performance.now()
  await runOnce()
  const cold = performance.now() - t0

  const samples = []
  for (let i = 0; i < runs; i++) {
    const s = performance.now()
    await runOnce()
    samples.push(performance.now() - s)
  }
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length
  const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length
  return { coldMs: cold, warmMeanMs: mean, warmStdMs: Math.sqrt(variance), runs }
}

// Peak memory during inference. Preferred: measureUserAgentSpecificMemory()
// (needs cross-origin isolation — configured via COOP/COEP in vite.config.js).
// Fallback: the non-standard performance.memory heap (Chromium only). Returns
// { bytes, method } or { bytes: null, method: 'unavailable' }.
export async function measureMemory() {
  if (globalThis.crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
    try {
      const result = await performance.measureUserAgentSpecificMemory()
      return { bytes: result.bytes, method: 'measureUserAgentSpecificMemory' }
    } catch (_) { /* fall through */ }
  }
  if (performance.memory && performance.memory.usedJSHeapSize) {
    return { bytes: performance.memory.usedJSHeapSize, method: 'performance.memory (heap only)' }
  }
  return { bytes: null, method: 'unavailable' }
}

export function bytesToMB(bytes) {
  return bytes == null ? null : bytes / (1024 * 1024)
}
