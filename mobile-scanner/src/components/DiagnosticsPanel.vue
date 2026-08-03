<script setup>
// Report §1.1 — Deployment Performance Measurement, run on the target device.
// Reports model file size, inference latency (cold vs warm mean±std), and
// peak memory during inference.
import { ref } from 'vue'
import { benchmarkLatency, measureMemory, bytesToMB, modelSizeMB } from '../services/metrics.js'
import { makeBenchmarkRunner, getModelBytes, getLoadedVariant } from '../services/inference.js'

const running = ref(false)
const report = ref(null)

async function run() {
  running.value = true
  report.value = null
  try {
    const runner = makeBenchmarkRunner()
    const latency = await benchmarkLatency(runner, 30)
    const mem = await measureMemory()
    report.value = {
      variant: getLoadedVariant(),
      sizeMB: modelSizeMB(getModelBytes()),
      latency,
      memMB: bytesToMB(mem.bytes),
      memMethod: mem.method,
    }
  } finally {
    running.value = false
  }
}
</script>

<template>
  <div class="card">
    <div class="row">
      <strong>On-device performance (§1.1)</strong>
      <button class="btn-secondary" :disabled="running" @click="run">
        {{ running ? 'Measuring…' : 'Measure' }}
      </button>
    </div>
    <p class="muted">Model size, inference latency, and peak memory measured here on this device.</p>

    <table v-if="report" class="metrics">
      <tbody>
        <tr><td>Model variant</td><td>{{ report.variant.toUpperCase() }}</td></tr>
        <tr><td>Model size</td><td>{{ report.sizeMB.toFixed(2) }} MB</td></tr>
        <tr><td>Cold inference (1st)</td><td>{{ report.latency.coldMs.toFixed(1) }} ms</td></tr>
        <tr>
          <td>Warm inference (avg of {{ report.latency.runs }})</td>
          <td>{{ report.latency.warmMeanMs.toFixed(1) }} ± {{ report.latency.warmStdMs.toFixed(1) }} ms</td>
        </tr>
        <tr>
          <td>Peak memory</td>
          <td>{{ report.memMB != null ? report.memMB.toFixed(1) + ' MB' : 'unavailable' }}</td>
        </tr>
      </tbody>
    </table>
    <p v-if="report" class="muted method">Memory method: {{ report.memMethod }}</p>
  </div>
</template>

<style scoped>
.row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 4px; }
.metrics { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9rem; }
.metrics td { padding: 8px 0; border-top: 1px solid var(--border); }
.metrics td:last-child { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.method { margin-top: 8px; font-size: 0.78rem; }
</style>
