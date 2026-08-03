<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import CameraCapture from './components/CameraCapture.vue'
import ResultCard from './components/ResultCard.vue'
import DiagnosticsPanel from './components/DiagnosticsPanel.vue'
import { loadModel, predict } from './services/inference.js'

const modelReady = ref(false)
const modelError = ref('')
const variant = ref('fp32')
const busy = ref(false)
const result = ref(null)
const previewUrl = ref('')
const diseaseInfo = ref({})
const showDiagnostics = ref(false)

async function init(v) {
  modelReady.value = false
  modelError.value = ''
  try {
    await loadModel(v)
    modelReady.value = true
  } catch (e) {
    modelError.value = 'Failed to load model: ' + (e?.message || e)
  }
}

async function onVariantChange() {
  clearResult()
  await init(variant.value)
}

function clearResult() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
  result.value = null
}

async function onImage(bitmap) {
  if (!modelReady.value || busy.value) return
  busy.value = true
  try {
    const out = await predict(bitmap)
    // Preview from the exact 224 crop the model saw.
    out.previewCanvas.toBlob((blob) => {
      if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = URL.createObjectURL(blob)
    }, 'image/jpeg', 0.9)
    result.value = out
    // Privacy §3.8: the source image is not stored — drop the reference now.
    if (bitmap.close) bitmap.close()
  } catch (e) {
    modelError.value = 'Prediction failed: ' + (e?.message || e)
  } finally {
    busy.value = false
  }
}

function scanAgain() {
  clearResult()
}

onMounted(async () => {
  diseaseInfo.value = await (await fetch('/disease_info.json')).json()
  await init(variant.value)
})
onBeforeUnmount(clearResult)
</script>

<template>
  <header class="app-head">
    <h1>🍅 Leaf Scan</h1>
    <p class="muted">On-device tomato leaf disease detection</p>
  </header>

  <div v-if="modelError" class="card err">{{ modelError }}</div>

  <div v-if="!modelReady && !modelError" class="card loading">
    <div class="spinner"></div>
    <span class="muted">Loading model…</span>
  </div>

  <template v-if="modelReady">
    <template v-if="!result">
      <CameraCapture @image="onImage" />
      <div v-if="busy" class="card loading">
        <div class="spinner"></div><span class="muted">Analysing leaf…</span>
      </div>
    </template>

    <template v-else>
      <ResultCard :result="result" :info="diseaseInfo" :preview-url="previewUrl" />
      <button class="btn-primary" @click="scanAgain">Scan another leaf</button>
    </template>

    <div class="controls card">
      <label class="ctl">
        <span>Model</span>
        <select v-model="variant" @change="onVariantChange">
          <option value="fp32">FP32 (full accuracy)</option>
          <option value="int8">INT8 (smaller / faster)</option>
        </select>
      </label>
      <button class="btn-secondary" @click="showDiagnostics = !showDiagnostics">
        {{ showDiagnostics ? 'Hide' : 'Show' }} performance
      </button>
    </div>

    <DiagnosticsPanel v-if="showDiagnostics" />
  </template>

  <footer class="privacy">
    <p class="muted">
      🔒 Images are processed entirely on your device. Nothing is uploaded, and
      photos (with any GPS/EXIF data) are never stored.
    </p>
  </footer>
</template>

<style scoped>
.app-head { text-align: center; padding: 20px 0 12px; }
.app-head h1 { margin: 0; font-size: 1.5rem; color: var(--green-dark); }
.loading { display: flex; align-items: center; gap: 12px; justify-content: center; }
.err { color: var(--danger); }
.spinner {
  width: 20px; height: 20px; border-radius: 50%;
  border: 3px solid #dfe7df; border-top-color: var(--green);
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.controls { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.ctl { display: flex; flex-direction: column; gap: 4px; font-size: 0.8rem; color: var(--muted); }
.ctl select {
  font: inherit; padding: 8px 10px; border-radius: 10px;
  border: 1px solid var(--border); background: #fff; color: var(--text);
}
.privacy { padding: 8px 4px 24px; text-align: center; }
</style>
