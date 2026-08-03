<script setup>
// Image input: live camera (getUserMedia) with a file/gallery fallback.
// Every image is decoded through exif.decodeClean() so metadata never survives.
import { ref, onBeforeUnmount } from 'vue'
import { decodeClean } from '../services/exif.js'

const emit = defineEmits(['image'])

const videoEl = ref(null)
const fileEl = ref(null)
const streaming = ref(false)
const cameraError = ref('')
let stream = null

async function startCamera() {
  cameraError.value = ''
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } },
      audio: false,
    })
    streaming.value = true
    // wait for the <video> to render before attaching
    requestAnimationFrame(() => {
      if (videoEl.value) {
        videoEl.value.srcObject = stream
        videoEl.value.play()
      }
    })
  } catch (e) {
    cameraError.value = 'Camera unavailable — use "Choose photo" instead.'
    streaming.value = false
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop())
    stream = null
  }
  streaming.value = false
}

async function capture() {
  if (!videoEl.value) return
  const v = videoEl.value
  const canvas = document.createElement('canvas')
  canvas.width = v.videoWidth
  canvas.height = v.videoHeight
  canvas.getContext('2d').drawImage(v, 0, 0)
  stopCamera()
  // Canvas pixels carry no EXIF; emit an ImageBitmap for a uniform pipeline.
  const bitmap = await createImageBitmap(canvas)
  emit('image', bitmap)
}

async function onFile(e) {
  const file = e.target.files?.[0]
  if (!file) return
  const bitmap = await decodeClean(file) // strips EXIF/GPS
  emit('image', bitmap)
  e.target.value = '' // allow re-selecting the same file
}

onBeforeUnmount(stopCamera)
</script>

<template>
  <div class="card capture">
    <div v-if="streaming" class="video-wrap">
      <video ref="videoEl" playsinline muted></video>
      <button class="btn-primary shutter" @click="capture">Capture</button>
      <button class="btn-secondary cancel" @click="stopCamera">Cancel</button>
    </div>

    <div v-else class="idle">
      <div class="hint">
        <div class="leaf">🍅</div>
        <p class="muted">Point the camera at a tomato leaf, or choose a photo.</p>
      </div>
      <button class="btn-primary" @click="startCamera">Open camera</button>
      <button class="btn-secondary full" @click="fileEl.click()">Choose photo</button>
      <input
        ref="fileEl"
        type="file"
        accept="image/*"
        capture="environment"
        hidden
        @change="onFile"
      />
      <p v-if="cameraError" class="muted err">{{ cameraError }}</p>
    </div>
  </div>
</template>

<style scoped>
.capture { padding: 16px; }
.idle { display: flex; flex-direction: column; gap: 10px; }
.hint { text-align: center; padding: 18px 0; }
.leaf { font-size: 2.4rem; }
.full { width: 100%; }
.video-wrap { position: relative; display: flex; flex-direction: column; gap: 10px; }
video {
  width: 100%;
  border-radius: 12px;
  background: #000;
  aspect-ratio: 3 / 4;
  object-fit: cover;
}
.err { color: var(--danger); text-align: center; }
</style>
