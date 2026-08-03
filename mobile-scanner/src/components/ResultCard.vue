<script setup>
// Prediction result: preview thumbnail, top disease + confidence, treatment
// tip, and the ranked alternatives.
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, required: true }, // { top, results }
  info: { type: Object, default: () => ({}) }, // disease_info.json
  previewUrl: { type: String, default: '' },
})

const top = computed(() => props.result.top)
const topInfo = computed(() => props.info[top.value.className] || {})
const confidencePct = computed(() => (top.value.prob * 100).toFixed(1))
const alternatives = computed(() => props.result.results.slice(1, 4))

function pct(p) {
  return (p * 100).toFixed(1)
}
</script>

<template>
  <div class="card">
    <div class="head">
      <img v-if="previewUrl" :src="previewUrl" class="thumb" alt="scanned leaf" />
      <div class="headline">
        <div class="type muted">{{ topInfo.type || 'Prediction' }}</div>
        <h2 class="name">{{ topInfo.name || top.label }}</h2>
        <div class="confbar">
          <div class="fill" :style="{ width: confidencePct + '%' }"></div>
        </div>
        <div class="muted">{{ confidencePct }}% confidence</div>
      </div>
    </div>

    <p v-if="topInfo.tip" class="tip">💡 {{ topInfo.tip }}</p>

    <div class="alts">
      <div class="alts-title muted">Other possibilities</div>
      <div v-for="a in alternatives" :key="a.index" class="alt-row">
        <span>{{ (info[a.className] && info[a.className].name) || a.label }}</span>
        <span class="muted">{{ pct(a.prob) }}%</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.head { display: flex; gap: 14px; align-items: center; }
.thumb {
  width: 84px; height: 84px;
  border-radius: 12px; object-fit: cover;
  flex-shrink: 0; border: 1px solid var(--border);
}
.headline { flex: 1; min-width: 0; }
.name { margin: 2px 0 8px; font-size: 1.25rem; line-height: 1.2; }
.type { text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.72rem; }
.confbar {
  height: 8px; background: #eef2ee; border-radius: 6px; overflow: hidden; margin-bottom: 4px;
}
.fill { height: 100%; background: var(--green); border-radius: 6px; transition: width 0.4s ease; }
.tip {
  background: #f1f8f1; border-left: 3px solid var(--green);
  padding: 12px 14px; border-radius: 8px; font-size: 0.9rem; line-height: 1.45; margin: 16px 0;
}
.alts-title { margin-bottom: 6px; }
.alt-row {
  display: flex; justify-content: space-between;
  padding: 7px 0; border-top: 1px solid var(--border); font-size: 0.9rem;
}
</style>
