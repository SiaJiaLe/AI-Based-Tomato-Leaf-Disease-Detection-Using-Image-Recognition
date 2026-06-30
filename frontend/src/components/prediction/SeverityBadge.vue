<template>
  <span class="severity-badge" :class="level">
    {{ label }}
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  severity: { type: String, default: null },
})

const level = computed(() => props.severity || 'unknown')
const label = computed(() => {
  const map = { mild: 'Mild', moderate: 'Moderate', severe: 'Severe', unknown: 'N/A' }
  return map[level.value] || level.value
})
</script>

<style scoped>
.severity-badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.mild   { background: #e8f5e9; color: #2e7d32; }
.moderate { background: #fff8e1; color: #f57f17; }
.severe { background: #ffebee; color: #c62828; }
.unknown { background: var(--border-color); color: var(--text-secondary); }
</style>
