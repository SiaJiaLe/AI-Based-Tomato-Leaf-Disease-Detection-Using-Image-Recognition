<template>
  <div class="banner" :class="certLabel" v-if="certLabel !== 'high_certainty'">
    <span class="icon">{{ icon }}</span>
    <span class="text">{{ message }}</span>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  certLabel: { type: String, default: 'high_certainty' },
  isLowConfidence: { type: Boolean, default: false },
})

const icon = computed(() => props.certLabel === 'low_certainty_seek_confirmation' ? '⚠️' : 'ℹ️')
const message = computed(() => {
  if (props.certLabel === 'low_certainty_seek_confirmation')
    return 'Low confidence — the AI is uncertain. Consider taking another photo or consulting an extension officer.'
  if (props.certLabel === 'moderate_certainty')
    return 'Moderate confidence — result is likely correct but verify by checking leaf symptoms.'
  return ''
})
</script>

<style scoped>
.banner {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  margin-bottom: 1rem;
}
.low_certainty_seek_confirmation {
  background: #ffebee;
  border: 1px solid #ef9a9a;
  color: #b71c1c;
}
.moderate_certainty {
  background: #e3f2fd;
  border: 1px solid #90caf9;
  color: #1565c0;
}
.icon { flex-shrink: 0; }
</style>
