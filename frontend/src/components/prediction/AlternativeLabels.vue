<template>
  <div class="alternatives" v-if="labels && labels.length">
    <p class="alt-title">Other possible diagnoses:</p>
    <ul class="alt-list">
      <li v-for="alt in labels" :key="alt.label" class="alt-item">
        <span class="alt-name">{{ formatName(alt.label) }}</span>
        <span class="alt-conf">{{ alt.confidence.toFixed(1) }}%</span>
      </li>
    </ul>
  </div>
</template>

<script setup>
defineProps({
  labels: { type: Array, default: () => [] },
})

const formatName = (raw) => {
  if (!raw) return 'Unknown'
  return raw.replace('Tomato___', '').replace(/_/g, ' ')
}
</script>

<style scoped>
.alternatives {
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: var(--radius-md);
  padding: 1rem;
  margin-top: 1rem;
}
.alt-title {
  font-size: 0.85rem;
  font-weight: 600;
  color: #f57f17;
  margin-bottom: 0.5rem;
}
.alt-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.alt-item {
  display: flex;
  justify-content: space-between;
  padding: 0.25rem 0;
  font-size: 0.85rem;
  color: var(--text-primary);
  border-bottom: 1px solid #ffe082;
}
.alt-item:last-child { border-bottom: none; }
.alt-conf { color: var(--text-secondary); font-weight: 600; }
</style>
