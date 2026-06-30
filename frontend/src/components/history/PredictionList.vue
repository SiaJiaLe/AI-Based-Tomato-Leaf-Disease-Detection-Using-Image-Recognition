<template>
  <div class="list-wrapper">
    <div
      v-for="p in predictions"
      :key="p.prediction_id"
      class="prediction-row"
    >
      <div class="row-disease">
        <span class="disease-name" :class="{ healthy: isHealthy(p.label) }">
          {{ formatName(p.label) }}
        </span>
        <SeverityBadge :severity="p.severity_level" />
      </div>
      <div class="row-meta">
        <span class="confidence">{{ p.confidence?.toFixed(1) }}% confidence</span>
        <span class="ts">{{ formatDate(p.timestamp) }}</span>
      </div>
      <div class="row-actions">
        <div v-if="p.is_low_confidence" class="low-conf-pill">Low confidence</div>
        <RouterLink :to="`/treatment-log/${p.prediction_id}`" class="log-link">
          Log treatment →
        </RouterLink>
      </div>
    </div>
  </div>
</template>

<script setup>
import SeverityBadge from '../prediction/SeverityBadge.vue'

defineProps({
  predictions: { type: Array, required: true },
})

const formatName = (raw) => (raw || '').replace('Tomato___', '').replace(/_/g, ' ')
const isHealthy = (label) => (label || '').toLowerCase().includes('healthy')
const formatDate = (ts) => {
  if (!ts) return ''
  return new Date(ts).toLocaleString('en-MY', {
    day: 'numeric', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}
</script>

<style scoped>
.list-wrapper { max-width: 800px; margin: 0 auto; display: flex; flex-direction: column; gap: 0.75rem; }
.prediction-row {
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1rem 1.25rem;
  box-shadow: var(--shadow-sm);
}
.row-disease {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.4rem;
  flex-wrap: wrap;
}
.disease-name {
  font-weight: 600;
  font-size: 1rem;
  color: var(--error);
}
.disease-name.healthy { color: var(--success); }
.row-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
  color: var(--text-secondary);
}
.row-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.4rem;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.low-conf-pill {
  display: inline-block;
  font-size: 0.75rem;
  background: #fff3e0;
  color: #e65100;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-weight: 600;
}
.log-link {
  font-size: 0.8rem;
  color: var(--primary-color);
  text-decoration: none;
  font-weight: 600;
  margin-left: auto;
}
.log-link:hover { text-decoration: underline; }
</style>
