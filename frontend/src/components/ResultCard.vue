<template>
  <div class="result-card">
    <div class="result-header">
      <h3>Analysis Complete</h3>
    </div>

    <div class="result-body">
      <!-- Confidence warning banner -->
      <ConfidenceBanner
        :cert-label="result.certainty_label"
        :is-low-confidence="result.is_low_confidence"
      />

      <!-- Disease name + severity -->
      <div class="disease-row">
        <div class="disease-name">
          <span class="label">Detected Disease</span>
          <h2 :class="{ 'is-healthy': isHealthy }">{{ displayName }}</h2>
        </div>
        <SeverityBadge v-if="result.severity_level" :severity="result.severity_level" />
      </div>

      <!-- Confidence bar -->
      <div class="confidence-meter">
        <div class="confidence-header">
          <span class="label">AI Confidence Score</span>
          <span class="score">{{ result.confidence?.toFixed(1) }}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: result.confidence + '%' }"></div>
        </div>
      </div>

      <!-- Affected area -->
      <div v-if="result.affected_area_ratio != null" class="affected-area">
        <span class="label">Estimated Leaf Coverage</span>
        <span class="area-value">{{ (result.affected_area_ratio * 100).toFixed(1) }}%</span>
      </div>

      <!-- Alternative labels (when low confidence) -->
      <AlternativeLabels
        v-if="result.is_low_confidence && result.alternative_labels?.length"
        :labels="result.alternative_labels"
      />

      <!-- Treatment advice -->
      <div v-if="result.advice" class="advice-section">
        <span class="label">Recommended Action</span>
        <p class="advice-text">{{ result.advice }}</p>
      </div>

      <!-- Treatment options -->
      <div v-if="result.treatment_options?.length" class="treatments-section">
        <span class="label">Treatment Options</span>
        <TreatmentCard
          v-for="t in result.treatment_options"
          :key="t.id"
          :treatment="t"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import SeverityBadge from './prediction/SeverityBadge.vue'
import TreatmentCard from './prediction/TreatmentCard.vue'
import AlternativeLabels from './prediction/AlternativeLabels.vue'
import ConfidenceBanner from './prediction/ConfidenceBanner.vue'

const props = defineProps({
  result: { type: Object, required: true },
})

const displayName = computed(() => {
  const raw = props.result.label || ''
  return raw.replace('Tomato___', '').replace(/_/g, ' ')
})

const isHealthy = computed(() =>
  (props.result.label || '').toLowerCase().includes('healthy')
)
</script>

<style scoped>
.result-card {
  background-color: var(--surface-color);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  animation: slideUp 0.4s ease-out;
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.result-header {
  background-color: var(--bg-color);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border-color);
}
.result-header h3 { font-size: 1.1rem; color: var(--text-primary); margin: 0; }
.result-body { padding: 1.5rem; }

.disease-row {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.disease-name h2 {
  font-size: 1.6rem;
  color: var(--error);
  margin-bottom: 0;
}
.disease-name h2.is-healthy { color: var(--success); }

.label {
  display: block;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 0.35rem;
  font-weight: 600;
}
.confidence-meter { margin-bottom: 1.25rem; }
.confidence-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.4rem;
}
.score { font-weight: 700; font-size: 1.05rem; color: var(--primary-color); }
.progress-bar {
  height: 8px;
  background: var(--border-color);
  border-radius: 4px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--primary-color);
  border-radius: 4px;
  transition: width 1s ease-out;
}

.affected-area {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: var(--bg-color);
  border-radius: var(--radius-md);
}
.area-value { font-weight: 700; color: var(--text-primary); }

.advice-section {
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #f1f8e9;
  border-radius: var(--radius-md);
  border-left: 4px solid var(--primary-color);
}
.advice-text { font-size: 0.9rem; line-height: 1.6; color: var(--text-primary); margin: 0; }

.treatments-section { margin-top: 1rem; }
</style>
