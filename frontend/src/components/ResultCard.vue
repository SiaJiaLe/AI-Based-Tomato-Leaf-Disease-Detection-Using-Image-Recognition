<template>
  <div class="result-card">
    <div class="result-header">
      <h3>Analysis Complete</h3>
    </div>
    
    <div class="result-body">
      <div class="disease-name">
        <span class="label">Detected Disease</span>
        <h2 :class="{'is-healthy': formatName(result.disease) === 'Healthy'}">
          {{ formatName(result.disease) }}
        </h2>
      </div>
      
      <div class="confidence-meter">
        <div class="confidence-header">
          <span class="label">AI Confidence Score</span>
          <span class="score">{{ result.confidence }}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: result.confidence + '%' }"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  result: {
    type: Object,
    required: true
  }
})

// Cleans up strings like "Tomato___Early_blight" to "Early Blight"
const formatName = (rawName) => {
  if (!rawName) return "Unknown"
  let cleanName = rawName.replace('Tomato___', '').replace(/_/g, ' ')
  if (cleanName.toLowerCase() === 'healthy') return 'Healthy'
  return cleanName
}
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
  to { opacity: 1; transform: translateY(0); }
}

.result-header {
  background-color: var(--bg-color);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border-color);
}

.result-header h3 {
  font-size: 1.1rem;
  color: var(--text-primary);
  margin: 0;
}

.result-body {
  padding: 2rem 1.5rem;
}

.label {
  display: block;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  font-weight: 600;
}

.disease-name h2 {
  font-size: 1.8rem;
  color: var(--error);
  margin-bottom: 2rem;
}

.disease-name h2.is-healthy {
  color: var(--success);
}

.confidence-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.score {
  font-weight: 700;
  font-size: 1.1rem;
  color: var(--primary-color);
}

.progress-bar {
  height: 8px;
  background-color: var(--border-color);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background-color: var(--primary-color);
  border-radius: 4px;
  transition: width 1s ease-out;
}
</style>
