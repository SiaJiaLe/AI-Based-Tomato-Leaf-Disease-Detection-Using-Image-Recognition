<template>
  <div class="treatment-log-view">
    <div class="page-header">
      <RouterLink to="/history" class="back-link">← Back to History</RouterLink>
      <h1>Log Treatment Outcome</h1>
      <p class="subtitle">Record how the treatment worked for prediction <code>{{ predictionId }}</code></p>
    </div>

    <div v-if="loading" class="state-msg">Loading prediction…</div>
    <div v-else-if="error" class="state-msg error">{{ error }}</div>

    <div v-else class="content">
      <!-- Prediction summary -->
      <div v-if="prediction" class="prediction-summary card">
        <h3>Diagnosed: <span class="disease-name">{{ formatName(prediction.label) }}</span></h3>
        <p class="meta">Confidence: {{ prediction.confidence?.toFixed(1) }}% &nbsp;|&nbsp; Severity: {{ prediction.severity_level || '—' }}</p>
      </div>

      <!-- Log form -->
      <form class="log-form card" @submit.prevent="submitLog">
        <h3>Select Treatment Applied</h3>

        <div v-if="treatments.length === 0" class="no-treatments">
          No treatments available for this disease.
        </div>

        <div v-else class="treatments-list">
          <label
            v-for="t in treatments"
            :key="t.id"
            class="treatment-option"
            :class="{ selected: selectedTreatmentId === t.id }"
          >
            <input
              type="radio"
              :value="t.id"
              v-model="selectedTreatmentId"
              name="treatment"
            />
            <div class="treatment-info">
              <span class="type-badge" :class="t.treatment_type">
                {{ t.treatment_type === 'organic' ? '🌿 Organic' : '🧪 Chemical' }}
              </span>
              <strong>{{ t.product_name }}</strong>
              <span v-if="t.active_ingredient" class="ingredient">{{ t.active_ingredient }}</span>
            </div>
          </label>
        </div>

        <div class="form-group">
          <label for="outcome">Outcome after treatment</label>
          <select id="outcome" v-model="outcome" required>
            <option value="">— Select outcome —</option>
            <option value="improved">Improved — symptoms reduced</option>
            <option value="no_change">No change — symptoms stayed the same</option>
            <option value="worsened">Worsened — symptoms got worse</option>
            <option value="recovered">Fully recovered</option>
          </select>
        </div>

        <div v-if="submitError" class="submit-error">{{ submitError }}</div>
        <div v-if="submitSuccess" class="submit-success">Treatment log saved successfully!</div>

        <button
          type="submit"
          class="btn-primary"
          :disabled="submitting || !selectedTreatmentId || !outcome"
        >
          {{ submitting ? 'Saving…' : 'Save Treatment Log' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getPrediction, listTreatments, logTreatment } from '../services/api.js'

const route = useRoute()
const predictionId = route.params.id

const loading = ref(true)
const error = ref(null)
const prediction = ref(null)
const treatments = ref([])

const selectedTreatmentId = ref(null)
const outcome = ref('')
const submitting = ref(false)
const submitError = ref(null)
const submitSuccess = ref(false)

const formatName = (raw) => (raw || '').replace('Tomato___', '').replace(/_/g, ' ')

onMounted(async () => {
  try {
    const { data: pred } = await getPrediction(predictionId)
    prediction.value = pred
    const { data: tx } = await listTreatments(pred.label).catch(() => ({ data: [] }))
    treatments.value = tx ?? []
  } catch (e) {
    error.value = e?.response?.data?.detail || 'Failed to load prediction.'
  } finally {
    loading.value = false
  }
})

async function submitLog() {
  if (!selectedTreatmentId.value || !outcome.value) return
  submitting.value = true
  submitError.value = null
  submitSuccess.value = false
  try {
    await logTreatment(predictionId, selectedTreatmentId.value, outcome.value)
    submitSuccess.value = true
    selectedTreatmentId.value = null
    outcome.value = ''
  } catch (e) {
    submitError.value = e?.response?.data?.detail || 'Failed to save log.'
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.treatment-log-view {
  max-width: 720px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.page-header { margin-bottom: 2rem; }
.back-link {
  font-size: 0.875rem;
  color: var(--primary-color);
  text-decoration: none;
  display: inline-block;
  margin-bottom: 0.75rem;
}
.back-link:hover { text-decoration: underline; }
h1 { font-size: 1.6rem; color: var(--text-primary); margin-bottom: 0.25rem; }
.subtitle { color: var(--text-secondary); font-size: 0.9rem; }
code { background: var(--bg-color); padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.8rem; }

.state-msg { text-align: center; padding: 3rem; color: var(--text-secondary); }
.state-msg.error { color: var(--error); }

.content { display: flex; flex-direction: column; gap: 1.5rem; }

.card {
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  box-shadow: var(--shadow-sm);
}

.prediction-summary h3 { margin-bottom: 0.4rem; font-size: 1.1rem; }
.disease-name { color: var(--error); }
.meta { font-size: 0.85rem; color: var(--text-secondary); margin: 0; }

.log-form h3 { margin-bottom: 1rem; font-size: 1rem; }

.no-treatments { color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem; }

.treatments-list { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 1.5rem; }

.treatment-option {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}
.treatment-option:hover { border-color: var(--primary-color); }
.treatment-option.selected {
  border-color: var(--primary-color);
  background: #f1f8e9;
}
.treatment-option input[type="radio"] { flex-shrink: 0; }

.treatment-info { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.type-badge {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
}
.type-badge.organic  { background: #e8f5e9; color: #2e7d32; }
.type-badge.chemical { background: #e3f2fd; color: #1565c0; }
.ingredient { font-size: 0.8rem; color: var(--text-secondary); }

.form-group { margin-bottom: 1.25rem; }
.form-group label {
  display: block;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 0.4rem;
}
.form-group select {
  width: 100%;
  padding: 0.6rem 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  background: var(--bg-color);
  color: var(--text-primary);
}

.submit-error  { color: var(--error); font-size: 0.85rem; margin-bottom: 0.75rem; }
.submit-success { color: var(--success); font-size: 0.85rem; margin-bottom: 0.75rem; font-weight: 600; }

.btn-primary {
  width: 100%;
  padding: 0.75rem;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary:not(:disabled):hover { opacity: 0.9; }
</style>
