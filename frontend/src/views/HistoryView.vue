<template>
  <main class="container">
    <header class="page-header">
      <h1>Prediction History</h1>
      <p class="subtitle">All past disease detections, newest first.</p>
    </header>

    <div v-if="loading" class="state-msg">Loading history…</div>
    <div v-else-if="error" class="error-message">{{ error }}</div>
    <div v-else-if="history.length === 0" class="state-msg">
      No predictions yet. Go to <router-link to="/detect">Detect</router-link> to analyse your first leaf.
    </div>
    <PredictionList v-else :predictions="history" />
  </main>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import PredictionList from '../components/history/PredictionList.vue'
import { listPredictions } from '../services/api.js'

const history = ref([])
const loading = ref(false)
const error = ref(null)

onMounted(async () => {
  loading.value = true
  try {
    const res = await listPredictions()
    history.value = res.data
  } catch (err) {
    error.value = 'Failed to load history. Make sure the backend is running.'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.page-header {
  text-align: center;
  margin-bottom: 2rem;
  padding-top: 2rem;
}
.state-msg {
  text-align: center;
  color: var(--text-secondary);
  padding: 3rem 0;
}
.error-message {
  background-color: rgba(211, 47, 47, 0.08);
  color: var(--error);
  padding: 1rem;
  border-radius: var(--radius-md);
  border: 1px solid rgba(211, 47, 47, 0.2);
  text-align: center;
}
</style>
