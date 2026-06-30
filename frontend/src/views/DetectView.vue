<template>
  <main class="container">
    <header class="page-header">
      <h1>Disease Detection</h1>
      <p class="subtitle">Upload a clear, close-up photo of the tomato leaf.</p>
    </header>

    <div class="content-grid">
      <div class="upload-section">
        <ImageDropzone @analyze="handleAnalysis" @reset="resetResult" />
        <div v-if="error" class="error-message">{{ error }}</div>
      </div>

      <div class="result-section" v-if="result">
        <ResultCard :result="result" />
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import ImageDropzone from '../components/ImageDropzone.vue'
import ResultCard from '../components/ResultCard.vue'
import { predict } from '../services/api.js'

const result = ref(null)
const error = ref(null)

const resetResult = () => {
  result.value = null
  error.value = null
}

const handleAnalysis = async (file, done) => {
  error.value = null
  result.value = null
  try {
    const res = await predict(file)
    result.value = res.data
  } catch (err) {
    error.value = err?.response?.data?.detail || 'Unable to reach the AI server. Make sure the backend is running.'
  } finally {
    done()
  }
}
</script>

<style scoped>
.page-header {
  text-align: center;
  margin-bottom: 2rem;
  padding-top: 2rem;
}
.content-grid {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  max-width: 700px;
  margin: 0 auto;
}
.error-message {
  background-color: rgba(211, 47, 47, 0.08);
  color: var(--error);
  padding: 1rem;
  border-radius: var(--radius-md);
  border: 1px solid rgba(211, 47, 47, 0.2);
  text-align: center;
  margin-top: 1rem;
}
</style>
