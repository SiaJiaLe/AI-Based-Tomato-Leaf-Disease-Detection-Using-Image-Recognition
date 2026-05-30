<template>
  <main class="container">
    <header class="app-header">
      <h1>Tomato Leaf Disease Analyzer</h1>
      <p class="subtitle">Powered by AI to instantly diagnose crop health</p>
    </header>

    <div class="content-grid">
      <div class="upload-section">
        <ImageDropzone @analyze="handleAnalysis" @reset="analysisResult = null; error = null" />
        
        <div v-if="error" class="error-message">
          <p>{{ error }}</p>
        </div>
      </div>
      
      <div class="result-section" v-if="analysisResult">
        <ResultCard :result="analysisResult" />
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import ImageDropzone from '../components/ImageDropzone.vue'
import ResultCard from '../components/ResultCard.vue'
import { predictDisease } from '../api/client'

const analysisResult = ref(null)
const error = ref(null)

const handleAnalysis = async (file, done) => {
  error.value = null
  analysisResult.value = null
  
  try {
    const res = await predictDisease(file)
    if (res.success) {
      analysisResult.value = res.prediction
    } else {
      error.value = "Failed to analyze image."
    }
  } catch (err) {
    error.value = "Unable to reach the AI server. Please make sure the backend is running."
  } finally {
    done()
  }
}
</script>

<style scoped>
.app-header {
  text-align: center;
  margin-bottom: 3rem;
  padding-top: 2rem;
}

.content-grid {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  max-width: 600px;
  margin: 0 auto;
}

.error-message {
  background-color: rgba(211, 47, 47, 0.1);
  color: var(--error);
  padding: 1rem;
  border-radius: var(--radius-md);
  margin-top: 1rem;
  text-align: center;
  border: 1px solid rgba(211, 47, 47, 0.2);
}
</style>
