<template>
  <div 
    class="dropzone-container"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="handleDrop"
    :class="{ 'is-dragging': isDragging }"
  >
    <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" class="hidden-input" />

    <div class="dropzone-content" v-if="!previewUrl">
      <svg class="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
      <h3>Drag and drop an image of a tomato leaf</h3>
      <p>or click to browse from your computer</p>
      <button class="btn-primary" @click="$refs.fileInput.click()">Browse Files</button>
    </div>
    
    <div class="preview-container" v-else>
      <img :src="previewUrl" alt="Selected leaf" class="image-preview" />
      <div class="preview-actions">
        <button type="button" class="btn-secondary" @click="$refs.fileInput.click()" :disabled="isLoading">Choose Different Image</button>
        <button type="button" class="btn-primary" @click="submitImage" :disabled="isLoading">
          <span v-if="isLoading">Analyzing...</span>
          <span v-else>Analyze Disease</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['analyze', 'reset'])
const isDragging = ref(false)
const selectedFile = ref(null)
const previewUrl = ref(null)
const isLoading = ref(false)

const handleDrop = (e) => {
  isDragging.value = false
  const files = e.dataTransfer.files
  if (files.length > 0 && files[0].type.startsWith('image/')) {
    processFile(files[0])
  }
}

const handleFileSelect = (e) => {
  const files = e.target.files
  if (files.length > 0) {
    processFile(files[0])
  }
}

const processFile = (file) => {
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  emit('reset')
}

const clearSelection = () => {
  selectedFile.value = null
  previewUrl.value = null
}

const submitImage = async () => {
  if (!selectedFile.value) return
  
  isLoading.value = true
  emit('analyze', selectedFile.value, () => {
    isLoading.value = false
  })
}
</script>

<style scoped>
.dropzone-container {
  background-color: var(--surface-color);
  border: 2px dashed var(--border-color);
  border-radius: var(--radius-lg);
  padding: 3rem 2rem;
  text-align: center;
  transition: all 0.3s ease;
  box-shadow: var(--shadow-sm);
  margin-bottom: 2rem;
}

.dropzone-container.is-dragging {
  border-color: var(--primary-color);
  background-color: rgba(46, 125, 50, 0.02);
}

.upload-icon {
  width: 48px;
  height: 48px;
  color: var(--primary-color);
  margin-bottom: 1rem;
}

h3 {
  margin-bottom: 0.5rem;
}

p {
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
}

.hidden-input {
  display: none;
}

.btn-primary {
  background-color: var(--primary-color);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s ease, transform 0.1s ease;
}

.btn-primary:hover:not(:disabled) {
  background-color: var(--primary-hover);
}

.btn-primary:active:not(:disabled) {
  transform: scale(0.98);
}

.btn-primary:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.btn-secondary {
  background-color: transparent;
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-secondary:hover:not(:disabled) {
  background-color: var(--bg-color);
}

.preview-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image-preview {
  max-width: 100%;
  max-height: 400px;
  border-radius: var(--radius-md);
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
}

.preview-actions {
  display: flex;
  gap: 1rem;
}
</style>
