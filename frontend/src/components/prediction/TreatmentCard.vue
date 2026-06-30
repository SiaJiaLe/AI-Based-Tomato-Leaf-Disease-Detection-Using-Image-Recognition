<template>
  <div class="treatment-card">
    <div class="tc-header">
      <span class="type-badge" :class="treatment.treatment_type">
        {{ treatment.treatment_type === 'organic' ? '🌿 Organic' : '🧪 Chemical' }}
      </span>
      <strong class="product-name">{{ treatment.product_name }}</strong>
    </div>
    <p v-if="treatment.active_ingredient" class="meta">
      <span class="meta-label">Active ingredient:</span> {{ treatment.active_ingredient }}
    </p>
    <p v-if="treatment.application_method" class="method">
      {{ treatment.application_method }}
    </p>
    <p v-if="treatment.estimated_cost_myr" class="cost">
      Estimated cost: <strong>RM {{ treatment.estimated_cost_myr.toFixed(2) }}</strong>
    </p>
  </div>
</template>

<script setup>
defineProps({
  treatment: { type: Object, required: true },
})
</script>

<style scoped>
.treatment-card {
  background: var(--bg-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 1rem;
  margin-bottom: 0.75rem;
}
.tc-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}
.type-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}
.type-badge.organic  { background: #e8f5e9; color: #2e7d32; }
.type-badge.chemical { background: #e3f2fd; color: #1565c0; }
.product-name { font-size: 0.95rem; color: var(--text-primary); }
.meta { font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 0.4rem; }
.meta-label { font-weight: 600; }
.method { font-size: 0.85rem; color: var(--text-primary); line-height: 1.5; margin-bottom: 0.5rem; }
.cost { font-size: 0.85rem; color: var(--text-secondary); }
</style>
