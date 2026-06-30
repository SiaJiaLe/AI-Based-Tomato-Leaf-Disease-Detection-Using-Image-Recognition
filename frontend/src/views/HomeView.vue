<template>
  <main class="container home">
    <div class="hero">
      <h1>Tomato Leaf Disease Detection</h1>
      <p class="subtitle">
        Upload a photo of a tomato leaf — our AI model identifies the disease,
        estimates severity, and recommends the right treatment.
      </p>
      <router-link to="/detect" class="btn-primary cta">Start Detection</router-link>
    </div>

    <div class="features">
      <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <h3>98.6% Accuracy</h3>
        <p>ResNet34 model trained on 35,000+ leaf images across 10 disease classes.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">💊</div>
        <h3>Treatment Advice</h3>
        <p>Get specific product recommendations with estimated costs for Malaysian markets.</p>
      </div>
      <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3>Severity Estimation</h3>
        <p>Know if your crop is mildly affected or needs urgent action.</p>
      </div>
    </div>
  </main>
</template>

<style scoped>
.home {
  padding-top: 4rem;
}
.hero {
  text-align: center;
  margin-bottom: 4rem;
}
.hero h1 {
  font-size: 2.4rem;
  margin-bottom: 1rem;
}
.subtitle {
  max-width: 560px;
  margin: 0 auto 2rem;
  font-size: 1.1rem;
  color: var(--text-secondary);
}
.cta {
  display: inline-block;
  background-color: var(--primary-color);
  color: white;
  padding: 0.9rem 2.5rem;
  border-radius: var(--radius-md);
  font-weight: 600;
  font-size: 1.05rem;
  text-decoration: none;
  transition: background-color 0.2s, transform 0.1s;
}
.cta:hover {
  background-color: var(--primary-hover);
  transform: translateY(-1px);
}
.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.5rem;
}
.feature-card {
  background: var(--surface-color);
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-color);
}
.feature-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}
.feature-card h3 {
  margin-bottom: 0.5rem;
}
.feature-card p {
  color: var(--text-secondary);
  font-size: 0.9rem;
}
</style>
