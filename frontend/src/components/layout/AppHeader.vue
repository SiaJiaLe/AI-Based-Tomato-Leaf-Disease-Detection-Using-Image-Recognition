<template>
  <header class="header">
    <div class="header-inner">
      <router-link to="/" class="logo">
        🌿 TomatoScan
      </router-link>
      <nav class="nav">
        <router-link to="/" class="nav-link" active-class="active" exact>Home</router-link>
        <router-link to="/detect" class="nav-link" active-class="active">Detect</router-link>
        <router-link to="/history" class="nav-link" active-class="active">History</router-link>
      </nav>
    </div>
  </header>
</template>

<style scoped>
.header {
  background-color: var(--primary-color);
  padding: 0 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.header-inner {
  max-width: 1000px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
}
.logo {
  color: white;
  font-weight: 700;
  font-size: 1.2rem;
  text-decoration: none;
}
.nav {
  display: flex;
  gap: 0.5rem;
}
.nav-link {
  color: rgba(255,255,255,0.85);
  text-decoration: none;
  padding: 0.4rem 1rem;
  border-radius: var(--radius-md);
  font-weight: 500;
  transition: background-color 0.2s;
}
.nav-link:hover,
.nav-link.active {
  background-color: rgba(255,255,255,0.15);
  color: white;
}
</style>
