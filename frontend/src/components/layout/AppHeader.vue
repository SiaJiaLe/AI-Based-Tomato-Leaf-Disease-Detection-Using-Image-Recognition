<template>
  <header class="header">
    <nav>
      <router-link to="/">Home</router-link>
      <router-link to="/detect">Detect</router-link>
      <router-link to="/history">History</router-link>
    </nav>
  </header>
</template>
<style scoped>.header { padding: 1rem; background: var(--color-primary); }
.header a { color: white; margin-right: 1rem; }</style>
