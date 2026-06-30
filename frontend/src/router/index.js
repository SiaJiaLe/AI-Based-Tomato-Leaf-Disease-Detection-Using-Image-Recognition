import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import DetectView from '../views/DetectView.vue'
import HistoryView from '../views/HistoryView.vue'
import TreatmentLogView from '../views/TreatmentLogView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/detect', name: 'detect', component: DetectView },
    { path: '/history', name: 'history', component: HistoryView },
    { path: '/treatment-log/:id', name: 'treatment-log', component: TreatmentLogView },
  ],
})

export default router
