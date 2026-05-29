import { createRouter, createWebHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import DetectView from "../views/DetectView.vue";
import HistoryView from "../views/HistoryView.vue";

const routes = [
  { path: "/", name: "home", component: HomeView },
  { path: "/detect", name: "detect", component: DetectView },
  { path: "/history", name: "history", component: HistoryView },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
