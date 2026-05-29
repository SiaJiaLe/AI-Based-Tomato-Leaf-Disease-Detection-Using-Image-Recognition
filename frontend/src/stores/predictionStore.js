/**
 * Pinia store — calls api.js only (scaffold).
 */
import { defineStore } from "pinia";
import * as api from "../services/api.js";

export const usePredictionStore = defineStore("prediction", {
  state: () => ({
    currentResult: null,
    history: [],
    loading: false,
    error: null,
  }),
  actions: {
    async predict(imageFile) {
      this.loading = true;
      this.error = null;
      try {
        const response = await api.predict(imageFile);
        this.currentResult = response.data;
        this.history.unshift(response.data);
      } catch (err) {
        this.error = err.message || "Prediction failed";
        throw err;
      } finally {
        this.loading = false;
      }
    },
    async fetchHistory() {
      const response = await api.listPredictions();
      this.history = response.data;
    },
  },
});
