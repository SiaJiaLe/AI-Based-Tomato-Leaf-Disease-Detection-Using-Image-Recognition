/**
 * API client — sole bridge to backend. Maps 1:1 to FastAPI router methods.
 */
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 30000,
});

export const getHealth = () => api.get("/api/v1/health");

export const predict = (file) => {
  const formData = new FormData();
  formData.append("image", file);
  return api.post("/api/v1/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const listPredictions = () => api.get("/api/v1/predictions");

export const getPrediction = (predictionId) =>
  api.get(`/api/v1/predictions/${predictionId}`);

export default api;
