import axios from 'axios'

/**
 * Axios instance for the Tomato Disease API.
 *
 * In development: VITE_API_BASE_URL is not set, so we use '/api' which
 *                 Vite's dev-server proxy forwards to HF Space.
 * In production:  VITE_API_BASE_URL is set on Vercel to the full HF
 *                 Space URL including the /api prefix, e.g.:
 *                 https://aqibniazi-tomato-disease-api.hf.space/api
 */
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
});

// Unwrap the Flask { success, data } envelope once — at the interceptor level.
// Every caller receives the inner `data` object directly.
api.interceptors.response.use(
  (response) => response.data.data, // ← unwrap ONLY here, not in each function
  (error) => {
    const message =
      error.response?.data?.error?.message ||
      error.response?.data?.message ||
      error.message ||
      "An unexpected error occurred.";
    const code = error.response?.data?.error?.code || "NETWORK_ERROR";
    return Promise.reject({ message, code, status: error.response?.status });
  },
);

// ── API methods ───────────────────────────────────────────────────────
// Each function returns the unwrapped payload directly — no .data needed.

export async function predictDisease(imageFile, onProgress) {
  const formData = new FormData()
  formData.append('image', imageFile)

  return api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100));
      }
    },
  });
}

export async function fetchClasses() {
  return api.get("/classes");
}

export async function fetchHealth() {
  return api.get("/health");
}