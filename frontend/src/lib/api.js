import axios from 'axios'

/** Axios instance — all requests routed through Vite's /api proxy */

// ✅ Dynamic base URL (env-based)
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
});

// ── Response interceptor: unwrap the success envelope ─────────────────
api.interceptors.response.use(
  (response) => response.data,        // { success, data }
  (error) => {
    const message =
      error.response?.data?.error?.message ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred.'
    const code = error.response?.data?.error?.code || 'NETWORK_ERROR'
    return Promise.reject({ message, code, status: error.response?.status })
  }
)

// ── API methods ───────────────────────────────────────────────────────

/**
 * POST /api/predict
 * @param {File} imageFile  - raw File object from the browser
 * @param {function} onProgress - optional (0-100) progress callback
 * @returns {Promise<{predictions, top_prediction, is_healthy, inference_ms}>}
 */
export async function predictDisease(imageFile, onProgress) {
  const formData = new FormData()
  formData.append('image', imageFile)

  const response = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100))
      }
    },
  })
  return response.data
}

/**
 * GET /api/classes
 * @returns {Promise<{classes: Array, total: number}>}
 */
export async function fetchClasses() {
  const response = await api.get('/classes')
  return response.data
}

/**
 * GET /api/health
 * @returns {Promise<{status, model_loaded, num_classes, device}>}
 */
export async function fetchHealth() {
  const response = await api.get('/health')
  return response.data
}
