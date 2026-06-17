/**
 * apiClient.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Centralised Axios client for all communication with the GitBrain FastAPI
 * backend. The React components never call fetch() or import backend Python
 * files — they only use the functions in this file.
 *
 * Base URL is read from the Vite environment variable:
 *   VITE_API_BASE_URL=http://127.0.0.1:8000
 *
 * Functions exported:
 *   healthCheck()        → GET  /health
 *   queryRepository()    → POST /query
 *   ingestRepository()   → POST /ingest
 *   getBackendStatus()   → combined health + collection check
 */

import axios from 'axios'

// ── Axios instance ─────────────────────────────────────────────────────────
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300_000,           // 5 minutes — indexing can take a while
  headers: { 'Content-Type': 'application/json' },
})

// ── Response interceptor — normalise errors ───────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      return Promise.reject(new Error(
        'Cannot reach the backend. Make sure the FastAPI server is running:\n' +
        '  uvicorn api.main:app --reload --port 8000'
      ))
    }
    if (error.response) {
      const detail = error.response.data?.detail || error.response.statusText
      return Promise.reject(new Error(`API Error ${error.response.status}: ${detail}`))
    }
    return Promise.reject(error)
  }
)


// ── healthCheck ───────────────────────────────────────────────────────────
/**
 * Ping the backend health endpoint.
 *
 * @returns {Promise<{status, service, groq_key, chroma}>}
 */
export async function healthCheck() {
  const { data } = await api.get('/health')
  return data
}


// ── queryRepository ────────────────────────────────────────────────────────
/**
 * Ask a natural-language question about an indexed repository.
 *
 * @param {string} question   - The user's question
 * @param {string} repoName   - ChromaDB collection name, e.g. "psf__requests"
 * @param {number} [topK=5]   - Number of chunks to retrieve
 * @returns {Promise<{answer, sources, chunks_retrieved, repo_name}>}
 */
export async function queryRepository(question, repoName, topK = 5) {
  const { data } = await api.post('/query', {
    question,
    repo_name: repoName,
    top_k:     topK,
  })
  return data
}


// ── ingestRepository ──────────────────────────────────────────────────────
/**
 * Trigger ingestion of a GitHub repository.
 *
 * This endpoint (POST /ingest) will be added in Week 6 or can be called
 * directly via the CLI build_vector_index.py script.
 *
 * For Week 5, the UI shows a "manual indexing" guide when /ingest is
 * not yet available, so this function handles 404 gracefully.
 *
 * @param {string} repoUrl   - Full GitHub URL
 * @param {string} [token]   - Optional GitHub personal access token
 * @returns {Promise<{status, chunks_indexed, repo_name}>}
 */
export async function ingestRepository(repoUrl, token = '') {
  try {
    const { data } = await api.post('/ingest', {
      repo_url: repoUrl,
      token:    token || undefined,
    })
    return data
  } catch (error) {
    // /ingest endpoint not yet implemented — return helpful message
    if (error.message?.includes('404')) {
      throw new Error(
        'The /ingest API endpoint is not yet available.\n\n' +
        'Index the repository manually using the terminal:\n\n' +
        '  1. python scripts\\export_chunks.py ' + repoUrl + '\n' +
        '  2. python scripts\\build_vector_index.py data\\chunks\\REPO_chunks.jsonl'
      )
    }
    throw error
  }
}


// ── getBackendStatus ──────────────────────────────────────────────────────
/**
 * Fetch backend health and return a structured status object for the UI.
 *
 * @returns {Promise<{online, groqConfigured, chromaDir, error}>}
 */
export async function getBackendStatus() {
  try {
    const health = await healthCheck()
    return {
      online:         true,
      groqConfigured: health.groq_key === 'configured',
      chromaDir:      health.chroma || 'data/chroma_db',
      error:          null,
    }
  } catch (err) {
    return {
      online:         false,
      groqConfigured: false,
      chromaDir:      null,
      error:          err.message,
    }
  }
}

export default api
