/**
 * repoUtils.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Utility functions for repository name handling, matching the sanitization
 * logic used by the Python backend (utils/repo_parser.py).
 */

/**
 * Convert a GitHub URL into a ChromaDB-safe collection name.
 *
 * Examples:
 *   "https://github.com/psf/requests"      → "psf__requests"
 *   "https://github.com/tiangolo/fastapi"  → "tiangolo__fastapi"
 *   "https://github.com/my-org/my-repo"    → "my_org__my_repo"
 *
 * @param {string} url - Full GitHub repository URL
 * @returns {string|null} Collection name or null if URL is invalid
 */
export function urlToRepoName(url) {
  if (!url || typeof url !== 'string') return null
  try {
    let clean = url.trim().replace(/\/$/, '').replace(/\.git$/, '')
    if (!clean.includes('github.com')) return null
    const parts = clean.split('github.com/').pop().split('/')
    if (parts.length < 2) return null
    const owner = parts[0].toLowerCase().replace(/[-. ]/g, '_')
    const repo  = parts[1].toLowerCase().replace(/[-. ]/g, '_')
    if (!owner || !repo) return null
    return `${owner}__${repo}`
  } catch {
    return null
  }
}

/**
 * Validate that a string looks like a GitHub repository URL.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isValidGitHubUrl(url) {
  if (!url || typeof url !== 'string') return false
  const trimmed = url.trim()
  return (
    trimmed.includes('github.com/') &&
    trimmed.split('github.com/').pop().split('/').filter(Boolean).length >= 2
  )
}

/**
 * Extract the display name (owner/repo) from a GitHub URL.
 *
 * @param {string} url
 * @returns {string} e.g. "psf/requests"
 */
export function getDisplayName(url) {
  if (!url) return ''
  try {
    let clean = url.trim().replace(/\/$/, '').replace(/\.git$/, '')
    const parts = clean.split('github.com/').pop().split('/')
    if (parts.length < 2) return ''
    return `${parts[0]}/${parts[1]}`
  } catch {
    return ''
  }
}

/**
 * Format a line range string for display.
 * Handles both "42-58" and "42" (single line).
 *
 * @param {string} lines
 * @returns {string}
 */
export function formatLineRange(lines) {
  if (!lines) return ''
  return lines.includes('-') ? `Lines ${lines}` : `Line ${lines}`
}

/**
 * Format a cosine similarity score as a percentage badge.
 *
 * @param {number} score  0.0 – 1.0
 * @returns {string}      e.g. "87%"
 */
export function formatScore(score) {
  return `${Math.round((score || 0) * 100)}%`
}

/**
 * Get a colour class based on similarity score for the score badge.
 *
 * @param {number} score
 * @returns {string} Tailwind colour classes
 */
export function scoreColour(score) {
  if (score >= 0.75) return 'bg-green-900/50 text-green-300 border-green-700'
  if (score >= 0.5)  return 'bg-blue-900/50  text-blue-300  border-blue-700'
  return                     'bg-gray-800     text-gray-400  border-gray-600'
}
