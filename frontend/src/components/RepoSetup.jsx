/**
 * RepoSetup.jsx
 * Panel for entering a GitHub repository URL, optionally providing a
 * GitHub token, and triggering indexing.
 *
 * Props:
 *   onRepoIndexed(repoName, repoUrl)  — called when indexing succeeds
 *   currentRepo  {string|null}        — currently indexed repo name
 */
import { useState } from 'react'
import { Github, Key, Zap, CheckCircle2, AlertCircle, Loader2, Terminal } from 'lucide-react'
import { ingestRepository } from '../services/apiClient'
import { urlToRepoName, isValidGitHubUrl, getDisplayName } from '../utils/repoUtils'

export default function RepoSetup({ onRepoIndexed, currentRepo }) {
  const [url,       setUrl]       = useState('')
  const [token,     setToken]     = useState('')
  const [status,    setStatus]    = useState('idle')   // idle | loading | success | error | manual
  const [message,   setMessage]   = useState('')
  const [chunks,    setChunks]    = useState(null)
  const [showToken, setShowToken] = useState(false)

  const repoName  = urlToRepoName(url)
  const validUrl  = isValidGitHubUrl(url)

  async function handleIngest() {
    if (!validUrl) return
    setStatus('loading')
    setMessage('')
    setChunks(null)

    try {
      const result = await ingestRepository(url, token)
      setChunks(result.chunks_indexed ?? result.total_chunks ?? '?')
      setStatus('success')
      setMessage(`Repository indexed successfully`)
      onRepoIndexed(repoName, url)
    } catch (err) {
      const msg = err.message || 'Indexing failed'
      if (msg.includes('/ingest API endpoint')) {
        setStatus('manual')
        setMessage(msg)
      } else {
        setStatus('error')
        setMessage(msg)
      }
    }
  }

  function handleManualSet() {
    if (!validUrl) return
    onRepoIndexed(repoName, url)
    setStatus('success')
    setMessage('Repository set manually — make sure it is indexed via terminal.')
    setChunks(null)
  }

  return (
    <div className="rounded-2xl border border-gray-700/60 bg-gray-900/60 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-700/40 bg-gray-800/40">
        <Github size={16} className="text-gray-400" />
        <h2 className="text-sm font-semibold text-gray-200">Repository Setup</h2>
        {currentRepo && (
          <span className="ml-auto flex items-center gap-1.5 text-xs text-green-300">
            <CheckCircle2 size={12} />
            {currentRepo}
          </span>
        )}
      </div>

      <div className="p-4 space-y-3">
        {/* URL input */}
        <div>
          <label className="block text-xs text-gray-400 mb-1.5 font-medium">
            GitHub Repository URL
          </label>
          <input
            type="text"
            value={url}
            onChange={(e) => { setUrl(e.target.value); setStatus('idle') }}
            placeholder="https://github.com/owner/repo"
            className="w-full rounded-lg bg-gray-800 border border-gray-700 text-gray-200 text-sm
                       px-3 py-2 placeholder-gray-600 focus:outline-none focus:border-blue-500
                       focus:ring-1 focus:ring-blue-500/40 transition-colors"
          />
          {url && !validUrl && (
            <p className="mt-1 text-xs text-red-400">Enter a valid GitHub URL</p>
          )}
          {url && validUrl && (
            <p className="mt-1 text-xs text-gray-500">
              Collection: <span className="text-blue-400 font-mono">{repoName}</span>
            </p>
          )}
        </div>

        {/* Token input (toggleable) */}
        <div>
          <button
            onClick={() => setShowToken((v) => !v)}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            <Key size={12} />
            {showToken ? 'Hide' : 'Add'} GitHub token (optional, for private repos / higher rate limit)
          </button>
          {showToken && (
            <input
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="ghp_your_personal_access_token"
              className="mt-2 w-full rounded-lg bg-gray-800 border border-gray-700 text-gray-200 text-sm
                         px-3 py-2 placeholder-gray-600 focus:outline-none focus:border-blue-500
                         focus:ring-1 focus:ring-blue-500/40 font-mono transition-colors"
            />
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleIngest}
            disabled={!validUrl || status === 'loading'}
            className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-blue-600
                       hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed
                       text-white text-sm font-medium py-2.5 transition-colors"
          >
            {status === 'loading'
              ? <><Loader2 size={14} className="animate-spin" /> Indexing…</>
              : <><Zap size={14} /> Index Repository</>
            }
          </button>
          <button
            onClick={handleManualSet}
            disabled={!validUrl}
            title="Use if you already indexed via terminal"
            className="px-3 rounded-lg bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800
                       disabled:cursor-not-allowed text-gray-300 text-sm transition-colors"
          >
            <Terminal size={14} />
          </button>
        </div>

        {/* Status feedback */}
        {status === 'success' && (
          <div className="flex items-start gap-2 rounded-lg bg-green-950/40 border border-green-800/40 p-3 text-xs text-green-300">
            <CheckCircle2 size={14} className="flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">{message}</p>
              {chunks !== null && <p className="mt-0.5 text-green-400">{chunks} chunks indexed</p>}
            </div>
          </div>
        )}

        {status === 'error' && (
          <div className="flex items-start gap-2 rounded-lg bg-red-950/40 border border-red-800/40 p-3 text-xs text-red-300">
            <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
            <p className="whitespace-pre-line">{message}</p>
          </div>
        )}

        {status === 'manual' && (
          <div className="rounded-lg bg-yellow-950/30 border border-yellow-700/40 p-3 text-xs text-yellow-300 space-y-2">
            <div className="flex items-center gap-2 font-medium">
              <Terminal size={12} />
              Index via terminal
            </div>
            <p className="text-gray-400">Run these commands in your terminal:</p>
            <pre className="bg-gray-900/60 rounded p-2 font-mono text-gray-300 text-xs overflow-x-auto">
{`python scripts\\export_chunks.py ${url}
python scripts\\build_vector_index.py data\\chunks\\${repoName}_chunks.jsonl`}
            </pre>
            <p className="text-gray-500">Then click the <span className="text-gray-300">terminal icon ↑</span> button to set the repo manually.</p>
          </div>
        )}
      </div>
    </div>
  )
}
