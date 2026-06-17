/**
 * Sidebar.jsx
 * Left sidebar: branding, backend status indicators, current repo info,
 * model information, and navigation links.
 *
 * Props:
 *   backendStatus  { online, groqConfigured, chromaDir, error }
 *   currentRepo    string | null   — sanitised collection name
 *   repoUrl        string | null   — original GitHub URL
 */
import { BrainCircuit, Github, Database, Cpu, RefreshCw, ExternalLink } from 'lucide-react'
import StatusCard from './StatusCard'

export default function Sidebar({ backendStatus, currentRepo, repoUrl, onRefreshStatus }) {
  const { online, groqConfigured, chromaDir } = backendStatus || {}

  return (
    <aside className="w-64 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col h-full">

      {/* ── Logo & title ────────────────────────────────────────────── */}
      <div className="px-5 py-5 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-600/20">
            <BrainCircuit size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight">GitBrain</h1>
            <p className="text-xs text-gray-500">AI Repository Intelligence</p>
          </div>
        </div>
      </div>

      {/* ── Scrollable body ─────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-4 space-y-5">

        {/* Backend status */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Backend</h2>
            <button
              onClick={onRefreshStatus}
              title="Refresh status"
              className="text-gray-600 hover:text-gray-300 transition-colors"
            >
              <RefreshCw size={12} />
            </button>
          </div>
          <div className="space-y-2">
            <StatusCard
              label="FastAPI Server"
              status={online === true ? 'ok' : online === false ? 'error' : 'loading'}
              detail={online === false ? 'Run: uvicorn api.main:app' : 'http://127.0.0.1:8000'}
            />
            <StatusCard
              label="Groq / Llama 3"
              status={groqConfigured ? 'ok' : groqConfigured === false ? 'warning' : 'loading'}
              detail={groqConfigured ? 'API key configured' : 'Add GROQ_API_KEY to .env'}
            />
            <StatusCard
              label="Vector Index"
              status={chromaDir ? 'ok' : 'loading'}
              detail={chromaDir || 'Not connected'}
            />
          </div>
        </section>

        {/* Current repository */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Repository</h2>
          {currentRepo ? (
            <div className="rounded-lg bg-blue-950/30 border border-blue-800/40 p-3 space-y-2">
              <div className="flex items-center gap-2">
                <Github size={13} className="text-blue-400 flex-shrink-0" />
                <span className="text-xs text-blue-300 font-mono truncate">{currentRepo}</span>
              </div>
              {repoUrl && (
                <a
                  href={repoUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                >
                  <ExternalLink size={11} />
                  View on GitHub
                </a>
              )}
            </div>
          ) : (
            <div className="rounded-lg bg-gray-800/50 border border-gray-700/40 p-3">
              <p className="text-xs text-gray-500">No repository indexed yet.</p>
              <p className="mt-1 text-xs text-gray-600">Enter a GitHub URL in the setup panel.</p>
            </div>
          )}
        </section>

        {/* Model info */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Model</h2>
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <Cpu size={12} className="text-gray-500" />
              <span>Llama 3 (Groq)</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <Database size={12} className="text-gray-500" />
              <span>all-MiniLM-L6-v2</span>
            </div>
          </div>
        </section>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <div className="px-4 py-3 border-t border-gray-800">
        <p className="text-xs text-gray-600 text-center">GitBrain v1.0 · Week 5</p>
      </div>
    </aside>
  )
}
