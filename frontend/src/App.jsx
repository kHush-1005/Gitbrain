/**
 * App.jsx
 * Root application component.
 *
 * Responsibilities:
 *   - Top-level layout (sidebar + main content)
 *   - Global state: backendStatus, currentRepo, messages, isLoading
 *   - Backend health polling
 *   - Orchestrates API calls → passes results to child components
 */
import { useState, useEffect, useCallback } from 'react'
import Sidebar    from './components/Sidebar'
import RepoSetup  from './components/RepoSetup'
import ChatWindow from './components/ChatWindow'
import { getBackendStatus, queryRepository } from './services/apiClient'

const HEALTH_POLL_MS = 15_000   // re-check backend health every 15 seconds

export default function App() {
  // ── Backend status ─────────────────────────────────────────────────────────
  const [backendStatus, setBackendStatus] = useState({
    online: null, groqConfigured: null, chromaDir: null, error: null,
  })

  // ── Current repository ─────────────────────────────────────────────────────
  const [currentRepo, setCurrentRepo] = useState(null)   // e.g. "psf__requests"
  const [repoUrl,     setRepoUrl]     = useState(null)   // original GitHub URL

  // ── Chat state ─────────────────────────────────────────────────────────────
  const [messages,  setMessages]  = useState([])
  const [isLoading, setIsLoading] = useState(false)

  // ── Fetch backend health ───────────────────────────────────────────────────
  const refreshStatus = useCallback(async () => {
    const status = await getBackendStatus()
    setBackendStatus(status)
  }, [])

  useEffect(() => {
    refreshStatus()
    const id = setInterval(refreshStatus, HEALTH_POLL_MS)
    return () => clearInterval(id)
  }, [refreshStatus])

  // ── Handle repo indexed ───────────────────────────────────────────────────
  function handleRepoIndexed(repoName, url) {
    setCurrentRepo(repoName)
    setRepoUrl(url)
    setMessages([])   // clear previous chat when switching repos
  }

  // ── Handle chat send ──────────────────────────────────────────────────────
  async function handleSend(question) {
    if (!currentRepo || isLoading) return

    // Add user message immediately
    const userMsg = {
      role:      'user',
      content:   question,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])
    setIsLoading(true)

    try {
      const result = await queryRepository(question, currentRepo)
      const assistantMsg = {
        role:      'assistant',
        content:   result.answer,
        sources:   result.sources || [],
        timestamp: new Date().toISOString(),
        isError:   false,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (err) {
      const errorMsg = {
        role:      'assistant',
        content:   err.message || 'An unexpected error occurred. Please try again.',
        sources:   [],
        timestamp: new Date().toISOString(),
        isError:   true,
      }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  function handleClearChat() {
    setMessages([])
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen bg-gray-950 overflow-hidden">

      {/* ── Left Sidebar ──────────────────────────────────────────────── */}
      <Sidebar
        backendStatus={backendStatus}
        currentRepo={currentRepo}
        repoUrl={repoUrl}
        onRefreshStatus={refreshStatus}
      />

      {/* ── Main Content ──────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Repo setup panel (collapsible once repo is set) */}
        <div className={`flex-shrink-0 border-b border-gray-800 transition-all duration-300 ${
          currentRepo ? 'max-h-20' : 'max-h-96'
        } overflow-hidden`}>
          <div className="p-4">
            {currentRepo ? (
              /* Compact summary bar when repo is already set */
              <div className="flex items-center gap-3 text-sm text-gray-400">
                <span className="text-xs bg-blue-950/40 border border-blue-800/40 text-blue-300
                                 px-2.5 py-1 rounded-full font-mono">
                  {currentRepo}
                </span>
                <span className="text-gray-600">indexed and ready</span>
                <button
                  onClick={() => { setCurrentRepo(null); setRepoUrl(null); setMessages([]) }}
                  className="ml-auto text-xs text-gray-600 hover:text-gray-300 transition-colors"
                >
                  Change repo
                </button>
              </div>
            ) : (
              <RepoSetup
                onRepoIndexed={handleRepoIndexed}
                currentRepo={currentRepo}
              />
            )}
          </div>
        </div>

        {/* Chat window fills remaining height */}
        <div className="flex-1 min-h-0">
          <ChatWindow
            messages={messages}
            onSend={handleSend}
            isLoading={isLoading}
            onClearChat={handleClearChat}
            disabled={!currentRepo}
          />
        </div>
      </main>
    </div>
  )
}
