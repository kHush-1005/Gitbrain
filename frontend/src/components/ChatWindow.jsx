/**
 * ChatWindow.jsx
 * Main chat interface: input bar, message history, loading state.
 *
 * Props:
 *   messages       Array<{role, content, sources, timestamp, isError}>
 *   onSend(text)   called when user submits a question
 *   isLoading      boolean
 *   onClearChat()  called when user clears history
 *   disabled       boolean  — true when no repo is indexed
 */
import { useState, useRef, useEffect } from 'react'
import { Send, Trash2, BrainCircuit } from 'lucide-react'
import ChatMessage from './ChatMessage'
import LoadingDots from './LoadingDots'

const EXAMPLE_QUESTIONS = [
  'How does authentication work?',
  'What are the main entry points?',
  'How are errors handled?',
  'Explain the project structure',
]

export default function ChatWindow({ messages, onSend, isLoading, onClearChat, disabled }) {
  const [input, setInput]   = useState('')
  const bottomRef           = useRef(null)
  const inputRef            = useRef(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  function handleSubmit(e) {
    e.preventDefault()
    const q = input.trim()
    if (!q || isLoading || disabled) return
    onSend(q)
    setInput('')
    inputRef.current?.focus()
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900/80 flex-shrink-0">
        <div className="flex items-center gap-2">
          <BrainCircuit size={16} className="text-blue-400" />
          <h2 className="text-sm font-semibold text-gray-200">Chat</h2>
          {isLoading && (
            <span className="text-xs text-gray-500 animate-pulse">answering…</span>
          )}
        </div>
        {!isEmpty && (
          <button
            onClick={onClearChat}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-red-400 transition-colors"
          >
            <Trash2 size={12} />
            Clear
          </button>
        )}
      </div>

      {/* ── Message list ───────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 space-y-6">

        {/* Empty state */}
        {isEmpty && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-6 pb-20">
            <div className="w-16 h-16 rounded-2xl bg-gray-800 border border-gray-700 flex items-center justify-center">
              <BrainCircuit size={32} className="text-blue-500 opacity-70" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-200 mb-1">Ask about your repository</h3>
              <p className="text-sm text-gray-500 max-w-xs">
                {disabled
                  ? 'Index a GitHub repository using the panel above to start chatting.'
                  : 'Type a question below or choose an example.'}
              </p>
            </div>
            {!disabled && (
              <div className="flex flex-wrap justify-center gap-2 max-w-sm">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => onSend(q)}
                    className="px-3 py-1.5 rounded-full text-xs bg-gray-800 hover:bg-gray-700
                               border border-gray-700 text-gray-300 hover:text-white
                               transition-colors cursor-pointer"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, idx) => (
          <ChatMessage
            key={idx}
            role={msg.role}
            content={msg.content}
            sources={msg.sources}
            timestamp={msg.timestamp}
            isError={msg.isError}
          />
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-gray-700 border border-gray-600 flex items-center justify-center flex-shrink-0">
              <BrainCircuit size={14} className="text-blue-400" />
            </div>
            <div className="bg-gray-800 border border-gray-700/50 rounded-2xl rounded-tl-sm px-4 py-3">
              <LoadingDots />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input bar ──────────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-6 py-4 border-t border-gray-800 bg-gray-900/60">
        <form onSubmit={handleSubmit} className="flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={disabled || isLoading}
              placeholder={
                disabled
                  ? 'Index a repository first…'
                  : 'Ask a question about the codebase…'
              }
              rows={1}
              className="w-full resize-none rounded-xl bg-gray-800 border border-gray-700
                         text-gray-200 text-sm px-4 py-3 pr-12 placeholder-gray-600
                         focus:outline-none focus:border-blue-500 focus:ring-1
                         focus:ring-blue-500/40 disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors scrollbar-thin max-h-40"
              style={{ lineHeight: '1.5rem' }}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || disabled || isLoading}
            className="w-11 h-11 rounded-xl bg-blue-600 hover:bg-blue-500 flex-shrink-0
                       disabled:bg-gray-700 disabled:cursor-not-allowed
                       flex items-center justify-center transition-colors shadow-lg"
          >
            <Send size={16} className="text-white" />
          </button>
        </form>
        <p className="mt-2 text-xs text-gray-600 text-center">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
