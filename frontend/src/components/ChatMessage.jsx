/**
 * ChatMessage.jsx
 * Renders a single message bubble — either the user's question or the
 * AI's answer (with optional source citations).
 *
 * Props:
 *   role      {"user" | "assistant"}
 *   content   {string}
 *   sources   {Array}           — only for assistant messages
 *   timestamp {string}          — optional ISO string
 *   isError   {boolean}
 */
import { User, BrainCircuit, AlertCircle } from 'lucide-react'
import SourcePanel from './SourcePanel'

/** Minimal inline-code + code-block renderer (no heavy markdown library needed) */
function renderContent(text) {
  if (!text) return null

  // Split on triple-backtick code blocks
  const parts = text.split(/(```[\s\S]*?```)/g)

  return parts.map((part, i) => {
    if (part.startsWith('```')) {
      const lines  = part.replace(/^```[a-z]*\n?/, '').replace(/```$/, '')
      return (
        <pre key={i} className="my-2 rounded-lg bg-gray-900 border border-gray-700/60 overflow-x-auto p-3 text-xs font-mono text-gray-300 whitespace-pre-wrap">
          {lines}
        </pre>
      )
    }
    // Inline code: `code`
    const inlineParts = part.split(/(`[^`]+`)/g)
    return (
      <span key={i}>
        {inlineParts.map((ip, j) =>
          ip.startsWith('`') ? (
            <code key={j} className="rounded bg-gray-800 px-1.5 py-0.5 text-xs font-mono text-blue-300">
              {ip.slice(1, -1)}
            </code>
          ) : (
            <span key={j} className="whitespace-pre-wrap">{ip}</span>
          )
        )}
      </span>
    )
  })
}

function formatTime(iso) {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch { return '' }
}

export default function ChatMessage({ role, content, sources, timestamp, isError }) {
  const isUser = role === 'user'

  return (
    <div className={`flex gap-3 animate-slide-up ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser
          ? 'bg-blue-600'
          : isError
            ? 'bg-red-900 border border-red-700'
            : 'bg-gray-700 border border-gray-600'
      }`}>
        {isUser
          ? <User size={14} className="text-white" />
          : isError
            ? <AlertCircle size={14} className="text-red-400" />
            : <BrainCircuit size={14} className="text-blue-400" />
        }
      </div>

      {/* Bubble */}
      <div className={`max-w-[85%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
        {/* Label */}
        <span className={`text-xs font-medium ${isUser ? 'text-blue-400 text-right' : 'text-gray-400'}`}>
          {isUser ? 'You' : 'GitBrain'}
          {timestamp && (
            <span className="ml-2 text-gray-600 font-normal">{formatTime(timestamp)}</span>
          )}
        </span>

        {/* Message body */}
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-blue-600 text-white rounded-tr-sm'
            : isError
              ? 'bg-red-950/60 border border-red-800/50 text-red-300 rounded-tl-sm'
              : 'bg-gray-800 border border-gray-700/50 text-gray-200 rounded-tl-sm'
        }`}>
          {renderContent(content)}
        </div>

        {/* Source citations (assistant only) */}
        {!isUser && !isError && sources && sources.length > 0 && (
          <div className="w-full">
            <SourcePanel sources={sources} />
          </div>
        )}
      </div>
    </div>
  )
}
