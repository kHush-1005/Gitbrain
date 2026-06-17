/**
 * SourcePanel.jsx
 * Collapsible panel showing the source file citations for an AI answer.
 *
 * Props:
 *   sources  {Array<{file, lines, score}>}
 */
import { useState } from 'react'
import { ChevronDown, ChevronUp, FileCode2, Hash } from 'lucide-react'
import { formatLineRange, formatScore, scoreColour } from '../utils/repoUtils'

export default function SourcePanel({ sources = [] }) {
  const [open, setOpen] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-3 rounded-lg border border-gray-700/60 overflow-hidden text-sm">
      {/* Toggle header */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-800/70 hover:bg-gray-800 transition-colors text-gray-400 hover:text-gray-200"
      >
        <div className="flex items-center gap-2">
          <FileCode2 size={14} />
          <span className="font-medium">
            {sources.length} source{sources.length !== 1 ? 's' : ''}
          </span>
        </div>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {/* Citation cards */}
      {open && (
        <div className="divide-y divide-gray-700/40">
          {sources.map((src, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 px-3 py-2.5 bg-gray-900/50 hover:bg-gray-900/80 transition-colors"
            >
              {/* Index number */}
              <span className="mt-0.5 flex-shrink-0 w-5 h-5 rounded-full bg-blue-900/60 text-blue-300 text-xs flex items-center justify-center font-mono">
                {idx + 1}
              </span>

              {/* File + lines */}
              <div className="min-w-0 flex-1">
                <p className="text-blue-300 font-mono text-xs truncate">
                  {src.file}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="flex items-center gap-1 text-gray-500 text-xs">
                    <Hash size={10} />
                    {formatLineRange(src.lines)}
                  </span>
                </div>
              </div>

              {/* Score badge */}
              <span
                className={`flex-shrink-0 text-xs font-medium px-2 py-0.5 rounded border ${scoreColour(src.score)}`}
              >
                {formatScore(src.score)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
