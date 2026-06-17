/**
 * StatusCard.jsx
 * Small pill/badge that shows backend or repo status with a coloured dot.
 *
 * Props:
 *   label   {string}   — display label
 *   status  {string}   — "ok" | "warning" | "error" | "loading"
 *   detail  {string}   — optional sub-text
 */
export default function StatusCard({ label, status = 'loading', detail }) {
  const dot = {
    ok:      'bg-green-400',
    warning: 'bg-yellow-400',
    error:   'bg-red-400',
    loading: 'bg-gray-500 animate-pulse',
  }[status] ?? 'bg-gray-500 animate-pulse'

  const text = {
    ok:      'text-green-300',
    warning: 'text-yellow-300',
    error:   'text-red-300',
    loading: 'text-gray-400',
  }[status] ?? 'text-gray-400'

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/60 border border-gray-700/50">
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />
      <div className="min-w-0">
        <p className={`text-xs font-medium truncate ${text}`}>{label}</p>
        {detail && (
          <p className="text-xs text-gray-500 truncate">{detail}</p>
        )}
      </div>
    </div>
  )
}
