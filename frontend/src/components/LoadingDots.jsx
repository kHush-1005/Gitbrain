/**
 * LoadingDots.jsx
 * Animated three-dot loader displayed while the AI is generating a response.
 */
export default function LoadingDots({ label = 'GitBrain is thinking…' }) {
  return (
    <div className="flex items-center gap-3 text-gray-400 text-sm py-2">
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-2 h-2 rounded-full bg-blue-400 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
      <span className="text-gray-500">{label}</span>
    </div>
  )
}
