/** Shared ScoreBar component used across views. */
export function ScoreBar({ label, value, max = 10 }) {
  const pct = Math.min(100, (value / max) * 100);
  const color = value >= 7 ? '#22c55e' : value >= 5 ? '#f59e0b' : '#ef4444';

  return (
    <div className="score-bar">
      <div className="score-bar-header">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-value" style={{ color }}>{value.toFixed(1)}</span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

/** Signal badge */
export function SignalBadge({ signal }) {
  const colors = {
    BUY: { bg: '#dcfce7', text: '#15803d', emoji: '🟢' },
    HOLD: { bg: '#fef9c3', text: '#854d0e', emoji: '🟡' },
    SELL: { bg: '#fee2e2', text: '#991b1b', emoji: '🔴' },
  };
  const style = colors[signal] || colors.HOLD;

  return (
    <span
      className="signal-badge"
      style={{ backgroundColor: style.bg, color: style.text }}
    >
      {style.emoji} {signal}
    </span>
  );
}

/** Spinner */
export function Spinner() {
  return <div className="spinner" aria-label="Loading…" />;
}

/** Error message */
export function ErrorMessage({ message }) {
  return <div className="error-message">⚠️ {message}</div>;
}
