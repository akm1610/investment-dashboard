import { useEffect, useState } from 'react';
import { fetchMetrics, fetchHealth } from '../services/api';
import { Spinner, ErrorMessage } from './shared';

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [m, h] = await Promise.all([fetchMetrics(), fetchHealth()]);
      setMetrics(m);
      setHealth(h);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { refresh(); }, []);

  const uptimeFmt = (sec) => {
    if (sec === undefined || sec === null) return '—';
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  const totalRequests = metrics
    ? Object.values(metrics.request_counts).reduce((a, b) => a + b, 0)
    : 0;

  const totalErrors = metrics
    ? Object.values(metrics.error_counts).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <div className="section">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2>API Metrics</h2>
        <button className="btn-primary" onClick={refresh} disabled={loading} style={{ padding: '0.4rem 1rem', fontSize: '0.85rem' }}>
          {loading ? 'Refreshing…' : '↻ Refresh'}
        </button>
      </div>
      <p className="section-desc">
        Live runtime metrics for the Flask API – uptime, request counts, and error rates.
      </p>

      {error && <ErrorMessage message={error} />}
      {loading && !metrics && <Spinner />}

      {health && (
        <div className="metrics-grid">
          <MetricCard
            label="API Status"
            value={health.status === 'healthy' ? '✅ Healthy' : '❌ Unhealthy'}
            sub={health.status === 'healthy' ? 'All systems operational' : 'Check API logs'}
            accent={health.status === 'healthy' ? '#22c55e' : '#ef4444'}
          />
          <MetricCard
            label="Uptime"
            value={uptimeFmt(health.uptime_seconds)}
            sub="Since last restart"
            accent="#2563eb"
          />
          <MetricCard
            label="Total Requests"
            value={totalRequests.toLocaleString()}
            sub="Across all endpoints"
            accent="#7c3aed"
          />
          <MetricCard
            label="Total Errors"
            value={totalErrors.toLocaleString()}
            sub={totalErrors === 0 ? 'No errors 🎉' : 'Check server logs'}
            accent={totalErrors === 0 ? '#22c55e' : '#f59e0b'}
          />
        </div>
      )}

      {metrics && Object.keys(metrics.request_counts).length > 0 && (
        <div className="card" style={{ marginTop: '1.25rem' }}>
          <h3 style={{ marginBottom: '1rem', fontSize: '1rem', fontWeight: 700 }}>
            Requests by Endpoint
          </h3>
          <table className="results-table">
            <thead>
              <tr>
                <th>Endpoint</th>
                <th>Requests</th>
                <th>Errors</th>
                <th>Error Rate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.request_counts)
                .sort(([, a], [, b]) => b - a)
                .map(([endpoint, count]) => {
                  const errs = metrics.error_counts[endpoint] || 0;
                  const rate = count > 0 ? ((errs / count) * 100).toFixed(1) : '0.0';
                  return (
                    <tr key={endpoint}>
                      <td className="ticker-cell" style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                        {endpoint}
                      </td>
                      <td>{count}</td>
                      <td style={{ color: errs > 0 ? '#b91c1c' : 'inherit' }}>{errs}</td>
                      <td style={{ color: parseFloat(rate) > 5 ? '#b91c1c' : '#64748b' }}>
                        {rate}%
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}

      {lastRefresh && (
        <p className="timestamp" style={{ marginTop: '1rem' }}>
          Last refreshed: {lastRefresh.toLocaleString()}
        </p>
      )}
    </div>
  );
}

function MetricCard({ label, value, sub, accent }) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value" style={{ color: accent }}>{value}</p>
      <p className="metric-sub">{sub}</p>
    </div>
  );
}
