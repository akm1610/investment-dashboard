import { useState } from 'react';
import { fetchPortfolio, exportPortfolioCsv } from '../services/api';
import { SignalBadge, Spinner, ErrorMessage } from './shared';

const DEFAULT_TICKERS = 'AAPL,MSFT,GOOGL,NVDA';

export default function PortfolioAnalysis() {
  const [input, setInput] = useState(DEFAULT_TICKERS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState(null);

  function parseTickers() {
    return input
      .split(/[,\s]+/)
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
  }

  async function analyse() {
    const tickers = parseTickers();
    if (tickers.length === 0) {
      setError('Enter at least one ticker.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await fetchPortfolio(tickers);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleExport() {
    const tickers = result
      ? result.results.map((r) => r.ticker)
      : parseTickers();
    if (tickers.length === 0) {
      setError('Analyse a portfolio before exporting.');
      return;
    }
    setExporting(true);
    try {
      await exportPortfolioCsv(tickers);
    } catch (err) {
      setError(`Export failed: ${err.message}`);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="section">
      <h2>Portfolio Analysis</h2>
      <p className="section-desc">
        Analyse multiple stocks at once. Enter comma-separated tickers and get
        ranked recommendations with signal buckets.
      </p>

      <div className="search-row">
        <input
          className="ticker-input wide"
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          placeholder="AAPL, MSFT, GOOGL, NVDA"
        />
        <button className="btn-primary" onClick={analyse} disabled={loading}>
          {loading ? 'Analysing…' : 'Analyse Portfolio'}
        </button>
        <button
          className="btn-secondary"
          onClick={handleExport}
          disabled={exporting || loading}
          title="Download results as CSV"
        >
          {exporting ? 'Exporting…' : '⬇ CSV'}
        </button>
      </div>

      {error && <ErrorMessage message={error} />}
      {loading && <Spinner />}

      {result && (
        <>
          <SummaryBuckets summary={result.summary} />
          <ResultsTable results={result.results} />
          {result.errors.length > 0 && (
            <div className="errors-list">
              <h4>Errors</h4>
              {result.errors.map((e) => (
                <p key={e.ticker} className="error-row">
                  {e.ticker}: {e.error}
                </p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function SummaryBuckets({ summary }) {
  const buckets = [
    { label: '🟢 Strong Buy (≥8.0)', key: 'strong_buy', color: '#dcfce7' },
    { label: '🟢 Buy (7–8)', key: 'buy', color: '#d1fae5' },
    { label: '🟡 Hold (5–7)', key: 'hold', color: '#fef9c3' },
    { label: '🔴 Sell (<5)', key: 'sell', color: '#fee2e2' },
  ];

  return (
    <div className="bucket-grid">
      {buckets.map(({ label, key, color }) => (
        <div key={key} className="bucket-card" style={{ backgroundColor: color }}>
          <p className="bucket-label">{label}</p>
          <p className="bucket-tickers">
            {summary[key].length > 0 ? summary[key].join(', ') : '—'}
          </p>
        </div>
      ))}
    </div>
  );
}

function ResultsTable({ results }) {
  return (
    <div className="table-wrapper">
      <table className="results-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Price</th>
            <th>Total</th>
            <th>Fund.</th>
            <th>Tech.</th>
            <th>Risk</th>
            <th>ML</th>
            <th>Sent.</th>
            <th>ETF</th>
            <th>Signal</th>
            <th>Conf.</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.ticker}>
              <td className="ticker-cell">{r.ticker}</td>
              <td>${r.price.toLocaleString()}</td>
              <td className="score-cell">{r.scores.total.toFixed(2)}</td>
              <td>{r.scores.fundamentals.toFixed(1)}</td>
              <td>{r.scores.technicals.toFixed(1)}</td>
              <td>{r.scores.risk.toFixed(1)}</td>
              <td>{r.scores.ml.toFixed(1)}</td>
              <td>{r.scores.sentiment.toFixed(1)}</td>
              <td>{r.scores.etf.toFixed(1)}</td>
              <td><SignalBadge signal={r.signal} /></td>
              <td>{r.confidence}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

