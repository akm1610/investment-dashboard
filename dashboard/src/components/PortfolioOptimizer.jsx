import { useState } from 'react';
import { fetchPortfolioOptimize } from '../services/api';
import { Spinner, ErrorMessage, SignalBadge } from './shared';

const DEFAULT_TICKERS = 'AAPL,MSFT,GOOGL,NVDA,TSLA';

export default function PortfolioOptimizer() {
  const [input, setInput] = useState(DEFAULT_TICKERS);
  const [excludeSell, setExcludeSell] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function optimise() {
    const tickers = input
      .split(/[,\s]+/)
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);

    if (tickers.length === 0) {
      setError('Enter at least one ticker.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await fetchPortfolioOptimize(tickers, excludeSell);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="section">
      <h2>Portfolio Optimiser</h2>
      <p className="section-desc">
        Score-weighted allocation: portfolio weights are proportional to each
        stock's composite score. Stocks with a SELL signal can optionally be
        excluded from the allocation.
      </p>

      <div className="search-row">
        <input
          className="ticker-input wide"
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          placeholder="AAPL, MSFT, GOOGL, NVDA, TSLA"
        />
        <button className="btn-primary" onClick={optimise} disabled={loading}>
          {loading ? 'Optimising…' : 'Optimise'}
        </button>
      </div>

      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={excludeSell}
          onChange={(e) => setExcludeSell(e.target.checked)}
        />
        Exclude SELL signals from allocation
      </label>

      {error && <ErrorMessage message={error} />}
      {loading && <Spinner />}

      {result && <OptimizationResult result={result} />}
    </div>
  );
}

function OptimizationResult({ result }) {
  const { weights, scores, signals } = result;
  const tickers = Object.keys(weights).sort((a, b) => weights[b] - weights[a]);

  return (
    <div className="card">
      <h3>Optimised Allocation</h3>
      <div className="allocation-list">
        {tickers.map((ticker) => {
          const weight = weights[ticker];
          const score = scores[ticker];
          const signal = signals[ticker];
          const pct = (weight * 100).toFixed(1);

          return (
            <div key={ticker} className="allocation-row">
              <div className="allocation-meta">
                <span className="allocation-ticker">{ticker}</span>
                <SignalBadge signal={signal} />
                <span className="allocation-score">Score: {score?.toFixed(2)}</span>
              </div>
              <div className="allocation-bar-row">
                <div className="allocation-track">
                  <div
                    className="allocation-fill"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: weight > 0 ? '#3b82f6' : '#e5e7eb',
                    }}
                  />
                </div>
                <span className="allocation-pct">{pct}%</span>
              </div>
            </div>
          );
        })}
      </div>

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
    </div>
  );
}
