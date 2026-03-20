import { useState } from 'react';
import { fetchPrediction } from '../services/api';
import { ScoreBar, SignalBadge, Spinner, ErrorMessage } from './shared';

const QUICK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'INTC'];

export default function StockSearch() {
  const [ticker, setTicker] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function search(sym) {
    const symbol = (sym || ticker).trim().toUpperCase();
    if (!symbol) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchPrediction(symbol);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter') search();
  }

  return (
    <div className="section">
      <h2>Stock Analysis</h2>
      <p className="section-desc">
        Search for a stock to get a real-time composite score based on fundamentals,
        technicals, risk, ML predictions, sentiment, and ETF inclusion.
      </p>

      <div className="search-row">
        <input
          className="ticker-input"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          onKeyDown={handleKey}
          placeholder="Ticker (e.g. AAPL)"
          maxLength={10}
        />
        <button className="btn-primary" onClick={() => search()} disabled={loading}>
          {loading ? 'Loading…' : 'Analyse'}
        </button>
      </div>

      <div className="quick-tickers">
        {QUICK_TICKERS.map((t) => (
          <button key={t} className="chip" onClick={() => { setTicker(t); search(t); }}>
            {t}
          </button>
        ))}
      </div>

      {error && <ErrorMessage message={error} />}
      {loading && <Spinner />}
      {result && <PredictionCard data={result} />}
    </div>
  );
}

function PredictionCard({ data }) {
  const { ticker, price, scores, signal, confidence, timestamp } = data;
  const scoreComponents = [
    { label: 'Fundamentals', key: 'fundamentals' },
    { label: 'Technicals', key: 'technicals' },
    { label: 'Risk', key: 'risk' },
    { label: 'ML', key: 'ml' },
    { label: 'Sentiment', key: 'sentiment' },
    { label: 'ETF Exposure', key: 'etf' },
  ];

  return (
    <div className="card prediction-card">
      <div className="prediction-header">
        <div>
          <h3 className="prediction-ticker">{ticker}</h3>
          <p className="prediction-price">${price.toLocaleString()}</p>
        </div>
        <div className="prediction-right">
          <SignalBadge signal={signal} />
          <p className="confidence">Confidence: {confidence}%</p>
        </div>
      </div>

      <div className="total-score-row">
        <span className="total-score-label">Total Score</span>
        <span className="total-score-value">{scores.total.toFixed(2)} / 10</span>
      </div>

      <div className="score-components">
        {scoreComponents.map(({ label, key }) => (
          <ScoreBar key={key} label={label} value={scores[key]} />
        ))}
      </div>

      <p className="timestamp">
        Last updated: {new Date(timestamp).toLocaleString()}
      </p>
    </div>
  );
}
