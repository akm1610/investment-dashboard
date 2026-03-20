/**
 * Investment Dashboard API client.
 * All requests are proxied through Vite's dev server to http://localhost:9000.
 */

const BASE_URL = '/api';

async function _request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return response.json();
}

/** GET /health */
export function fetchHealth() {
  return _request('/health');
}

/** GET /metrics */
export function fetchMetrics() {
  return _request('/metrics');
}

/** GET /predict/:ticker */
export function fetchPrediction(ticker) {
  return _request(`/predict/${encodeURIComponent(ticker.toUpperCase())}`);
}

/** GET /sentiment/:ticker */
export function fetchSentiment(ticker) {
  return _request(`/sentiment/${encodeURIComponent(ticker.toUpperCase())}`);
}

/** POST /portfolio */
export function fetchPortfolio(tickers) {
  return _request('/portfolio', {
    method: 'POST',
    body: JSON.stringify({ tickers }),
  });
}

/** POST /portfolio/optimize */
export function fetchPortfolioOptimize(tickers, excludeSell = true) {
  return _request('/portfolio/optimize', {
    method: 'POST',
    body: JSON.stringify({ tickers, exclude_sell: excludeSell }),
  });
}

/**
 * Trigger a CSV download for the given tickers via GET /portfolio/export.
 * Bypasses the JSON _request helper because we need a raw Response for blob handling.
 */
export async function exportPortfolioCsv(tickers) {
  const params = new URLSearchParams({ tickers: tickers.join(',') });
  const response = await fetch(`${BASE_URL}/portfolio/export?${params}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  // Try to use the filename from Content-Disposition, fall back to a default
  const cd = response.headers.get('Content-Disposition') || '';
  const match = cd.match(/filename=([^;]+)/);
  a.download = match ? match[1].trim() : 'portfolio_analysis.csv';
  a.href = url;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

