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
 * Trigger a CSV export for the given tickers.
 * Opens a download in the browser by navigating to the export URL.
 *
 * @param {string[]} tickers
 */
export function exportPortfolioCSV(tickers) {
  const params = new URLSearchParams({ tickers: tickers.join(',') });
  window.location.href = `${BASE_URL}/portfolio/export?${params.toString()}`;
}
