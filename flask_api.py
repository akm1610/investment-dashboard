"""
flask_api.py
------------
Lightweight Flask REST API for the investment dashboard.

Run with:
    python flask_api.py

The server starts on port 9000 by default.
The port can be overridden via the ``API_PORT`` environment variable.

Endpoints
---------
GET  /health                   – Liveness check
GET  /metrics                  – Runtime metrics (uptime, request counts)
GET  /predict/<ticker>         – Single-stock prediction and score
GET  /sentiment/<ticker>       – News sentiment and recent headlines
POST /portfolio                – Portfolio analysis for a list of tickers
POST /portfolio/optimize       – Score-weighted portfolio optimisation
GET  /portfolio/export         – Export portfolio results as CSV

Authentication
--------------
When the ``API_KEY`` environment variable is set, every request must include
the header ``X-API-Key: <value>``.  Requests without the correct key receive
HTTP 401.  When ``API_KEY`` is not set authentication is disabled (dev mode).

Rate Limiting
-------------
Requests are limited to 200/day and 60/minute per IP by default.
Override via ``RATELIMIT_DEFAULT`` environment variable.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Path setup – make root-level and src/ modules importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
for _path in [_HERE, _SRC]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from src.config import API_PORT
from src.data_fetcher import DataFetcher
from src.ml_engine import RecommendationEngine, FeatureEngineer
from src.scoring_engine import (
    score_fundamentals_intelligent,
    score_technicals_intelligent,
    contextualize_risk,
    score_ml_intelligently,
    score_sentiment,
    score_etf_exposure,
    stretch_distribution,
    get_news_headlines,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the dashboard on port 3000

# ---------------------------------------------------------------------------
# Optional rate limiting
# ---------------------------------------------------------------------------
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    _ratelimit_default = os.environ.get("RATELIMIT_DEFAULT", "200 per day;60 per minute")
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[_ratelimit_default],
        storage_uri="memory://",
    )
    _limiter_available = True
except ImportError:  # pragma: no cover
    limiter = None  # type: ignore[assignment]
    _limiter_available = False

# ---------------------------------------------------------------------------
# Monitoring counters
# ---------------------------------------------------------------------------
_START_TIME: float = time.monotonic()
_request_counts: dict[str, int] = {}
_error_counts: dict[str, int] = {}


@app.before_request
def _track_request() -> None:
    endpoint = request.endpoint or "unknown"
    _request_counts[endpoint] = _request_counts.get(endpoint, 0) + 1


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

@app.after_request
def _add_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# ---------------------------------------------------------------------------
# Optional API key authentication
# ---------------------------------------------------------------------------

_API_KEY: str = os.environ.get("API_KEY", "").strip()


@app.before_request
def _check_api_key() -> Response | None:
    """Enforce API key auth when ``API_KEY`` env var is configured."""
    if not _API_KEY:
        return None  # auth disabled
    # Skip auth for the health endpoint so load-balancers can probe it
    if request.endpoint == "health":
        return None
    provided = request.headers.get("X-API-Key", "")
    if provided != _API_KEY:
        return jsonify({"error": "Unauthorized – missing or invalid X-API-Key header"}), 401
    return None


# Lazily initialised to avoid slow startup when running tests
_engine: RecommendationEngine | None = None
_feature_engineer: FeatureEngineer | None = None


def _get_engine() -> tuple[RecommendationEngine, FeatureEngineer]:
    global _engine, _feature_engineer
    if _engine is None:
        _engine = RecommendationEngine()
        _feature_engineer = FeatureEngineer()
    return _engine, _feature_engineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prediction(ticker: str) -> dict[str, Any]:
    """Fetch data and build a full prediction dict for *ticker*."""
    fetcher = DataFetcher(ticker)
    price_data = fetcher.fetch_stock_data(period="2y")

    # Current price
    try:
        price = float(price_data["Close"].iloc[-1])
    except Exception:
        price = 0.0

    fundamentals = fetcher.fetch_all_fundamentals(ticker)
    technicals = fetcher.calculate_technical_indicators(price_data)
    risk_metrics = fetcher.calculate_risk_metrics(price_data)

    engine, feat_eng = _get_engine()
    features = feat_eng.extract_features(price_data, fundamentals)
    ml_prediction = engine.predict(features)

    # Component scores
    fund_score = score_fundamentals_intelligent(fundamentals)
    tech_score = score_technicals_intelligent(technicals, price_data)
    risk_score = contextualize_risk(risk_metrics, fundamentals, technicals)
    ml_score = score_ml_intelligently(ml_prediction, fundamentals, technicals)
    sent_score = score_sentiment(ticker)
    etf_score = score_etf_exposure(ticker)

    raw_score = (
        fund_score * 0.40
        + tech_score * 0.25
        + risk_score * 0.15
        + ml_score * 0.12
        + sent_score * 0.05
        + etf_score * 0.03
    )
    total_score = round(stretch_distribution(raw_score), 2)

    # Signal and confidence
    if total_score >= 7.0:
        signal = "BUY"
        confidence = round(min(99.0, 50.0 + (total_score - 7.0) * 20.0), 1)
    elif total_score >= 5.0:
        signal = "HOLD"
        confidence = round(50.0, 1)
    else:
        signal = "SELL"
        confidence = round(min(99.0, 50.0 + (5.0 - total_score) * 20.0), 1)

    return {
        "ticker": ticker.upper(),
        "price": round(price, 2),
        "scores": {
            "fundamentals": round(fund_score, 2),
            "technicals": round(tech_score, 2),
            "risk": round(risk_score, 2),
            "ml": round(ml_score, 2),
            "sentiment": round(sent_score, 2),
            "etf": round(etf_score, 2),
            "total": total_score,
        },
        "signal": signal,
        "confidence": confidence,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def _validate_ticker(ticker: str) -> str | None:
    """Return an error message string if the ticker is invalid, else None."""
    cleaned = ticker.upper().strip()
    if not cleaned:
        return "Ticker symbol is required"
    if not cleaned.replace(".", "").replace("-", "").isalpha():
        return "Invalid ticker symbol – use letters only (e.g. AAPL, BRK.B)"
    if len(cleaned) > 10:
        return "Ticker symbol too long (max 10 characters)"
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Any:
    """Liveness check."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
        }
    )


@app.get("/metrics")
def metrics() -> Any:
    """Runtime metrics: uptime, per-endpoint request/error counts."""
    return jsonify(
        {
            "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
            "request_counts": dict(_request_counts),
            "error_counts": dict(_error_counts),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


@app.get("/predict/<ticker>")
def predict(ticker: str) -> Any:
    """Return a prediction and composite score for a single *ticker*."""
    err = _validate_ticker(ticker)
    if err:
        return jsonify({"error": err}), 400
    ticker = ticker.upper().strip()
    try:
        result = _build_prediction(ticker)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error predicting %s", ticker)
        _error_counts["predict"] = _error_counts.get("predict", 0) + 1
        return jsonify({"error": str(exc), "ticker": ticker}), 500


@app.get("/sentiment/<ticker>")
def sentiment(ticker: str) -> Any:
    """Return sentiment score and recent news headlines for *ticker*.

    Response body (JSON)::

        {
          "ticker": "AAPL",
          "sentiment_score": 6.5,
          "news_api_active": false,
          "headlines": [
            {
              "title": "Apple beats earnings…",
              "url": "https://…",
              "published_at": "2024-11-01T12:00:00Z",
              "source": "Yahoo Finance",
              "polarity": 0.5
            }
          ],
          "timestamp": "…"
        }
    """
    err = _validate_ticker(ticker)
    if err:
        return jsonify({"error": err}), 400
    ticker = ticker.upper().strip()
    try:
        sent_score = score_sentiment(ticker)
        headlines = get_news_headlines(ticker, max_articles=10)
        news_api_active = bool(os.environ.get("NEWS_API_KEY", "").strip())
        return jsonify(
            {
                "ticker": ticker,
                "sentiment_score": round(sent_score, 2),
                "news_api_active": news_api_active,
                "headlines": headlines,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error fetching sentiment for %s", ticker)
        _error_counts["sentiment"] = _error_counts.get("sentiment", 0) + 1
        return jsonify({"error": str(exc), "ticker": ticker}), 500


@app.post("/portfolio")
def portfolio() -> Any:
    """Analyse a list of tickers and return ranked results.

    Request body (JSON):
        { "tickers": ["AAPL", "MSFT", "NVDA"] }

    Response body (JSON):
        {
          "results": [...],        # list of prediction dicts, sorted by score
          "summary": {
            "strong_buy": [...],   # total >= 8.0
            "buy": [...],          # 7.0 <= total < 8.0
            "hold": [...],         # 5.0 <= total < 7.0
            "sell": [...]          # total < 5.0
          }
        }
    """
    body = request.get_json(silent=True) or {}
    tickers = body.get("tickers", [])

    if not isinstance(tickers, list) or not tickers:
        return jsonify({"error": "Provide a non-empty 'tickers' list"}), 400

    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    results: list[dict] = []
    errors: list[dict] = []

    for ticker in tickers:
        if _validate_ticker(ticker):
            errors.append({"ticker": ticker, "error": "Invalid ticker symbol"})
            continue
        try:
            results.append(_build_prediction(ticker))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error predicting %s", ticker)
            _error_counts["portfolio"] = _error_counts.get("portfolio", 0) + 1
            errors.append({"ticker": ticker, "error": str(exc)})

    results.sort(key=lambda r: r["scores"]["total"], reverse=True)

    summary = {
        "strong_buy": [r["ticker"] for r in results if r["scores"]["total"] >= 8.0],
        "buy": [r["ticker"] for r in results if 7.0 <= r["scores"]["total"] < 8.0],
        "hold": [r["ticker"] for r in results if 5.0 <= r["scores"]["total"] < 7.0],
        "sell": [r["ticker"] for r in results if r["scores"]["total"] < 5.0],
    }

    return jsonify({"results": results, "summary": summary, "errors": errors})


@app.post("/portfolio/optimize")
def portfolio_optimize() -> Any:
    """Return optimised portfolio weights for a list of tickers.

    Uses a score-weighted allocation approach: each ticker's weight is
    proportional to its composite score, then normalised to sum to 1.0.
    Tickers with a SELL signal (score < 5.0) receive zero weight.

    Request body (JSON):
        {
          "tickers": ["AAPL", "MSFT", "NVDA"],
          "exclude_sell": true          # optional, default true
        }

    Response body (JSON):
        {
          "weights": {"AAPL": 0.45, "MSFT": 0.35, "NVDA": 0.20},
          "scores":  {"AAPL": 8.09, "MSFT": 7.32, "NVDA": 7.69},
          "signals": {"AAPL": "BUY", "MSFT": "BUY", "NVDA": "BUY"},
          "errors":  []
        }
    """
    body = request.get_json(silent=True) or {}
    tickers = body.get("tickers", [])
    exclude_sell = body.get("exclude_sell", True)

    if not isinstance(tickers, list) or not tickers:
        return jsonify({"error": "Provide a non-empty 'tickers' list"}), 400

    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    predictions: list[dict] = []
    errors: list[dict] = []

    for ticker in tickers:
        if _validate_ticker(ticker):
            errors.append({"ticker": ticker, "error": "Invalid ticker symbol"})
            continue
        try:
            predictions.append(_build_prediction(ticker))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error predicting %s for optimisation", ticker)
            _error_counts["portfolio_optimize"] = _error_counts.get("portfolio_optimize", 0) + 1
            errors.append({"ticker": ticker, "error": str(exc)})

    # Build score map; optionally exclude SELL signals
    score_map: dict[str, float] = {}
    for pred in predictions:
        score = pred["scores"]["total"]
        if exclude_sell and pred["signal"] == "SELL":
            score_map[pred["ticker"]] = 0.0
        else:
            score_map[pred["ticker"]] = max(0.0, score)

    total_score = sum(score_map.values())
    if total_score == 0.0:
        # All signals are SELL; distribute equally among all tickers
        n = len(predictions)
        total_score = float(n)
        score_map = {pred["ticker"]: 1.0 for pred in predictions}

    weights = {
        ticker: round(score / total_score, 4)
        for ticker, score in score_map.items()
    }

    return jsonify(
        {
            "weights": weights,
            "scores": {pred["ticker"]: pred["scores"]["total"] for pred in predictions},
            "signals": {pred["ticker"]: pred["signal"] for pred in predictions},
            "errors": errors,
        }
    )


@app.get("/portfolio/export")
def portfolio_export() -> Any:
    """Export portfolio analysis results as a CSV file.

    Query parameters:
        tickers – comma-separated list of ticker symbols (e.g. ``?tickers=AAPL,MSFT``)

    Returns a ``text/csv`` response with the analysis for each ticker,
    or a JSON error if the ``tickers`` parameter is missing / empty.
    """
    raw = request.args.get("tickers", "")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    if not tickers:
        return jsonify({"error": "Provide a 'tickers' query parameter (comma-separated)"}), 400

    results: list[dict] = []
    errors: list[dict] = []

    for ticker in tickers:
        if _validate_ticker(ticker):
            errors.append({"ticker": ticker, "error": "Invalid ticker symbol"})
            continue
        try:
            results.append(_build_prediction(ticker))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error building prediction for CSV export: %s", ticker)
            errors.append({"ticker": ticker, "error": str(exc)})

    results.sort(key=lambda r: r["scores"]["total"], reverse=True)

    # Build CSV in memory
    output = io.StringIO()
    fieldnames = [
        "ticker", "price", "signal", "confidence",
        "score_total", "score_fundamentals", "score_technicals",
        "score_risk", "score_ml", "score_sentiment", "score_etf",
        "timestamp",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(
            {
                "ticker": r["ticker"],
                "price": r["price"],
                "signal": r["signal"],
                "confidence": r["confidence"],
                "score_total": r["scores"]["total"],
                "score_fundamentals": r["scores"]["fundamentals"],
                "score_technicals": r["scores"]["technicals"],
                "score_risk": r["scores"]["risk"],
                "score_ml": r["scores"]["ml"],
                "score_sentiment": r["scores"]["sentiment"],
                "score_etf": r["scores"]["etf"],
                "timestamp": r["timestamp"],
            }
        )

    csv_content = output.getvalue()
    filename = f"portfolio_analysis_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", API_PORT))
    logger.info("Starting Investment API on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
