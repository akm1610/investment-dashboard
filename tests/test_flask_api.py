"""
tests/test_flask_api.py
-----------------------
Unit tests for the Flask REST API (flask_api.py).

Strategy
--------
* The heavy ML / data-fetching code is mocked so tests run quickly without
  network access or trained model files.
* Tests cover:
  - /health endpoint returns 200 with status field
  - /predict/<ticker> happy path and validation error paths
  - /portfolio endpoint happy path and error handling
"""

from __future__ import annotations

import sys
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in [_REPO_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_PREDICTION: dict[str, Any] = {
    "ticker": "AAPL",
    "price": 248.20,
    "scores": {
        "fundamentals": 9.5,
        "technicals": 6.5,
        "risk": 6.0,
        "ml": 9.2,
        "sentiment": 7.5,
        "etf": 9.5,
        "total": 8.09,
    },
    "signal": "BUY",
    "confidence": 59.6,
    "timestamp": "2026-03-20T16:00:00+00:00",
}


@pytest.fixture()
def client(monkeypatch):
    """Create a Flask test client with _build_prediction mocked."""
    import flask_api as api_module

    monkeypatch.setattr(
        api_module,
        "_build_prediction",
        lambda ticker: {**_FAKE_PREDICTION, "ticker": ticker.upper()},
    )
    api_module.app.config["TESTING"] = True
    with api_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_field_is_healthy(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "healthy"

    def test_timestamp_present(self, client):
        data = client.get("/health").get_json()
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# /predict/<ticker>
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_valid_ticker_returns_200(self, client):
        resp = client.get("/predict/AAPL")
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client):
        data = client.get("/predict/AAPL").get_json()
        assert data["ticker"] == "AAPL"
        assert "price" in data
        assert "scores" in data
        assert "signal" in data
        assert "confidence" in data

    def test_scores_contains_all_components(self, client):
        scores = client.get("/predict/MSFT").get_json()["scores"]
        for key in ("fundamentals", "technicals", "risk", "ml", "sentiment", "etf", "total"):
            assert key in scores, f"Missing score component: {key}"

    def test_ticker_is_uppercased(self, client):
        data = client.get("/predict/aapl").get_json()
        assert data["ticker"] == "AAPL"

    def test_numeric_ticker_returns_400(self, client):
        # Digits are not valid ticker symbols; they should trigger a 400
        resp = client.get("/predict/123")
        assert resp.status_code == 400

    def test_error_propagated_as_500(self, monkeypatch):
        import flask_api as api_module

        monkeypatch.setattr(
            api_module,
            "_build_prediction",
            lambda ticker: (_ for _ in ()).throw(RuntimeError("network error")),
        )
        api_module.app.config["TESTING"] = True
        with api_module.app.test_client() as c:
            resp = c.get("/predict/AAPL")
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# /portfolio
# ---------------------------------------------------------------------------


class TestPortfolioEndpoint:
    def test_valid_request_returns_200(self, client):
        resp = client.post("/portfolio", json={"tickers": ["AAPL", "MSFT"]})
        assert resp.status_code == 200

    def test_results_sorted_by_total_score(self, client, monkeypatch):
        import flask_api as api_module

        def _fake_pred(ticker: str) -> dict:
            scores = {"AAPL": 8.5, "MSFT": 7.0, "NVDA": 9.0}
            total = scores.get(ticker, 5.0)
            return {
                **_FAKE_PREDICTION,
                "ticker": ticker,
                "scores": {**_FAKE_PREDICTION["scores"], "total": total},
            }

        monkeypatch.setattr(api_module, "_build_prediction", _fake_pred)
        resp = client.post("/portfolio", json={"tickers": ["AAPL", "MSFT", "NVDA"]})
        data = resp.get_json()
        totals = [r["scores"]["total"] for r in data["results"]]
        assert totals == sorted(totals, reverse=True)

    def test_summary_buckets_populated(self, client, monkeypatch):
        import flask_api as api_module

        def _fake_pred(ticker: str) -> dict:
            scores = {"AAPL": 8.5, "MSFT": 6.0, "NVDA": 4.0}
            total = scores.get(ticker, 5.0)
            return {
                **_FAKE_PREDICTION,
                "ticker": ticker,
                "scores": {**_FAKE_PREDICTION["scores"], "total": total},
            }

        monkeypatch.setattr(api_module, "_build_prediction", _fake_pred)
        summary = client.post("/portfolio", json={"tickers": ["AAPL", "MSFT", "NVDA"]}).get_json()["summary"]
        assert "AAPL" in summary["strong_buy"]
        assert "MSFT" in summary["hold"]
        assert "NVDA" in summary["sell"]

    def test_empty_tickers_returns_400(self, client):
        resp = client.post("/portfolio", json={"tickers": []})
        assert resp.status_code == 400

    def test_missing_tickers_key_returns_400(self, client):
        resp = client.post("/portfolio", json={})
        assert resp.status_code == 400

    def test_failed_ticker_goes_to_errors_list(self, client, monkeypatch):
        import flask_api as api_module

        def _fake_pred(ticker: str) -> dict:
            if ticker == "BAD":
                raise ValueError("bad ticker")
            return {**_FAKE_PREDICTION, "ticker": ticker}

        monkeypatch.setattr(api_module, "_build_prediction", _fake_pred)
        data = client.post("/portfolio", json={"tickers": ["AAPL", "BAD"]}).get_json()
        assert any(e["ticker"] == "BAD" for e in data["errors"])
        assert any(r["ticker"] == "AAPL" for r in data["results"])


# ---------------------------------------------------------------------------
# Config: API_PORT
# ---------------------------------------------------------------------------


class TestConfig:
    def test_api_port_is_8001(self):
        from src.config import API_PORT
        assert API_PORT == 8001

    def test_dashboard_port_is_3000(self):
        from src.config import DASHBOARD_PORT
        assert DASHBOARD_PORT == 3000
