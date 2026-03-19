"""
tests/test_recommendation_generator.py
----------------------------------------
Unit tests for src/recommendation_generator.py.

All network calls and ML library imports are mocked so the tests run
fully offline without real market data or trained models.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in [_REPO_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.recommendation_generator import RecommendationGenerator, WatchlistBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _mock_ml_engine(signal: str = "BUY", confidence: float = 72.0) -> MagicMock:
    engine = MagicMock()
    engine.predict.return_value = {
        "ticker": "TEST",
        "signal": signal,
        "confidence": confidence,
        "strength": "MODERATE",
        "short_term": {"signal": signal, "confidence": confidence},
        "long_term": {"signal": signal, "confidence": confidence},
        "model_votes": {"lgb": signal, "xgb": signal},
        "feature_importance": {},
        "key_drivers": ["Bullish MACD", "RSI 45"],
    }
    return engine


def _mock_feature_engineer(price_df: pd.DataFrame) -> MagicMock:
    from src.ml_engine import FeatureEngineer

    real = FeatureEngineer()
    tech = real.extract_technical_features(price_df)

    fe = MagicMock()
    fe.extract_technical_features.return_value = tech
    fe.extract_fundamental_features.return_value = pd.Series({"pe_ratio": 25.0, "beta": 1.2})
    return fe


def _make_generator(signal: str = "BUY", confidence: float = 72.0) -> RecommendationGenerator:
    price_df = _make_price_df()
    ml_engine = _mock_ml_engine(signal, confidence)
    feature_engineer = _mock_feature_engineer(price_df)
    gen = RecommendationGenerator(ml_engine, feature_engineer)
    return gen, price_df


# ---------------------------------------------------------------------------
# RecommendationGenerator._compute_signal
# ---------------------------------------------------------------------------


class TestComputeSignal:
    def _rng_features(self, rsi: float, macd_hist: float, pct_sma200: float) -> pd.DataFrame:
        """Build a minimal feature row for testing signal computation."""
        return pd.DataFrame(
            {"rsi_14": [rsi], "macd_hist": [macd_hist], "price_vs_sma200": [pct_sma200]}
        )

    def test_buy_when_oversold_and_bullish_macd(self):
        features = self._rng_features(rsi=25.0, macd_hist=0.5, pct_sma200=0.0)
        ml_rec = {"signal": "HOLD", "confidence": 50.0, "model_votes": {}}
        result = RecommendationGenerator._compute_signal(ml_rec, features)
        assert result == "BUY"

    def test_sell_when_overbought_and_bearish_macd(self):
        features = self._rng_features(rsi=75.0, macd_hist=-0.5, pct_sma200=-0.1)
        ml_rec = {"signal": "HOLD", "confidence": 50.0, "model_votes": {}}
        result = RecommendationGenerator._compute_signal(ml_rec, features)
        assert result == "SELL"

    def test_hold_when_neutral(self):
        features = self._rng_features(rsi=50.0, macd_hist=0.0, pct_sma200=0.0)
        ml_rec = {"signal": "HOLD", "confidence": 50.0, "model_votes": {}}
        result = RecommendationGenerator._compute_signal(ml_rec, features)
        assert result == "HOLD"

    def test_uses_ml_signal_when_models_loaded(self):
        features = self._rng_features(rsi=25.0, macd_hist=0.5, pct_sma200=0.2)
        ml_rec = {"signal": "SELL", "confidence": 80.0, "model_votes": {"lgb": "SELL"}}
        result = RecommendationGenerator._compute_signal(ml_rec, features)
        assert result == "SELL"

    def test_hold_on_empty_features(self):
        ml_rec = {"signal": "HOLD", "confidence": 50.0, "model_votes": {}}
        result = RecommendationGenerator._compute_signal(ml_rec, pd.DataFrame())
        assert result == "HOLD"


# ---------------------------------------------------------------------------
# RecommendationGenerator._compute_composite_score
# ---------------------------------------------------------------------------


class TestComputeCompositeScore:
    def test_score_range(self):
        price_df = _make_price_df()
        from src.ml_engine import FeatureEngineer
        features = FeatureEngineer().extract_technical_features(price_df)
        ml_rec = {"confidence": 50.0, "model_votes": {}}
        score = RecommendationGenerator._compute_composite_score(ml_rec, features, "HOLD")
        assert 0 <= score <= 100

    def test_uses_ml_confidence_when_models_loaded(self):
        ml_rec = {"confidence": 78.5, "model_votes": {"lgb": "BUY"}}
        score = RecommendationGenerator._compute_composite_score(ml_rec, pd.DataFrame(), "BUY")
        assert score == int(round(78.5))  # Python banker's rounding

    def test_oversold_rsi_increases_score(self):
        oversold_features = pd.DataFrame(
            {"rsi_14": [22.0], "macd_hist": [0.1], "price_vs_sma200": [0.02]}
        )
        neutral_features = pd.DataFrame(
            {"rsi_14": [50.0], "macd_hist": [0.0], "price_vs_sma200": [0.0]}
        )
        ml_rec = {"confidence": 50.0, "model_votes": {}}
        score_oversold = RecommendationGenerator._compute_composite_score(
            ml_rec, oversold_features, "BUY"
        )
        score_neutral = RecommendationGenerator._compute_composite_score(
            ml_rec, neutral_features, "HOLD"
        )
        assert score_oversold > score_neutral


# ---------------------------------------------------------------------------
# RecommendationGenerator._calculate_target_price
# ---------------------------------------------------------------------------


class TestCalculateTargetPrice:
    def test_buy_signal_target_above_entry(self):
        target = RecommendationGenerator._calculate_target_price(100.0, "BUY", 75)
        assert target > 100.0

    def test_sell_signal_target_below_entry(self):
        target = RecommendationGenerator._calculate_target_price(100.0, "SELL", 75)
        assert target < 100.0

    def test_hold_signal_target_slightly_above(self):
        target = RecommendationGenerator._calculate_target_price(100.0, "HOLD", 50)
        assert target == pytest.approx(102.0, rel=1e-3)

    def test_higher_confidence_buy_gives_higher_target(self):
        target_low = RecommendationGenerator._calculate_target_price(100.0, "BUY", 40)
        target_high = RecommendationGenerator._calculate_target_price(100.0, "BUY", 90)
        assert target_high > target_low


# ---------------------------------------------------------------------------
# RecommendationGenerator._matches_risk_profile
# ---------------------------------------------------------------------------


class TestMatchesRiskProfile:
    def test_conservative_requires_high_score_and_buy_hold(self):
        assert RecommendationGenerator._matches_risk_profile(60, "BUY", "conservative")
        assert RecommendationGenerator._matches_risk_profile(60, "HOLD", "conservative")
        assert not RecommendationGenerator._matches_risk_profile(60, "SELL", "conservative")
        assert not RecommendationGenerator._matches_risk_profile(50, "BUY", "conservative")

    def test_moderate_accepts_lower_score(self):
        assert RecommendationGenerator._matches_risk_profile(45, "BUY", "moderate")
        assert RecommendationGenerator._matches_risk_profile(50, "HOLD", "moderate")
        assert not RecommendationGenerator._matches_risk_profile(40, "BUY", "moderate")

    def test_aggressive_lowest_threshold(self):
        assert RecommendationGenerator._matches_risk_profile(35, "BUY", "aggressive")
        assert not RecommendationGenerator._matches_risk_profile(30, "BUY", "aggressive")

    def test_unknown_profile_uses_default_threshold(self):
        # Unknown profile falls back to "moderate" threshold (45)
        assert RecommendationGenerator._matches_risk_profile(50, "BUY", "unknown")


# ---------------------------------------------------------------------------
# RecommendationGenerator._compute_drivers
# ---------------------------------------------------------------------------


class TestComputeDrivers:
    def test_uses_ml_key_drivers(self):
        ml_rec = {
            "key_drivers": ["AI demand (+15pts)", "Positive momentum", "Analyst upgrades"],
            "model_votes": {"lgb": "BUY"},
        }
        result = RecommendationGenerator._compute_drivers(ml_rec, pd.DataFrame())
        assert "AI demand" in result
        assert "Positive momentum" in result

    def test_technical_fallback_rsi(self):
        ml_rec = {"key_drivers": [], "model_votes": {}}
        features = pd.DataFrame({"rsi_14": [28.0], "macd_hist": [0.3], "price_vs_sma200": [0.05]})
        result = RecommendationGenerator._compute_drivers(ml_rec, features)
        assert "Oversold RSI" in result

    def test_default_when_no_features(self):
        ml_rec = {"key_drivers": [], "model_votes": {}}
        result = RecommendationGenerator._compute_drivers(ml_rec, pd.DataFrame())
        assert result == "Technical signals"


# ---------------------------------------------------------------------------
# RecommendationGenerator.generate_recommendations (integration-style)
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def _make_patched_generator(self, signal: str = "BUY", confidence: float = 72.0):
        price_df = _make_price_df()
        ml_engine = _mock_ml_engine(signal, confidence)
        feature_engineer = _mock_feature_engineer(price_df)

        gen = RecommendationGenerator(ml_engine, feature_engineer)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_stock_data.return_value = price_df
        mock_fetcher.fetch_company_info.return_value = {"trailingPE": 25.0}

        return gen, mock_fetcher

    def test_returns_list_sorted_by_score(self):
        gen, mock_fetcher = self._make_patched_generator(signal="BUY", confidence=72.0)

        with patch("src.data_fetcher.DataFetcher", return_value=mock_fetcher):
            recs = gen.generate_recommendations(["AAPL", "MSFT"], "moderate", count=10)

        assert isinstance(recs, list)
        if len(recs) >= 2:
            scores = [r["score"] for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_count_limits_results(self):
        gen, mock_fetcher = self._make_patched_generator(signal="BUY", confidence=72.0)

        with patch("src.data_fetcher.DataFetcher", return_value=mock_fetcher):
            recs = gen.generate_recommendations(
                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], "aggressive", count=3
            )

        assert len(recs) <= 3

    def test_rec_has_required_fields(self):
        gen, mock_fetcher = self._make_patched_generator(signal="BUY", confidence=72.0)

        with patch("src.data_fetcher.DataFetcher", return_value=mock_fetcher):
            recs = gen.generate_recommendations(["NVDA"], "moderate", count=5)

        if recs:
            rec = recs[0]
            for field in ("ticker", "score", "signal", "confidence", "entry_price",
                          "target_price", "drivers", "model_votes"):
                assert field in rec, f"Missing field: {field}"

    def test_filters_by_risk_profile_conservative(self):
        """With low-confidence SELL signals, conservative profile should return nothing."""
        gen, mock_fetcher = self._make_patched_generator(signal="SELL", confidence=30.0)

        with patch("src.data_fetcher.DataFetcher", return_value=mock_fetcher):
            recs = gen.generate_recommendations(["AAPL"], "conservative", count=5)

        # SELL signals are not allowed for conservative
        assert all(r["signal"] != "SELL" for r in recs)

    def test_skips_ticker_on_error(self):
        """A failing ticker should be silently skipped."""
        price_df = _make_price_df()
        ml_engine = _mock_ml_engine()
        feature_engineer = _mock_feature_engineer(price_df)
        gen = RecommendationGenerator(ml_engine, feature_engineer)

        good_fetcher = MagicMock()
        good_fetcher.fetch_stock_data.return_value = price_df
        good_fetcher.fetch_company_info.return_value = {}

        bad_fetcher = MagicMock()
        bad_fetcher.fetch_stock_data.side_effect = RuntimeError("Network error")

        def fetcher_factory(ticker):
            return bad_fetcher if ticker == "BAD" else good_fetcher

        with patch("src.data_fetcher.DataFetcher", side_effect=fetcher_factory):
            recs = gen.generate_recommendations(["AAPL", "BAD"], "moderate", count=5)

        tickers = [r["ticker"] for r in recs]
        assert "BAD" not in tickers


# ---------------------------------------------------------------------------
# WatchlistBuilder
# ---------------------------------------------------------------------------


class TestWatchlistBuilder:
    def _make_holdings(self, n: int = 4) -> list:
        return [
            {
                "ticker": f"T{i}",
                "score": 60 + i * 5,
                "signal": "BUY",
                "confidence": 60 + i * 5,
                "entry_price": 100.0 + i * 10,
                "target_price": 120.0 + i * 10,
                "drivers": "Test driver",
            }
            for i in range(n)
        ]

    def test_build_watchlist_structure(self):
        builder = WatchlistBuilder()
        mock_gen = MagicMock()
        mock_gen.generate_recommendations.return_value = self._make_holdings(3)

        wl = builder.build_watchlist(
            name="Test WL",
            strategy="Growth",
            description="Test description",
            tickers=["AAPL", "MSFT", "GOOGL"],
            risk_profiles=["moderate"],
            risk_level="MEDIUM",
            generator=mock_gen,
            risk_profile="moderate",
        )

        assert wl["name"] == "Test WL"
        assert wl["strategy"] == "Growth"
        assert wl["risk_level"] == "MEDIUM"
        assert "performance" in wl
        assert "holdings" in wl
        assert len(wl["holdings"]) == 3

    def test_estimate_performance_empty_holdings(self):
        perf = WatchlistBuilder._estimate_performance([])
        assert perf["win_rate"] == 0.50
        assert perf["avg_return"] == 0.05
        assert perf["sharpe"] == 0.80

    def test_estimate_performance_with_holdings(self):
        holdings = self._make_holdings(4)
        perf = WatchlistBuilder._estimate_performance(holdings)
        assert 0.0 < perf["win_rate"] <= 1.0
        assert perf["avg_return"] > 0
        assert perf["sharpe"] > 0

    def test_estimate_performance_win_rate_scales_with_confidence(self):
        low_conf = [
            {"confidence": 40, "entry_price": 100, "target_price": 105}
        ]
        high_conf = [
            {"confidence": 90, "entry_price": 100, "target_price": 115}
        ]
        perf_low = WatchlistBuilder._estimate_performance(low_conf)
        perf_high = WatchlistBuilder._estimate_performance(high_conf)
        assert perf_high["win_rate"] >= perf_low["win_rate"]
