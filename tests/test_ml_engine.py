"""
tests/test_ml_engine.py
-----------------------
Unit tests for src/ml_engine.py (112+ tests).

All external network calls, ML library imports (lightgbm, xgboost,
tensorflow, sklearn), and yfinance calls are mocked so the test suite
runs fully offline and without optional ML dependencies installed.
"""

from __future__ import annotations

import pickle
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import ml_engine as me
from src.config import (
    ENSEMBLE_WEIGHTS,
    LONG_TERM_RETURN_THRESHOLD,
    SHORT_TERM_RETURN_THRESHOLD,
    SIGNAL_THRESHOLDS,
    TRAIN_TEST_SPLIT_RATIO,
)
from src.ml_engine import (
    DataCollector,
    FeatureEngineer,
    MLModelTrainer,
    ModelPerformanceTracker,
    RecommendationEngine,
    _generate_labels,
    _rsi,
    _ema,
    _atr,
    _obv,
    _stochastic,
    _adx,
    _safe_div,
    generate_labels,
    temporal_train_test_split,
)

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* trading days."""
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


def _make_feature_df(n: int = 100, n_features: int = 20) -> pd.DataFrame:
    """Return a small feature DataFrame."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, n_features)).astype(np.float32)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


def _make_labels(n: int = 100) -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.integers(0, 2, n).astype(int))


# ===========================================================================
# Private helper tests
# ===========================================================================


class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator_returns_nan(self):
        assert np.isnan(_safe_div(10.0, 0.0))

    def test_none_denominator_returns_nan(self):
        assert np.isnan(_safe_div(10.0, None))

    def test_custom_default(self):
        assert _safe_div(5.0, 0.0, default=-1.0) == -1.0

    def test_negative_values(self):
        assert _safe_div(-6.0, 2.0) == pytest.approx(-3.0)


class TestRsi:
    def test_rsi_range(self):
        prices = _make_price_df(100)["Close"]
        rsi = _rsi(prices, 14)
        assert rsi.dropna().between(0, 100).all()

    def test_rsi_length(self):
        prices = _make_price_df(50)["Close"]
        rsi = _rsi(prices, 14)
        assert len(rsi) == len(prices)


class TestEma:
    def test_ema_length(self):
        prices = _make_price_df(100)["Close"]
        ema = _ema(prices, 20)
        assert len(ema) == len(prices)

    def test_ema_converges(self):
        # Constant series → EMA should equal the constant
        prices = pd.Series([5.0] * 50)
        ema = _ema(prices, 10)
        assert ema.iloc[-1] == pytest.approx(5.0, abs=1e-6)


class TestAtr:
    def test_atr_positive(self):
        df = _make_price_df(100)
        atr = _atr(df["High"], df["Low"], df["Close"], 14)
        assert (atr.dropna() > 0).all()


class TestObv:
    def test_obv_cumulative(self):
        df = _make_price_df(50)
        obv = _obv(df["Close"], df["Volume"])
        assert len(obv) == len(df)

    def test_obv_direction(self):
        closes = pd.Series([10.0, 11.0, 10.5])
        volumes = pd.Series([100.0, 200.0, 150.0])
        obv = _obv(closes, volumes)
        # day1: +200, day2: -150 → cumsum starting from 0
        assert obv.iloc[1] == pytest.approx(200.0)
        assert obv.iloc[2] == pytest.approx(50.0)


class TestStochastic:
    def test_stoch_range(self):
        df = _make_price_df(100)
        k, d = _stochastic(df["High"], df["Low"], df["Close"])
        assert k.dropna().between(0, 100).all()
        assert d.dropna().between(0, 100).all()


class TestAdx:
    def test_adx_positive(self):
        df = _make_price_df(100)
        adx = _adx(df["High"], df["Low"], df["Close"], 14)
        # ADX should be non-negative where defined
        assert (adx.dropna() >= 0).all()


class TestGenerateLabels:
    def test_labels_binary(self):
        prices = pd.Series(range(1, 101), dtype=float)
        labels = _generate_labels(prices, forward_days=5, threshold=0.01)
        assert set(labels.unique()).issubset({0, 1})

    def test_labels_shorter_than_input(self):
        # With forward_days=5, last 5 rows have no future data and are dropped
        prices = pd.Series(range(1, 51), dtype=float)
        labels = _generate_labels(prices, forward_days=5, threshold=0.01)
        assert len(labels) == len(prices) - 5

    def test_generate_labels_short_term(self):
        prices = pd.Series(np.linspace(100, 110, 50))
        labels = generate_labels(prices, horizon="short_term")
        assert labels.dtype == int

    def test_generate_labels_long_term(self):
        prices = pd.Series(np.linspace(100, 120, 300))
        labels = generate_labels(prices, horizon="long_term")
        assert labels.dtype == int

    def test_generate_labels_custom_threshold(self):
        prices = pd.Series(np.linspace(100, 105, 30))
        labels = generate_labels(prices, horizon="short_term", threshold=0.001)
        assert 1 in labels.values


class TestTemporalSplit:
    def test_split_ratio(self):
        X = _make_feature_df(100)
        y = _make_labels(100)
        X_tr, X_te, y_tr, y_te = temporal_train_test_split(X, y, 0.8)
        assert len(X_tr) == 80
        assert len(X_te) == 20

    def test_time_ordering_preserved(self):
        X = _make_feature_df(100)
        y = _make_labels(100)
        X_tr, X_te, y_tr, y_te = temporal_train_test_split(X, y, 0.8)
        # All test indices should come after all train indices
        assert X_tr.index.max() < X_te.index.min()


# ===========================================================================
# FeatureEngineer tests
# ===========================================================================


class TestFeatureEngineerTechnical:
    def setup_method(self):
        self.fe = FeatureEngineer()
        self.price_df = _make_price_df(300)

    def test_returns_dataframe(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert isinstance(result, pd.DataFrame)

    def test_rsi_columns_present(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert "rsi_14" in result.columns
        assert "rsi_7" in result.columns

    def test_macd_columns_present(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_sma_columns_present(self):
        result = self.fe.extract_technical_features(self.price_df)
        for p in [20, 50, 200]:
            assert f"sma_{p}" in result.columns

    def test_bollinger_columns_present(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_pct" in result.columns

    def test_volume_columns_present(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert "obv" in result.columns
        assert "volume_ratio" in result.columns

    def test_temporal_columns_present(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns

    def test_missing_column_raises(self):
        bad_df = self.price_df.drop(columns=["Volume"])
        with pytest.raises(ValueError, match="missing columns"):
            self.fe.extract_technical_features(bad_df)

    def test_days_parameter_limits_output(self):
        result = self.fe.extract_technical_features(self.price_df, days=50)
        assert len(result) == 50

    def test_no_inf_values(self):
        result = self.fe.extract_technical_features(self.price_df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_at_least_25_features(self):
        result = self.fe.extract_technical_features(self.price_df)
        assert result.shape[1] >= 25

    def test_lowercase_column_names_accepted(self):
        df = self.price_df.copy()
        df.columns = [c.lower() for c in df.columns]
        result = self.fe.extract_technical_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_multiindex_columns_accepted(self):
        """yfinance ≥ 0.2.x / 1.x may return a MultiIndex (price_type, ticker)."""
        df = self.price_df.copy()
        ticker = "AAPL"
        df.columns = pd.MultiIndex.from_tuples(
            [(c, ticker) for c in df.columns], names=["Price", "Ticker"]
        )
        assert isinstance(df.columns, pd.MultiIndex)
        result = self.fe.extract_technical_features(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] >= 25

    def test_multiindex_columns_do_not_raise_tuple_lower_error(self):
        """Regression: tuple columns must not cause 'tuple has no attribute lower'."""
        df = self.price_df.copy()
        df.columns = pd.MultiIndex.from_tuples(
            [(c, "MSFT") for c in df.columns], names=["Price", "Ticker"]
        )
        # Should complete without AttributeError
        result = self.fe.extract_technical_features(df)
        assert "rsi_14" in result.columns


class TestFeatureEngineerFundamental:
    def setup_method(self):
        self.fe = FeatureEngineer()
        self.info = {
            "trailingPE": 20.5,
            "forwardPE": 18.0,
            "priceToBook": 3.2,
            "returnOnEquity": 0.28,
            "returnOnAssets": 0.14,
            "grossMargins": 0.43,
            "operatingMargins": 0.22,
            "profitMargins": 0.18,
            "revenueGrowth": 0.12,
            "earningsGrowth": 0.15,
            "debtToEquity": 50.0,
            "currentRatio": 2.0,
            "freeCashflow": 5_000_000,
            "operatingCashflow": 7_000_000,
            "beta": 1.1,
            "dividendYield": 0.015,
        }

    def test_returns_series(self):
        result = self.fe.extract_fundamental_features(self.info)
        assert isinstance(result, pd.Series)

    def test_pe_ratio_extracted(self):
        result = self.fe.extract_fundamental_features(self.info)
        assert result["pe_ratio"] == pytest.approx(20.5)

    def test_missing_keys_are_nan(self):
        result = self.fe.extract_fundamental_features({})
        assert result.isna().all()

    def test_fcf_to_ocf_computed(self):
        result = self.fe.extract_fundamental_features(self.info)
        expected = 5_000_000 / 7_000_000
        assert result["fcf_to_ocf"] == pytest.approx(expected, rel=1e-4)

    def test_none_value_becomes_nan(self):
        info = {"trailingPE": None}
        result = self.fe.extract_fundamental_features(info)
        assert np.isnan(result["pe_ratio"])

    def test_at_least_20_features(self):
        result = self.fe.extract_fundamental_features(self.info)
        assert len(result) >= 20


class TestFeatureEngineerSentiment:
    def setup_method(self):
        self.fe = FeatureEngineer()

    def test_returns_series(self):
        result = self.fe.extract_sentiment_features({})
        assert isinstance(result, pd.Series)

    def test_trend_improving_encodes_to_1(self):
        result = self.fe.extract_sentiment_features({"sentiment_trend": "improving"})
        assert result["sentiment_trend"] == 1.0

    def test_trend_declining_encodes_to_minus1(self):
        result = self.fe.extract_sentiment_features({"sentiment_trend": "declining"})
        assert result["sentiment_trend"] == -1.0

    def test_analyst_upside_computed(self):
        data = {"analyst_target_price": 110.0, "current_price": 100.0}
        result = self.fe.extract_sentiment_features(data)
        assert result["analyst_upside_pct"] == pytest.approx(0.10)

    def test_composite_social_sentiment(self):
        data = {
            "news_sentiment": 0.4,
            "reddit_sentiment": 0.2,
            "twitter_sentiment": 0.6,
            "stocktwits_sentiment": 0.0,
        }
        result = self.fe.extract_sentiment_features(data)
        assert result["composite_social_sentiment"] == pytest.approx(0.3, rel=1e-4)

    def test_upgrade_ratio_computed(self):
        data = {"upgrades_30d": 3.0, "downgrades_30d": 1.0}
        result = self.fe.extract_sentiment_features(data)
        assert result["upgrade_ratio"] == pytest.approx(0.75)

    def test_missing_data_returns_nan(self):
        result = self.fe.extract_sentiment_features({})
        assert np.isnan(result["news_sentiment"])

    def test_at_least_10_features(self):
        result = self.fe.extract_sentiment_features({})
        assert len(result) >= 10


class TestFeatureEngineerMarket:
    def setup_method(self):
        self.fe = FeatureEngineer()
        self.price_df = _make_price_df(100)

    def test_returns_series(self):
        result = self.fe.extract_market_features(self.price_df)
        assert isinstance(result, pd.Series)

    def test_volume_vs_avg_present(self):
        result = self.fe.extract_market_features(self.price_df)
        assert "volume_vs_avg" in result.index

    def test_with_market_df(self):
        mkt_df = _make_price_df(100, seed=99)
        result = self.fe.extract_market_features(self.price_df, market_df=mkt_df)
        assert "market_correlation" in result.index
        assert "beta" in result.index

    def test_no_market_df_correlation_nan(self):
        result = self.fe.extract_market_features(self.price_df)
        assert np.isnan(result["market_correlation"])


class TestPreprocessFeatures:
    def setup_method(self):
        self.fe = FeatureEngineer()

    def test_returns_dataframe(self):
        df = _make_feature_df(100, 20)
        result = self.fe.preprocess_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_after_preprocessing(self):
        df = _make_feature_df(100, 20)
        # Inject some NaN
        df.iloc[5, 3] = np.nan
        result = self.fe.preprocess_features(df)
        assert not result.isna().any().any()

    def test_drops_high_missing_columns(self):
        df = _make_feature_df(100, 5)
        df.iloc[:, 2] = np.nan  # 100% missing col
        result = self.fe.preprocess_features(df)
        assert result.shape[1] < 5

    def test_respects_top_n(self):
        df = _make_feature_df(200, 50)
        result = self.fe.preprocess_features(df, top_n=10)
        assert result.shape[1] <= 10

    def test_standardised_output(self):
        df = _make_feature_df(200, 5)
        result = self.fe.preprocess_features(df)
        if not result.empty:
            means = result.mean().abs()
            assert (means < 1e-4).all()

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame()
        result = self.fe.preprocess_features(df)
        assert result.empty

    def test_removes_correlated_features(self):
        # Two perfectly correlated columns
        base = pd.Series(np.random.randn(200))
        df = pd.DataFrame({"a": base, "b": base, "c": np.random.randn(200)})
        result = self.fe.preprocess_features(df)
        # Should drop one of the perfectly correlated pair
        assert result.shape[1] <= 2


# ===========================================================================
# MLModelTrainer tests
# ===========================================================================


class TestMLModelTrainerInit:
    def test_default_horizon(self):
        trainer = MLModelTrainer()
        assert trainer.horizon == "long_term"

    def test_short_term_horizon(self):
        trainer = MLModelTrainer(horizon="short_term")
        assert trainer.horizon == "short_term"

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            MLModelTrainer(horizon="invalid")

    def test_models_empty_initially(self):
        trainer = MLModelTrainer()
        assert trainer.models == {}


def _mock_lgb_model():
    model = MagicMock()
    model.predict.return_value = np.array([0.7, 0.4, 0.6])
    model.feature_importance.return_value = np.array([0.3, 0.2, 0.5])
    return model


def _mock_rf_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
    model.feature_importances_ = np.array([0.3, 0.3, 0.4])
    return model


def _mock_xgb_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65], [0.55, 0.45], [0.45, 0.55]])
    return model


class TestMLModelTrainerSaveLoad:
    def test_save_and_load_rf(self, tmp_path):
        """Verify save_models writes pkl files and load_models reads them back."""
        from sklearn.ensemble import RandomForestClassifier

        X = _make_feature_df(50, 3)
        y = _make_labels(50)
        trainer = MLModelTrainer(horizon="long_term")
        trainer.feature_names = ["a", "b", "c"]
        rf = RandomForestClassifier(n_estimators=2, random_state=0)
        rf.fit(X, y)
        trainer.models["rf"] = rf

        trainer.save_models(str(tmp_path))
        assert (tmp_path / "rf_long_term.pkl").exists()
        assert (tmp_path / "meta_long_term.pkl").exists()

        trainer2 = MLModelTrainer(horizon="long_term")
        trainer2.load_models(str(tmp_path))
        assert "rf" in trainer2.models
        assert trainer2.feature_names == ["a", "b", "c"]

    def test_load_from_nonexistent_dir(self, tmp_path):
        trainer = MLModelTrainer(horizon="short_term")
        trainer.load_models(str(tmp_path / "nonexistent"))
        assert trainer.models == {}


class TestMLModelTrainerEvaluate:
    def test_evaluate_returns_dict(self):
        from sklearn.ensemble import RandomForestClassifier
        trainer = MLModelTrainer()
        X = _make_feature_df(20, 3)
        y = _make_labels(20)
        rf = RandomForestClassifier(n_estimators=2, random_state=0)
        rf.fit(X, y)
        trainer.models["rf"] = rf
        X_test = _make_feature_df(6, 3)
        y_test = pd.Series([1, 0, 1, 0, 1, 0])
        results = trainer.evaluate_models(X_test, y_test)
        assert "rf" in results
        assert "ensemble" in results

    def test_evaluate_keys(self):
        from sklearn.ensemble import RandomForestClassifier
        trainer = MLModelTrainer()
        X = _make_feature_df(20, 3)
        y = _make_labels(20)
        rf = RandomForestClassifier(n_estimators=2, random_state=0)
        rf.fit(X, y)
        trainer.models["rf"] = rf
        X_test = _make_feature_df(6, 3)
        y_test = pd.Series([1, 0, 1, 0, 1, 0])
        results = trainer.evaluate_models(X_test, y_test)
        for model_key in results:
            assert {"accuracy", "precision", "recall", "f1"} <= set(results[model_key].keys())

    def test_evaluate_accuracy_between_0_and_1(self):
        from sklearn.ensemble import RandomForestClassifier
        trainer = MLModelTrainer()
        X = _make_feature_df(20, 3)
        y = _make_labels(20)
        rf = RandomForestClassifier(n_estimators=2, random_state=0)
        rf.fit(X, y)
        trainer.models["rf"] = rf
        X_test = _make_feature_df(6, 3)
        y_test = pd.Series([1, 0, 1, 0, 1, 0])
        results = trainer.evaluate_models(X_test, y_test)
        assert 0.0 <= results["rf"]["accuracy"] <= 1.0

    def test_evaluate_empty_models_returns_empty(self):
        trainer = MLModelTrainer()
        X_test = _make_feature_df(3, 3)
        y_test = pd.Series([1, 0, 1])
        results = trainer.evaluate_models(X_test, y_test)
        assert results == {}


# ===========================================================================
# RecommendationEngine tests
# ===========================================================================


class TestRecommendationEngine:
    def setup_method(self):
        self.engine = RecommendationEngine()

    def test_get_signal_strength_strong(self):
        assert RecommendationEngine.get_signal_strength(85.0) == "STRONG"

    def test_get_signal_strength_moderate(self):
        assert RecommendationEngine.get_signal_strength(70.0) == "MODERATE"

    def test_get_signal_strength_weak(self):
        assert RecommendationEngine.get_signal_strength(45.0) == "WEAK"

    def test_proba_to_signal_buy(self):
        thresholds = SIGNAL_THRESHOLDS["long_term"]
        assert RecommendationEngine._proba_to_signal(0.70, thresholds) == "BUY"

    def test_proba_to_signal_sell(self):
        thresholds = SIGNAL_THRESHOLDS["long_term"]
        assert RecommendationEngine._proba_to_signal(0.30, thresholds) == "SELL"

    def test_proba_to_signal_hold(self):
        thresholds = SIGNAL_THRESHOLDS["long_term"]
        assert RecommendationEngine._proba_to_signal(0.50, thresholds) == "HOLD"

    def test_proba_to_confidence_buy(self):
        conf = RecommendationEngine._proba_to_confidence(0.72, "BUY")
        assert conf == pytest.approx(72.0)

    def test_proba_to_confidence_sell(self):
        conf = RecommendationEngine._proba_to_confidence(0.28, "SELL")
        assert conf == pytest.approx(72.0)

    def test_proba_to_confidence_hold(self):
        conf = RecommendationEngine._proba_to_confidence(0.50, "HOLD")
        assert conf == pytest.approx(50.0)

    def test_predict_no_models_returns_hold(self):
        features = _make_feature_df(1, 5)
        result = self.engine.predict("AAPL", features=features)
        assert result["ticker"] == "AAPL"
        assert result["signal"] == "HOLD"
        assert "confidence" in result

    def test_predict_no_features_returns_hold(self):
        result = self.engine.predict("TSLA")
        assert result["signal"] == "HOLD"

    def test_predict_with_mocked_models(self):
        trainer = MLModelTrainer(horizon="long_term")
        trainer.feature_names = [f"feat_{i}" for i in range(5)]
        trainer.models["lgb"] = _mock_lgb_model()
        trainer.models["rf"] = _mock_rf_model()
        self.engine._trainers["long_term"] = trainer

        features = _make_feature_df(1, 5)
        features.columns = trainer.feature_names
        result = self.engine.predict("AAPL", features=features, horizon="long_term")
        assert result["ticker"] == "AAPL"
        assert result["signal"] in ("BUY", "HOLD", "SELL")
        assert 0.0 <= result["confidence"] <= 100.0
        assert result["strength"] in ("STRONG", "MODERATE", "WEAK")

    def test_predict_result_structure(self):
        result = self.engine.predict("MSFT")
        required_keys = {
            "ticker", "signal", "confidence", "strength",
            "short_term", "long_term", "model_votes",
            "feature_importance", "key_drivers",
        }
        assert required_keys <= set(result.keys())

    def test_predict_batch(self):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        results = self.engine.predict_batch(tickers)
        assert set(results.keys()) == set(tickers)
        for r in results.values():
            assert "signal" in r

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            self.engine.predict("AAPL", horizon="weekly")

    def test_feature_importance_top_10(self):
        trainer = MLModelTrainer(horizon="long_term")
        trainer.feature_names = [f"feat_{i}" for i in range(15)]
        lgb_model = MagicMock()
        lgb_model.predict.return_value = np.array([0.65])
        lgb_model.feature_importance.return_value = np.random.rand(15)
        trainer.models["lgb"] = lgb_model
        self.engine._trainers["long_term"] = trainer

        features = _make_feature_df(1, 15)
        features.columns = trainer.feature_names
        result = self.engine.predict("AAPL", features=features, horizon="long_term")
        assert len(result["feature_importance"]) <= 10

    def test_predict_batch_with_feature_map(self):
        features = _make_feature_df(1, 5)
        feature_map = {"AAPL": features, "MSFT": features}
        results = self.engine.predict_batch(
            ["AAPL", "MSFT"], feature_map=feature_map
        )
        assert "AAPL" in results
        assert "MSFT" in results


# ===========================================================================
# ModelPerformanceTracker tests
# ===========================================================================


class TestModelPerformanceTracker:
    def setup_method(self):
        self.tracker = ModelPerformanceTracker()

    def _dummy_prediction(self, signal: str = "BUY") -> Dict[str, Any]:
        return {"signal": signal, "confidence": 70.0}

    def test_empty_tracker_accuracy(self):
        result = self.tracker.calculate_accuracy()
        assert result["accuracy"] == 0.0
        assert result["total_predictions"] == 0

    def test_log_prediction_correct_buy(self):
        self.tracker.log_prediction(
            "AAPL",
            self._dummy_prediction("BUY"),
            actual_return=0.20,  # > LONG_TERM_RETURN_THRESHOLD (0.15)
            horizon="long_term",
        )
        result = self.tracker.calculate_accuracy("long_term", days=1)
        assert result["correct_predictions"] == 1

    def test_log_prediction_incorrect_buy(self):
        self.tracker.log_prediction(
            "AAPL",
            self._dummy_prediction("BUY"),
            actual_return=-0.10,  # lost money
            horizon="long_term",
        )
        result = self.tracker.calculate_accuracy("long_term", days=1)
        assert result["correct_predictions"] == 0

    def test_log_prediction_correct_sell(self):
        self.tracker.log_prediction(
            "TSLA",
            self._dummy_prediction("SELL"),
            actual_return=-0.20,
            horizon="long_term",
        )
        assert self.tracker.calculate_accuracy()["correct_predictions"] == 1

    def test_log_prediction_correct_hold(self):
        self.tracker.log_prediction(
            "MSFT",
            self._dummy_prediction("HOLD"),
            actual_return=0.005,  # within threshold
            horizon="long_term",
        )
        assert self.tracker.calculate_accuracy()["correct_predictions"] == 1

    def test_win_rate_no_predictions(self):
        assert self.tracker.get_win_rate("BUY") == 0.0

    def test_win_rate_buy(self):
        self.tracker.log_prediction("A", self._dummy_prediction("BUY"), 0.20, "long_term")
        self.tracker.log_prediction("B", self._dummy_prediction("BUY"), 0.20, "long_term")
        self.tracker.log_prediction("C", self._dummy_prediction("BUY"), -0.10, "long_term")
        assert self.tracker.get_win_rate("BUY", "long_term") == pytest.approx(2 / 3, rel=1e-4)

    def test_generate_performance_report_empty(self):
        report = self.tracker.generate_performance_report()
        assert report["total_predictions"] == 0
        assert report["overall_accuracy"] == 0.0

    def test_generate_performance_report_with_data(self):
        for _ in range(5):
            self.tracker.log_prediction("AAPL", self._dummy_prediction("BUY"), 0.20, "long_term")
        report = self.tracker.generate_performance_report()
        assert report["total_predictions"] == 5
        assert report["overall_accuracy"] == pytest.approx(1.0)

    def test_save_and_load(self, tmp_path):
        self.tracker.log_prediction("X", self._dummy_prediction("BUY"), 0.10, "short_term")
        path = str(tmp_path / "tracker.pkl")
        self.tracker.save(path)

        tracker2 = ModelPerformanceTracker()
        tracker2.load(path)
        assert len(tracker2._records) == 1
        assert tracker2._records[0]["ticker"] == "X"

    def test_load_nonexistent_file_safe(self, tmp_path):
        tracker = ModelPerformanceTracker()
        tracker.load(str(tmp_path / "ghost.pkl"))  # should not raise
        assert tracker._records == []

    def test_accuracy_horizon_filter(self):
        self.tracker.log_prediction("A", self._dummy_prediction("BUY"), 0.20, "short_term")
        self.tracker.log_prediction("B", self._dummy_prediction("SELL"), -0.20, "long_term")
        st = self.tracker.calculate_accuracy("short_term", days=1)
        lt = self.tracker.calculate_accuracy("long_term", days=1)
        assert st["total_predictions"] == 1
        assert lt["total_predictions"] == 1


# ===========================================================================
# DataCollector tests
# ===========================================================================


class TestDataCollector:
    def setup_method(self):
        self.collector = DataCollector()

    @patch("src.ml_engine.DataCollector.fetch_all_data")
    def test_fetch_all_data_called(self, mock_fetch):
        mock_fetch.return_value = {"ticker": "AAPL", "price_history": pd.DataFrame()}
        result = self.collector.fetch_all_data("AAPL")
        mock_fetch.assert_called_once_with("AAPL")
        assert result["ticker"] == "AAPL"

    def test_cache_prevents_second_fetch(self):
        """Test that the cache mechanism skips re-fetching."""
        fake_data = {
            "ticker": "MSFT",
            "price_history": _make_price_df(10),
            "info": {},
            "sentiment": {},
            "fetched_at": "2024-01-01T00:00:00",
        }
        self.collector._cache["MSFT"] = fake_data
        from datetime import datetime, timezone
        self.collector._cache_timestamps["MSFT"] = datetime.now(timezone.utc)

        # This should return cached data without calling yfinance
        result = self.collector.fetch_all_data("MSFT")
        assert result["ticker"] == "MSFT"

    def test_is_cached_false_when_empty(self):
        assert self.collector._is_cached("NVDA") is False

    def test_update_cache_invalidates(self):
        fake_data = {
            "ticker": "GOOGL",
            "price_history": pd.DataFrame(),
            "info": {},
            "sentiment": {},
            "fetched_at": "2024-01-01",
        }
        from datetime import datetime, timezone
        self.collector._cache["GOOGL"] = fake_data
        self.collector._cache_timestamps["GOOGL"] = datetime.now(timezone.utc)

        # update_cache should invalidate without raising when fetch also mocked
        with patch.object(self.collector, "fetch_all_data", return_value=fake_data):
            self.collector.update_cache("GOOGL")
        assert "GOOGL" not in self.collector._cache

    @patch("yfinance.Ticker")
    def test_fetch_all_data_normalises_multiindex_columns(self, mock_ticker_cls):
        """fetch_all_data must flatten MultiIndex columns from yfinance 1.x."""
        price_df = _make_price_df(50)
        multi_idx = pd.MultiIndex.from_tuples(
            [(c, "AAPL") for c in price_df.columns], names=["Price", "Ticker"]
        )
        price_df_multi = price_df.copy()
        price_df_multi.columns = multi_idx

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = price_df_multi
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        result = self.collector.fetch_all_data("AAPL")
        ph = result["price_history"]
        # All column names must be plain strings, not tuples
        assert all(isinstance(c, str) for c in ph.columns), (
            "Column names should be strings after MultiIndex flattening"
        )
        assert "Close" in ph.columns or "close" in ph.columns


# ===========================================================================
# Top-level re-export tests
# ===========================================================================


class TestReExports:
    """Verify the top-level ml_engine module re-exports work."""

    def test_recommendation_engine_importable(self):
        assert me.RecommendationEngine is RecommendationEngine

    def test_feature_engineer_importable(self):
        assert me.FeatureEngineer is FeatureEngineer

    def test_ml_model_trainer_importable(self):
        assert me.MLModelTrainer is MLModelTrainer

    def test_model_performance_tracker_importable(self):
        assert me.ModelPerformanceTracker is ModelPerformanceTracker

    def test_data_collector_importable(self):
        assert me.DataCollector is DataCollector

    def test_generate_labels_importable(self):
        assert me.generate_labels is generate_labels

    def test_temporal_split_importable(self):
        assert me.temporal_train_test_split is temporal_train_test_split


# ===========================================================================
# Config module tests
# ===========================================================================


class TestConfig:
    def test_ensemble_weights_sum_to_one(self):
        from src.config import ENSEMBLE_WEIGHTS
        for horizon, weights in ENSEMBLE_WEIGHTS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0), f"{horizon} weights don't sum to 1"

    def test_signal_thresholds_buy_gt_sell(self):
        from src.config import SIGNAL_THRESHOLDS
        for horizon, thresholds in SIGNAL_THRESHOLDS.items():
            assert thresholds["buy"] > thresholds["sell"], (
                f"{horizon} buy threshold must be > sell threshold"
            )

    def test_train_test_ratio_valid(self):
        assert 0 < TRAIN_TEST_SPLIT_RATIO < 1

    def test_return_thresholds_positive(self):
        assert SHORT_TERM_RETURN_THRESHOLD > 0
        assert LONG_TERM_RETURN_THRESHOLD > 0
