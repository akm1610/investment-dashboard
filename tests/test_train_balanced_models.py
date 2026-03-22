"""
tests/test_train_balanced_models.py
------------------------------------
Unit tests for scripts/train_balanced_models.py.

All external network calls (yfinance) and ML library calls are mocked so the
suite runs fully offline without any optional dependencies.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on the path so the script module can be imported.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.train_balanced_models as tbm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 300, ticker: str = "AAPL") -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with plain string columns."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close)
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_multiindex_price_df(n: int = 300, ticker: str = "AAPL") -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with yfinance 1.x MultiIndex columns."""
    df = _make_price_df(n, ticker)
    df.columns = pd.MultiIndex.from_tuples(
        [(c, ticker) for c in df.columns], names=["Price", "Ticker"]
    )
    return df


def _fake_fetch_all_data(price_df: pd.DataFrame) -> Dict[str, Any]:
    """Build a fake DataCollector.fetch_all_data return value."""
    return {
        "ticker": "AAPL",
        "price_history": price_df,
        "info": {"currentPrice": 150.0},
        "sentiment": {
            "news_sentiment": float("nan"),
            "news_volume_week": float("nan"),
            "analyst_rating": float("nan"),
            "analyst_target_price": float("nan"),
            "current_price": 150.0,
        },
        "fetched_at": "2024-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Tests for _build_dataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    """Tests for the _build_dataset helper in train_balanced_models.py."""

    def _run_build(self, price_df: pd.DataFrame):
        """Patch DataCollector and run _build_dataset for ['AAPL']."""
        fake_data = _fake_fetch_all_data(price_df)
        with patch("scripts.train_balanced_models.DataCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.fetch_all_data.return_value = fake_data
            X, y = tbm._build_dataset(["AAPL"])
        return X, y

    def test_returns_dataframe_and_series(self):
        price_df = _make_price_df(300)
        X, y = self._run_build(price_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_labels_are_binary(self):
        price_df = _make_price_df(300)
        X, y = self._run_build(price_df)
        assert set(y.unique()).issubset({0, 1})

    def test_multiindex_columns_handled(self):
        """
        Regression: MultiIndex columns (yfinance 1.x) must not cause
        'tuple object has no attribute lower'.
        """
        price_df = _make_multiindex_price_df(300)
        # Should not raise
        X, y = self._run_build(price_df)
        assert len(X) > 0

    def test_lowercase_columns_handled(self):
        """Lowercase column names must work correctly."""
        price_df = _make_price_df(300)
        price_df.columns = [c.lower() for c in price_df.columns]
        X, y = self._run_build(price_df)
        assert len(X) > 0

    def test_insufficient_data_skipped(self):
        """Tickers with < 100 rows should be skipped gracefully."""
        price_df = _make_price_df(50)  # only 50 rows
        fake_data = _fake_fetch_all_data(price_df)
        with patch("scripts.train_balanced_models.DataCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.fetch_all_data.return_value = fake_data
            with pytest.raises(RuntimeError, match="No usable training data"):
                tbm._build_dataset(["AAPL"])

    def test_fetch_error_skips_ticker(self):
        """If fetch_all_data raises, the ticker is skipped and others continue."""
        good_df = _make_price_df(300)
        good_data = _fake_fetch_all_data(good_df)
        good_data["ticker"] = "MSFT"

        call_count = 0

        def side_effect(ticker):
            nonlocal call_count
            call_count += 1
            if ticker == "AAPL":
                raise ValueError("network error")
            return good_data

        with patch("scripts.train_balanced_models.DataCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.fetch_all_data.side_effect = side_effect
            X, y = tbm._build_dataset(["AAPL", "MSFT"])
        assert len(X) > 0
        assert call_count == 2

    def test_no_close_column_skips_ticker(self):
        """A DataFrame missing any 'close' variant should be skipped gracefully."""
        price_df = _make_price_df(300).drop(columns=["Close"])
        # Should raise RuntimeError (no data collected) rather than crash
        fake_data = _fake_fetch_all_data(price_df)
        with patch("scripts.train_balanced_models.DataCollector") as MockCollector:
            instance = MockCollector.return_value
            instance.fetch_all_data.return_value = fake_data
            with pytest.raises(RuntimeError, match="No usable training data"):
                tbm._build_dataset(["AAPL"])


# ---------------------------------------------------------------------------
# Tests for CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_default_tickers(self):
        args = tbm._parse_args.__wrapped__() if hasattr(tbm._parse_args, "__wrapped__") else None
        # Just verify the module attribute is set correctly
        assert "AAPL" in tbm.DEFAULT_TICKERS
        assert "MSFT" in tbm.DEFAULT_TICKERS
        assert "GOOGL" in tbm.DEFAULT_TICKERS
        assert "NVDA" in tbm.DEFAULT_TICKERS

    def test_tickers_are_strings(self):
        """All default tickers must be plain strings, not tuples."""
        for t in tbm.DEFAULT_TICKERS:
            assert isinstance(t, str), f"Ticker {t!r} is not a string"
            assert t == t.upper(), f"Ticker {t!r} should be uppercase"
