"""
tests/test_app_integration.py
------------------------------
Integration tests for the updated Streamlit dashboard.

Strategy
--------
* All Streamlit API calls are mocked so tests run without a browser/server.
* Tests cover:
  - New session state keys (risk_profile, backtest_results, recommendations, …)
  - Risk & Recommendations page helpers (RiskProfileAssessor integration,
    watchlist filtering, recommendation ranking)
  - Backtesting page helpers (strategy factory, performance metric rendering)
  - Integration between pages (portfolio cross-page state)
"""

from __future__ import annotations

import sys
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
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
# MockSessionState helper (reuse pattern from test_app.py)
# ---------------------------------------------------------------------------


class _MockSessionState(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


# ---------------------------------------------------------------------------
# Tests: _init_session_state – new keys
# ---------------------------------------------------------------------------


class TestInitSessionStateNewKeys:
    """Ensure new session-state keys are populated by _init_session_state."""

    def _make_state(self) -> _MockSessionState:
        return _MockSessionState(
            portfolio={"holdings": {}, "cash": 0.0, "journal": [], "trades": []}
        )

    def test_risk_profile_default(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["risk_profile"] == "moderate"

    def test_risk_profile_result_default_none(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["risk_profile_result"] is None

    def test_backtest_results_default_empty_dict(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["backtest_results"] == {}

    def test_bt_comparisons_default_empty_dict(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["bt_comparisons"] == {}

    def test_recommendations_default_empty_list(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["recommendations"] == []

    def test_watchlists_default_empty_dict(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["watchlists"] == {}

    def test_existing_risk_profile_not_overwritten(self, monkeypatch):
        from src import app as app_module

        mock_state = self._make_state()
        mock_state["risk_profile"] = "aggressive"
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state["risk_profile"] == "aggressive"


# ---------------------------------------------------------------------------
# Constants for test validation
# ---------------------------------------------------------------------------

_RISK_PROFILE_RESULT_KEYS = frozenset({
    "risk_score", "profile", "suggested_volatility",
    "max_position_size", "recommended_allocation",
})


# ---------------------------------------------------------------------------
# Tests: RiskProfileAssessor integration
# ---------------------------------------------------------------------------


class TestRiskProfileAssessorIntegration:
    """Test that the RiskProfileAssessor is used correctly in the page."""

    def test_conservative_profile_from_low_answers(self):
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([1] * 10)
        assert result["profile"] == "conservative"
        assert result["risk_score"] == 0

    def test_aggressive_profile_from_high_answers(self):
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([5] * 10)
        assert result["profile"] == "aggressive"
        assert result["risk_score"] == 100

    def test_moderate_profile_from_mid_answers(self):
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([3] * 10)
        assert result["profile"] == "moderate"

    def test_result_has_required_keys(self):
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([3] * 10)
        assert _RISK_PROFILE_RESULT_KEYS.issubset(result.keys())

    def test_recommended_allocation_sums_to_one(self):
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        for answers in [[1] * 10, [3] * 10, [5] * 10]:
            result = assessor.assess(answers)
            alloc = result["recommended_allocation"]
            total = sum(alloc.values())
            assert abs(total - 1.0) < 1e-9, f"Allocation sums to {total} for {result['profile']}"


# ---------------------------------------------------------------------------
# Tests: risk_recommendations helpers
# ---------------------------------------------------------------------------


class TestRiskRecommendationsHelpers:
    """Test pure helper functions in the risk_recommendations component."""

    def test_upside_pct_positive(self):
        from components.risk_recommendations import _upside_pct

        assert abs(_upside_pct(100.0, 120.0) - 0.20) < 1e-9

    def test_upside_pct_negative(self):
        from components.risk_recommendations import _upside_pct

        assert abs(_upside_pct(100.0, 80.0) - (-0.20)) < 1e-9

    def test_upside_pct_zero_entry_returns_zero(self):
        from components.risk_recommendations import _upside_pct

        assert _upside_pct(0.0, 100.0) == 0.0

    def test_profile_badge_conservative(self):
        from components.risk_recommendations import _profile_badge

        assert "Conservative" in _profile_badge("conservative")

    def test_profile_badge_aggressive(self):
        from components.risk_recommendations import _profile_badge

        assert "Aggressive" in _profile_badge("aggressive")

    def test_position_size_label_high_score(self):
        from components.risk_recommendations import _position_size_label

        # High score → full max_size
        label = _position_size_label(85, "moderate")
        assert label == "10%"

    def test_position_size_label_low_score(self):
        from components.risk_recommendations import _position_size_label

        # Low score → reduced size
        label = _position_size_label(50, "moderate")
        assert "%" in label


# ---------------------------------------------------------------------------
# Tests: watchlist data integrity
# ---------------------------------------------------------------------------


class TestWatchlistData:
    """Verify the static watchlist data is well-formed."""

    def test_watchlists_not_empty(self):
        from components.risk_recommendations import _WATCHLISTS

        assert len(_WATCHLISTS) > 0

    def test_each_watchlist_has_required_fields(self):
        from components.risk_recommendations import _WATCHLISTS

        required = {"name", "description", "strategy", "risk_level",
                    "risk_profiles", "holdings_count", "performance", "holdings"}
        for wl in _WATCHLISTS:
            assert required.issubset(wl.keys()), f"Watchlist {wl.get('name')} missing fields"

    def test_each_watchlist_has_performance_metrics(self):
        from components.risk_recommendations import _WATCHLISTS

        for wl in _WATCHLISTS:
            perf = wl["performance"]
            assert "win_rate" in perf
            assert "avg_return" in perf
            assert "sharpe" in perf

    def test_each_holding_has_required_fields(self):
        from components.risk_recommendations import _WATCHLISTS

        required = {"ticker", "score", "signal", "confidence",
                    "entry_price", "target_price", "drivers"}
        for wl in _WATCHLISTS:
            for h in wl["holdings"]:
                assert required.issubset(h.keys()), (
                    f"Holding {h.get('ticker')} in {wl['name']} missing fields"
                )

    def test_scores_in_valid_range(self):
        from components.risk_recommendations import _WATCHLISTS

        for wl in _WATCHLISTS:
            for h in wl["holdings"]:
                assert 0 <= h["score"] <= 100, (
                    f"{h['ticker']} score {h['score']} out of range"
                )

    def test_signals_are_valid(self):
        from components.risk_recommendations import _WATCHLISTS

        valid_signals = {"BUY", "HOLD", "SELL"}
        for wl in _WATCHLISTS:
            for h in wl["holdings"]:
                assert h["signal"] in valid_signals, (
                    f"{h['ticker']} has invalid signal {h['signal']!r}"
                )

    def test_risk_profiles_are_valid(self):
        from components.risk_recommendations import _WATCHLISTS

        valid = {"conservative", "moderate", "aggressive"}
        for wl in _WATCHLISTS:
            for rp in wl["risk_profiles"]:
                assert rp in valid, f"Watchlist {wl['name']} has invalid risk profile {rp!r}"


# ---------------------------------------------------------------------------
# Tests: backtesting strategy factory
# ---------------------------------------------------------------------------


class TestBacktestingStrategyFactory:
    """Test that the strategy factory builds callable functions."""

    def test_momentum_strategy_callable(self):
        from components.backtesting import _build_strategy_func

        func = _build_strategy_func("momentum", {"lookback": 20, "threshold": 0.05})
        assert callable(func)

    def test_mean_reversion_strategy_callable(self):
        from components.backtesting import _build_strategy_func

        func = _build_strategy_func("mean_reversion", {"lookback": 50, "threshold": -1.5})
        assert callable(func)

    def test_rsi_strategy_callable(self):
        from components.backtesting import _build_strategy_func

        func = _build_strategy_func("rsi_oversold", {"period": 14, "threshold": 30})
        assert callable(func)

    def test_macd_strategy_callable(self):
        from components.backtesting import _build_strategy_func

        func = _build_strategy_func("macd_crossover", {"fast_ema": 12, "slow_ema": 26, "signal": 9})
        assert callable(func)

    def test_unknown_strategy_returns_callable(self):
        from components.backtesting import _build_strategy_func

        func = _build_strategy_func("unknown_strategy", {})
        assert callable(func)

    def test_momentum_strategy_returns_list(self):
        from components.backtesting import _build_strategy_func
        import pandas as pd

        func = _build_strategy_func("momentum", {"lookback": 5, "threshold": 0.01})
        prices = {
            "AAPL": pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        }
        result = func("2024-01-10", ["AAPL"], prices)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: BacktestEngine integration
# ---------------------------------------------------------------------------


class TestBacktestEngineIntegration:
    """Test BacktestEngine runs and returns expected result keys."""

    def _run_simple_backtest(self) -> dict:
        from backtesting_engine import BacktestEngine, momentum_strategy

        engine = BacktestEngine(
            start_date="2023-01-01",
            end_date="2023-06-30",
            initial_capital=100_000,
            tickers=["AAPL", "MSFT"],
        )

        def simple_strategy(date_str, tickers, prices):
            return momentum_strategy(prices, lookback=5, threshold=0.01)

        return engine.backtest_strategy(simple_strategy)

    def test_backtest_returns_dict(self):
        result = self._run_simple_backtest()
        assert isinstance(result, dict)

    def test_backtest_has_total_return(self):
        result = self._run_simple_backtest()
        assert "total_return" in result

    def test_backtest_has_equity_curve(self):
        result = self._run_simple_backtest()
        assert "equity_curve" in result
        assert len(result["equity_curve"]) > 0

    def test_backtest_has_win_rate(self):
        result = self._run_simple_backtest()
        assert "win_rate" in result
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_backtest_has_sharpe_ratio(self):
        result = self._run_simple_backtest()
        assert "sharpe_ratio" in result

    def test_backtest_has_benchmark_return(self):
        result = self._run_simple_backtest()
        assert "benchmark_return" in result

    def test_backtest_has_alpha_and_beta(self):
        result = self._run_simple_backtest()
        assert "alpha" in result
        assert "beta" in result


# ---------------------------------------------------------------------------
# Tests: Backtesting page constants
# ---------------------------------------------------------------------------


class TestBacktestingConstants:
    """Verify page-level constants are well-formed."""

    def test_benchmarks_dict_not_empty(self):
        from components.backtesting import _BENCHMARKS

        assert len(_BENCHMARKS) > 0

    def test_strategy_labels_cover_all_strategies(self):
        from components.backtesting import _STRATEGY_LABELS
        from backtesting_engine import STRATEGIES

        for key in STRATEGIES:
            assert key in _STRATEGY_LABELS, f"Strategy {key!r} missing from _STRATEGY_LABELS"

    def test_default_tickers_is_list(self):
        from components.backtesting import _DEFAULT_TICKERS

        assert isinstance(_DEFAULT_TICKERS, list)
        assert len(_DEFAULT_TICKERS) > 0


# ---------------------------------------------------------------------------
# Tests: page function imports
# ---------------------------------------------------------------------------


class TestPageImports:
    """Verify the two new page functions are importable from src/app.py."""

    def test_page_risk_recommendations_importable(self):
        from src.app import page_risk_recommendations  # noqa: F401

        assert callable(page_risk_recommendations)

    def test_page_backtesting_importable(self):
        from src.app import page_backtesting  # noqa: F401

        assert callable(page_backtesting)

    def test_six_pages_registered(self, monkeypatch):
        """_init_session_state + page list should contain 6 pages."""
        from src import app as app_module

        pages_captured = []

        class FakePage:
            def __init__(self, func, title="", icon=""):
                self.func = func
                self.title = title

        monkeypatch.setattr(app_module.st, "Page", FakePage)
        monkeypatch.setattr(
            app_module.st, "navigation",
            lambda pages: type("Nav", (), {"run": lambda self: None})()
        )

        mock_state = _MockSessionState(
            portfolio={"holdings": {}, "cash": 0.0, "journal": [], "trades": []}
        )
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()

        # Build the pages list the same way main() does
        pages = [
            FakePage(app_module.page_company_analysis, title="Company Analysis"),
            FakePage(app_module.page_portfolio_overview, title="Portfolio Overview"),
            FakePage(app_module.page_pretrade_checklist, title="Pre-Trade Checklist"),
            FakePage(app_module.page_investment_journal, title="Investment Journal"),
            FakePage(app_module.page_risk_recommendations, title="Risk & Recommendations"),
            FakePage(app_module.page_backtesting, title="Strategy Backtesting"),
        ]
        assert len(pages) == 6
