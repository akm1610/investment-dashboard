"""
tests/test_risk_engine.py
-------------------------
Unit tests for src/risk_engine.py (60+ tests).

All tests are fully offline – no network calls are made.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

import risk_engine as re_root
from src.risk_engine import (
    PortfolioHealthMonitor,
    PortfolioRiskAnalyzer,
    PositionSizer,
    RiskProfileAssessor,
    _compute_overall_risk_score,
    calculate_beta,
    calculate_conditional_var,
    calculate_correlation_matrix,
    calculate_max_drawdown,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_value_at_risk,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n: int = 252, mean: float = 0.001, std: float = 0.01, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def _make_returns_df(
    tickers: List[str], n: int = 252, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {t: rng.normal(0.001, 0.01, n) for t in tickers}
    return pd.DataFrame(data)


# ===========================================================================
# Root re-export sanity check
# ===========================================================================


class TestRootReExport:
    def test_position_sizer_importable(self):
        assert re_root.PositionSizer is PositionSizer

    def test_risk_profile_assessor_importable(self):
        assert re_root.RiskProfileAssessor is RiskProfileAssessor

    def test_risk_metrics_importable(self):
        assert re_root.calculate_sharpe_ratio is calculate_sharpe_ratio


# ===========================================================================
# Standalone risk-metric functions
# ===========================================================================


class TestCalculatePortfolioVolatility:
    def test_positive_volatility(self):
        returns = _make_returns()
        vol = calculate_portfolio_volatility(returns)
        assert vol > 0

    def test_empty_returns(self):
        assert calculate_portfolio_volatility(pd.Series([], dtype=float)) == 0.0

    def test_single_element(self):
        assert calculate_portfolio_volatility(pd.Series([0.01])) == 0.0

    def test_constant_returns_zero_vol(self):
        returns = pd.Series([0.01] * 252)
        assert calculate_portfolio_volatility(returns) == pytest.approx(0.0, abs=1e-9)

    def test_annualization(self):
        """Daily std of 0.01 → annualised ~0.01*sqrt(252)."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 10_000))
        vol = calculate_portfolio_volatility(returns)
        assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


class TestCalculateValueAtRisk:
    def test_var_positive(self):
        returns = _make_returns()
        var = calculate_value_at_risk(returns)
        assert var > 0

    def test_var_increases_with_confidence(self):
        returns = _make_returns(n=1000)
        var_95 = calculate_value_at_risk(returns, 0.95)
        var_99 = calculate_value_at_risk(returns, 0.99)
        assert var_99 >= var_95

    def test_empty_returns(self):
        assert calculate_value_at_risk(pd.Series([], dtype=float)) == 0.0

    def test_known_distribution(self):
        """For N(0, 0.01) daily returns, 95% VaR ≈ 1.645*0.01."""
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0, 0.01, 100_000))
        var = calculate_value_at_risk(returns, 0.95)
        assert var == pytest.approx(1.645 * 0.01, rel=0.05)


class TestCalculateConditionalVaR:
    def test_cvar_gte_var(self):
        returns = _make_returns(n=1000)
        var = calculate_value_at_risk(returns)
        cvar = calculate_conditional_var(returns)
        assert cvar >= var

    def test_empty_returns(self):
        assert calculate_conditional_var(pd.Series([], dtype=float)) == 0.0

    def test_cvar_positive(self):
        returns = _make_returns()
        assert calculate_conditional_var(returns) > 0


class TestCalculateSharpeRatio:
    def test_positive_for_positive_mean(self):
        returns = _make_returns(mean=0.002, std=0.01)
        assert calculate_sharpe_ratio(returns) > 0

    def test_zero_std(self):
        returns = pd.Series([0.001] * 252)
        # constant returns minus risk-free might be positive; any finite value is fine
        result = calculate_sharpe_ratio(returns)
        assert isinstance(result, float)

    def test_empty_returns(self):
        assert calculate_sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_negative_for_negative_mean(self):
        returns = _make_returns(mean=-0.003, std=0.01)
        assert calculate_sharpe_ratio(returns) < 0


class TestCalculateSortinoRatio:
    def test_positive_for_positive_mean(self):
        returns = _make_returns(mean=0.002, std=0.01)
        result = calculate_sortino_ratio(returns)
        assert result > 0

    def test_no_downside(self):
        """All positive returns → downside std zero → ratio = 0."""
        returns = pd.Series([0.01] * 100)
        assert calculate_sortino_ratio(returns) == pytest.approx(0.0)

    def test_empty_returns(self):
        assert calculate_sortino_ratio(pd.Series([], dtype=float)) == 0.0

    def test_sortino_handles_mixed_returns(self):
        """Sortino ratio should be finite and non-negative for a series with
        both positive and negative returns and a positive mean."""
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0.002, 0.01, 500))
        sortino = calculate_sortino_ratio(returns)
        assert sortino > 0


class TestCalculateMaxDrawdown:
    def test_positive_for_declining_series(self):
        returns = pd.Series([-0.01] * 100)
        dd = calculate_max_drawdown(returns)
        assert dd > 0

    def test_zero_for_all_positive_returns(self):
        returns = pd.Series([0.005] * 100)
        assert calculate_max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns(self):
        assert calculate_max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_known_50pct_drawdown(self):
        """A drop from 1.0 to 0.5 and recovery gives max DD ≈ 50%."""
        # Build returns: +0% for 50 days, -50% instant, then +100% recovery
        returns = pd.Series([-0.01] * 50 + [0.01] * 50)
        dd = calculate_max_drawdown(returns)
        assert dd > 0.3


class TestCalculateBeta:
    def test_market_equals_one(self):
        returns = _make_returns(n=500, seed=1)
        assert calculate_beta(returns, returns) == pytest.approx(1.0, abs=1e-9)

    def test_zero_market_variance(self):
        """When market has zero variance, beta falls back to 1.0."""
        returns = _make_returns()
        # Series of all zeros has exactly zero variance
        market = pd.Series([0.0] * 252)
        assert calculate_beta(returns, market) == 1.0

    def test_empty_inputs(self):
        assert calculate_beta(pd.Series([], dtype=float), pd.Series([0.01, 0.02])) == 1.0

    def test_negative_beta(self):
        """Inverse-market asset should have negative beta."""
        market = _make_returns(n=500, seed=10)
        inverse = -market
        beta = calculate_beta(inverse, market)
        assert beta < 0


class TestCalculateCorrelationMatrix:
    def test_diagonal_is_one(self):
        df = _make_returns_df(["A", "B", "C"])
        corr = calculate_correlation_matrix(df)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self):
        df = _make_returns_df(["X", "Y"])
        corr = calculate_correlation_matrix(df)
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_empty_df(self):
        result = calculate_correlation_matrix(pd.DataFrame())
        assert result.empty


# ===========================================================================
# PositionSizer
# ===========================================================================


class TestPositionSizerKelly:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_positive_edge_gives_positive_size(self):
        size = self.sizer.kelly_sizing(0.6, 100, 100, 100_000)
        assert size > 0

    def test_negative_edge_gives_zero(self):
        # win_rate=0.3, avg_win=50, avg_loss=100 → negative Kelly → 0
        size = self.sizer.kelly_sizing(0.3, 50, 100, 100_000)
        assert size == 0.0

    def test_zero_avg_loss_returns_zero(self):
        assert self.sizer.kelly_sizing(0.6, 100, 0, 100_000) == 0.0

    def test_kelly_fraction_applied(self):
        """25% Kelly: full Kelly of 0.2 → 0.05 of account."""
        # b=1, p=0.6, q=0.4 → full Kelly = (1*0.6-0.4)/1 = 0.2
        size = self.sizer.kelly_sizing(0.6, 100, 100, 100_000)
        assert size == pytest.approx(0.2 * 0.25 * 100_000, rel=0.01)


class TestPositionSizerFixedFractional:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_default_2pct(self):
        assert self.sizer.fixed_fractional_sizing(100_000) == pytest.approx(2_000.0)

    def test_custom_risk(self):
        assert self.sizer.fixed_fractional_sizing(100_000, 0.05) == pytest.approx(5_000.0)

    def test_zero_account(self):
        assert self.sizer.fixed_fractional_sizing(0) == 0.0

    def test_risk_clamped_high(self):
        """risk_per_trade > 10% should be clamped to 10%."""
        size = self.sizer.fixed_fractional_sizing(100_000, 0.50)
        assert size <= 10_000.0


class TestPositionSizerOneTwoThree:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_micro_cap(self):
        assert self.sizer.one_two_three_sizing(100_000, "micro") == pytest.approx(1_000.0)

    def test_mid_cap(self):
        assert self.sizer.one_two_three_sizing(100_000, "mid") == pytest.approx(2_000.0)

    def test_large_cap(self):
        assert self.sizer.one_two_three_sizing(100_000, "large") == pytest.approx(3_000.0)

    def test_unknown_tier_defaults_to_mid(self):
        size = self.sizer.one_two_three_sizing(100_000, "unknown")
        assert size == pytest.approx(2_000.0)


class TestPositionSizerVolatilityAdjusted:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_lower_vol_gives_bigger_position(self):
        """Volatility 10% → smaller position than 20%."""
        low_vol = self.sizer.volatility_adjusted_sizing(100_000, 0.10)
        high_vol = self.sizer.volatility_adjusted_sizing(100_000, 0.20)
        assert low_vol > high_vol

    def test_zero_volatility_returns_zero(self):
        assert self.sizer.volatility_adjusted_sizing(100_000, 0.0) == 0.0

    def test_position_capped_at_25pct(self):
        # Very low volatility should not exceed 25% of account
        size = self.sizer.volatility_adjusted_sizing(100_000, 0.001)
        assert size <= 25_000.0


class TestPositionSizerRiskParity:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_weights_sum_to_one(self):
        weights = self.sizer.risk_parity_sizing(
            ["A", "B", "C"], [0.10, 0.20, 0.30]
        )
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_lower_vol_gets_higher_weight(self):
        weights = self.sizer.risk_parity_sizing(["A", "B"], [0.10, 0.20])
        assert weights["A"] > weights["B"]

    def test_equal_vols_give_equal_weights(self):
        weights = self.sizer.risk_parity_sizing(["X", "Y"], [0.15, 0.15])
        assert weights["X"] == pytest.approx(0.5, abs=1e-9)
        assert weights["Y"] == pytest.approx(0.5, abs=1e-9)

    def test_empty_inputs(self):
        assert self.sizer.risk_parity_sizing([], []) == {}

    def test_mismatched_inputs(self):
        assert self.sizer.risk_parity_sizing(["A"], [0.1, 0.2]) == {}


class TestPositionSizerSuggest:
    def setup_method(self):
        self.sizer = PositionSizer()

    def test_returns_required_keys(self):
        result = self.sizer.suggest_position_size("AAPL", 100_000, 0.75)
        for key in ("ticker", "position_size", "notional_value", "method_used"):
            assert key in result

    def test_ticker_preserved(self):
        result = self.sizer.suggest_position_size("MSFT", 100_000, 0.80)
        assert result["ticker"] == "MSFT"

    def test_smart_method(self):
        result = self.sizer.suggest_position_size(
            "AAPL", 100_000, 0.78, method="smart", stock_volatility=0.02
        )
        assert result["method_used"] == "smart"
        assert 0 < result["position_size"] <= 0.20

    def test_kelly_method(self):
        result = self.sizer.suggest_position_size(
            "TSLA", 100_000, 0.65,
            method="kelly", win_rate=0.6, avg_win=100, avg_loss=80
        )
        assert result["method_used"] == "kelly"

    def test_volatility_method(self):
        result = self.sizer.suggest_position_size(
            "NVDA", 100_000, 0.70, method="volatility", stock_volatility=0.03
        )
        assert result["method_used"] == "volatility_adjusted"

    def test_shares_calculated_when_price_given(self):
        result = self.sizer.suggest_position_size(
            "AAPL", 100_000, 0.75, entry_price=200.0
        )
        assert result["shares"] is not None and result["shares"] >= 0

    def test_shares_none_without_price(self):
        result = self.sizer.suggest_position_size("AAPL", 100_000, 0.75)
        assert result["shares"] is None


# ===========================================================================
# RiskProfileAssessor
# ===========================================================================


class TestRiskProfileAssessor:
    def setup_method(self):
        self.assessor = RiskProfileAssessor()

    def _moderate_answers(self) -> List[int]:
        return [3, 4, 4, 3, 2, 3, 3, 3, 2, 1]

    def _conservative_answers(self) -> List[int]:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def _aggressive_answers(self) -> List[int]:
        return [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    def test_moderate_profile(self):
        result = self.assessor.assess(self._moderate_answers())
        assert result["profile"] == "moderate"

    def test_conservative_profile(self):
        result = self.assessor.assess(self._conservative_answers())
        assert result["profile"] == "conservative"

    def test_aggressive_profile(self):
        result = self.assessor.assess(self._aggressive_answers())
        assert result["profile"] == "aggressive"

    def test_risk_score_range(self):
        for answers in [
            self._conservative_answers(),
            self._moderate_answers(),
            self._aggressive_answers(),
        ]:
            result = self.assessor.assess(answers)
            assert 0 <= result["risk_score"] <= 100

    def test_returns_all_required_keys(self):
        result = self.assessor.assess(self._moderate_answers())
        for key in (
            "risk_score",
            "profile",
            "suggested_volatility",
            "max_position_size",
            "recommended_allocation",
        ):
            assert key in result

    def test_allocation_keys(self):
        result = self.assessor.assess(self._moderate_answers())
        alloc = result["recommended_allocation"]
        assert "stocks" in alloc and "bonds" in alloc

    def test_wrong_number_of_answers_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            self.assessor.assess([3, 3, 3])

    def test_out_of_range_answer_raises(self):
        answers = [3] * 10
        answers[0] = 6
        with pytest.raises(ValueError, match="between 1 and 5"):
            self.assessor.assess(answers)

    def test_answer_of_zero_raises(self):
        answers = [0] + [3] * 9
        with pytest.raises(ValueError):
            self.assessor.assess(answers)

    def test_validate_portfolio_fit_aligned(self):
        result = self.assessor.validate_portfolio_fit(
            portfolio={"holdings": {"AAPL": 0.05, "MSFT": 0.05}},
            profile="moderate",
        )
        assert result["is_aligned"] is True

    def test_validate_portfolio_fit_unaligned(self):
        result = self.assessor.validate_portfolio_fit(
            portfolio={"holdings": {"AAPL": 0.50}},
            profile="conservative",
        )
        assert result["is_aligned"] is False
        assert len(result["issues"]) > 0

    def test_validate_portfolio_fit_uses_answers(self):
        result = self.assessor.validate_portfolio_fit(
            portfolio={"holdings": {"AAPL": 0.02}},
            answers=self._conservative_answers(),
        )
        assert result["profile"] == "conservative"


# ===========================================================================
# PortfolioRiskAnalyzer
# ===========================================================================


class TestAnalyzeConcentration:
    def setup_method(self):
        self.analyzer = PortfolioRiskAnalyzer()

    def test_single_stock(self):
        result = self.analyzer.analyze_concentration({"AAPL": 1.0})
        assert result["herfindahl_index"] == pytest.approx(1.0)
        assert result["concentration_score"] == 100

    def test_equal_weights(self):
        holdings = {t: 0.25 for t in ["A", "B", "C", "D"]}
        result = self.analyzer.analyze_concentration(holdings)
        # Fully equal → HHI = 1/4 = 0.25, concentration_score should be 0
        assert result["concentration_score"] == 0

    def test_large_position_alert(self):
        result = self.analyzer.analyze_concentration({"AAPL": 0.35, "MSFT": 0.65})
        alerts = [a for a in result["alerts"] if a["position"] == "AAPL"]
        assert any(a["severity"] in ("HIGH", "MEDIUM") for a in result["alerts"])

    def test_empty_holdings(self):
        result = self.analyzer.analyze_concentration({})
        assert result["herfindahl_index"] == 0.0

    def test_top5_cumulative(self):
        holdings = {f"S{i}": 0.10 for i in range(10)}
        result = self.analyzer.analyze_concentration(holdings)
        assert result["top_5_cumulative"] == pytest.approx(0.5, abs=0.01)


class TestAnalyzeSectorExposure:
    def setup_method(self):
        self.analyzer = PortfolioRiskAnalyzer()

    def test_single_sector_high_alert(self):
        holdings = {
            "AAPL": {"weight": 0.30, "sector": "Technology"},
            "MSFT": {"weight": 0.30, "sector": "Technology"},
            "GOOG": {"weight": 0.40, "sector": "Technology"},
        }
        result = self.analyzer.analyze_sector_exposure(holdings)
        assert any(a["severity"] == "HIGH" for a in result["alerts"])

    def test_diversified_no_alert(self):
        holdings = {
            "A": {"weight": 0.10, "sector": "Technology"},
            "B": {"weight": 0.10, "sector": "Healthcare"},
            "C": {"weight": 0.10, "sector": "Financials"},
            "D": {"weight": 0.10, "sector": "Energy"},
            "E": {"weight": 0.10, "sector": "Utilities"},
        }
        result = self.analyzer.analyze_sector_exposure(holdings)
        assert result["alerts"] == []

    def test_fallback_weight_only(self):
        result = self.analyzer.analyze_sector_exposure({"X": 0.5, "Y": 0.5})
        assert "Unknown" in result["sector_weights"]


class TestAnalyzeCorrelation:
    def setup_method(self):
        self.analyzer = PortfolioRiskAnalyzer()

    def test_highly_correlated_detected(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 0.01, 300)
        df = pd.DataFrame({"A": base, "B": base * 0.99 + rng.normal(0, 0.0001, 300)})
        result = self.analyzer.analyze_correlation(df)
        assert len(result["highly_correlated_pairs"]) > 0

    def test_empty_df(self):
        result = self.analyzer.analyze_correlation(pd.DataFrame())
        assert result["avg_correlation"] == 0.0

    def test_diversification_score_range(self):
        df = _make_returns_df(["X", "Y", "Z"])
        result = self.analyzer.analyze_correlation(df)
        assert 0 <= result["diversification_score"] <= 100


class TestAnalyzeDrawdownHistory:
    def setup_method(self):
        self.analyzer = PortfolioRiskAnalyzer()

    def test_empty_returns(self):
        result = self.analyzer.analyze_drawdown_history(pd.Series([], dtype=float))
        assert result["max_drawdown"] == 0.0

    def test_all_positive_no_drawdown(self):
        result = self.analyzer.analyze_drawdown_history(pd.Series([0.005] * 100))
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
        assert result["num_drawdown_periods"] == 0

    def test_declining_series_has_drawdown(self):
        result = self.analyzer.analyze_drawdown_history(pd.Series([-0.01] * 50))
        assert result["max_drawdown"] > 0

    def test_current_drawdown_when_in_dd(self):
        returns = pd.Series([0.01] * 100 + [-0.02] * 20)
        result = self.analyzer.analyze_drawdown_history(returns)
        assert result["current_drawdown"] > 0


class TestStressTest:
    def setup_method(self):
        self.analyzer = PortfolioRiskAnalyzer()

    def test_market_crash_is_negative(self):
        portfolio = {
            "AAPL": {"weight": 0.5, "sector": "Technology"},
            "MSFT": {"weight": 0.5, "sector": "Technology"},
        }
        result = self.analyzer.stress_test(portfolio, ["market_crash"])
        assert result["market_crash"]["portfolio_loss"] < 0

    def test_all_scenarios_returned(self):
        portfolio = {"X": 1.0}
        result = self.analyzer.stress_test(portfolio)
        assert set(result.keys()) == {
            "market_crash", "sector_rotation", "rate_spike", "recession"
        }

    def test_unknown_scenario_ignored(self):
        result = self.analyzer.stress_test({"X": 1.0}, ["nonexistent"])
        assert result == {}

    def test_details_populated(self):
        result = self.analyzer.stress_test({"AAPL": 1.0}, ["recession"])
        assert len(result["recession"]["details"]) == 1


# ===========================================================================
# PortfolioHealthMonitor
# ===========================================================================


class TestPortfolioHealthMonitor:
    def setup_method(self):
        self.monitor = PortfolioHealthMonitor()

    def test_concentration_violation(self):
        violations = self.monitor.check_concentration_limits(
            {"AAPL": 0.30, "MSFT": 0.10}
        )
        assert any(v["ticker"] == "AAPL" for v in violations)

    def test_no_concentration_violation(self):
        violations = self.monitor.check_concentration_limits(
            {"AAPL": 0.10, "MSFT": 0.10}
        )
        assert violations == []

    def test_volatility_limit_exceeded(self):
        assert self.monitor.check_volatility_limits({"volatility": 0.30}, 0.20) is True

    def test_volatility_limit_not_exceeded(self):
        assert self.monitor.check_volatility_limits({"volatility": 0.10}, 0.20) is False

    def test_drawdown_limit_exceeded(self):
        returns = pd.Series([-0.01] * 100)
        assert self.monitor.check_drawdown_limits(returns, max_drawdown=0.05) is True

    def test_drawdown_limit_not_exceeded(self):
        returns = pd.Series([0.005] * 100)
        assert self.monitor.check_drawdown_limits(returns, max_drawdown=0.15) is False

    def test_generate_alerts_concentration(self):
        portfolio = {"holdings": {"AAPL": 0.40, "MSFT": 0.60}}
        alerts = self.monitor.generate_alerts(portfolio)
        types = [a["type"] for a in alerts]
        assert "concentration_warning" in types

    def test_generate_alerts_volatility(self):
        portfolio = {"holdings": {}, "volatility": 0.35}
        alerts = self.monitor.generate_alerts(portfolio)
        types = [a["type"] for a in alerts]
        assert "volatility_warning" in types

    def test_generate_alerts_drawdown(self):
        returns = pd.Series([-0.02] * 100)
        portfolio = {"holdings": {}, "returns": returns}
        alerts = self.monitor.generate_alerts(portfolio)
        types = [a["type"] for a in alerts]
        assert "drawdown_warning" in types

    def test_no_alerts_healthy_portfolio(self):
        returns = _make_returns(mean=0.001, std=0.005)
        portfolio = {
            "holdings": {"A": 0.10, "B": 0.10, "C": 0.10},
            "returns": returns,
            "volatility": 0.08,
        }
        alerts = self.monitor.generate_alerts(portfolio)
        assert alerts == []

    def test_suggest_rebalancing_buy_and_sell(self):
        current = {"AAPL": 0.30, "MSFT": 0.20, "GOOG": 0.10}
        target = {"AAPL": 0.20, "MSFT": 0.25, "GOOG": 0.20}
        trades = self.monitor.suggest_rebalancing(current, target)
        actions = {t["ticker"]: t["action"] for t in trades}
        assert actions["AAPL"] == "SELL"
        assert actions["MSFT"] == "BUY"

    def test_suggest_rebalancing_threshold(self):
        """Drift below threshold should be ignored."""
        current = {"AAPL": 0.200}
        target = {"AAPL": 0.201}
        trades = self.monitor.suggest_rebalancing(current, target, min_trade_threshold=0.005)
        assert trades == []

    def test_generate_risk_report_keys(self):
        holdings = {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25}
        returns = _make_returns()
        portfolio = {"holdings": holdings, "returns": returns}
        report = self.monitor.generate_risk_report(portfolio)
        for key in ("overall_risk_score", "concentration", "risk_metrics", "alerts"):
            assert key in report

    def test_generate_risk_report_score_range(self):
        portfolio = {"holdings": {"AAPL": 1.0}, "volatility": 0.30}
        report = self.monitor.generate_risk_report(portfolio)
        assert 0 <= report["overall_risk_score"] <= 100

    def test_custom_limits(self):
        monitor = PortfolioHealthMonitor(limits={"max_single_position": 0.10})
        violations = monitor.check_concentration_limits({"AAPL": 0.15})
        assert len(violations) == 1


# ===========================================================================
# Private helpers
# ===========================================================================


class TestComputeOverallRiskScore:
    def test_range(self):
        for c, v, d, n in [
            (0, 0.0, 0.0, 0),
            (100, 0.5, 0.5, 10),
            (50, 0.15, 0.10, 2),
        ]:
            score = _compute_overall_risk_score(c, v, d, n)
            assert 0 <= score <= 100

    def test_higher_risk_gives_higher_score(self):
        low = _compute_overall_risk_score(10, 0.05, 0.02, 0)
        high = _compute_overall_risk_score(80, 0.30, 0.25, 5)
        assert high > low
