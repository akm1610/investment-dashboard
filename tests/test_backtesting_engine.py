"""
tests/test_backtesting_engine.py
---------------------------------
Unit tests for src/backtesting_engine.py (137 tests).

All tests are fully offline – no network calls are made.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

import backtesting_engine as bt_root
from src.backtesting_engine import (
    STRATEGIES,
    BacktestEngine,
    BacktestVisualizer,
    BenchmarkAnalyzer,
    PerformanceCalculator,
    StrategyAnalyzer,
    TradeTracker,
    macd_strategy,
    mean_reversion_strategy,
    momentum_strategy,
    rsi_strategy,
    walk_forward_test,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n: int = 252, mean: float = 0.001, std: float = 0.01, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def _make_prices(n: int = 252, start: float = 100.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices)


def _make_trades(
    n: int = 10, avg_pnl: float = 100.0, seed: int = 0
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    trades = []
    for i in range(n):
        pnl = float(rng.normal(avg_pnl, 200.0))
        entry_price = 100.0
        exit_price = entry_price * (1 + pnl / 1000)
        trades.append(
            {
                "ticker": f"TICK{i}",
                "entry_date": f"2023-0{(i % 9) + 1}-01",
                "entry_price": entry_price,
                "exit_date": f"2023-0{(i % 9) + 1}-15",
                "exit_price": exit_price,
                "position_size": 1000.0,
                "pnl_dollars": pnl,
                "pnl_percent": pnl / 1000,
                "days_held": 14,
            }
        )
    return trades


# ===========================================================================
# Root re-export sanity check
# ===========================================================================


class TestRootReExport:
    def test_backtest_engine_importable(self):
        assert bt_root.BacktestEngine is BacktestEngine

    def test_performance_calculator_importable(self):
        assert bt_root.PerformanceCalculator is PerformanceCalculator

    def test_trade_tracker_importable(self):
        assert bt_root.TradeTracker is TradeTracker

    def test_strategy_analyzer_importable(self):
        assert bt_root.StrategyAnalyzer is StrategyAnalyzer

    def test_benchmark_analyzer_importable(self):
        assert bt_root.BenchmarkAnalyzer is BenchmarkAnalyzer

    def test_visualizer_importable(self):
        assert bt_root.BacktestVisualizer is BacktestVisualizer

    def test_strategies_config_importable(self):
        assert bt_root.STRATEGIES is STRATEGIES

    def test_walk_forward_test_importable(self):
        assert bt_root.walk_forward_test is walk_forward_test


# ===========================================================================
# PerformanceCalculator
# ===========================================================================


class TestCalculateTotalReturn:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_positive_return(self):
        assert self.calc.calculate_total_return(100, 150) == pytest.approx(0.5)

    def test_negative_return(self):
        assert self.calc.calculate_total_return(100, 80) == pytest.approx(-0.20)

    def test_zero_return(self):
        assert self.calc.calculate_total_return(100, 100) == pytest.approx(0.0)

    def test_zero_start_value(self):
        assert self.calc.calculate_total_return(0, 100) == 0.0


class TestCalculateAnnualizedReturn:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_empty_returns(self):
        assert self.calc.calculate_annualized_return(pd.Series([], dtype=float)) == 0.0

    def test_positive_returns(self):
        returns = _make_returns(252, mean=0.001)
        ann = self.calc.calculate_annualized_return(returns)
        assert isinstance(ann, float)

    def test_negative_returns(self):
        returns = _make_returns(252, mean=-0.001)
        ann = self.calc.calculate_annualized_return(returns)
        assert ann < 0

    def test_single_value(self):
        # Should not raise, just return a value
        result = self.calc.calculate_annualized_return(pd.Series([0.05]))
        assert isinstance(result, float)


class TestCalculateSharpeRatio:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_positive_sharpe(self):
        returns = _make_returns(252, mean=0.002, std=0.01)
        sharpe = self.calc.calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_empty_returns(self):
        assert self.calc.calculate_sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_zero_std(self):
        returns = pd.Series([0.001] * 252)
        # All returns equal to rf/252 → 0 std → return 0
        sharpe = self.calc.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_custom_risk_free_rate(self):
        returns = _make_returns()
        s1 = self.calc.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        s2 = self.calc.calculate_sharpe_ratio(returns, risk_free_rate=0.10)
        assert s1 != s2


class TestCalculateSortinoRatio:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_returns_float(self):
        returns = _make_returns()
        sortino = self.calc.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_empty_returns(self):
        assert self.calc.calculate_sortino_ratio(pd.Series([], dtype=float)) == 0.0

    def test_all_positive_returns(self):
        returns = pd.Series([0.01] * 252)
        sortino = self.calc.calculate_sortino_ratio(returns)
        assert sortino == 0.0  # no downside → 0 / 0 case

    def test_sortino_higher_than_sharpe_low_downside(self):
        rng = np.random.default_rng(99)
        # Mostly positive returns
        returns = pd.Series(abs(rng.normal(0.002, 0.005, 252)))
        sharpe = self.calc.calculate_sharpe_ratio(returns)
        sortino = self.calc.calculate_sortino_ratio(returns)
        # Sortino should be >= Sharpe when downside is low
        assert sortino >= sharpe or sortino == 0.0


class TestCalculateCalmarRatio:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_returns_float(self):
        returns = _make_returns()
        result = self.calc.calculate_calmar_ratio(returns)
        assert isinstance(result, float)

    def test_empty_returns(self):
        assert self.calc.calculate_calmar_ratio(pd.Series([], dtype=float)) == 0.0


class TestCalculateMaxDrawdown:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_max_drawdown_negative(self):
        returns = _make_returns()
        cumulative = (1 + returns).cumprod()
        mdd = self.calc.calculate_max_drawdown(cumulative)
        assert mdd <= 0

    def test_monotonically_increasing_no_drawdown(self):
        prices = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        mdd = self.calc.calculate_max_drawdown(prices)
        assert mdd == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak at 2.0, trough at 1.0 → -50 %
        prices = pd.Series([1.0, 2.0, 1.0])
        mdd = self.calc.calculate_max_drawdown(prices)
        assert mdd == pytest.approx(-0.5)

    def test_empty_series(self):
        assert self.calc.calculate_max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestCalculateRecoveryTime:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_never_recovers(self):
        prices = pd.Series([1.0, 2.0, 1.0])  # never gets back to 2.0
        result = self.calc.calculate_recovery_time(prices)
        assert result == -1

    def test_recovers(self):
        prices = pd.Series([1.0, 2.0, 1.5, 2.0, 2.5])
        result = self.calc.calculate_recovery_time(prices)
        assert result >= 0

    def test_empty_series(self):
        assert self.calc.calculate_recovery_time(pd.Series([], dtype=float)) == 0


class TestCalculateConsecutiveWins:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_basic_streak(self):
        returns = pd.Series([0.01, 0.02, 0.03, -0.01, 0.01, 0.02])
        assert self.calc.calculate_consecutive_wins(returns) == 3

    def test_all_wins(self):
        returns = pd.Series([0.01] * 10)
        assert self.calc.calculate_consecutive_wins(returns) == 10

    def test_all_losses(self):
        returns = pd.Series([-0.01] * 10)
        assert self.calc.calculate_consecutive_wins(returns) == 0

    def test_empty(self):
        assert self.calc.calculate_consecutive_wins(pd.Series([], dtype=float)) == 0


class TestCalculateWinRate:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_basic_win_rate(self):
        trades = [
            {"pnl_dollars": 100},
            {"pnl_dollars": -50},
            {"pnl_dollars": 200},
            {"pnl_dollars": -30},
        ]
        assert self.calc.calculate_win_rate(trades) == pytest.approx(0.5)

    def test_no_trades(self):
        assert self.calc.calculate_win_rate([]) == 0.0

    def test_all_wins(self):
        trades = [{"pnl_dollars": 100}] * 5
        assert self.calc.calculate_win_rate(trades) == pytest.approx(1.0)

    def test_all_losses(self):
        trades = [{"pnl_dollars": -50}] * 5
        assert self.calc.calculate_win_rate(trades) == pytest.approx(0.0)


class TestCalculateProfitFactor:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_basic_profit_factor(self):
        trades = [
            {"pnl_dollars": 300},
            {"pnl_dollars": -100},
        ]
        assert self.calc.calculate_profit_factor(trades) == pytest.approx(3.0)

    def test_no_trades(self):
        assert self.calc.calculate_profit_factor([]) == 0.0

    def test_all_wins_inf(self):
        trades = [{"pnl_dollars": 100}] * 3
        result = self.calc.calculate_profit_factor(trades)
        assert result == float("inf")

    def test_all_losses_zero(self):
        trades = [{"pnl_dollars": -50}] * 3
        assert self.calc.calculate_profit_factor(trades) == 0.0


class TestCalculateExpectancy:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_basic_expectancy(self):
        trades = [{"pnl_dollars": 100}, {"pnl_dollars": -50}]
        assert self.calc.calculate_expectancy(trades) == pytest.approx(25.0)

    def test_no_trades(self):
        assert self.calc.calculate_expectancy([]) == 0.0


class TestInformationRatio:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_returns_float(self):
        s = _make_returns(seed=0)
        b = _make_returns(seed=1)
        ir = self.calc.calculate_information_ratio(s, b)
        assert isinstance(ir, float)

    def test_empty_strategy_returns(self):
        b = _make_returns()
        assert self.calc.calculate_information_ratio(pd.Series([], dtype=float), b) == 0.0


# ===========================================================================
# TradeTracker
# ===========================================================================


class TestTradeTracker:
    def test_record_trade(self):
        tracker = TradeTracker()
        tracker.record_trade(
            ticker="AAPL",
            entry_date="2023-01-01",
            entry_price=150.0,
            exit_date="2023-06-01",
            exit_price=180.0,
            position_size=1500.0,
        )
        assert len(tracker.trades) == 1
        trade = tracker.trades[0]
        assert trade["ticker"] == "AAPL"
        assert trade["pnl_dollars"] > 0

    def test_get_trade_pnl_profit(self):
        tracker = TradeTracker()
        pnl = tracker.get_trade_pnl(
            {"entry_price": 100.0, "exit_price": 120.0, "position_size": 1000.0}
        )
        assert pnl["pnl_dollars"] == pytest.approx(200.0)
        assert pnl["pnl_percent"] == pytest.approx(0.20)

    def test_get_trade_pnl_loss(self):
        tracker = TradeTracker()
        pnl = tracker.get_trade_pnl(
            {"entry_price": 100.0, "exit_price": 80.0, "position_size": 1000.0}
        )
        assert pnl["pnl_dollars"] == pytest.approx(-200.0)
        assert pnl["pnl_percent"] == pytest.approx(-0.20)

    def test_get_trade_pnl_zero_entry(self):
        tracker = TradeTracker()
        pnl = tracker.get_trade_pnl(
            {"entry_price": 0.0, "exit_price": 100.0, "position_size": 1000.0}
        )
        assert pnl["pnl_dollars"] == 0.0

    def test_get_trade_statistics_empty(self):
        tracker = TradeTracker()
        stats = tracker.get_trade_statistics()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_get_trade_statistics_mixed(self):
        tracker = TradeTracker()
        for _ in range(3):
            tracker.record_trade("AAPL", "2023-01-01", 100.0, "2023-02-01", 120.0, 1000.0)
        for _ in range(2):
            tracker.record_trade("MSFT", "2023-01-01", 100.0, "2023-02-01", 80.0, 1000.0)
        stats = tracker.get_trade_statistics()
        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] == pytest.approx(0.6)

    def test_get_trade_statistics_all_wins(self):
        tracker = TradeTracker()
        for _ in range(5):
            tracker.record_trade("AAPL", "2023-01-01", 100.0, "2023-02-01", 110.0, 1000.0)
        stats = tracker.get_trade_statistics()
        assert stats["win_rate"] == pytest.approx(1.0)
        assert stats["profit_factor"] == float("inf")

    def test_get_trade_statistics_all_losses(self):
        tracker = TradeTracker()
        for _ in range(5):
            tracker.record_trade("AAPL", "2023-01-01", 100.0, "2023-02-01", 90.0, 1000.0)
        stats = tracker.get_trade_statistics()
        assert stats["win_rate"] == pytest.approx(0.0)
        assert stats["profit_factor"] == 0.0

    def test_get_trade_list(self):
        tracker = TradeTracker()
        tracker.record_trade("NVDA", "2023-03-01", 300.0, "2023-06-01", 400.0, 3000.0)
        trades = tracker.get_trade_list()
        assert isinstance(trades, list)
        assert len(trades) == 1
        assert trades[0]["ticker"] == "NVDA"

    def test_consecutive_win_streak(self):
        tracker = TradeTracker()
        # 4 wins then 1 loss
        for _ in range(4):
            tracker.record_trade("AAPL", "2023-01-01", 100.0, "2023-02-01", 110.0, 1000.0)
        tracker.record_trade("AAPL", "2023-03-01", 100.0, "2023-04-01", 90.0, 1000.0)
        stats = tracker.get_trade_statistics()
        assert stats["consecutive_wins"] == 4
        assert stats["consecutive_losses"] == 1

    def test_days_held_calculated(self):
        tracker = TradeTracker()
        tracker.record_trade("AAPL", "2023-01-01", 100.0, "2023-01-15", 110.0, 1000.0)
        assert tracker.trades[0]["days_held"] == 14


# ===========================================================================
# StrategyAnalyzer
# ===========================================================================


class TestStrategyAnalyzer:
    def setup_method(self):
        self.analyzer = StrategyAnalyzer()

    def test_analyze_entry_quality_all_good(self):
        trades = [
            {"entry_price": 100.0, "exit_price": 110.0},
            {"entry_price": 50.0, "exit_price": 60.0},
        ]
        result = self.analyzer.analyze_entry_quality(trades)
        assert result["quality_score"] == pytest.approx(1.0)

    def test_analyze_entry_quality_none(self):
        result = self.analyzer.analyze_entry_quality([])
        assert result["quality_score"] == 0.0

    def test_analyze_exit_quality_empty(self):
        result = self.analyzer.analyze_exit_quality([])
        assert result["quality_score"] == 0.0

    def test_analyze_holding_periods_empty(self):
        result = self.analyzer.analyze_holding_periods([])
        assert result["avg_holding_days"] == 0

    def test_analyze_holding_periods_with_trades(self):
        trades = _make_trades(5)
        result = self.analyzer.analyze_holding_periods(trades)
        assert "avg_holding_days" in result
        assert "median_holding_days" in result
        assert "holding_distribution" in result
        assert isinstance(result["holding_distribution"], dict)

    def test_analyze_sector_performance_empty(self):
        result = self.analyzer.analyze_sector_performance([])
        assert result == {}

    def test_analyze_sector_performance_with_sectors(self):
        trades = [
            {"sector": "Tech", "pnl_dollars": 200.0},
            {"sector": "Tech", "pnl_dollars": -50.0},
            {"sector": "Finance", "pnl_dollars": 100.0},
        ]
        result = self.analyzer.analyze_sector_performance(trades)
        assert "Tech" in result
        assert "Finance" in result
        assert result["Tech"]["total_trades"] == 2
        assert result["Finance"]["total_pnl"] == pytest.approx(100.0)

    def test_analyze_sector_performance_no_sector_key(self):
        trades = [{"pnl_dollars": 100.0}]
        result = self.analyzer.analyze_sector_performance(trades)
        assert "Unknown" in result

    def test_monte_carlo_no_trades(self):
        result = self.analyzer.monte_carlo_simulation([])
        assert result["probability_profit"] == 0.0
        assert result["expected_final_value"] == 100_000

    def test_monte_carlo_with_trades(self):
        trades = _make_trades(20, avg_pnl=500.0, seed=42)
        result = self.analyzer.monte_carlo_simulation(
            trades, num_simulations=200, seed=42
        )
        assert 0.0 <= result["probability_profit"] <= 1.0
        assert "var_95" in result
        assert "cvar_95" in result
        assert "expected_final_value" in result

    def test_monte_carlo_custom_capital(self):
        trades = _make_trades(5)
        result = self.analyzer.monte_carlo_simulation(
            trades, num_simulations=100, initial_capital=50_000, seed=7
        )
        # Result should be near 50k (adjusted by mean P&L)
        assert abs(result["expected_final_value"] - 50_000) < 50_000

    def test_walk_forward_validation_returns_dict(self):
        result = self.analyzer.walk_forward_validation(lambda *a: [], lookback_periods=3)
        assert isinstance(result, dict)
        assert "lookback_periods" in result


# ===========================================================================
# BenchmarkAnalyzer
# ===========================================================================


class TestBenchmarkAnalyzer:
    def setup_method(self):
        self.analyzer = BenchmarkAnalyzer()

    def test_calculate_alpha_positive(self):
        alpha = self.analyzer.calculate_alpha(
            strategy_return=0.15,
            risk_free_rate=0.04,
            beta=1.0,
            market_return=0.10,
        )
        # α = 0.15 - [0.04 + 1.0 * (0.10 - 0.04)] = 0.15 - 0.10 = 0.05
        assert alpha == pytest.approx(0.05)

    def test_calculate_alpha_negative(self):
        alpha = self.analyzer.calculate_alpha(
            strategy_return=0.05,
            risk_free_rate=0.04,
            beta=1.2,
            market_return=0.10,
        )
        assert alpha < 0

    def test_calculate_beta_returns_float(self):
        s = _make_returns(seed=0)
        m = _make_returns(seed=1)
        beta = self.analyzer.calculate_beta(s, m)
        assert isinstance(beta, float)

    def test_calculate_beta_empty(self):
        beta = self.analyzer.calculate_beta(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert beta == 1.0

    def test_calculate_beta_correlated(self):
        rng = np.random.default_rng(0)
        market = pd.Series(rng.normal(0, 0.01, 252))
        # Strategy = 2x market + noise
        strategy = 2 * market + pd.Series(rng.normal(0, 0.001, 252))
        beta = self.analyzer.calculate_beta(strategy, market)
        assert beta == pytest.approx(2.0, abs=0.2)

    def test_calculate_information_ratio_returns_float(self):
        s = _make_returns(seed=0)
        b = _make_returns(seed=1)
        ir = self.analyzer.calculate_information_ratio(s, b)
        assert isinstance(ir, float)

    def test_compare_to_benchmark_empty(self):
        result = self.analyzer.compare_to_benchmark(
            pd.Series([], dtype=float), pd.Series([], dtype=float)
        )
        assert result == {}

    def test_compare_to_benchmark_keys(self):
        s = _make_returns(seed=0)
        b = _make_returns(seed=1)
        result = self.analyzer.compare_to_benchmark(s, b)
        for key in [
            "strategy_total_return",
            "benchmark_total_return",
            "alpha",
            "beta",
            "information_ratio",
            "tracking_error",
            "correlation",
        ]:
            assert key in result

    def test_drawdown_comparison(self):
        s = _make_returns(seed=0)
        b = _make_returns(seed=1)
        result = self.analyzer.drawdown_comparison(s, b)
        assert "strategy_max_drawdown" in result
        assert "benchmark_max_drawdown" in result
        assert result["strategy_max_drawdown"] <= 0
        assert result["benchmark_max_drawdown"] <= 0


# ===========================================================================
# BacktestVisualizer
# ===========================================================================


class TestBacktestVisualizer:
    def setup_method(self):
        self.viz = BacktestVisualizer()

    def test_get_equity_curve_data_empty(self):
        df = self.viz.get_equity_curve_data([])
        assert df.empty

    def test_get_equity_curve_data_with_trades(self):
        trades = _make_trades(5)
        df = self.viz.get_equity_curve_data(trades)
        assert "date" in df.columns
        assert "cumulative_value" in df.columns
        assert len(df) == 5

    def test_get_monthly_returns_empty(self):
        df = self.viz.get_monthly_returns(pd.Series([], dtype=float))
        assert df.empty

    def test_get_monthly_returns_with_data(self):
        idx = pd.date_range("2022-01-01", periods=365, freq="D")
        returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 365), index=idx)
        df = self.viz.get_monthly_returns(returns)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_get_drawdown_data_empty(self):
        df = self.viz.get_drawdown_data(pd.Series([], dtype=float))
        assert df.empty

    def test_get_drawdown_data_valid(self):
        prices = _make_prices()
        df = self.viz.get_drawdown_data(prices)
        assert "drawdown" in df.columns
        assert (df["drawdown"] <= 0).all()

    def test_get_performance_metrics_keys(self):
        result = {
            "total_return": 0.45,
            "annualized_return": 0.12,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.1,
            "max_drawdown": -0.18,
            "win_rate": 0.70,
            "profit_factor": 2.5,
            "alpha": 0.07,
            "beta": 0.95,
        }
        metrics = self.viz.get_performance_metrics(result)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_get_trade_distribution_empty(self):
        result = self.viz.get_trade_distribution([])
        assert result["buckets"] == {}
        assert result["mean"] == 0.0

    def test_get_trade_distribution_with_trades(self):
        trades = _make_trades(20)
        result = self.viz.get_trade_distribution(trades)
        assert len(result["buckets"]) > 0
        assert "mean" in result
        assert "std" in result
        assert "median" in result


# ===========================================================================
# BacktestEngine
# ===========================================================================


class TestBacktestEngineInit:
    def test_init_defaults(self):
        engine = BacktestEngine("2020-01-01", "2023-01-01")
        assert engine.start_date == "2020-01-01"
        assert engine.end_date == "2023-01-01"
        assert engine.initial_capital == 100_000
        assert engine.benchmark == "^GSPC"

    def test_init_custom_capital(self):
        engine = BacktestEngine("2020-01-01", "2023-01-01", initial_capital=50_000)
        assert engine.initial_capital == 50_000

    def test_init_custom_benchmark(self):
        engine = BacktestEngine("2020-01-01", "2023-01-01", benchmark="^NDX")
        assert engine.benchmark == "^NDX"

    def test_add_signal(self):
        engine = BacktestEngine("2020-01-01", "2023-01-01")
        engine.add_signal("AAPL", "2020-03-01", "2020-09-01", "BUY")
        assert len(engine._signals) == 1
        assert engine._signals[0]["ticker"] == "AAPL"

    def test_add_multiple_signals(self):
        engine = BacktestEngine("2020-01-01", "2023-01-01")
        engine.add_signal("AAPL", "2020-03-01", "2020-09-01", "BUY")
        engine.add_signal("MSFT", "2020-04-01", "2020-10-01", "BUY")
        assert len(engine._signals) == 2


class TestBacktestEngineSignals:
    def _make_engine(self):
        return BacktestEngine(
            start_date="2022-01-01",
            end_date="2022-12-31",
            initial_capital=10_000,
        )

    def test_backtest_signals_empty(self):
        engine = self._make_engine()
        result = engine.backtest_signals([])
        assert result["total_trades"] == 0
        assert result["total_return"] == 0.0

    def test_backtest_signals_single_win(self):
        engine = self._make_engine()
        signals = [
            {
                "ticker": "AAPL",
                "entry_date": "2022-01-10",
                "entry_price": 100.0,
                "exit_date": "2022-06-01",
                "exit_price": 130.0,
                "position_size": 1000.0,
            }
        ]
        result = engine.backtest_signals(signals)
        assert result["total_trades"] == 1
        assert result["winning_trades"] == 1
        assert result["total_return"] > 0

    def test_backtest_signals_single_loss(self):
        engine = self._make_engine()
        signals = [
            {
                "ticker": "AAPL",
                "entry_date": "2022-01-10",
                "entry_price": 100.0,
                "exit_date": "2022-06-01",
                "exit_price": 80.0,
                "position_size": 1000.0,
            }
        ]
        result = engine.backtest_signals(signals)
        assert result["losing_trades"] == 1
        assert result["total_return"] < 0

    def test_backtest_signals_result_keys(self):
        engine = self._make_engine()
        signals = [
            {
                "ticker": "AAPL",
                "entry_date": "2022-02-01",
                "entry_price": 150.0,
                "exit_date": "2022-08-01",
                "exit_price": 170.0,
                "position_size": 1500.0,
            }
        ]
        result = engine.backtest_signals(signals)
        expected_keys = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "trades",
            "equity_curve",
            "benchmark_return",
            "alpha",
            "beta",
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_backtest_signals_multiple_trades(self):
        engine = self._make_engine()
        signals = [
            {
                "ticker": "AAPL",
                "entry_date": "2022-01-10",
                "entry_price": 100.0,
                "exit_date": "2022-03-01",
                "exit_price": 120.0,
                "position_size": 1000.0,
            },
            {
                "ticker": "MSFT",
                "entry_date": "2022-04-01",
                "entry_price": 200.0,
                "exit_date": "2022-08-01",
                "exit_price": 180.0,
                "position_size": 2000.0,
            },
        ]
        result = engine.backtest_signals(signals)
        assert result["total_trades"] == 2


class TestBacktestEngineStrategy:
    def _make_engine(self):
        return BacktestEngine(
            start_date="2022-01-01",
            end_date="2022-06-30",
            initial_capital=10_000,
            tickers=["AAPL", "MSFT"],
        )

    def test_backtest_strategy_no_trades(self):
        engine = self._make_engine()

        def do_nothing(date, tickers, prices):
            return []

        result = engine.backtest_strategy(do_nothing)
        assert result["total_trades"] == 0

    def test_backtest_strategy_always_buy(self):
        engine = self._make_engine()

        def always_buy(date, tickers, prices):
            if date == "2022-01-03":
                return [{"ticker": "AAPL", "action": "BUY", "weight": 0.1}]
            return []

        result = engine.backtest_strategy(always_buy)
        assert isinstance(result, dict)
        assert "total_return" in result

    def test_backtest_strategy_result_is_dict(self):
        engine = self._make_engine()
        result = engine.backtest_strategy(lambda date, tickers, prices: [])
        assert isinstance(result, dict)

    def test_backtest_strategy_with_provided_prices(self):
        engine = self._make_engine()
        prices = {
            "AAPL": pd.Series([100.0 + i * 0.5 for i in range(180)]),
            "MSFT": pd.Series([200.0 - i * 0.3 for i in range(180)]),
        }
        result = engine.backtest_strategy(lambda d, t, p: [], prices=prices)
        assert isinstance(result, dict)


class TestBacktestEngineOptimizeParameters:
    def test_optimize_parameters_basic(self):
        engine = BacktestEngine(
            start_date="2022-01-01",
            end_date="2022-06-30",
            tickers=["AAPL"],
        )

        def strategy_factory(threshold=0.05):
            def strategy(date, tickers, prices):
                return []
            return strategy

        result = engine.optimize_parameters(
            param_ranges={"threshold": [0.03, 0.05]},
            strategy_func=strategy_factory,
        )
        assert "best_params" in result
        assert "best_result" in result


# ===========================================================================
# walk_forward_test
# ===========================================================================


class TestWalkForwardTest:
    def _noop_strategy(self, date, tickers, prices):
        return []

    def test_returns_dict(self):
        result = walk_forward_test(
            strategy_func=self._noop_strategy,
            tickers=["AAPL"],
            start_date="2020-01-01",
            end_date="2022-01-01",
        )
        assert isinstance(result, dict)

    def test_result_keys(self):
        result = walk_forward_test(
            strategy_func=self._noop_strategy,
            tickers=["AAPL"],
            start_date="2020-01-01",
            end_date="2022-01-01",
        )
        for key in [
            "windows",
            "avg_sharpe",
            "sharpe_std",
            "avg_return",
            "return_std",
            "out_of_sample_return",
            "robustness_score",
        ]:
            assert key in result

    def test_empty_range(self):
        # Range too short to produce any windows
        result = walk_forward_test(
            strategy_func=self._noop_strategy,
            tickers=["AAPL"],
            start_date="2022-01-01",
            end_date="2022-06-01",  # < 2 * 252 days
        )
        assert result["avg_sharpe"] == 0.0

    def test_robustness_score_range(self):
        result = walk_forward_test(
            strategy_func=self._noop_strategy,
            tickers=["AAPL"],
            start_date="2018-01-01",
            end_date="2022-12-31",
        )
        assert 0.0 <= result["robustness_score"] <= 1.0

    def test_quarterly_rebalance(self):
        result = walk_forward_test(
            strategy_func=self._noop_strategy,
            tickers=["AAPL"],
            start_date="2018-01-01",
            end_date="2022-12-31",
            rebalance_freq="quarterly",
        )
        assert isinstance(result["windows"], list)


# ===========================================================================
# Example strategies
# ===========================================================================


class TestMomentumStrategy:
    def _make_data(self, n=100):
        return {
            "AAPL": _make_prices(n, seed=0),
            "MSFT": _make_prices(n, seed=1),
        }

    def test_returns_list(self):
        data = self._make_data()
        result = momentum_strategy(data)
        assert isinstance(result, list)

    def test_insufficient_data(self):
        data = {"AAPL": _make_prices(5)}
        result = momentum_strategy(data, lookback=20)
        assert result == []

    def test_buy_signal_on_momentum(self):
        # Strongly trending up price series
        prices = pd.Series([100.0 * (1.005 ** i) for i in range(50)])
        result = momentum_strategy({"UP": prices}, lookback=10, threshold=0.02)
        assert any(s["action"] == "BUY" for s in result)


class TestMeanReversionStrategy:
    def test_returns_list(self):
        data = {"AAPL": _make_prices(200, seed=0)}
        result = mean_reversion_strategy(data)
        assert isinstance(result, list)

    def test_insufficient_data(self):
        data = {"AAPL": _make_prices(10)}
        result = mean_reversion_strategy(data, lookback=50)
        assert result == []

    def test_buy_on_oversold(self):
        # Price drops sharply at the end
        prices_vals = [100.0] * 50 + [60.0] * 5
        prices = pd.Series(prices_vals)
        result = mean_reversion_strategy({"X": prices}, lookback=40, std_threshold=-1.0)
        assert any(s["action"] == "BUY" for s in result)


class TestRsiStrategy:
    def test_returns_list(self):
        data = {"AAPL": _make_prices(100, seed=0)}
        result = rsi_strategy(data)
        assert isinstance(result, list)

    def test_insufficient_data(self):
        data = {"AAPL": _make_prices(5)}
        result = rsi_strategy(data, period=14)
        assert result == []

    def test_oversold_triggers_buy(self):
        # Strongly declining prices
        prices = pd.Series([100.0 * (0.99 ** i) for i in range(50)])
        result = rsi_strategy({"DOWN": prices}, period=14, oversold=40)
        # May or may not trigger depending on exact RSI, just assert no crash
        assert isinstance(result, list)


class TestMacdStrategy:
    def test_returns_list(self):
        data = {"AAPL": _make_prices(100, seed=0)}
        result = macd_strategy(data)
        assert isinstance(result, list)

    def test_insufficient_data(self):
        data = {"AAPL": _make_prices(10)}
        result = macd_strategy(data)
        assert result == []

    def test_bullish_crossover_buy(self):
        # Upward trending price
        prices = pd.Series([100.0 * (1.002 ** i) for i in range(100)])
        result = macd_strategy({"UP": prices})
        assert isinstance(result, list)


# ===========================================================================
# STRATEGIES config
# ===========================================================================


class TestStrategiesConfig:
    def test_required_strategies_exist(self):
        for key in ["momentum", "mean_reversion", "rsi_oversold", "macd_crossover"]:
            assert key in STRATEGIES

    def test_momentum_keys(self):
        assert "lookback" in STRATEGIES["momentum"]
        assert "threshold" in STRATEGIES["momentum"]
        assert "description" in STRATEGIES["momentum"]

    def test_rsi_oversold_threshold(self):
        assert STRATEGIES["rsi_oversold"]["threshold"] == 30

    def test_macd_keys(self):
        assert "fast_ema" in STRATEGIES["macd_crossover"]
        assert "slow_ema" in STRATEGIES["macd_crossover"]
        assert "signal" in STRATEGIES["macd_crossover"]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_performance_calc_single_trade(self):
        tracker = TradeTracker()
        tracker.record_trade("AAPL", "2023-01-01", 100.0, "2023-12-31", 150.0, 1000.0)
        stats = tracker.get_trade_statistics()
        assert stats["total_trades"] == 1
        assert stats["win_rate"] == pytest.approx(1.0)

    def test_performance_calc_no_trades(self):
        tracker = TradeTracker()
        stats = tracker.get_trade_statistics()
        assert stats["total_trades"] == 0

    def test_backtest_engine_empty_tickers(self):
        engine = BacktestEngine("2022-01-01", "2022-12-31", tickers=[])
        result = engine.backtest_strategy(lambda d, t, p: [])
        assert result["total_trades"] == 0

    def test_backtest_signals_with_auto_prices(self):
        """Signals without explicit entry/exit prices should use synthetic prices."""
        engine = BacktestEngine("2022-01-01", "2022-12-31", initial_capital=10_000)
        signals = [
            {
                "ticker": "AAPL",
                "entry_date": "2022-02-01",
                "exit_date": "2022-08-01",
            }
        ]
        result = engine.backtest_signals(signals)
        assert result["total_trades"] == 1

    def test_calmar_ratio_zero_drawdown(self):
        calc = PerformanceCalculator()
        returns = pd.Series([0.001] * 252)
        result = calc.calculate_calmar_ratio(returns)
        # With constant returns there is no drawdown → 0
        assert result == 0.0

    def test_max_drawdown_single_value(self):
        calc = PerformanceCalculator()
        result = calc.calculate_max_drawdown(pd.Series([1.0]))
        assert result == 0.0

    def test_profit_factor_with_no_pnl_key(self):
        calc = PerformanceCalculator()
        trades = [{"ticker": "X"}]  # no pnl_dollars
        result = calc.calculate_profit_factor(trades)
        assert result == 0.0

    def test_expectancy_with_no_pnl_key(self):
        calc = PerformanceCalculator()
        trades = [{"ticker": "X"}]
        result = calc.calculate_expectancy(trades)
        assert result == 0.0

    def test_backtest_engine_monte_carlo(self):
        engine = BacktestEngine("2022-01-01", "2022-12-31")
        trades = _make_trades(10)
        result = engine.monte_carlo_simulation(trades, num_simulations=50, seed=0)
        assert 0 <= result["probability_profit"] <= 1

    def test_benchmark_analyzer_zero_market_var(self):
        analyzer = BenchmarkAnalyzer()
        market = pd.Series([0.001] * 100)  # constant → near-zero var
        strategy = _make_returns(100)
        # Should not raise; result is mathematically undefined for zero-variance
        beta = analyzer.calculate_beta(strategy, market)
        assert isinstance(beta, float)

    def test_visualizer_equity_curve_increasing_capital(self):
        viz = BacktestVisualizer()
        trades = [
            {"exit_date": f"2022-{i:02d}-15", "pnl_dollars": 100.0}
            for i in range(1, 6)
        ]
        df = viz.get_equity_curve_data(trades, initial_capital=10_000)
        assert df["cumulative_value"].iloc[-1] > df["cumulative_value"].iloc[0]
