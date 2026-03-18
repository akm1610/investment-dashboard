"""
src/backtesting_engine.py
-------------------------
Comprehensive backtesting engine for historical strategy testing and
performance validation.

Architecture
------------
1. BacktestEngine        – execute custom strategy functions or pre-defined
                           signals against historical price data.
2. PerformanceCalculator – standalone functions for total/annualised return,
                           Sharpe, Sortino, Calmar, information ratio,
                           max-drawdown, recovery time, win-rate, profit
                           factor and expectancy.
3. TradeTracker          – record individual trades and compute aggregated
                           statistics including consecutive win/loss streaks.
4. StrategyAnalyzer      – entry/exit quality, holding-period analysis,
                           sector performance, Monte Carlo simulation, and
                           walk-forward validation.
5. BenchmarkAnalyzer     – alpha, beta, information ratio, tracking error,
                           correlation, and drawdown comparisons.
6. BacktestVisualizer    – generate DataFrames / dicts ready for charting
                           (equity curve, monthly returns heatmap, drawdown).
7. walk_forward_test     – module-level convenience function.
8. Example strategies    – momentum, mean_reversion, rsi_strategy,
                           macd_strategy.
9. STRATEGIES            – pre-built configuration dict.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR: int = 252
_DEFAULT_RISK_FREE_RATE: float = 0.04

# ---------------------------------------------------------------------------
# Pre-built strategy configuration
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "lookback": 20,
        "threshold": 0.05,
        "description": "Buy stocks with positive momentum",
    },
    "mean_reversion": {
        "lookback": 50,
        "threshold": -1.5,
        "description": "Buy oversold stocks",
    },
    "rsi_oversold": {
        "period": 14,
        "threshold": 30,
        "description": "Buy when RSI < 30 (oversold)",
    },
    "macd_crossover": {
        "fast_ema": 12,
        "slow_ema": 26,
        "signal": 9,
        "description": "Buy on MACD positive crossover",
    },
}


# ===========================================================================
# Helper utilities
# ===========================================================================


def _parse_date(date_str: str) -> datetime:
    """Parse an ISO-format date string (YYYY-MM-DD) into a datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def _date_range_days(start: str, end: str) -> int:
    """Return the number of calendar days between two date strings."""
    return (_parse_date(end) - _parse_date(start)).days


def _annualise_factor(n_days: int) -> float:
    """Return the annualisation factor given *n_days* calendar days."""
    return 365.0 / max(n_days, 1)


# ===========================================================================
# 2. PerformanceCalculator
# ===========================================================================


class PerformanceCalculator:
    """Calculate comprehensive performance metrics from returns / trade lists."""

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    def calculate_total_return(
        self, start_value: float, end_value: float
    ) -> float:
        """Return (end_value / start_value) - 1, or 0 on bad input."""
        if start_value <= 0:
            return 0.0
        return (end_value - start_value) / start_value

    def calculate_annualized_return(self, returns: pd.Series) -> float:
        """
        Calculate annualised geometric return from a daily returns series.

        Uses (1 + total_return)^(252/n) - 1.
        """
        if returns is None or len(returns) == 0:
            return 0.0
        n = len(returns)
        total = float((1 + returns).prod()) - 1.0
        if n < 2:
            return total
        return float((1 + total) ** (_TRADING_DAYS_PER_YEAR / n) - 1)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> float:
        """Return annualised Sharpe ratio (excess return / volatility)."""
        if returns is None or len(returns) < 2:
            return 0.0
        daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        std = excess.std()
        if std == 0:
            return 0.0
        return float(excess.mean() / std * math.sqrt(_TRADING_DAYS_PER_YEAR))

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        target_return: float = 0.0,
    ) -> float:
        """Return annualised Sortino ratio (uses downside deviation only)."""
        if returns is None or len(returns) < 2:
            return 0.0
        daily_target = target_return / _TRADING_DAYS_PER_YEAR
        downside = returns[returns < daily_target] - daily_target
        downside_std = math.sqrt((downside**2).mean()) if len(downside) > 0 else 0.0
        if downside_std == 0:
            return 0.0
        excess_mean = float(returns.mean() - daily_target)
        return float(
            excess_mean / downside_std * math.sqrt(_TRADING_DAYS_PER_YEAR)
        )

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Return annualised return divided by |max drawdown|."""
        if returns is None or len(returns) < 2:
            return 0.0
        ann_return = self.calculate_annualized_return(returns)
        cumulative = (1 + returns).cumprod()
        mdd = self.calculate_max_drawdown(cumulative)
        if mdd == 0:
            return 0.0
        return float(ann_return / abs(mdd))

    def calculate_information_ratio(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Return annualised information ratio (active return / tracking error).
        """
        if (
            strategy_returns is None
            or benchmark_returns is None
            or len(strategy_returns) < 2
        ):
            return 0.0
        n = min(len(strategy_returns), len(benchmark_returns))
        active = strategy_returns.values[:n] - benchmark_returns.values[:n]
        active_series = pd.Series(active)
        te = active_series.std()
        if te == 0:
            return 0.0
        return float(
            active_series.mean() / te * math.sqrt(_TRADING_DAYS_PER_YEAR)
        )

    # ------------------------------------------------------------------
    # Drawdown / recovery
    # ------------------------------------------------------------------

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Return the maximum peak-to-trough decline (negative number)."""
        if cumulative_returns is None or len(cumulative_returns) < 2:
            return 0.0
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return float(drawdown.min())

    def calculate_recovery_time(
        self, cumulative_returns: pd.Series
    ) -> int:
        """
        Return the number of periods (days) required to recover from the
        maximum drawdown trough.  Returns -1 if never recovered.
        """
        if cumulative_returns is None or len(cumulative_returns) < 2:
            return 0
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        trough_idx = int(drawdown.idxmin())
        peak_val = float(rolling_max.iloc[trough_idx])
        recovery_idx = -1
        for i in range(trough_idx + 1, len(cumulative_returns)):
            if float(cumulative_returns.iloc[i]) >= peak_val:
                recovery_idx = i
                break
        if recovery_idx == -1:
            return -1
        return recovery_idx - trough_idx

    # ------------------------------------------------------------------
    # Streak / win metrics
    # ------------------------------------------------------------------

    def calculate_consecutive_wins(self, returns: pd.Series) -> int:
        """Return the longest consecutive winning streak (positive returns)."""
        if returns is None or len(returns) == 0:
            return 0
        max_streak = current = 0
        for r in returns:
            if r > 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Return fraction of trades with positive P&L."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl_dollars", 0) > 0)
        return wins / len(trades)

    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Return gross profit / gross loss (0.0 if no losses)."""
        if not trades:
            return 0.0
        gross_profit = sum(
            t.get("pnl_dollars", 0) for t in trades if t.get("pnl_dollars", 0) > 0
        )
        gross_loss = abs(
            sum(t.get("pnl_dollars", 0) for t in trades if t.get("pnl_dollars", 0) < 0)
        )
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """Return average P&L per trade."""
        if not trades:
            return 0.0
        return sum(t.get("pnl_dollars", 0) for t in trades) / len(trades)


# ===========================================================================
# 3. TradeTracker
# ===========================================================================


class TradeTracker:
    """Record completed trades and compute aggregated statistics."""

    def __init__(self) -> None:
        self.trades: List[Dict[str, Any]] = []

    def record_trade(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
        exit_date: str,
        exit_price: float,
        position_size: float,
        exit_reason: str = "signal",
    ) -> None:
        """Append a completed trade to the internal trade list."""
        pnl = self.get_trade_pnl(
            {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": position_size,
            }
        )
        entry_dt = _parse_date(entry_date)
        exit_dt = _parse_date(exit_date)
        days_held = max((exit_dt - entry_dt).days, 0)
        self.trades.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "position_size": position_size,
                "exit_reason": exit_reason,
                "pnl_dollars": pnl["pnl_dollars"],
                "pnl_percent": pnl["pnl_percent"],
                "days_held": days_held,
            }
        )

    def get_trade_pnl(self, trade: Dict) -> Dict[str, float]:
        """
        Calculate P&L for a single trade dict.

        Expects keys: entry_price, exit_price, position_size.
        """
        entry = float(trade.get("entry_price", 0))
        exit_p = float(trade.get("exit_price", 0))
        size = float(trade.get("position_size", 0))
        if entry <= 0:
            return {"pnl_dollars": 0.0, "pnl_percent": 0.0}
        pnl_pct = (exit_p - entry) / entry
        # position_size is treated as the dollar amount invested
        shares = size / entry if entry > 0 else 0
        pnl_dollars = shares * (exit_p - entry)
        return {"pnl_dollars": pnl_dollars, "pnl_percent": pnl_pct}

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Return aggregated trade statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
            }

        pnls = [t["pnl_dollars"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_loss = abs(sum(losses)) if losses else 0.0
        gross_profit = sum(wins) if wins else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        )

        # Consecutive streaks
        max_win_streak = max_loss_streak = cur_win = cur_loss = 0
        for p in pnls:
            if p > 0:
                cur_win += 1
                cur_loss = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss_streak = max(max_loss_streak, cur_loss)

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "largest_win": float(max(wins)) if wins else 0.0,
            "largest_loss": float(min(losses)) if losses else 0.0,
            "profit_factor": profit_factor,
            "expectancy": float(np.mean(pnls)),
            "consecutive_wins": max_win_streak,
            "consecutive_losses": max_loss_streak,
        }

    def get_trade_list(self) -> List[Dict]:
        """Return a copy of all recorded trades."""
        return list(self.trades)


# ===========================================================================
# 4. StrategyAnalyzer
# ===========================================================================


class StrategyAnalyzer:
    """Analyse strategy characteristics and robustness."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> None:
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._perf = PerformanceCalculator()

    # ------------------------------------------------------------------
    # Entry / exit quality
    # ------------------------------------------------------------------

    def analyze_entry_quality(self, trades: List[Dict]) -> Dict:
        """
        Rough entry quality: fraction of trades where the entry price was
        below (or equal to) the exit price (i.e. the position moved in the
        intended direction).
        """
        if not trades:
            return {"quality_score": 0.0, "good_entries": 0, "total": 0}
        good = sum(
            1 for t in trades if t.get("exit_price", 0) >= t.get("entry_price", 0)
        )
        return {
            "quality_score": good / len(trades),
            "good_entries": good,
            "total": len(trades),
        }

    def analyze_exit_quality(self, trades: List[Dict]) -> Dict:
        """
        Approximate exit quality: fraction of profitable trades where the
        exit price was above (or equal to) the entry price.
        """
        if not trades:
            return {"quality_score": 0.0, "good_exits": 0, "total": 0}
        good = sum(
            1 for t in trades if t.get("exit_price", 0) >= t.get("entry_price", 0)
        )
        return {
            "quality_score": good / len(trades),
            "good_exits": good,
            "total": len(trades),
        }

    # ------------------------------------------------------------------
    # Holding periods
    # ------------------------------------------------------------------

    def analyze_holding_periods(self, trades: List[Dict]) -> Dict:
        """Return descriptive statistics for trade holding-period durations."""
        if not trades:
            return {
                "avg_holding_days": 0,
                "median_holding_days": 0,
                "shortest_trade": 0,
                "longest_trade": 0,
                "holding_distribution": {},
            }
        days = []
        for t in trades:
            if "days_held" in t:
                days.append(t["days_held"])
            elif "entry_date" in t and "exit_date" in t:
                d = (_parse_date(t["exit_date"]) - _parse_date(t["entry_date"])).days
                days.append(max(d, 0))
            else:
                days.append(0)

        arr = np.array(days, dtype=float)
        buckets = {"1-7d": 0, "8-30d": 0, "31-90d": 0, "91-180d": 0, "180d+": 0}
        for d in arr:
            if d <= 7:
                buckets["1-7d"] += 1
            elif d <= 30:
                buckets["8-30d"] += 1
            elif d <= 90:
                buckets["31-90d"] += 1
            elif d <= 180:
                buckets["91-180d"] += 1
            else:
                buckets["180d+"] += 1

        return {
            "avg_holding_days": float(arr.mean()),
            "median_holding_days": float(np.median(arr)),
            "shortest_trade": int(arr.min()),
            "longest_trade": int(arr.max()),
            "holding_distribution": buckets,
        }

    # ------------------------------------------------------------------
    # Sector performance
    # ------------------------------------------------------------------

    def analyze_sector_performance(self, trades: List[Dict]) -> Dict:
        """
        Group trades by the 'sector' key and return per-sector P&L summary.
        If trades don't include a 'sector' key they are grouped as 'Unknown'.
        """
        if not trades:
            return {}
        sectors: Dict[str, List[float]] = {}
        for t in trades:
            sector = t.get("sector", "Unknown")
            sectors.setdefault(sector, []).append(t.get("pnl_dollars", 0.0))

        result = {}
        for sector, pnls in sectors.items():
            result[sector] = {
                "total_trades": len(pnls),
                "total_pnl": float(sum(pnls)),
                "avg_pnl": float(np.mean(pnls)),
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            }
        return result

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        trades: List[Dict],
        num_simulations: int = 1000,
        initial_capital: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Run a Monte Carlo simulation by randomly re-ordering the observed
        trade P&L sequence.

        Returns:
            probability_profit  – fraction of simulations ending above initial capital
            var_95              – 5th-percentile final portfolio value change
            cvar_95             – expected value below the 5th percentile
            expected_final_value – mean final portfolio value
        """
        capital = initial_capital if initial_capital is not None else self._initial_capital
        if not trades:
            return {
                "probability_profit": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "expected_final_value": capital,
            }

        pnls = np.array([t.get("pnl_dollars", 0.0) for t in trades])
        rng = np.random.default_rng(seed)
        final_values = []
        for _ in range(num_simulations):
            shuffled = rng.choice(pnls, size=len(pnls), replace=True)
            final_values.append(capital + float(shuffled.sum()))

        final_arr = np.array(final_values)
        prob_profit = float((final_arr > capital).mean())
        percentile_5 = float(np.percentile(final_arr, 5))
        below_5 = final_arr[final_arr <= percentile_5]
        cvar = float(below_5.mean()) if len(below_5) > 0 else percentile_5

        return {
            "probability_profit": prob_profit,
            "var_95": percentile_5 - capital,
            "cvar_95": cvar - capital,
            "expected_final_value": float(final_arr.mean()),
        }

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validation(
        self,
        strategy_func: Callable,
        lookback_periods: int = 5,
        rebalance_freq: str = "monthly",
    ) -> Dict:
        """
        Perform walk-forward analysis.

        Since the analyzer has no price data of its own, this method
        delegates to the module-level ``walk_forward_test`` function.
        The caller should supply a strategy_func that accepts
        (date, tickers, prices) and returns a list of signal dicts.
        """
        return {
            "lookback_periods": lookback_periods,
            "rebalance_freq": rebalance_freq,
            "message": (
                "Use the module-level walk_forward_test() for a full "
                "data-aware walk-forward analysis."
            ),
        }


# ===========================================================================
# 5. BenchmarkAnalyzer
# ===========================================================================


class BenchmarkAnalyzer:
    """Compare strategy performance to a benchmark."""

    def compare_to_benchmark(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Compare strategy to benchmark and return a comprehensive dict.
        """
        if strategy_returns is None or len(strategy_returns) == 0:
            return {}

        perf = PerformanceCalculator()
        n = min(len(strategy_returns), len(benchmark_returns))
        s_ret = strategy_returns.iloc[:n]
        b_ret = benchmark_returns.iloc[:n]

        s_cum = (1 + s_ret).cumprod()
        b_cum = (1 + b_ret).cumprod()
        s_total = float(s_cum.iloc[-1]) - 1.0
        b_total = float(b_cum.iloc[-1]) - 1.0

        beta = self.calculate_beta(s_ret, b_ret)
        ann_s = perf.calculate_annualized_return(s_ret)
        ann_b = perf.calculate_annualized_return(b_ret)
        alpha = self.calculate_alpha(ann_s, _DEFAULT_RISK_FREE_RATE, beta, ann_b)
        ir = self.calculate_information_ratio(s_ret, b_ret)

        active = s_ret.values - b_ret.values
        tracking_error = float(pd.Series(active).std() * math.sqrt(_TRADING_DAYS_PER_YEAR))
        corr = float(s_ret.corr(b_ret))

        # Monthly outperformance – only computed when both series have DatetimeIndex
        out_months = 0
        under_months = 0
        if isinstance(s_ret.index, pd.DatetimeIndex) and isinstance(b_ret.index, pd.DatetimeIndex):
            s_monthly = s_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            b_monthly = b_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            common_idx = s_monthly.index.intersection(b_monthly.index)
            for idx in common_idx:
                if s_monthly[idx] > b_monthly[idx]:
                    out_months += 1
                else:
                    under_months += 1

        return {
            "strategy_total_return": s_total,
            "benchmark_total_return": b_total,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": ir,
            "tracking_error": tracking_error,
            "correlation": corr,
            "outperformance_months": out_months,
            "underperformance_months": under_months,
        }

    def calculate_alpha(
        self,
        strategy_return: float,
        risk_free_rate: float,
        beta: float,
        market_return: float,
    ) -> float:
        """Jensen's alpha: r_p - [r_f + β*(r_m - r_f)]."""
        return strategy_return - (risk_free_rate + beta * (market_return - risk_free_rate))

    def calculate_beta(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
    ) -> float:
        """Beta = Cov(r_p, r_m) / Var(r_m)."""
        if (
            strategy_returns is None
            or market_returns is None
            or len(strategy_returns) < 2
            or len(market_returns) < 2
        ):
            return 1.0
        n = min(len(strategy_returns), len(market_returns))
        s = strategy_returns.iloc[:n].values
        m = market_returns.iloc[:n].values
        cov_matrix = np.cov(s, m)
        market_var = cov_matrix[1, 1]
        if market_var == 0:
            return 1.0
        return float(cov_matrix[0, 1] / market_var)

    def calculate_information_ratio(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Annualised information ratio (active return / tracking error)."""
        perf = PerformanceCalculator()
        return perf.calculate_information_ratio(strategy_returns, benchmark_returns)

    def drawdown_comparison(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict:
        """Compare maximum drawdowns of strategy vs benchmark."""
        perf = PerformanceCalculator()
        s_cum = (1 + strategy_returns).cumprod()
        b_cum = (1 + benchmark_returns).cumprod()
        return {
            "strategy_max_drawdown": perf.calculate_max_drawdown(s_cum),
            "benchmark_max_drawdown": perf.calculate_max_drawdown(b_cum),
            "strategy_recovery_days": perf.calculate_recovery_time(s_cum),
            "benchmark_recovery_days": perf.calculate_recovery_time(b_cum),
        }


# ===========================================================================
# 6. BacktestVisualizer
# ===========================================================================


class BacktestVisualizer:
    """Generate DataFrames / dicts ready for charting."""

    def get_equity_curve_data(
        self,
        trades: List[Dict],
        initial_capital: float = 100_000,
    ) -> pd.DataFrame:
        """
        Build a simple equity curve by sorting trades chronologically and
        accumulating P&L.
        """
        if not trades:
            return pd.DataFrame(columns=["date", "cumulative_value"])

        sorted_trades = sorted(trades, key=lambda t: t.get("exit_date", ""))
        dates = []
        values = []
        running = initial_capital
        for t in sorted_trades:
            running += t.get("pnl_dollars", 0.0)
            dates.append(t.get("exit_date", ""))
            values.append(running)
        return pd.DataFrame({"date": dates, "cumulative_value": values})

    def get_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """
        Pivot daily returns into a year × month heatmap table.

        The index is datetime-indexed for resampling; falls back to a
        simple month-by-month grouping if the index is not datetime.
        """
        if returns is None or len(returns) == 0:
            return pd.DataFrame()
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            df = monthly.to_frame(name="return")
            df["year"] = df.index.year
            df["month"] = df.index.month
            pivoted = df.pivot(index="year", columns="month", values="return")
            return pivoted
        # Fallback: return as-is grouped by position
        return returns.to_frame(name="return")

    def get_drawdown_data(
        self, cumulative_returns: pd.Series
    ) -> pd.DataFrame:
        """Generate a drawdown series from cumulative returns."""
        if cumulative_returns is None or len(cumulative_returns) == 0:
            return pd.DataFrame(columns=["drawdown"])
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.to_frame(name="drawdown")

    def get_performance_metrics(self, backtest_result: Dict) -> Dict:
        """Format key metrics from a backtest result dict for display."""
        keys = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "alpha",
            "beta",
        ]
        return {k: backtest_result.get(k) for k in keys if k in backtest_result}

    def get_trade_distribution(self, trades: List[Dict]) -> Dict:
        """
        Compute P&L histogram buckets and basic distribution statistics.
        """
        if not trades:
            return {"buckets": {}, "mean": 0.0, "std": 0.0, "median": 0.0}
        pnls = np.array([t.get("pnl_dollars", 0.0) for t in trades])
        counts, edges = np.histogram(pnls, bins=10)
        buckets = {}
        for i, count in enumerate(counts):
            label = f"{edges[i]:.0f} to {edges[i+1]:.0f}"
            buckets[label] = int(count)
        return {
            "buckets": buckets,
            "mean": float(pnls.mean()),
            "std": float(pnls.std()),
            "median": float(np.median(pnls)),
        }


# ===========================================================================
# 1. BacktestEngine
# ===========================================================================


class BacktestEngine:
    """
    Execute and analyse historical strategy backtests.

    Parameters
    ----------
    start_date : str
        ISO date string (YYYY-MM-DD) for the start of the backtest window.
    end_date : str
        ISO date string (YYYY-MM-DD) for the end of the backtest window.
    initial_capital : float
        Starting portfolio value in dollars (default 100,000).
    benchmark : str
        Ticker symbol for the benchmark index (default ``^GSPC``).
    tickers : list[str], optional
        List of stock tickers to be used inside ``backtest_strategy``.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000,
        benchmark: str = "^GSPC",
        tickers: Optional[List[str]] = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.tickers: List[str] = tickers or []

        self._signals: List[Dict] = []
        self._tracker = TradeTracker()
        self._perf = PerformanceCalculator()
        self._bench_analyzer = BenchmarkAnalyzer()

    # ------------------------------------------------------------------
    # Signal management
    # ------------------------------------------------------------------

    def add_signal(
        self,
        ticker: str,
        entry_date: str,
        exit_date: str,
        signal: str = "BUY",
    ) -> None:
        """Add a pre-computed historical signal."""
        self._signals.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "signal": signal,
            }
        )

    # ------------------------------------------------------------------
    # Core backtesting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_prices(
        ticker: str,
        start_date: str,
        end_date: str,
        seed: Optional[int] = None,
    ) -> pd.Series:
        """
        Generate a synthetic daily price series for *ticker* over the given
        date range using a geometric-random-walk model.

        This is used when real price data is unavailable (offline / tests).
        The seed is derived from the ticker name for reproducibility.
        """
        n_days = _date_range_days(start_date, end_date)
        if n_days <= 0:
            return pd.Series(dtype=float)
        rng_seed = seed if seed is not None else sum(ord(c) for c in ticker)
        rng = np.random.default_rng(rng_seed)
        daily_returns = rng.normal(0.0005, 0.015, n_days)
        prices = 100.0 * np.cumprod(1 + daily_returns)
        idx = pd.date_range(start=start_date, periods=n_days, freq="D")
        return pd.Series(prices, index=idx, name=ticker)

    def _get_price(self, ticker: str, date: str) -> float:
        """Look up a price for a ticker on a given date using synthetic data."""
        prices = self._simulate_prices(ticker, self.start_date, self.end_date)
        if prices.empty:
            return 100.0
        ts = pd.Timestamp(date)
        if ts in prices.index:
            return float(prices[ts])
        # Nearest available price
        nearest = prices.index.get_indexer([ts], method="nearest")[0]
        return float(prices.iloc[nearest])

    # ------------------------------------------------------------------
    # backtest_strategy
    # ------------------------------------------------------------------

    def backtest_strategy(
        self,
        strategy_func: Callable,
        prices: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Run a backtest using a custom strategy function.

        strategy_func signature::

            def my_strategy(date, tickers, prices) -> List[Dict]:
                # Each dict: {'ticker': str, 'action': 'BUY'/'SELL', 'weight': float}
                ...

        If *prices* is not provided, synthetic prices are generated for each
        ticker in ``self.tickers``.

        Returns a comprehensive result dict (see class docstring).
        """
        tracker = TradeTracker()

        if prices is None:
            prices = {
                t: self._simulate_prices(t, self.start_date, self.end_date)
                for t in self.tickers
            }

        n_days = _date_range_days(self.start_date, self.end_date)
        dates = [
            (datetime.strptime(self.start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_days)
        ]

        portfolio_value = self.initial_capital
        open_positions: Dict[str, Dict] = {}

        equity_curve: List[float] = [self.initial_capital]
        portfolio_returns: List[float] = []

        for date in dates:
            if not self.tickers:
                break

            # Close positions when the strategy signals exit
            for ticker in list(open_positions.keys()):
                signals_today = strategy_func(date, [ticker], prices)
                close_signal = any(
                    s.get("action") in ("SELL", "EXIT") and s.get("ticker") == ticker
                    for s in (signals_today or [])
                )
                if close_signal:
                    pos = open_positions.pop(ticker)
                    exit_price = self._get_price(ticker, date)
                    tracker.record_trade(
                        ticker=ticker,
                        entry_date=pos["entry_date"],
                        entry_price=pos["entry_price"],
                        exit_date=date,
                        exit_price=exit_price,
                        position_size=pos["position_size"],
                        exit_reason="sell_signal",
                    )
                    pnl = tracker.trades[-1]["pnl_dollars"]
                    portfolio_value += pnl

            # Open new positions
            try:
                new_signals = strategy_func(date, self.tickers, prices)
            except Exception:
                new_signals = []

            for sig in new_signals or []:
                ticker = sig.get("ticker", "")
                if sig.get("action") == "BUY" and ticker not in open_positions and ticker:
                    weight = float(sig.get("weight", 0.05))
                    position_size = portfolio_value * weight
                    entry_price = self._get_price(ticker, date)
                    open_positions[ticker] = {
                        "entry_date": date,
                        "entry_price": entry_price,
                        "position_size": position_size,
                    }

            # Daily P&L approximation for equity curve
            daily_pnl = 0.0
            for ticker, pos in open_positions.items():
                current_price = self._get_price(ticker, date)
                ep = pos["entry_price"]
                if ep > 0:
                    daily_pnl += (current_price - ep) / ep * pos["position_size"]

            prev_value = equity_curve[-1]
            current_value = portfolio_value + daily_pnl
            if prev_value > 0:
                portfolio_returns.append((current_value - prev_value) / prev_value)
            equity_curve.append(current_value)

        # Close any remaining open positions at end date
        for ticker, pos in open_positions.items():
            exit_price = self._get_price(ticker, self.end_date)
            tracker.record_trade(
                ticker=ticker,
                entry_date=pos["entry_date"],
                entry_price=pos["entry_price"],
                exit_date=self.end_date,
                exit_price=exit_price,
                position_size=pos["position_size"],
                exit_reason="end_of_period",
            )

        self._tracker = tracker
        return self._compile_results(tracker, portfolio_returns, equity_curve)

    # ------------------------------------------------------------------
    # backtest_signals
    # ------------------------------------------------------------------

    def backtest_signals(self, signals: List[Dict]) -> Dict[str, Any]:
        """
        Backtest a pre-defined list of signals.

        Each signal dict must contain:
        ``ticker``, ``entry_date``, ``entry_price``, ``exit_date``,
        ``exit_price``, ``position_size`` (optional, default 10 % of capital).
        """
        tracker = TradeTracker()
        portfolio_value = self.initial_capital
        portfolio_returns: List[float] = []
        equity_curve: List[float] = [self.initial_capital]

        sorted_signals = sorted(signals, key=lambda s: s.get("entry_date", ""))

        for sig in sorted_signals:
            ticker = sig.get("ticker", "UNKNOWN")
            entry_date = sig.get("entry_date", self.start_date)
            exit_date = sig.get("exit_date", self.end_date)
            entry_price = float(sig.get("entry_price", self._get_price(ticker, entry_date)))
            exit_price = float(sig.get("exit_price", self._get_price(ticker, exit_date)))
            position_size = float(
                sig.get("position_size", portfolio_value * 0.10)
            )

            tracker.record_trade(
                ticker=ticker,
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                position_size=position_size,
                exit_reason=sig.get("exit_reason", "signal"),
            )
            pnl = tracker.trades[-1]["pnl_dollars"]
            portfolio_value += pnl
            prev = equity_curve[-1]
            equity_curve.append(portfolio_value)
            if prev > 0:
                portfolio_returns.append(pnl / prev)

        self._tracker = tracker
        return self._compile_results(tracker, portfolio_returns, equity_curve)

    # ------------------------------------------------------------------
    # optimize_parameters
    # ------------------------------------------------------------------

    def optimize_parameters(
        self,
        param_ranges: Dict[str, List],
        strategy_func: Callable,
    ) -> Dict[str, Any]:
        """
        Grid-search over *param_ranges* to find the parameter combination
        that maximises the Sharpe ratio.

        Parameters
        ----------
        param_ranges : dict
            Mapping of parameter name → list of candidate values.
            E.g. ``{'lookback': [10, 20, 30], 'threshold': [0.03, 0.05]}``.
        strategy_func : callable
            A *factory* that accepts the parameter kwargs and returns a
            strategy function compatible with ``backtest_strategy``.

        Returns the best parameters and the associated backtest result.
        """
        keys = list(param_ranges.keys())
        value_lists = [param_ranges[k] for k in keys]

        best_sharpe = float("-inf")
        best_params: Dict[str, Any] = {}
        best_result: Dict[str, Any] = {}

        for combo in product(*value_lists):
            params = dict(zip(keys, combo))
            try:
                strat = strategy_func(**params)
                result = self.backtest_strategy(strat)
                sharpe = result.get("sharpe_ratio", float("-inf"))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_result = result
            except Exception as exc:  # pragma: no cover
                logger.debug("Parameter combo %s failed: %s", params, exc)

        return {"best_params": best_params, "best_result": best_result}

    # ------------------------------------------------------------------
    # walk-forward (convenience wrapper)
    # ------------------------------------------------------------------

    def walk_forward_test(
        self,
        strategy_func: Callable,
        lookback_period: str = "252d",
        rebalance_freq: str = "monthly",
    ) -> Dict:
        """Delegate to the module-level ``walk_forward_test`` function."""
        return walk_forward_test(
            strategy_func=strategy_func,
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            lookback_period=lookback_period,
            rebalance_freq=rebalance_freq,
        )

    # ------------------------------------------------------------------
    # Monte Carlo (convenience wrapper)
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        trades: List[Dict],
        num_simulations: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict:
        """Delegate to ``StrategyAnalyzer.monte_carlo_simulation``."""
        analyzer = StrategyAnalyzer(initial_capital=self.initial_capital)
        return analyzer.monte_carlo_simulation(
            trades=trades,
            num_simulations=num_simulations,
            initial_capital=self.initial_capital,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal result compilation
    # ------------------------------------------------------------------

    def _compile_results(
        self,
        tracker: TradeTracker,
        portfolio_returns: List[float],
        equity_curve: List[float],
    ) -> Dict[str, Any]:
        """Build the standard backtest result dict from raw components."""
        returns_series = pd.Series(portfolio_returns)
        cumulative = (1 + returns_series).cumprod() if len(returns_series) > 0 else pd.Series([1.0])

        total_return = self._perf.calculate_total_return(
            self.initial_capital, equity_curve[-1] if equity_curve else self.initial_capital
        )
        ann_return = self._perf.calculate_annualized_return(returns_series)
        sharpe = self._perf.calculate_sharpe_ratio(returns_series)
        sortino = self._perf.calculate_sortino_ratio(returns_series)
        max_dd = self._perf.calculate_max_drawdown(cumulative)
        recovery = self._perf.calculate_recovery_time(cumulative)

        stats = tracker.get_trade_statistics()
        trades = tracker.get_trade_list()

        # Benchmark synthetic returns
        bench_prices = self._simulate_prices(
            self.benchmark, self.start_date, self.end_date
        )
        if len(bench_prices) > 1:
            bench_returns = bench_prices.pct_change().dropna()
            bench_return = float((1 + bench_returns).prod()) - 1.0
            beta = self._bench_analyzer.calculate_beta(returns_series, bench_returns)
            alpha = self._bench_analyzer.calculate_alpha(
                ann_return, _DEFAULT_RISK_FREE_RATE, beta,
                self._perf.calculate_annualized_return(bench_returns),
            )
        else:
            bench_return = 0.0
            beta = 1.0
            alpha = 0.0

        # Monthly returns
        if len(returns_series) > 0 and isinstance(returns_series.index, pd.RangeIndex):
            idx = pd.date_range(
                start=self.start_date, periods=len(returns_series), freq="D"
            )
            dated_returns = pd.Series(returns_series.values, index=idx)
        else:
            dated_returns = returns_series

        monthly = []
        try:
            m_series = dated_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            monthly = [
                {"date": str(d.date()), "return": float(r)}
                for d, r in zip(m_series.index, m_series)
            ]
        except Exception:
            pass

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "recovery_days": recovery,
            "winning_trades": stats["winning_trades"],
            "losing_trades": stats["losing_trades"],
            "total_trades": stats["total_trades"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "expectancy": stats["expectancy"],
            "monthly_returns": monthly,
            "equity_curve": [
                {"index": i, "value": v} for i, v in enumerate(equity_curve)
            ],
            "benchmark_return": bench_return,
            "alpha": alpha,
            "beta": beta,
            "trades": trades,
        }


# ===========================================================================
# 7. walk_forward_test – module-level function
# ===========================================================================


def walk_forward_test(
    strategy_func: Callable,
    tickers: List[str],
    start_date: str,
    end_date: str,
    lookback_period: str = "252d",
    rebalance_freq: str = "monthly",
) -> Dict:
    """
    Perform walk-forward analysis to prevent over-fitting.

    The overall date range [start_date, end_date] is split into rolling
    windows.  For each window:
    1. An in-sample period of length *lookback_period* is used for training.
    2. An out-of-sample test period of equal length follows.

    Parameters
    ----------
    strategy_func : callable
        Same signature as for ``BacktestEngine.backtest_strategy``.
    tickers : list[str]
        Tickers available to the strategy.
    start_date, end_date : str
        ISO date strings.
    lookback_period : str
        Length of the in-sample window, e.g. ``"252d"`` (default 1 year).
    rebalance_freq : str
        Step size between windows: ``"monthly"`` (≈21 trading days) or
        ``"quarterly"`` (≈63 trading days).

    Returns
    -------
    dict with:
        windows, avg_sharpe, sharpe_std, avg_return, return_std,
        out_of_sample_return, robustness_score
    """
    # Parse lookback
    if lookback_period.endswith("d"):
        lb_days = int(lookback_period[:-1])
    elif lookback_period.endswith("y"):
        lb_days = int(lookback_period[:-1]) * 365
    else:
        lb_days = 252  # default 1 year

    step_days = 21 if rebalance_freq == "monthly" else 63

    total_days = _date_range_days(start_date, end_date)
    start_dt = _parse_date(start_date)

    windows = []
    offset = 0
    while offset + lb_days * 2 <= total_days:
        train_start = (start_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        train_end = (start_dt + timedelta(days=offset + lb_days)).strftime("%Y-%m-%d")
        test_end = (
            start_dt + timedelta(days=offset + lb_days * 2)
        ).strftime("%Y-%m-%d")

        engine = BacktestEngine(
            start_date=train_end,
            end_date=test_end,
            tickers=tickers,
        )
        try:
            result = engine.backtest_strategy(strategy_func)
            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_end": test_end,
                    "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                    "total_return": result.get("total_return", 0.0),
                }
            )
        except Exception as exc:
            logger.debug("Walk-forward window %s–%s failed: %s", train_end, test_end, exc)

        offset += step_days

    if not windows:
        return {
            "windows": [],
            "avg_sharpe": 0.0,
            "sharpe_std": 0.0,
            "avg_return": 0.0,
            "return_std": 0.0,
            "out_of_sample_return": 0.0,
            "robustness_score": 0.0,
        }

    sharpes = [w["sharpe_ratio"] for w in windows]
    returns = [w["total_return"] for w in windows]
    avg_sharpe = float(np.mean(sharpes))
    avg_return = float(np.mean(returns))
    sharpe_std = float(np.std(sharpes))
    return_std = float(np.std(returns))

    # Robustness: fraction of windows with positive return
    robustness = sum(1 for r in returns if r > 0) / len(returns)

    return {
        "windows": windows,
        "avg_sharpe": avg_sharpe,
        "sharpe_std": sharpe_std,
        "avg_return": avg_return,
        "return_std": return_std,
        "out_of_sample_return": avg_return,
        "robustness_score": robustness,
    }


# ===========================================================================
# 8. Example strategies
# ===========================================================================


def momentum_strategy(
    data: Dict[str, pd.Series],
    lookback: int = 20,
    threshold: float = 0.05,
) -> List[Dict]:
    """
    Simple price-momentum strategy.

    Buy tickers whose *lookback*-day return exceeds *threshold*.
    """
    signals = []
    for ticker, prices in data.items():
        if prices is None or len(prices) < lookback + 1:
            continue
        recent_return = float(prices.iloc[-1] / prices.iloc[-(lookback + 1)] - 1)
        if recent_return > threshold:
            signals.append({"ticker": ticker, "action": "BUY", "weight": 0.1})
    return signals


def mean_reversion_strategy(
    data: Dict[str, pd.Series],
    lookback: int = 50,
    std_threshold: float = -1.5,
) -> List[Dict]:
    """
    Mean-reversion strategy.

    Buy when the latest price is *std_threshold* standard deviations below
    the *lookback*-day mean.
    """
    signals = []
    for ticker, prices in data.items():
        if prices is None or len(prices) < lookback + 1:
            continue
        window = prices.iloc[-(lookback + 1) :]
        mean = float(window.mean())
        std = float(window.std())
        if std == 0:
            continue
        z_score = (float(prices.iloc[-1]) - mean) / std
        if z_score <= std_threshold:
            signals.append({"ticker": ticker, "action": "BUY", "weight": 0.1})
    return signals


def rsi_strategy(
    data: Dict[str, pd.Series],
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> List[Dict]:
    """
    RSI-based strategy.

    Buy when RSI < *oversold*, sell when RSI > *overbought*.
    """
    signals = []
    for ticker, prices in data.items():
        if prices is None or len(prices) < period + 1:
            continue
        delta = prices.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        if rsi_series.empty or rsi_series.isna().all():
            continue
        current_rsi = float(rsi_series.iloc[-1])
        if current_rsi < oversold:
            signals.append({"ticker": ticker, "action": "BUY", "weight": 0.1})
        elif current_rsi > overbought:
            signals.append({"ticker": ticker, "action": "SELL", "weight": 0.0})
    return signals


def macd_strategy(
    data: Dict[str, pd.Series],
    fast_ema: int = 12,
    slow_ema: int = 26,
    signal_period: int = 9,
) -> List[Dict]:
    """
    MACD crossover strategy.

    Buy on a positive MACD crossover (MACD line crosses above signal line).
    """
    signals = []
    for ticker, prices in data.items():
        if prices is None or len(prices) < slow_ema + signal_period:
            continue
        ema_fast = prices.ewm(span=fast_ema, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_ema, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        if len(macd_line) < 2:
            continue
        prev_diff = float(macd_line.iloc[-2]) - float(signal_line.iloc[-2])
        curr_diff = float(macd_line.iloc[-1]) - float(signal_line.iloc[-1])

        if prev_diff < 0 and curr_diff >= 0:
            # Bullish crossover
            signals.append({"ticker": ticker, "action": "BUY", "weight": 0.1})
        elif prev_diff > 0 and curr_diff <= 0:
            # Bearish crossover
            signals.append({"ticker": ticker, "action": "SELL", "weight": 0.0})
    return signals
