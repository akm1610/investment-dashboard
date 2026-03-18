"""
src/components/backtesting.py
------------------------------
Strategy Backtesting page.

Sections
--------
A. Configuration  – Strategy selection, date range, capital, benchmark.
B. Results        – Key metric cards, equity curve, monthly heatmap, drawdown.
C. Analysis       – Trade list, walk-forward validation, Monte Carlo.
D. Comparison     – Side-by-side strategy comparison.
"""

from __future__ import annotations

import sys
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtesting_engine import (
    STRATEGIES,
    BacktestEngine,
    BacktestVisualizer,
    StrategyAnalyzer,
    momentum_strategy,
    mean_reversion_strategy,
    rsi_strategy,
    macd_strategy,
    walk_forward_test,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BENCHMARKS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow Jones": "^DJI",
}

_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

_STRATEGY_LABELS = {
    "momentum": "📈 Momentum",
    "mean_reversion": "↩️ Mean Reversion",
    "rsi_oversold": "📉 RSI Oversold",
    "macd_crossover": "✂️ MACD Crossover",
}

_VISUALIZER = BacktestVisualizer()
_ANALYZER = StrategyAnalyzer()

# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def _build_strategy_func(
    strategy_name: str,
    params: Dict[str, Any],
):
    """Return a ``backtest_strategy``-compatible callable for the given strategy."""

    if strategy_name == "momentum":
        lookback = params.get("lookback", 20)
        threshold = params.get("threshold", 0.05)

        def _momentum_func(date_str, tickers, prices):  # noqa: ARG001
            return momentum_strategy(prices, lookback=lookback, threshold=threshold)

        return _momentum_func

    elif strategy_name == "mean_reversion":
        lookback = params.get("lookback", 50)
        threshold = params.get("threshold", -1.5)

        def _mean_reversion_func(date_str, tickers, prices):  # noqa: ARG001
            return mean_reversion_strategy(prices, lookback=lookback, std_threshold=threshold)

        return _mean_reversion_func

    elif strategy_name == "rsi_oversold":
        period = params.get("period", 14)
        threshold = params.get("threshold", 30)

        def _rsi_func(date_str, tickers, prices):  # noqa: ARG001
            return rsi_strategy(prices, period=period, oversold=threshold)

        return _rsi_func

    elif strategy_name == "macd_crossover":
        fast = params.get("fast_ema", 12)
        slow = params.get("slow_ema", 26)
        signal = params.get("signal", 9)

        def _macd_func(date_str, tickers, prices):  # noqa: ARG001
            return macd_strategy(prices, fast_ema=fast, slow_ema=slow, signal_period=signal)

        return _macd_func

    else:
        def _noop_func(date_str, tickers, prices):  # noqa: ARG001
            return []

        return _noop_func


# ---------------------------------------------------------------------------
# Section A: Backtest Configuration
# ---------------------------------------------------------------------------


def _section_configuration() -> Optional[Dict[str, Any]]:
    """Render configuration panel; return config dict when Run is clicked."""
    st.subheader("⚙️ Backtest Configuration")

    # Strategy selection
    strategy_name = st.selectbox(
        "Strategy",
        list(_STRATEGY_LABELS.keys()),
        format_func=lambda k: _STRATEGY_LABELS[k],
        key="bt_strategy",
    )

    st.markdown(f"*{STRATEGIES[strategy_name]['description']}*")

    # Strategy-specific parameters
    params: Dict[str, Any] = {}
    with st.expander("Strategy Parameters", expanded=True):
        if strategy_name == "momentum":
            params["lookback"] = st.slider("Lookback Period (days)", 5, 60,
                                           STRATEGIES["momentum"]["lookback"], key="bt_p_lb")
            params["threshold"] = st.slider("Return Threshold", 0.01, 0.20,
                                            float(STRATEGIES["momentum"]["threshold"]),
                                            step=0.01, key="bt_p_th")
        elif strategy_name == "mean_reversion":
            params["lookback"] = st.slider("Lookback Period (days)", 20, 100,
                                           STRATEGIES["mean_reversion"]["lookback"], key="bt_p_lb")
            params["threshold"] = st.slider("Z-Score Threshold", -3.0, -0.5,
                                            float(STRATEGIES["mean_reversion"]["threshold"]),
                                            step=0.1, key="bt_p_th")
        elif strategy_name == "rsi_oversold":
            params["period"] = st.slider("RSI Period", 5, 30,
                                         STRATEGIES["rsi_oversold"]["period"], key="bt_p_period")
            params["threshold"] = st.slider("Oversold Threshold", 10, 40,
                                            STRATEGIES["rsi_oversold"]["threshold"], key="bt_p_th")
        elif strategy_name == "macd_crossover":
            params["fast_ema"] = st.slider("Fast EMA", 5, 20,
                                           STRATEGIES["macd_crossover"]["fast_ema"], key="bt_p_fast")
            params["slow_ema"] = st.slider("Slow EMA", 20, 50,
                                           STRATEGIES["macd_crossover"]["slow_ema"], key="bt_p_slow")
            params["signal"] = st.slider("Signal Period", 5, 20,
                                         STRATEGIES["macd_crossover"]["signal"], key="bt_p_sig")

    st.markdown("---")

    # Date range & capital
    col1, col2 = st.columns(2)
    default_start = date.today() - timedelta(days=365 * 5)
    start_date = col1.date_input("Start Date", value=default_start, key="bt_start")
    end_date = col2.date_input("End Date", value=date.today(), key="bt_end")

    col3, col4 = st.columns(2)
    initial_capital = col3.number_input(
        "Initial Capital ($)", min_value=1_000, value=100_000, step=1_000, key="bt_capital"
    )
    benchmark_label = col4.selectbox(
        "Benchmark", list(_BENCHMARKS.keys()), index=0, key="bt_bench"
    )
    benchmark = _BENCHMARKS[benchmark_label]

    # Ticker selection
    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(_DEFAULT_TICKERS),
        key="bt_tickers",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    st.markdown("---")
    run_btn = st.button("🚀 Run Backtest", type="primary", key="bt_run")

    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return None
        if not tickers:
            st.error("Please enter at least one ticker.")
            return None
        return {
            "strategy_name": strategy_name,
            "params": params,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "initial_capital": float(initial_capital),
            "benchmark": benchmark,
            "tickers": tickers,
        }
    return None


# ---------------------------------------------------------------------------
# Section B: Performance Metrics Dashboard
# ---------------------------------------------------------------------------


def _section_performance(result: Dict[str, Any]) -> None:
    """Render key metric cards and main charts from a backtest result."""
    st.subheader("📊 Performance Metrics")

    # ── Row 1: return metrics ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{result.get('total_return', 0):.1%}")
    c2.metric("Annualised Return", f"{result.get('annualized_return', 0):.1%}")
    c3.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
    mdd = result.get("max_drawdown", 0)
    c4.metric("Max Drawdown", f"-{mdd:.1%}")

    # ── Row 2: trade metrics ───────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win Rate", f"{result.get('win_rate', 0):.1%}")
    c6.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
    c7.metric("Alpha", f"{result.get('alpha', 0):+.2%}")
    c8.metric("Beta", f"{result.get('beta', 1):.2f}")

    # ── Detailed table ─────────────────────────────────────────────────────
    with st.expander("📋 Full Metrics Table", expanded=False):
        metrics_data = {
            "Total Return": f"{result.get('total_return', 0):.2%}",
            "Annualised Return": f"{result.get('annualized_return', 0):.2%}",
            "Benchmark Return": f"{result.get('benchmark_return', 0):.2%}",
            "Sharpe Ratio": f"{result.get('sharpe_ratio', 0):.3f}",
            "Sortino Ratio": f"{result.get('sortino_ratio', 0):.3f}",
            "Max Drawdown": f"-{result.get('max_drawdown', 0):.2%}",
            "Recovery (days)": str(result.get("recovery_days", "N/A")),
            "Total Trades": str(result.get("total_trades", 0)),
            "Winning Trades": str(result.get("winning_trades", 0)),
            "Losing Trades": str(result.get("losing_trades", 0)),
            "Win Rate": f"{result.get('win_rate', 0):.2%}",
            "Profit Factor": f"{result.get('profit_factor', 0):.3f}",
            "Expectancy ($)": f"${result.get('expectancy', 0):,.2f}",
            "Alpha": f"{result.get('alpha', 0):+.3%}",
            "Beta": f"{result.get('beta', 1):.3f}",
        }
        metrics_df = pd.DataFrame(
            list(metrics_data.items()), columns=["Metric", "Value"]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Equity curve ──────────────────────────────────────────────────────
    st.subheader("📈 Equity Curve")
    equity = result.get("equity_curve", [])
    if equity:
        eq_df = pd.DataFrame(equity)
        initial = result.get("initial_capital", 100_000)
        bench_return = result.get("benchmark_return", 0)

        # Benchmark line (simple linear interpolation)
        bench_values = [
            initial * (1 + bench_return * i / max(len(equity) - 1, 1))
            for i in range(len(equity))
        ]

        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df["index"],
                y=eq_df["value"],
                mode="lines",
                name="Strategy",
                line={"color": "royalblue", "width": 2},
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df["index"],
                y=bench_values,
                mode="lines",
                name="Benchmark",
                line={"color": "grey", "width": 1, "dash": "dash"},
            )
        )
        fig_eq.update_layout(
            height=400,
            xaxis_title="Day",
            yaxis_title="Portfolio Value ($)",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Monthly returns heatmap ───────────────────────────────────────────
    monthly = result.get("monthly_returns", [])
    if monthly:
        st.subheader("📅 Monthly Returns Heatmap")
        monthly_df = pd.DataFrame(monthly)
        monthly_df["date"] = pd.to_datetime(monthly_df["date"])
        monthly_df["year"] = monthly_df["date"].dt.year
        monthly_df["month"] = monthly_df["date"].dt.month

        pivot = monthly_df.pivot_table(
            index="year", columns="month", values="return", aggfunc="sum"
        )
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        pivot.columns = [month_names.get(c, str(c)) for c in pivot.columns]

        z_vals = pivot.values * 100  # convert to %
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=z_vals,
                x=list(pivot.columns),
                y=[str(y) for y in pivot.index],
                colorscale=[
                    [0.0, "red"],
                    [0.5, "lightyellow"],
                    [1.0, "green"],
                ],
                zmid=0,
                text=[[f"{v:.1f}%" for v in row] for row in z_vals],
                texttemplate="%{text}",
                showscale=True,
                colorbar={"title": "Return %"},
            )
        )
        fig_heat.update_layout(
            height=max(200, len(pivot) * 40 + 100),
            xaxis_title="Month",
            yaxis_title="Year",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Drawdown chart ────────────────────────────────────────────────────
    if equity:
        st.subheader("📉 Drawdown Chart")
        values = [e["value"] for e in equity]
        values_series = pd.Series(values)
        running_max = values_series.cummax()
        drawdown = (values_series - running_max) / running_max * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=list(range(len(drawdown))),
                y=drawdown.tolist(),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255, 80, 80, 0.3)",
                line={"color": "red"},
                name="Drawdown %",
            )
        )
        min_dd_idx = int(drawdown.idxmin())
        fig_dd.add_annotation(
            x=min_dd_idx,
            y=float(drawdown.iloc[min_dd_idx]),
            text=f"Max DD: {drawdown.min():.1f}%",
            showarrow=True,
            arrowhead=2,
        )
        fig_dd.update_layout(
            height=300, xaxis_title="Day", yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Trade P&L distribution ────────────────────────────────────────────
    trades = result.get("trades", [])
    if trades:
        st.subheader("📊 Trade P&L Distribution")
        dist = _VISUALIZER.get_trade_distribution(trades)
        buckets = dist.get("buckets", {})
        if buckets:
            dist_df = pd.DataFrame(
                list(buckets.items()), columns=["P&L Range", "Count"]
            )
            fig_dist = px.bar(
                dist_df,
                x="P&L Range",
                y="Count",
                title=f"Trade P&L Distribution  |  Mean: ${dist['mean']:,.0f}  "
                      f"|  Median: ${dist['median']:,.0f}",
                color="Count",
                color_continuous_scale="RdYlGn",
            )
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------------------------
# Section C: Analysis (trade list, walk-forward, Monte Carlo)
# ---------------------------------------------------------------------------


def _section_analysis(result: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Render trade list, walk-forward validation, and Monte Carlo tabs."""
    st.subheader("🔬 Deep Analysis")

    a_tab1, a_tab2, a_tab3 = st.tabs(
        ["📑 Trade List", "🔄 Walk-Forward Validation", "🎲 Monte Carlo Simulation"]
    )

    with a_tab1:
        _render_trade_list(result.get("trades", []))

    with a_tab2:
        _render_walk_forward(config)

    with a_tab3:
        _render_monte_carlo(result.get("trades", []))


def _render_trade_list(trades: List[Dict]) -> None:
    """Render the full trade list with colour-coded rows."""
    st.subheader("Trade List")
    if not trades:
        st.info("No trades were executed in this backtest.")
        return

    stats_data = {
        "Total Trades": len(trades),
        "Avg Holding Days": f"{np.mean([t.get('days_held', 0) for t in trades]):.1f}",
        "Largest Win ($)": f"${max((t.get('pnl_dollars', 0) for t in trades), default=0):,.2f}",
        "Largest Loss ($)": f"${min((t.get('pnl_dollars', 0) for t in trades), default=0):,.2f}",
    }
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (k, v) in zip([sc1, sc2, sc3, sc4], stats_data.items()):
        col.metric(k, v)

    rows = []
    for t in trades:
        rows.append(
            {
                "Ticker": t.get("ticker", ""),
                "Entry Date": t.get("entry_date", ""),
                "Entry $": f"${t.get('entry_price', 0):,.2f}",
                "Exit Date": t.get("exit_date", ""),
                "Exit $": f"${t.get('exit_price', 0):,.2f}",
                "Days": t.get("days_held", 0),
                "P&L $": f"${t.get('pnl_dollars', 0):,.2f}",
                "P&L %": f"{t.get('pnl_pct', 0):.2%}",
                "Exit Reason": t.get("exit_reason", ""),
            }
        )

    trades_df = pd.DataFrame(rows)
    st.dataframe(trades_df, use_container_width=True, hide_index=True)


def _render_walk_forward(config: Dict[str, Any]) -> None:
    """Render walk-forward validation results."""
    st.subheader("🔄 Walk-Forward Validation")
    st.caption(
        "Walk-forward analysis splits the date range into rolling windows "
        "to assess whether the strategy is robust or over-fit."
    )

    if st.button("▶️ Run Walk-Forward Validation", key="bt_wf_run"):
        strategy_func = _build_strategy_func(config["strategy_name"], config["params"])
        with st.spinner("Running walk-forward validation …"):
            try:
                wf_result = walk_forward_test(
                    strategy_func=strategy_func,
                    tickers=config["tickers"],
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                )
                st.session_state.bt_wf_result = wf_result
            except Exception as exc:
                st.error(f"Walk-forward failed: {exc}")
                return

    wf = st.session_state.get("bt_wf_result")
    if not wf:
        st.info("Click the button above to run walk-forward validation.")
        return

    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Avg Sharpe", f"{wf['avg_sharpe']:.2f}")
    wc2.metric("Avg Return", f"{wf['avg_return']:.2%}")
    wc3.metric("Robustness Score", f"{wf['robustness_score']:.0%}")

    robustness = wf["robustness_score"]
    if robustness >= 0.70:
        st.success("✅ Strategy appears **robust** – performs consistently across windows.")
    elif robustness >= 0.50:
        st.warning("⚠️ Strategy shows **moderate robustness** – mixed performance across windows.")
    else:
        st.error("❌ Strategy may be **over-fit** – inconsistent out-of-sample performance.")

    windows = wf.get("windows", [])
    if windows:
        win_df = pd.DataFrame(windows)
        fig_wf = go.Figure()
        fig_wf.add_trace(
            go.Bar(
                x=list(range(len(windows))),
                y=[w["sharpe_ratio"] for w in windows],
                name="Sharpe by Window",
                marker_color=[
                    "green" if s > 0 else "red"
                    for s in [w["sharpe_ratio"] for w in windows]
                ],
            )
        )
        fig_wf.update_layout(
            height=300,
            xaxis_title="Window",
            yaxis_title="Sharpe Ratio",
            title="Sharpe Ratio by Walk-Forward Window",
        )
        st.plotly_chart(fig_wf, use_container_width=True)
        st.dataframe(win_df, use_container_width=True, hide_index=True)


def _render_monte_carlo(trades: List[Dict]) -> None:
    """Render Monte Carlo simulation results."""
    st.subheader("🎲 Monte Carlo Simulation")
    st.caption(
        "Resample historical trades to estimate the distribution of possible outcomes."
    )

    n_simulations = st.slider(
        "Number of Simulations", 100, 2000, 1000, step=100, key="bt_mc_n"
    )

    if st.button("▶️ Run Monte Carlo", key="bt_mc_run"):
        if not trades:
            st.error("No trades available. Run the backtest first.")
            return

        with st.spinner(f"Running {n_simulations} simulations …"):
            try:
                mc_result = _ANALYZER.monte_carlo_simulation(
                    trades=trades, n_simulations=n_simulations
                )
                st.session_state.bt_mc_result = mc_result
            except Exception as exc:
                st.error(f"Monte Carlo failed: {exc}")
                return

    mc = st.session_state.get("bt_mc_result")
    if not mc:
        st.info("Click the button above to run Monte Carlo simulation.")
        return

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Prob. of Profit", f"{mc.get('probability_of_profit', 0):.1%}")
    mc2.metric("Expected Return", f"{mc.get('mean_return', 0):.2%}")
    mc3.metric(
        "95% Confidence Interval",
        f"{mc.get('percentile_5', 0):.1%} → {mc.get('percentile_95', 0):.1%}",
    )

    mc4, mc5 = st.columns(2)
    mc4.metric("Best Case", f"{mc.get('max_return', 0):.2%}")
    mc5.metric("Worst Case", f"{mc.get('min_return', 0):.2%}")

    # Histogram of simulation outcomes
    all_returns = mc.get("all_returns", [])
    if all_returns:
        fig_mc = px.histogram(
            x=all_returns,
            nbins=50,
            title="Monte Carlo Return Distribution",
            labels={"x": "Simulated Return"},
            color_discrete_sequence=["steelblue"],
        )
        fig_mc.add_vline(
            x=mc.get("mean_return", 0),
            line_dash="dash",
            line_color="orange",
            annotation_text="Mean",
        )
        fig_mc.update_layout(height=350)
        st.plotly_chart(fig_mc, use_container_width=True)


# ---------------------------------------------------------------------------
# Section D: Strategy Comparison
# ---------------------------------------------------------------------------


def _section_comparison() -> None:
    """Allow side-by-side comparison of multiple strategy backtest results."""
    st.subheader("⚖️ Strategy Comparison")

    comparisons = st.session_state.get("bt_comparisons", {})
    if not comparisons:
        st.info(
            "Run backtests for multiple strategies and they will appear here for comparison. "
            "Each completed backtest is saved automatically."
        )
        return

    compare_labels = st.multiselect(
        "Select strategies to compare",
        list(comparisons.keys()),
        default=list(comparisons.keys()),
        key="bt_compare_sel",
    )

    if not compare_labels:
        st.info("Select at least one strategy to compare.")
        return

    comparison_rows = []
    for label in compare_labels:
        r = comparisons[label]
        comparison_rows.append(
            {
                "Strategy": label,
                "Total Return": f"{r.get('total_return', 0):.2%}",
                "Ann. Return": f"{r.get('annualized_return', 0):.2%}",
                "Sharpe": f"{r.get('sharpe_ratio', 0):.2f}",
                "Sortino": f"{r.get('sortino_ratio', 0):.2f}",
                "Max DD": f"-{r.get('max_drawdown', 0):.2%}",
                "Win Rate": f"{r.get('win_rate', 0):.2%}",
                "Profit Factor": f"{r.get('profit_factor', 0):.2f}",
                "Alpha": f"{r.get('alpha', 0):+.2%}",
                "Beta": f"{r.get('beta', 1):.2f}",
            }
        )

    compare_df = pd.DataFrame(comparison_rows)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig_cmp = go.Figure()
    for label in compare_labels:
        r = comparisons[label]
        fig_cmp.add_trace(
            go.Bar(
                name=label,
                x=["Total Return", "Ann. Return", "Alpha"],
                y=[
                    r.get("total_return", 0) * 100,
                    r.get("annualized_return", 0) * 100,
                    r.get("alpha", 0) * 100,
                ],
            )
        )
    fig_cmp.update_layout(
        barmode="group",
        title="Return Metrics Comparison (%)",
        height=350,
        yaxis_title="% Return",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def page_backtesting() -> None:
    """Strategy Backtesting page."""
    st.title("📈 Strategy Backtesting")

    # Ensure session state keys
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = {}
    if "bt_comparisons" not in st.session_state:
        st.session_state.bt_comparisons = {}
    if "bt_wf_result" not in st.session_state:
        st.session_state.bt_wf_result = None
    if "bt_mc_result" not in st.session_state:
        st.session_state.bt_mc_result = None

    tab1, tab2, tab3, tab4 = st.tabs(
        ["⚙️ Configuration", "📊 Results", "🔬 Analysis", "⚖️ Comparison"]
    )

    with tab1:
        config = _section_configuration()
        if config:
            strategy_func = _build_strategy_func(config["strategy_name"], config["params"])
            engine = BacktestEngine(
                start_date=config["start_date"],
                end_date=config["end_date"],
                initial_capital=config["initial_capital"],
                benchmark=config["benchmark"],
                tickers=config["tickers"],
            )
            with st.spinner("Running backtest …"):
                try:
                    result = engine.backtest_strategy(strategy_func)
                    result["initial_capital"] = config["initial_capital"]
                    st.session_state.backtest_results = result
                    # Save for comparison
                    label = (
                        f"{_STRATEGY_LABELS.get(config['strategy_name'], config['strategy_name'])}"
                        f" ({config['start_date'][:4]}–{config['end_date'][:4]})"
                    )
                    st.session_state.bt_comparisons[label] = result
                    # Clear previous analysis state
                    st.session_state.bt_wf_result = None
                    st.session_state.bt_mc_result = None
                    st.session_state._bt_config = config
                    st.success("✅ Backtest complete! Switch to the **Results** tab to view.")
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")

    with tab2:
        result = st.session_state.get("backtest_results")
        if result:
            _section_performance(result)
        else:
            st.info("Run a backtest in the **Configuration** tab to see results here.")

    with tab3:
        result = st.session_state.get("backtest_results")
        config = st.session_state.get("_bt_config", {})
        if result:
            _section_analysis(result, config)
        else:
            st.info("Run a backtest in the **Configuration** tab to see analysis here.")

    with tab4:
        _section_comparison()
