"""
src/components/portfolio_overview.py
--------------------------------------
Page 2 – monitor overall portfolio health, allocation, and rebalancing
opportunities.

Components rendered:
* Key KPI metric cards (total value, cost basis, P&L, # holdings, largest pos.)
* Holdings table (sortable, colour-coded P&L)
* Allocation pie charts (by symbol & by sector)
* Concentration alerts
* Rebalancing suggestions table
* Portfolio health score
* Add holding form (expander)
* Remove / sell form
* Export CSV button
"""

from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

import portfolio_manager as pm

from .utils import format_currency, format_large_number, format_pct


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_current_prices(portfolio: dict) -> Dict[str, float]:
    """Attempt to fetch live prices for all holdings."""
    try:
        import data_fetcher as df_mod  # lazy import – avoids cost if unused
    except ImportError:
        return {}

    prices: Dict[str, float] = {}
    for sym in portfolio.get("holdings", {}):
        try:
            ticker = df_mod.get_ticker(sym)
            fi = ticker.fast_info
            price = float(fi.get("last_price") or fi.get("regularMarketPrice") or 0)
            if price > 0:
                prices[sym] = price
        except Exception:
            continue
    return prices


def _holdings_dataframe(alloc: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready holdings DataFrame."""
    if alloc.empty:
        return pd.DataFrame()

    display = alloc.copy()
    rename_map = {
        "symbol": "Symbol",
        "shares": "Shares",
        "avg_cost": "Avg Cost",
        "current_price": "Current Price",
        "market_value": "Market Value",
        "cost_basis": "Cost Basis",
        "unrealized_pnl": "Unrealized P&L",
        "pnl_pct": "P&L %",
        "weight": "Weight",
        "sector": "Sector",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

    # Format money columns
    for col in ["Avg Cost", "Current Price", "Market Value", "Cost Basis", "Unrealized P&L"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda v: format_currency(v) if pd.notna(v) and v is not None else "N/A"
            )

    # Format percentage columns
    for col in ["P&L %", "Weight"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda v: format_pct(v) if pd.notna(v) and v is not None else "N/A"
            )

    return display


# ---------------------------------------------------------------------------
# Public page function
# ---------------------------------------------------------------------------


def page_portfolio_overview() -> None:
    """Render the Portfolio Overview page."""
    st.title("💼 Portfolio Overview")
    st.caption("Monitor portfolio health, allocation, and rebalancing opportunities.")

    portfolio = st.session_state.portfolio

    # --- Refresh prices button ---
    col_refresh, col_export = st.columns([1, 4])
    refresh_btn = col_refresh.button("🔄 Refresh Prices")

    if portfolio.get("holdings") == {}:
        st.info("No holdings in your portfolio yet.  Use the sidebar to add positions.")
        _render_add_holding_form(portfolio)
        return

    # Fetch prices (use cached if not refreshing)
    prices_key = "portfolio_prices"
    if refresh_btn or prices_key not in st.session_state:
        with st.spinner("Fetching current prices …"):
            st.session_state[prices_key] = _get_current_prices(portfolio)
    prices = st.session_state.get(prices_key, {})

    summary = pm.get_portfolio_summary(portfolio, prices)
    alloc: pd.DataFrame = summary["allocation_df"]

    # --- KPI cards ---
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Value", format_currency(summary["total_value"]))
    k2.metric("Cost Basis", format_currency(summary["cost_basis"]))
    pnl = summary["total_pnl"]
    pnl_pct = summary["pnl_pct"]
    k3.metric(
        "Unrealized P&L",
        format_currency(pnl),
        delta=format_pct(pnl_pct),
        delta_color="normal",
    )
    k4.metric("Cash", format_currency(summary["cash"]))
    k5.metric("Holdings", str(summary["num_holdings"]))

    # Largest position
    largest_ticker = ""
    largest_weight = 0.0
    if not alloc.empty and "weight" in alloc.columns:
        valid = alloc.dropna(subset=["weight"])
        if not valid.empty:
            idx = valid["weight"].idxmax()
            largest_ticker = valid.loc[idx, "symbol"]
            largest_weight = valid.loc[idx, "weight"]
    k6.metric(
        "Largest Position",
        f"{largest_ticker} ({format_pct(largest_weight)})" if largest_ticker else "N/A",
    )

    st.markdown("---")

    # --- Holdings table ---
    st.subheader("Holdings")
    display_df = _holdings_dataframe(alloc)
    if not display_df.empty:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Export CSV
        csv_buffer = io.StringIO()
        alloc.to_csv(csv_buffer, index=False)
        col_export.download_button(
            "📥 Export CSV",
            data=csv_buffer.getvalue(),
            file_name="portfolio_holdings.csv",
            mime="text/csv",
        )
    else:
        st.info("Allocation data unavailable – could not fetch current prices.")

    st.markdown("---")

    # --- Allocation charts ---
    st.subheader("Allocation")
    chart_col1, chart_col2 = st.columns(2)

    valid_alloc = alloc[alloc["market_value"].notna() & (alloc["market_value"] > 0)] if not alloc.empty else pd.DataFrame()
    if not valid_alloc.empty:
        fig_sym = px.pie(
            valid_alloc,
            names="symbol",
            values="market_value",
            title="By Symbol",
            hole=0.45,
        )
        fig_sym.update_traces(textposition="outside", textinfo="percent+label")
        fig_sym.update_layout(height=350, showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
        chart_col1.plotly_chart(fig_sym, use_container_width=True)

    if summary.get("sector_allocation"):
        sec_df = pd.DataFrame(
            list(summary["sector_allocation"].items()),
            columns=["Sector", "Weight"],
        )
        fig_sec = px.pie(
            sec_df,
            names="Sector",
            values="Weight",
            title="By Sector",
            hole=0.45,
        )
        fig_sec.update_traces(textposition="outside", textinfo="percent+label")
        fig_sec.update_layout(height=350, showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20})
        chart_col2.plotly_chart(fig_sec, use_container_width=True)

    st.markdown("---")

    # --- Concentration alerts ---
    conc_threshold = st.session_state.get("concentration_threshold", 0.20)
    alerts = pm.concentration_alerts(alloc, threshold=conc_threshold)
    if alerts:
        st.subheader("⚠️ Concentration Alerts")
        for alert in alerts:
            sev = "HIGH" if alert["weight"] > 0.35 else "MEDIUM"
            st.warning(f"**[{sev}]** {alert['message']}  \nConsider trimming this position to reduce concentration risk.")

    # --- Portfolio health score ---
    st.subheader("Portfolio Health")
    n = len(portfolio["holdings"])
    if n > 0 and not alloc.empty and "weight" in alloc.columns:
        weights = alloc["weight"].dropna().values
        import numpy as np

        herfindahl = float(np.sum(weights ** 2)) if len(weights) else 1.0
        diversification = round(1.0 - herfindahl, 3)
        # Simple composite: diversification * 100 capped at 100
        health_score = min(100.0, diversification * 100 * (1 + 1 / n))
        h1, h2, h3 = st.columns(3)
        h1.metric("Health Score", f"{health_score:.0f}/100")
        h2.metric("Diversification", f"{diversification:.3f}")
        h3.metric("Concentration Risk", "HIGH" if herfindahl > 0.25 else "LOW")

    st.markdown("---")

    # --- Rebalancing suggestions ---
    st.subheader("Rebalancing Suggestions")
    st.caption("Suggestions are based on an equal-weight target by default.")
    with st.form("rebalance_form"):
        equal_weight = round(1.0 / n, 4) if n > 0 else 0.0
        target_weights = {s: equal_weight for s in portfolio["holdings"]}
        if st.form_submit_button("Generate Equal-Weight Suggestions"):
            rebal_df = pm.rebalancing_suggestions(alloc, target_weights)
            if not rebal_df.empty:
                st.dataframe(rebal_df, use_container_width=True, hide_index=True)
            else:
                st.info("Portfolio is already well-balanced.")

    st.markdown("---")

    # --- Remove holding ---
    st.subheader("Sell / Remove Holding")
    with st.form("remove_holding_form"):
        symbols_list = sorted(portfolio["holdings"].keys())
        rem_sym = st.selectbox("Symbol", symbols_list)
        rem_col1, rem_col2 = st.columns(2)
        rem_shares = rem_col1.number_input("Shares to sell (0 = all)", min_value=0.0, value=0.0)
        rem_price = rem_col2.number_input("Execution price ($)", min_value=0.0, value=0.0)
        if st.form_submit_button("Sell", type="secondary"):
            pm.remove_holding(
                portfolio,
                rem_sym,
                shares=rem_shares if rem_shares > 0 else None,
                price=rem_price if rem_price > 0 else None,
            )
            pm.save_portfolio(portfolio)
            st.success(f"Sold **{rem_sym}** from portfolio.")
            st.rerun()

    st.markdown("---")

    # --- Add holding ---
    _render_add_holding_form(portfolio)


def _render_add_holding_form(portfolio: dict) -> None:
    """Render the Add Holding expander form."""
    with st.expander("➕ Add New Holding", expanded=False):
        with st.form("add_holding_form_po"):
            ah_col1, ah_col2 = st.columns(2)
            sym = ah_col1.text_input("Ticker Symbol", placeholder="e.g. AAPL")
            sector = ah_col2.text_input("Sector", placeholder="e.g. Technology")

            ah_col3, ah_col4 = st.columns(2)
            shares = ah_col3.number_input("Shares", min_value=0.001, value=1.0, step=0.001)
            cost = ah_col4.number_input("Purchase Price ($)", min_value=0.01, value=100.0)

            purchase_date = st.date_input("Purchase Date")
            notes = st.text_area("Notes (optional)", height=60)

            if st.form_submit_button("Add to Portfolio", type="primary"):
                if sym.strip():
                    pm.add_holding(
                        portfolio,
                        sym.strip().upper(),
                        shares,
                        cost,
                        sector=sector or "Unknown",
                        thesis=notes,
                    )
                    pm.save_portfolio(portfolio)
                    st.success(f"Added **{sym.upper()}** to portfolio.")
                    st.rerun()
                else:
                    st.error("Please enter a ticker symbol.")
