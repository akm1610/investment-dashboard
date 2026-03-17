"""
app.py
------
Streamlit dashboard integrating data_fetcher, analysis_engine,
and portfolio_manager for long-term investment analysis.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis_engine as ae
import data_fetcher as df_mod
import portfolio_manager as pm

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Long-Term Investment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pm.load_portfolio()

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}


# ---------------------------------------------------------------------------
# Sidebar – navigation & portfolio actions
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Investment Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Company Analysis",
        "Portfolio Overview",
        "Pre-Trade Checklist",
        "Investment Journal",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Add Holding")
with st.sidebar.form("add_holding_form"):
    sym_input = st.text_input("Symbol", placeholder="e.g. AAPL")
    shares_input = st.number_input("Shares", min_value=0.01, value=1.0)
    cost_input = st.number_input("Avg Cost ($)", min_value=0.01, value=100.0)
    sector_input = st.text_input("Sector", placeholder="e.g. Technology")
    thesis_input = st.text_area("Thesis (optional)", height=80)
    if st.form_submit_button("Add"):
        if sym_input.strip():
            pm.add_holding(
                st.session_state.portfolio,
                sym_input.strip(),
                shares_input,
                cost_input,
                sector=sector_input or "Unknown",
                thesis=thesis_input,
            )
            pm.save_portfolio(st.session_state.portfolio)
            st.sidebar.success(f"Added {sym_input.upper()}")
            st.rerun()


# ---------------------------------------------------------------------------
# Helper: get/cache analysis for a symbol
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_and_analyze(symbol: str) -> dict:
    fundamentals = df_mod.fetch_all_fundamentals(symbol)
    analysis = ae.analyze(fundamentals)
    analysis["fundamentals"] = fundamentals
    return analysis


def _get_current_price(symbol: str) -> float | None:
    try:
        ticker = df_mod.get_ticker(symbol)
        fi = ticker.fast_info
        return float(fi.get("last_price") or fi.get("regularMarketPrice") or 0)
    except Exception:
        return None


def _current_prices_for_portfolio() -> dict[str, float]:
    prices = {}
    for sym in st.session_state.portfolio.get("holdings", {}):
        p = _get_current_price(sym)
        if p:
            prices[sym] = p
    return prices


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _score_color(score: float) -> str:
    if score >= 70:
        return "green"
    if score >= 50:
        return "orange"
    return "red"


def _status_emoji(status: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(status, "❓")


# ---------------------------------------------------------------------------
# Page 1: Company Analysis
# ---------------------------------------------------------------------------
def page_company_analysis():
    st.title("🔍 Company Analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter ticker symbol", placeholder="e.g. MSFT", key="analysis_symbol").upper().strip()
    with col2:
        analyse_btn = st.button("Analyse", type="primary")

    if not symbol:
        st.info("Enter a ticker symbol to begin analysis.")
        return

    if analyse_btn or symbol:
        with st.spinner(f"Fetching data for {symbol} …"):
            try:
                result = _fetch_and_analyze(symbol)
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")
                return

        ratios = result["ratios"]
        scores = result["scores"]
        fundamentals = result["fundamentals"]
        stats = fundamentals.get("key_stats", {})

        # --- Company header ---
        company_name = stats.get("longName") or stats.get("shortName") or symbol
        sector = stats.get("sector", "N/A")
        industry = stats.get("industry", "N/A")
        market_cap = stats.get("marketCap")
        mc_str = f"${market_cap / 1e9:.1f}B" if market_cap else "N/A"

        st.header(company_name)
        st.caption(f"{sector} · {industry} · Market Cap: {mc_str}")

        # --- Score gauges ---
        st.subheader("Investment Scores")
        c1, c2, c3, c4 = st.columns(4)
        for col, (key, label) in zip(
            [c1, c2, c3, c4],
            [
                ("composite", "Composite"),
                ("stability", "Stability"),
                ("quality", "Quality"),
                ("valuation", "Valuation"),
            ],
        ):
            val = scores.get(key, 0)
            color = _score_color(val)
            col.metric(label, f"{val}/100")

        # Gauge chart for composite score
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=scores.get("composite", 0),
                title={"text": "Composite Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 40], "color": "#ff4444"},
                        {"range": [40, 60], "color": "#ffaa00"},
                        {"range": [60, 80], "color": "#88cc00"},
                        {"range": [80, 100], "color": "#00cc44"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- Key ratios table ---
        st.subheader("Key Ratios")
        ratio_display = {
            "P/E Ratio": ratios.get("pe_ratio"),
            "P/B Ratio": ratios.get("pb_ratio"),
            "EV/EBITDA": ratios.get("ev_ebitda"),
            "PEG Ratio": ratios.get("peg_ratio"),
            "Gross Margin": f"{ratios.get('gross_margin', 0) or 0:.1%}" if ratios.get("gross_margin") is not None else "N/A",
            "Operating Margin": f"{ratios.get('operating_margin', 0) or 0:.1%}" if ratios.get("operating_margin") is not None else "N/A",
            "Net Margin": f"{ratios.get('net_margin', 0) or 0:.1%}" if ratios.get("net_margin") is not None else "N/A",
            "ROE": f"{ratios.get('roe', 0) or 0:.1%}" if ratios.get("roe") is not None else "N/A",
            "ROA": f"{ratios.get('roa', 0) or 0:.1%}" if ratios.get("roa") is not None else "N/A",
            "Current Ratio": ratios.get("current_ratio"),
            "Debt/Equity": ratios.get("debt_to_equity"),
            "Debt/EBITDA": ratios.get("debt_to_ebitda"),
            "FCF Yield": f"{ratios.get('fcf_yield', 0) or 0:.1%}" if ratios.get("fcf_yield") is not None else "N/A",
        }

        ratio_df = pd.DataFrame(
            [(k, v if isinstance(v, str) else (f"{v:.2f}" if v is not None else "N/A"))
             for k, v in ratio_display.items()],
            columns=["Metric", "Value"],
        )
        col_r1, col_r2 = st.columns(2)
        half = len(ratio_df) // 2
        col_r1.dataframe(ratio_df.iloc[:half], use_container_width=True, hide_index=True)
        col_r2.dataframe(ratio_df.iloc[half:], use_container_width=True, hide_index=True)

        # --- Price chart ---
        st.subheader("Price History (5 Years)")
        price_df = fundamentals.get("price_history", pd.DataFrame())
        if not price_df.empty and "Close" in price_df.columns:
            fig_price = px.line(
                price_df,
                y="Close",
                title=f"{symbol} Closing Price",
                labels={"Close": "Price (USD)", "Date": ""},
            )
            fig_price.update_layout(height=350)
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning("Price data unavailable.")


# ---------------------------------------------------------------------------
# Page 2: Portfolio Overview
# ---------------------------------------------------------------------------
def page_portfolio_overview():
    st.title("💼 Portfolio Overview")

    portfolio = st.session_state.portfolio
    if not portfolio["holdings"]:
        st.info("No holdings yet. Add positions via the sidebar.")
        return

    with st.spinner("Fetching current prices …"):
        prices = _current_prices_for_portfolio()

    summary = pm.get_portfolio_summary(portfolio, prices)
    alloc = summary["allocation_df"]

    # --- Summary KPIs ---
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Value", f"${summary['total_value']:,.2f}")
    k2.metric("Cost Basis", f"${summary['cost_basis']:,.2f}")
    pnl = summary["total_pnl"]
    k3.metric("Unrealized P&L", f"${pnl:,.2f}", delta=f"{summary['pnl_pct']:.1%}")
    k4.metric("Cash", f"${summary['cash']:,.2f}")
    k5.metric("# Holdings", summary["num_holdings"])

    st.markdown("---")

    # --- Allocation table ---
    st.subheader("Holdings")
    display_cols = ["symbol", "shares", "avg_cost", "current_price",
                    "market_value", "cost_basis", "unrealized_pnl", "pnl_pct", "weight", "sector"]
    display_df = alloc[[c for c in display_cols if c in alloc.columns]].copy()
    for pct_col in ["pnl_pct", "weight"]:
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].apply(
                lambda v: f"{v:.1%}" if pd.notna(v) and v is not None else "N/A"
            )
    for money_col in ["avg_cost", "current_price", "market_value", "cost_basis", "unrealized_pnl"]:
        if money_col in display_df.columns:
            display_df[money_col] = display_df[money_col].apply(
                lambda v: f"${v:,.2f}" if pd.notna(v) and v is not None else "N/A"
            )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Concentration alerts ---
    alerts = pm.concentration_alerts(alloc)
    if alerts:
        st.subheader("⚠️ Concentration Alerts")
        for a in alerts:
            st.warning(a["message"])

    # --- Pie chart: allocation by symbol ---
    col_p1, col_p2 = st.columns(2)
    valid_alloc = alloc[alloc["market_value"].notna() & (alloc["market_value"] > 0)]
    if not valid_alloc.empty:
        fig_pie = px.pie(
            valid_alloc,
            names="symbol",
            values="market_value",
            title="Portfolio Allocation by Symbol",
            hole=0.4,
        )
        col_p1.plotly_chart(fig_pie, use_container_width=True)

    # Sector allocation
    if summary["sector_allocation"]:
        sector_df = pd.DataFrame(
            list(summary["sector_allocation"].items()),
            columns=["Sector", "Weight"],
        )
        fig_sector = px.pie(
            sector_df,
            names="Sector",
            values="Weight",
            title="Sector Allocation",
            hole=0.4,
        )
        col_p2.plotly_chart(fig_sector, use_container_width=True)

    # --- Remove holding ---
    st.markdown("---")
    st.subheader("Remove a Holding")
    with st.form("remove_form"):
        symbols_list = list(portfolio["holdings"].keys())
        if symbols_list:
            rem_sym = st.selectbox("Symbol", symbols_list)
            rem_shares = st.number_input("Shares to sell (0 = all)", min_value=0.0, value=0.0)
            rem_price = st.number_input("Execution price ($)", min_value=0.0, value=0.0)
            if st.form_submit_button("Sell"):
                pm.remove_holding(
                    portfolio,
                    rem_sym,
                    shares=rem_shares if rem_shares > 0 else None,
                    price=rem_price if rem_price > 0 else None,
                )
                pm.save_portfolio(portfolio)
                st.success(f"Sold {rem_sym}")
                st.rerun()

    # --- Rebalancing ---
    st.subheader("Rebalancing Suggestions")
    st.caption("Enter equal-weight targets (or leave blank to skip).")
    with st.form("rebalance_form"):
        n = len(portfolio["holdings"])
        default_weight = round(1.0 / n, 4) if n > 0 else 0.0
        target_weights = {s: default_weight for s in portfolio["holdings"]}
        if st.form_submit_button("Generate Suggestions"):
            rebal_df = pm.rebalancing_suggestions(alloc, target_weights)
            st.dataframe(rebal_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 3: Pre-Trade Checklist
# ---------------------------------------------------------------------------
def page_pretrade_checklist():
    st.title("✅ Pre-Trade Checklist")

    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Symbol to evaluate", placeholder="e.g. NVDA").upper().strip()
    with col2:
        run_btn = st.button("Run Checklist", type="primary")

    thesis = st.text_area(
        "Investment Thesis",
        height=100,
        placeholder="Why do you want to invest in this company?",
    )

    if symbol and run_btn:
        with st.spinner(f"Analysing {symbol} …"):
            try:
                result = _fetch_and_analyze(symbol)
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")
                return

        ratios = result["ratios"]
        scores = result["scores"]
        checklist = ae.pretrade_checklist(ratios, scores, thesis)

        st.subheader(f"Checklist for {symbol}")
        passed = sum(1 for i in checklist if i["status"] == "pass")
        total = len(checklist)
        st.progress(passed / total if total else 0, text=f"{passed}/{total} items passed")

        for item in checklist:
            emoji = _status_emoji(item["status"])
            st.markdown(f"{emoji} **{item['item']}**  \n{item['detail']}")

        # Score summary
        st.subheader("Scores")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Composite", f"{scores.get('composite', 'N/A')}/100")
        col_s2.metric("Stability", f"{scores.get('stability', 'N/A')}/100")
        col_s3.metric("Quality", f"{scores.get('quality', 'N/A')}/100")
        col_s4.metric("Valuation", f"{scores.get('valuation', 'N/A')}/100")

        # Option to save thesis to journal
        if thesis.strip() and st.button("💾 Save thesis to journal"):
            pm.add_journal_entry(st.session_state.portfolio, symbol, thesis)
            pm.save_portfolio(st.session_state.portfolio)
            st.success("Thesis saved to journal.")


# ---------------------------------------------------------------------------
# Page 4: Investment Journal
# ---------------------------------------------------------------------------
def page_investment_journal():
    st.title("📓 Investment Journal")

    portfolio = st.session_state.portfolio

    # Add new entry
    with st.expander("➕ New Journal Entry", expanded=False):
        with st.form("journal_form"):
            j_sym = st.text_input("Symbol")
            j_thesis = st.text_area("Thesis / Notes", height=120)
            j_tags = st.text_input("Tags (comma-separated)", placeholder="e.g. growth, moat, undervalued")
            if st.form_submit_button("Save Entry"):
                tags = [t.strip() for t in j_tags.split(",") if t.strip()]
                pm.add_journal_entry(portfolio, j_sym, j_thesis, tags)
                pm.save_portfolio(portfolio)
                st.success("Entry saved.")
                st.rerun()

    # Display entries
    journal = pm.get_journal(portfolio)
    if not journal:
        st.info("No journal entries yet.")
        return

    filter_sym = st.text_input("Filter by symbol (leave blank for all)").upper().strip()
    entries = pm.get_journal(portfolio, filter_sym if filter_sym else None)

    st.write(f"Showing {len(entries)} entries")
    for entry in reversed(entries):
        with st.expander(f"{entry['symbol']} — {entry['timestamp'][:10]}", expanded=False):
            st.write(entry["thesis"])
            if entry.get("tags"):
                st.caption("Tags: " + ", ".join(entry["tags"]))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Company Analysis":
    page_company_analysis()
elif page == "Portfolio Overview":
    page_portfolio_overview()
elif page == "Pre-Trade Checklist":
    page_pretrade_checklist()
elif page == "Investment Journal":
    page_investment_journal()
