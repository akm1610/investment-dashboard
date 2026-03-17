"""
src/components/pretrade_checklist.py
--------------------------------------
Page 3 – enforce disciplined decision-making before adding new positions.

Components rendered:
* Risk warning banner
* Ticker input & run button
* 7-item PASS / WARN / FAIL checklist
* Pass/fail decision gate (manual override for warnings)
* Investment thesis input (with structured prompts)
* Conviction level & target holding period selectors
* Concentration check against current portfolio
* Action buttons: Save Thesis, Add to Watchlist, Proceed to Trade, Cancel
"""

from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

import analysis_engine as ae
import data_fetcher as df_mod
import portfolio_manager as pm

from .utils import display_checklist, format_pct, score_color, status_emoji


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_and_analyze(symbol: str) -> dict:
    """Fetch & analyse a symbol; results are cached for 1 h."""
    fundamentals = df_mod.fetch_all_fundamentals(symbol)
    analysis = ae.analyze(fundamentals)
    analysis["fundamentals"] = fundamentals
    return analysis


def _concentration_check(symbol: str, portfolio: dict) -> str | None:
    """Return a warning string if adding *symbol* would cause concentration, else None."""
    holdings = portfolio.get("holdings", {})
    n = len(holdings)
    if n == 0:
        return None
    # Assume equal weighting for a rough estimate
    estimated_weight_after = 1.0 / (n + 1)
    threshold = st.session_state.get("concentration_threshold", 0.20)
    if estimated_weight_after > threshold:
        return None  # new position would be proportionally small
    # Check if symbol already in portfolio
    if symbol in holdings:
        return (
            f"⚠️ **{symbol}** is already in your portfolio. "
            f"Adding more increases concentration."
        )
    return None


# ---------------------------------------------------------------------------
# Public page function
# ---------------------------------------------------------------------------


def page_pretrade_checklist() -> None:
    """Render the Pre-Trade Checklist page."""
    st.title("✅ Pre-Trade Checklist")
    st.write("Complete this checklist **before** trading any stock.")

    st.warning(
        "⚠️ **Risk Warning:** Investing in individual stocks involves significant risk. "
        "Past performance is not indicative of future results. "
        "Be aware of concentration and sector exposure before adding new positions.",
        icon="⚠️",
    )

    # --- Ticker selection ---
    portfolio = st.session_state.portfolio
    holdings = list(portfolio.get("holdings", {}).keys())

    col_sym, col_btn = st.columns([3, 1])
    with col_sym:
        use_portfolio = st.toggle("Select from portfolio", value=False, key="ptc_toggle")
        if use_portfolio and holdings:
            symbol_raw = st.selectbox("Select ticker", holdings, key="ptc_select")
        else:
            symbol_raw = st.text_input(
                "Enter ticker symbol",
                placeholder="e.g. NVDA",
                key="ptc_symbol_input",
            )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Checklist", type="primary", use_container_width=True)

    symbol = (symbol_raw or "").upper().strip()

    if not symbol:
        st.info("Enter a ticker symbol above and click **Run Checklist**.")
        return

    if not run_btn and f"ptc_result_{symbol}" not in st.session_state:
        st.info("Click **Run Checklist** to analyse this symbol.")
        return

    # --- Fetch & analyse ---
    with st.spinner(f"Analysing {symbol} …"):
        try:
            result = _fetch_and_analyze(symbol)
            st.session_state[f"ptc_result_{symbol}"] = result
        except Exception as exc:
            st.error(f"Could not fetch data for **{symbol}**: {exc}")
            return

    result = st.session_state.get(f"ptc_result_{symbol}", {})
    ratios: dict = result.get("ratios", {})
    scores: dict = result.get("scores", {})

    st.markdown("---")

    # --- Investment thesis input (before checklist so thesis affects item 7) ---
    st.subheader("📝 Investment Thesis")
    st.caption("Answering these questions will improve checklist completeness.")

    thesis_placeholder = (
        "Why am I buying this stock?\n"
        "What are the key catalysts?\n"
        "What could go wrong?\n"
        "Time horizon?"
    )
    thesis = st.text_area(
        "Your thesis",
        height=120,
        key=f"ptc_thesis_{symbol}",
        placeholder=thesis_placeholder,
    )

    col_conv, col_horizon = st.columns(2)
    conviction = col_conv.select_slider(
        "Conviction level",
        options=["LOW", "MEDIUM", "HIGH"],
        value="MEDIUM",
        key=f"ptc_conviction_{symbol}",
    )
    holding_period = col_horizon.number_input(
        "Target holding period (years)",
        min_value=0.5,
        max_value=30.0,
        value=3.0,
        step=0.5,
        key=f"ptc_holding_{symbol}",
    )

    st.markdown("---")

    # --- 7-item checklist ---
    st.subheader(f"Checklist for {symbol}")
    checklist = ae.pretrade_checklist(ratios, scores, thesis)

    passed = sum(1 for i in checklist if i["status"] == "pass")
    warned = sum(1 for i in checklist if i["status"] == "warn")
    failed = sum(1 for i in checklist if i["status"] == "fail")
    total = len(checklist)

    progress_val = passed / total if total else 0.0
    st.progress(progress_val, text=f"{passed}/{total} passed · {warned} warnings · {failed} failed")

    display_checklist(checklist)

    # --- Decision gate ---
    st.markdown("---")
    st.subheader("Decision Gate")
    if failed > 0:
        st.error(
            f"❌ **{failed} item(s) FAILED.** "
            "This stock does not meet the minimum investment criteria. "
            "Proceeding requires manual override."
        )
        override = st.checkbox(
            "I understand the risks and want to override the failed checks.",
            key=f"ptc_override_{symbol}",
        )
        can_proceed = override
    elif warned > 0:
        st.warning(
            f"⚠️ **{warned} item(s) have warnings.** "
            "Review the details before proceeding."
        )
        can_proceed = True
    else:
        st.success("✅ All checklist items PASSED. This stock meets all criteria.")
        can_proceed = True

    # Score summary
    score_cols = st.columns(4)
    for col, (key, label) in zip(
        score_cols,
        [("composite", "Composite"), ("stability", "Stability"), ("quality", "Quality"), ("valuation", "Valuation")],
    ):
        val = scores.get(key, 0)
        col.metric(label, f"{val}/100")

    st.markdown("---")

    # --- Concentration check ---
    conc_warning = _concentration_check(symbol, portfolio)
    if conc_warning:
        st.subheader("📊 Concentration Check")
        st.warning(conc_warning)
        n = len(portfolio.get("holdings", {}))
        suggested_weight = 1.0 / (n + 1) if n >= 0 else 1.0
        st.caption(
            f"Suggested max position weight: **{format_pct(suggested_weight)}** "
            f"(assuming equal-weight portfolio with {n + 1} holdings)."
        )

    # --- Action buttons ---
    st.subheader("Actions")
    act_col1, act_col2, act_col3, act_col4 = st.columns(4)

    # Save thesis
    if act_col1.button("💾 Save Thesis", use_container_width=True):
        if thesis.strip():
            pm.add_journal_entry(portfolio, symbol, thesis, tags=[conviction.lower(), f"{holding_period:.0f}yr"])
            pm.save_portfolio(portfolio)
            st.success(f"Thesis for **{symbol}** saved to Investment Journal.")
        else:
            st.warning("Please write a thesis before saving.")

    # Add to watchlist (stored as a journal tag)
    if act_col2.button("👁️ Add to Watchlist", use_container_width=True):
        pm.add_journal_entry(portfolio, symbol, thesis or "On watchlist.", tags=["watchlist"])
        pm.save_portfolio(portfolio)
        st.success(f"**{symbol}** added to watchlist (tagged in journal).")

    # Proceed to trade
    with act_col3:
        if can_proceed:
            with st.expander("🚀 Proceed to Trade", expanded=False):
                with st.form(f"ptc_trade_form_{symbol}"):
                    t_shares = st.number_input("Shares", min_value=0.001, value=1.0, step=0.001)
                    t_price = st.number_input("Purchase Price ($)", min_value=0.01, value=100.0)
                    t_sector = st.text_input("Sector", placeholder="e.g. Technology")
                    if st.form_submit_button("Confirm Trade", type="primary"):
                        pm.add_holding(
                            portfolio,
                            symbol,
                            t_shares,
                            t_price,
                            sector=t_sector or "Unknown",
                            thesis=thesis,
                        )
                        pm.save_portfolio(portfolio)
                        st.success(f"✅ {symbol} added to portfolio!")
        else:
            st.button("🚀 Proceed to Trade", disabled=True, use_container_width=True)

    # Cancel
    if act_col4.button("❌ Cancel", use_container_width=True):
        # Clear session state for this symbol's checklist run
        for key in [f"ptc_result_{symbol}", f"ptc_thesis_{symbol}"]:
            st.session_state.pop(key, None)
        st.rerun()
