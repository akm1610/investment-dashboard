"""
src/components/sidebar.py
--------------------------
Shared sidebar: navigation, quick-add holding form, and settings panel.

Intended to be called from ``src/app.py`` inside a ``with st.sidebar:`` block.
"""

from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

import portfolio_manager as pm


def render_sidebar() -> None:
    """Render the sidebar content (title, quick-add form, settings)."""
    with st.sidebar:
        st.title("📊 Investment Dashboard")
        st.caption("Long-Term Investment Analysis System")

        st.markdown("---")

        # --- Quick Add Holding ---
        st.subheader("Quick Add Holding")
        with st.form("sidebar_add_holding"):
            sym = st.text_input("Ticker", placeholder="e.g. AAPL")
            s_col1, s_col2 = st.columns(2)
            shares = s_col1.number_input("Shares", min_value=0.001, value=1.0, step=0.001)
            cost = s_col2.number_input("Avg Cost ($)", min_value=0.01, value=100.0)
            sector = st.text_input("Sector", placeholder="e.g. Technology")
            thesis = st.text_area("Thesis (optional)", height=60)
            if st.form_submit_button("➕ Add"):
                sym_clean = sym.strip().upper()
                if sym_clean:
                    portfolio = st.session_state.portfolio
                    pm.add_holding(
                        portfolio,
                        sym_clean,
                        shares,
                        cost,
                        sector=sector or "Unknown",
                        thesis=thesis,
                    )
                    pm.save_portfolio(portfolio)
                    st.success(f"Added {sym_clean}")
                    st.rerun()
                else:
                    st.error("Enter a ticker symbol.")

        st.markdown("---")

        # --- Settings ---
        with st.expander("⚙️ Settings", expanded=False):
            # Concentration threshold
            threshold = st.slider(
                "Concentration Alert Threshold",
                min_value=0.05,
                max_value=0.50,
                value=st.session_state.get("concentration_threshold", 0.20),
                step=0.05,
                format="%.0f%%",
                help="Positions exceeding this weight will trigger a concentration alert.",
            )
            st.session_state["concentration_threshold"] = threshold

            # Portfolio file path
            portfolio_path = st.text_input(
                "Portfolio File",
                value=st.session_state.get("portfolio_path", "portfolio.json"),
            )
            if portfolio_path != st.session_state.get("portfolio_path", "portfolio.json"):
                st.session_state["portfolio_path"] = portfolio_path
                st.session_state.portfolio = pm.load_portfolio(portfolio_path)
                st.rerun()

            # Currency selector (cosmetic – formatting only)
            currency = st.selectbox(
                "Currency",
                options=["USD ($)", "EUR (€)", "GBP (£)"],
                index=0,
                help="Affects how monetary values are displayed.",
            )
            st.session_state["currency"] = currency

        st.markdown("---")

        # --- About ---
        with st.expander("ℹ️ About", expanded=False):
            st.markdown(
                """
                **Investment Dashboard** v1.0.0

                A long-term investment analysis tool integrating:
                - Data fetching (yfinance)
                - Financial scoring (stability, quality, valuation)
                - Portfolio management & rebalancing
                - Investment thesis journal

                *Not financial advice.*
                """
            )
