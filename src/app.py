"""
src/app.py
----------
Main entry point for the Long-Term Investment Analysis Dashboard.

Run with:
    streamlit run src/app.py

Architecture
------------
* Multi-page app using ``st.navigation()`` and ``st.Page()``
* Session state for portfolio persistence and analysis caching
* Sidebar for navigation, quick-add form, and settings
* Four pages:
  1. Company Analysis   – deep-dive into a single ticker
  2. Portfolio Overview – holdings, allocation, rebalancing
  3. Pre-Trade Checklist – 7-item decision gate
  4. Investment Journal  – timestamped thesis records
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Path setup – ensure root-level modules (analysis_engine, portfolio_manager,
# data_fetcher) are importable whether the app is launched from the repo root
# or from the src/ directory.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _path in [_ROOT, _HERE]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import logging

import streamlit as st

import portfolio_manager as pm

from components.company_analysis import page_company_analysis
from components.investment_journal import page_investment_journal
from components.portfolio_overview import page_portfolio_overview
from components.pretrade_checklist import page_pretrade_checklist
from components.sidebar import render_sidebar

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Investment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise required session-state keys with safe defaults."""
    if "portfolio" not in st.session_state:
        path = st.session_state.get("portfolio_path", "portfolio.json")
        st.session_state.portfolio = pm.load_portfolio(path)

    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}

    if "concentration_threshold" not in st.session_state:
        st.session_state.concentration_threshold = 0.20

    if "portfolio_path" not in st.session_state:
        st.session_state.portfolio_path = "portfolio.json"

    if "currency" not in st.session_state:
        st.session_state.currency = "USD ($)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: initialise state, render sidebar, and run the selected page."""
    _init_session_state()

    # Sidebar (navigation + quick-add + settings)
    render_sidebar()

    # Multi-page navigation using st.navigation / st.Page
    pages = [
        st.Page(page_company_analysis, title="Company Analysis", icon="🔍"),
        st.Page(page_portfolio_overview, title="Portfolio Overview", icon="💼"),
        st.Page(page_pretrade_checklist, title="Pre-Trade Checklist", icon="✅"),
        st.Page(page_investment_journal, title="Investment Journal", icon="📓"),
    ]

    page = st.navigation(pages)
    page.run()


if __name__ == "__main__":
    main()
