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
* Seven pages:
  1. Company Analysis        – deep-dive into a single ticker (incl. sentiment panel)
  2. Portfolio Overview      – holdings, allocation, rebalancing
  3. Pre-Trade Checklist     – 7-item decision gate
  4. Investment Journal      – timestamped thesis records
  5. Risk & Recommendations  – risk profile assessment + curated watchlists
  6. Strategy Backtesting    – historical strategy testing and validation
  7. Sentiment Analysis      – real-time news & market sentiment for any ticker
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
# Insert in reverse order so _ROOT ends up at position 0 (highest priority).
# This prevents src/analysis_engine.py (a re-export shim) from shadowing the
# root-level analysis_engine.py, which would cause a circular-import error
# and result in a blank Streamlit page.
for _path in [_HERE, _ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import logging

import streamlit as st

_import_error: str | None = None
try:
    import portfolio_manager as pm

    from components.company_analysis import page_company_analysis
    from components.investment_journal import page_investment_journal
    from components.portfolio_overview import page_portfolio_overview
    from components.pretrade_checklist import page_pretrade_checklist
    from components.risk_recommendations import page_risk_recommendations
    from components.backtesting import page_backtesting
    from components.sentiment_analysis import page_sentiment_analysis
    from components.sidebar import render_sidebar
except Exception as _e:
    _import_error = str(_e)
    logging.basicConfig(level=logging.WARNING)
    logging.exception("Failed to import dashboard components: %s", _e)

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

    if "risk_profile" not in st.session_state:
        st.session_state.risk_profile = "moderate"

    if "risk_profile_result" not in st.session_state:
        st.session_state.risk_profile_result = None

    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = {}

    if "bt_comparisons" not in st.session_state:
        st.session_state.bt_comparisons = {}

    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []

    if "watchlists" not in st.session_state:
        st.session_state.watchlists = {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: initialise state, render sidebar, and run the selected page."""
    if _import_error:
        st.error(
            f"⚠️ Failed to load dashboard components:\n\n```\n{_import_error}\n```\n\n"
            "Please check the terminal for the full traceback."
        )
        st.info(
            "**Common fixes:**\n"
            "- Run `pip install -r requirements.txt` to ensure all dependencies are installed.\n"
            "- Run the app from the repository root: `streamlit run src/app.py`"
        )
        return

    try:
        _init_session_state()
    except Exception as exc:
        st.error(f"⚠️ Failed to initialise session state: {exc}")
        st.exception(exc)
        return

    # Sidebar (navigation + quick-add + settings)
    try:
        render_sidebar()
    except Exception as exc:
        st.sidebar.error(f"Sidebar error: {exc}")

    # Multi-page navigation using st.navigation / st.Page
    try:
        pages = [
            st.Page(page_company_analysis, title="Company Analysis", icon="🔍"),
            st.Page(page_portfolio_overview, title="Portfolio Overview", icon="💼"),
            st.Page(page_pretrade_checklist, title="Pre-Trade Checklist", icon="✅"),
            st.Page(page_investment_journal, title="Investment Journal", icon="📓"),
            st.Page(page_risk_recommendations, title="Risk & Recommendations", icon="🛡️"),
            st.Page(page_backtesting, title="Strategy Backtesting", icon="📈"),
            st.Page(page_sentiment_analysis, title="Sentiment Analysis", icon="📰"),
        ]

        page = st.navigation(pages)
        page.run()
    except Exception as exc:
        st.error(f"⚠️ Page rendering error: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
