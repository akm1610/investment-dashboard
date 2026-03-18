"""
src/components/risk_recommendations.py
---------------------------------------
Risk Profile & Recommendations page.

Sections
--------
A. Risk Profile Assessment  – 10-question 1-5 slider questionnaire,
                              real-time score & profile badge, suggested
                              allocation, save/reset actions.
B. Curated Watchlists       – Static curated watchlists with mock performance
                              metrics; filterable by risk profile suitability.
C. Top Recommendations      – Personalised ranking filtered by risk profile,
                              expandable detail cards.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

from risk_engine import RiskProfileAssessor
import portfolio_manager as pm

# ---------------------------------------------------------------------------
# Curated watchlists (static seed data)
# ---------------------------------------------------------------------------

_WATCHLISTS: List[Dict[str, Any]] = [
    {
        "name": "🤖 AI Growth Leaders",
        "description": "Growth investing in AI/ML infrastructure and software",
        "strategy": "Growth",
        "risk_level": "HIGH",
        "risk_profiles": ["aggressive"],
        "holdings_count": 24,
        "performance": {"win_rate": 0.73, "avg_return": 0.082, "sharpe": 1.20},
        "holdings": [
            {"ticker": "NVDA", "score": 87, "signal": "BUY", "confidence": 82,
             "entry_price": 875.0, "target_price": 1050.0,
             "drivers": "AI demand, earnings growth"},
            {"ticker": "MSFT", "score": 81, "signal": "BUY", "confidence": 78,
             "entry_price": 420.0, "target_price": 490.0,
             "drivers": "Azure growth, Copilot adoption"},
            {"ticker": "GOOGL", "score": 76, "signal": "BUY", "confidence": 71,
             "entry_price": 175.0, "target_price": 205.0,
             "drivers": "Search dominance, cloud growth"},
            {"ticker": "META", "score": 74, "signal": "HOLD", "confidence": 65,
             "entry_price": 510.0, "target_price": 560.0,
             "drivers": "Ad revenue, AI integration"},
        ],
    },
    {
        "name": "💰 Dividend Aristocrats",
        "description": "Companies with 25+ years of consecutive dividend growth",
        "strategy": "Income",
        "risk_level": "LOW",
        "risk_profiles": ["conservative", "moderate"],
        "holdings_count": 18,
        "performance": {"win_rate": 0.68, "avg_return": 0.045, "sharpe": 1.05},
        "holdings": [
            {"ticker": "JNJ", "score": 79, "signal": "BUY", "confidence": 74,
             "entry_price": 147.0, "target_price": 165.0,
             "drivers": "Dividend growth, healthcare stability"},
            {"ticker": "PG", "score": 75, "signal": "BUY", "confidence": 70,
             "entry_price": 160.0, "target_price": 178.0,
             "drivers": "Pricing power, brand moat"},
            {"ticker": "KO", "score": 72, "signal": "HOLD", "confidence": 66,
             "entry_price": 62.0, "target_price": 68.0,
             "drivers": "Dividend yield, global distribution"},
            {"ticker": "MMM", "score": 55, "signal": "HOLD", "confidence": 52,
             "entry_price": 95.0, "target_price": 100.0,
             "drivers": "Restructuring, spin-off value"},
        ],
    },
    {
        "name": "📈 Value Opportunities",
        "description": "Undervalued stocks with strong fundamentals",
        "strategy": "Value",
        "risk_level": "MEDIUM",
        "risk_profiles": ["conservative", "moderate", "aggressive"],
        "holdings_count": 20,
        "performance": {"win_rate": 0.64, "avg_return": 0.062, "sharpe": 0.95},
        "holdings": [
            {"ticker": "BRK.B", "score": 83, "signal": "BUY", "confidence": 80,
             "entry_price": 362.0, "target_price": 415.0,
             "drivers": "Buffett track record, insurance float"},
            {"ticker": "JPM", "score": 78, "signal": "BUY", "confidence": 75,
             "entry_price": 198.0, "target_price": 225.0,
             "drivers": "Interest rate tailwind, loan growth"},
            {"ticker": "BAC", "score": 70, "signal": "BUY", "confidence": 68,
             "entry_price": 36.0, "target_price": 43.0,
             "drivers": "Mortgage business, dividend growth"},
            {"ticker": "WFC", "score": 66, "signal": "HOLD", "confidence": 60,
             "entry_price": 58.0, "target_price": 65.0,
             "drivers": "Regulatory easing, efficiency"},
        ],
    },
    {
        "name": "🌱 ESG Leaders",
        "description": "Top-rated Environmental, Social and Governance companies",
        "strategy": "ESG",
        "risk_level": "MEDIUM",
        "risk_profiles": ["moderate", "aggressive"],
        "holdings_count": 15,
        "performance": {"win_rate": 0.61, "avg_return": 0.055, "sharpe": 0.88},
        "holdings": [
            {"ticker": "TSLA", "score": 69, "signal": "HOLD", "confidence": 58,
             "entry_price": 200.0, "target_price": 240.0,
             "drivers": "EV growth, energy storage"},
            {"ticker": "NEE", "score": 73, "signal": "BUY", "confidence": 68,
             "entry_price": 65.0, "target_price": 76.0,
             "drivers": "Renewables growth, regulated utilities"},
            {"ticker": "ENPH", "score": 64, "signal": "HOLD", "confidence": 55,
             "entry_price": 110.0, "target_price": 128.0,
             "drivers": "Solar adoption, IRA incentives"},
        ],
    },
    {
        "name": "🏥 Healthcare Innovation",
        "description": "Biotech, medtech and healthcare leaders with growth potential",
        "strategy": "Sector",
        "risk_level": "HIGH",
        "risk_profiles": ["moderate", "aggressive"],
        "holdings_count": 22,
        "performance": {"win_rate": 0.58, "avg_return": 0.071, "sharpe": 0.82},
        "holdings": [
            {"ticker": "LLY", "score": 88, "signal": "BUY", "confidence": 84,
             "entry_price": 780.0, "target_price": 920.0,
             "drivers": "GLP-1 drugs, Alzheimer's pipeline"},
            {"ticker": "ABBV", "score": 76, "signal": "BUY", "confidence": 72,
             "entry_price": 165.0, "target_price": 190.0,
             "drivers": "Immunology pipeline, Skyrizi/Rinvoq"},
            {"ticker": "UNH", "score": 80, "signal": "BUY", "confidence": 77,
             "entry_price": 520.0, "target_price": 595.0,
             "drivers": "Managed care growth, Optum expansion"},
        ],
    },
]

_RISK_LEVEL_COLOR = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
_SIGNAL_COLOR = {"BUY": "green", "HOLD": "orange", "SELL": "red"}

_ASSESSOR = RiskProfileAssessor()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _upside_pct(entry: float, target: float) -> float:
    """Return the percentage upside from entry to target price."""
    if entry <= 0:
        return 0.0
    return (target - entry) / entry


def _profile_badge(profile: str) -> str:
    badges = {
        "conservative": "🟢 Conservative",
        "moderate": "🟡 Moderate",
        "aggressive": "🔴 Aggressive",
    }
    return badges.get(profile, profile.title())


def _position_size_label(score: int, profile: str) -> str:
    """Suggest a position size based on composite score & risk profile."""
    max_sizes = {"conservative": 0.05, "moderate": 0.10, "aggressive": 0.15}
    max_size = max_sizes.get(profile, 0.10)
    if score >= 80:
        pct = max_size
    elif score >= 65:
        pct = max_size * 0.7
    else:
        pct = max_size * 0.4
    return f"{pct:.0%}"


# ---------------------------------------------------------------------------
# Section A: Risk Profile Assessment
# ---------------------------------------------------------------------------


def _section_risk_assessment() -> Optional[Dict[str, Any]]:
    """Render the 10-question risk questionnaire; return profile dict or None."""
    st.subheader("📊 Risk Profile Assessment")
    st.caption(
        "Answer all 10 questions on a 1–5 scale. "
        "Your answers determine your risk profile and personalised recommendations."
    )

    questions = _ASSESSOR.questions
    answers: List[int] = []

    with st.form("risk_profile_form"):
        for i, question in enumerate(questions, start=1):
            val = st.slider(
                f"Q{i}. {question}",
                min_value=1,
                max_value=5,
                value=st.session_state.get(f"rp_q{i}", 3),
                key=f"rp_slider_{i}",
            )
            answers.append(val)

        col_save, col_reset = st.columns([1, 1])
        save_clicked = col_save.form_submit_button("💾 Save Profile", type="primary")
        reset_clicked = col_reset.form_submit_button("🔄 Reset")

    if reset_clicked:
        for i in range(1, 11):
            st.session_state[f"rp_q{i}"] = 3
        st.rerun()

    # Compute live profile from current answers
    profile_result = _ASSESSOR.assess(answers)
    score = profile_result["risk_score"]
    profile = profile_result["profile"]
    alloc = profile_result["recommended_allocation"]

    # --- Live score display ---
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Risk Score", f"{score}/100")
    c2.metric("Profile", _profile_badge(profile))
    c3.metric("Max Position Size", f"{profile_result['max_position_size']:.0%}")

    # Profile summary card
    with st.container(border=True):
        st.markdown(f"### Your Risk Profile: **{profile.upper()}**")
        st.markdown(f"**Risk Score:** {score}/100")
        st.markdown(
            f"**Max Portfolio Volatility:** {profile_result['suggested_volatility']:.0%}"
        )
        st.markdown("**Suggested Allocation:**")
        a_col1, a_col2, a_col3 = st.columns(3)
        a_col1.metric("Stocks", f"{alloc['stocks']:.0%}")
        a_col2.metric("Bonds", f"{alloc['bonds']:.0%}")
        a_col3.metric("Alternatives", f"{alloc['alternatives']:.0%}")

    if save_clicked:
        # Persist answers and result in session state
        for i, ans in enumerate(answers, start=1):
            st.session_state[f"rp_q{i}"] = ans
        st.session_state.risk_profile = profile
        st.session_state.risk_profile_result = profile_result
        st.success(f"✅ Profile saved: **{profile.title()}** (score {score}/100)")

    return profile_result


# ---------------------------------------------------------------------------
# Section B: Curated Watchlists
# ---------------------------------------------------------------------------


def _section_watchlists(profile: str) -> None:
    """Render curated watchlists, optionally filtered by risk profile."""
    st.subheader("📋 Curated Watchlists")

    filter_profile = st.checkbox(
        f"Show only watchlists suitable for {_profile_badge(profile)} profile",
        value=True,
    )
    sort_by = st.selectbox(
        "Sort watchlists by",
        ["Sharpe Ratio", "Win Rate", "Avg Return", "Risk Level"],
        index=0,
    )

    watchlists = _WATCHLISTS
    if filter_profile:
        watchlists = [w for w in watchlists if profile in w["risk_profiles"]]

    if sort_by == "Sharpe Ratio":
        watchlists = sorted(watchlists, key=lambda w: w["performance"]["sharpe"], reverse=True)
    elif sort_by == "Win Rate":
        watchlists = sorted(watchlists, key=lambda w: w["performance"]["win_rate"], reverse=True)
    elif sort_by == "Avg Return":
        watchlists = sorted(watchlists, key=lambda w: w["performance"]["avg_return"], reverse=True)
    elif sort_by == "Risk Level":
        _order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        watchlists = sorted(watchlists, key=lambda w: _order.get(w["risk_level"], 1))

    if not watchlists:
        st.info("No watchlists match the current filter. Try unchecking the profile filter.")
        return

    for wl in watchlists:
        risk_color = _RISK_LEVEL_COLOR.get(wl["risk_level"], "grey")
        perf = wl["performance"]

        with st.expander(
            f"{wl['name']}  |  Risk: :{risk_color}[{wl['risk_level']}]"
            f"  |  Holdings: {wl['holdings_count']}",
            expanded=False,
        ):
            st.markdown(f"**Strategy:** {wl['strategy']}  ·  {wl['description']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("90-Day Win Rate", f"{perf['win_rate']:.0%}")
            m2.metric("Avg Return", f"+{perf['avg_return']:.1%}")
            m3.metric("Sharpe Ratio", f"{perf['sharpe']:.2f}")

            st.markdown("**Top Holdings:**")

            rows = []
            for h in wl["holdings"]:
                upside = _upside_pct(h["entry_price"], h["target_price"])
                rows.append(
                    {
                        "Ticker": h["ticker"],
                        "Score": h["score"],
                        "Signal": h["signal"],
                        "Confidence": f"{h['confidence']}%",
                        "Entry Price": f"${h['entry_price']:,.2f}",
                        "Target Price": f"${h['target_price']:,.2f}",
                        "Upside": f"{upside:+.1%}",
                        "Position Size": _position_size_label(h["score"], profile),
                        "Key Drivers": h["drivers"],
                    }
                )

            holdings_df = pd.DataFrame(rows)
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)

            # Action buttons per row
            st.caption("Actions: select a ticker, then choose an action.")
            tickers = [h["ticker"] for h in wl["holdings"]]
            sel_ticker = st.selectbox(
                "Select ticker",
                tickers,
                key=f"wl_sel_{wl['name']}",
            )
            act_col1, act_col2, act_col3 = st.columns(3)
            if act_col1.button("➕ Add to Portfolio", key=f"wl_add_{wl['name']}_{sel_ticker}"):
                _add_to_portfolio(sel_ticker, wl["holdings"])
            act_col2.button("🔍 View Analysis", key=f"wl_view_{wl['name']}_{sel_ticker}",
                            help="Navigate to Company Analysis page")
            act_col3.button("📈 Backtest", key=f"wl_bt_{wl['name']}_{sel_ticker}",
                            help="Navigate to Strategy Backtesting page")


def _add_to_portfolio(ticker: str, holdings: List[Dict]) -> None:
    """Add a watchlist holding to the portfolio at its entry price."""
    h = next((h for h in holdings if h["ticker"] == ticker), None)
    if h is None:
        st.error(f"Ticker {ticker} not found in watchlist.")
        return
    portfolio = st.session_state.portfolio
    pm.add_holding(
        portfolio,
        ticker,
        shares=1.0,
        avg_cost=h["entry_price"],
        sector="Unknown",
        thesis=f"Added from watchlist. Target: ${h['target_price']:,.2f}. "
               f"Drivers: {h['drivers']}",
    )
    pm.save_portfolio(portfolio)
    st.success(f"✅ Added {ticker} to portfolio at ${h['entry_price']:,.2f}")


# ---------------------------------------------------------------------------
# Section C: Top Personalised Recommendations
# ---------------------------------------------------------------------------


def _section_recommendations(profile: str) -> None:
    """Show top recommendations filtered by the assessed risk profile."""
    st.subheader("🎯 Top Recommendations for Your Profile")
    st.caption(f"Filtered for a **{profile.title()}** risk profile, ranked by composite score.")

    # Flatten all watchlist holdings, deduplicate (keep highest score)
    seen: Dict[str, Dict] = {}
    for wl in _WATCHLISTS:
        if profile not in wl["risk_profiles"]:
            continue
        for h in wl["holdings"]:
            ticker = h["ticker"]
            if ticker not in seen or h["score"] > seen[ticker]["score"]:
                seen[ticker] = {**h, "watchlist": wl["name"], "risk_level": wl["risk_level"]}

    if not seen:
        st.info("No recommendations available for the current risk profile.")
        return

    recs = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:20]

    rows = []
    for rank, rec in enumerate(recs, start=1):
        upside = _upside_pct(rec["entry_price"], rec["target_price"])
        rows.append(
            {
                "Rank": rank,
                "Ticker": rec["ticker"],
                "Watchlist": rec["watchlist"],
                "Score": rec["score"],
                "Signal": rec["signal"],
                "Confidence": f"{rec['confidence']}%",
                "Entry $": f"${rec['entry_price']:,.2f}",
                "Target $": f"${rec['target_price']:,.2f}",
                "Upside": f"{upside:+.1%}",
                "Position Size": _position_size_label(rec["score"], profile),
                "Risk": rec["risk_level"],
            }
        )

    recs_df = pd.DataFrame(rows)
    st.dataframe(recs_df, use_container_width=True, hide_index=True)

    # Expandable detail cards
    st.markdown("---")
    st.subheader("Recommendation Details")
    for rec in recs:
        upside = _upside_pct(rec["entry_price"], rec["target_price"])
        signal_color = _SIGNAL_COLOR.get(rec["signal"], "grey")
        with st.expander(
            f"**{rec['ticker']}** – Score {rec['score']} – :{signal_color}[{rec['signal']}]",
            expanded=False,
        ):
            col1, col2 = st.columns(2)
            col1.metric("Composite Score", f"{rec['score']}/100")
            col2.metric("ML Signal", rec["signal"])
            col1.metric("Confidence", f"{rec['confidence']}%")
            col2.metric("Upside", f"{upside:+.1%}")
            col1.metric("Entry Price", f"${rec['entry_price']:,.2f}")
            col2.metric("Target Price", f"${rec['target_price']:,.2f}")
            st.markdown(f"**Key Drivers:** {rec['drivers']}")
            st.markdown(f"**Source Watchlist:** {rec['watchlist']}")
            st.markdown(
                f"**Recommended Position Size:** {_position_size_label(rec['score'], profile)}"
            )
            if st.button(f"➕ Add {rec['ticker']} to Portfolio", key=f"rec_add_{rec['ticker']}"):
                _add_to_portfolio(rec["ticker"], [rec])


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def page_risk_recommendations() -> None:
    """Risk Profile & Recommendations page."""
    st.title("🛡️ Risk Profile & Recommendations")

    # Ensure session state keys are present
    if "risk_profile" not in st.session_state:
        st.session_state.risk_profile = "moderate"
    if "risk_profile_result" not in st.session_state:
        st.session_state.risk_profile_result = None

    tab1, tab2, tab3 = st.tabs(
        ["📊 Risk Assessment", "📋 Watchlists", "🎯 Top Recommendations"]
    )

    with tab1:
        profile_result = _section_risk_assessment()
        current_profile = profile_result["profile"] if profile_result else st.session_state.risk_profile

    with tab2:
        current_profile_tab2 = st.session_state.get("risk_profile", "moderate")
        _section_watchlists(current_profile_tab2)

    with tab3:
        current_profile_tab3 = st.session_state.get("risk_profile", "moderate")
        _section_recommendations(current_profile_tab3)
