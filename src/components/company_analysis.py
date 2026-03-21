"""
src/components/company_analysis.py
-----------------------------------
Page 1 – deep-dive into a single company's financials and investment quality.

Provides:
* Composite and pillar score gauges (Plotly)
* Company information card
* Financial ratios table (organised by category)
* 5-year interactive price chart with optional moving averages
* Pre-trade checklist with PASS / WARN / FAIL status
* Sentiment Analysis panel (news headlines, analyst consensus, insider activity)
* "Add to Portfolio" action
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Ensure root-level modules are importable when running as `streamlit run
# src/app.py` (the working directory may be the repo root OR `src/`).
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis_engine as ae
import data_fetcher as df_mod
import portfolio_manager as pm

from .utils import (
    display_checklist,
    display_score_gauge,
    format_currency,
    format_large_number,
    format_pct,
    format_ratio,
    score_color,
    status_emoji,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_sentiment(symbol: str) -> dict:
    """Fetch news sentiment data for *symbol*; cached for 15 minutes."""
    try:
        from scoring_engine import (  # type: ignore[import]
            get_news_headlines,
            get_news_sentiment,
        )
        import yfinance as yf

        headlines = get_news_headlines(symbol, max_articles=10)
        news_sentiment = get_news_sentiment(symbol)

        analyst_rating: Optional[float] = None
        analyst_label: Optional[str] = None
        try:
            info = yf.Ticker(symbol).info
            rating = info.get("recommendationMean")
            if rating is not None and 1.0 <= float(rating) <= 5.0:
                analyst_rating = float(rating)
                key = info.get("recommendationKey", "").lower()
                analyst_label = key.replace("_", " ").title() if key else None
        except Exception:  # noqa: BLE001
            pass

        news_api_active = bool(os.environ.get("NEWS_API_KEY", "").strip())

        return {
            "headlines": headlines,
            "news_sentiment": news_sentiment,
            "analyst_rating": analyst_rating,
            "analyst_label": analyst_label,
            "news_api_active": news_api_active,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "headlines": [],
            "news_sentiment": None,
            "analyst_rating": None,
            "analyst_label": None,
            "news_api_active": False,
            "error": str(exc),
        }


def _render_sentiment_section(symbol: str) -> None:
    """Render the sentiment analysis panel for *symbol* inside Company Analysis."""
    with st.spinner("Loading sentiment data …"):
        data = _fetch_sentiment(symbol)

    if data["error"]:
        st.warning(f"Sentiment data unavailable: {data['error']}")
        return

    headlines: list = data["headlines"]
    news_sentiment: Optional[float] = data["news_sentiment"]
    analyst_rating: Optional[float] = data["analyst_rating"]
    analyst_label: Optional[str] = data["analyst_label"]
    news_api_active: bool = data["news_api_active"]

    # Overall verdict
    if news_sentiment is None:
        verdict, badge_color = "Neutral", "#e65100"
    elif news_sentiment > 0.05:
        verdict, badge_color = "Bullish", "#2e7d32"
    elif news_sentiment < -0.05:
        verdict, badge_color = "Bearish", "#c62828"
    else:
        verdict, badge_color = "Neutral", "#e65100"

    sent_col1, sent_col2 = st.columns([1, 2])

    with sent_col1:
        # Map polarity [-1, +1] → gauge [0, 100]
        gauge_score = max(0.0, min(100.0, ((news_sentiment or 0.0) + 1.0) * 50.0))
        st.plotly_chart(
            display_score_gauge(gauge_score, "Sentiment Score"),
            use_container_width=True,
            key=f"ca_sent_gauge_{symbol}",
        )
        st.markdown(
            f'<div style="text-align:center;">'
            f'<span style="background:{badge_color};color:white;padding:4px 16px;'
            f'border-radius:12px;font-weight:600;">{verdict}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if analyst_rating is not None:
            label_str = analyst_label or (
                "Strong Buy" if analyst_rating <= 1.5 else
                "Buy" if analyst_rating <= 2.5 else
                "Hold" if analyst_rating <= 3.5 else
                "Sell" if analyst_rating <= 4.5 else
                "Strong Sell"
            )
            st.metric("Analyst Consensus", label_str, f"{analyst_rating:.1f}/5.0")
        source_note = "NewsAPI" if news_api_active else "Yahoo Finance (fallback)"
        st.caption(f"📡 Source: {source_note}")

    with sent_col2:
        if not headlines:
            st.info(
                "No headlines available. "
                + ("Set `NEWS_API_KEY` in `.env` for live news." if not news_api_active else "")
            )
        else:
            rows = []
            for article in headlines:
                title = article.get("title", "")
                url = article.get("url", "")
                src = article.get("source", "")
                polarity = article.get("polarity")
                if polarity is None:
                    sent_str = "🟡 Neutral"
                elif polarity > 0.05:
                    sent_str = "🟢 Positive"
                else:
                    sent_str = "🔴 Negative"
                rows.append({
                    "Headline": f"[{title}]({url})" if url else title,
                    "Source": src,
                    "Sentiment": sent_str,
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )



    """Fetch fundamentals and run full analysis; results are cached for 1 h."""
    fundamentals = df_mod.fetch_all_fundamentals(symbol)
    analysis = ae.analyze(fundamentals)
    analysis["fundamentals"] = fundamentals
    return analysis


def _build_price_figure(price_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Build a Plotly price chart with volume bars and optional moving averages."""
    fig = go.Figure()

    # Closing price line
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df["Close"],
            name="Close",
            line={"color": "#1976D2", "width": 1.5},
        )
    )

    # 50-day MA
    if len(price_df) >= 50:
        ma50 = price_df["Close"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=ma50,
                name="50-day MA",
                line={"color": "#FF6F00", "width": 1, "dash": "dot"},
            )
        )

    # 200-day MA
    if len(price_df) >= 200:
        ma200 = price_df["Close"].rolling(200).mean()
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=ma200,
                name="200-day MA",
                line={"color": "#7B1FA2", "width": 1, "dash": "dash"},
            )
        )

    # Volume bars (secondary y-axis)
    if "Volume" in price_df.columns:
        fig.add_trace(
            go.Bar(
                x=price_df.index,
                y=price_df["Volume"],
                name="Volume",
                marker_color="rgba(128,128,128,0.3)",
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=f"{symbol} – 5-Year Price History",
        xaxis_title="",
        yaxis_title="Price (USD)",
        yaxis2={
            "title": "Volume",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        height=400,
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 40, "r": 60, "t": 50, "b": 40},
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Public page function
# ---------------------------------------------------------------------------


def page_company_analysis() -> None:
    """Render the Company Analysis page."""
    st.title("🔍 Company Analysis")
    st.caption("Deep-dive into a company's financials, scores, and investment quality.")

    # --- Ticker input ---
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        symbol_raw = st.text_input(
            "Ticker symbol",
            placeholder="e.g. AAPL, MSFT, NVDA",
            key="ca_symbol_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("Analyse", type="primary", use_container_width=True)

    symbol = symbol_raw.upper().strip()

    if not symbol:
        st.info("Enter a ticker symbol above and click **Analyse** to begin.")
        return

    # Trigger on button click OR if the symbol is already cached
    cache_key = f"analysis_{symbol}"
    if not analyse_btn and cache_key not in st.session_state.get("analysis_cache", {}):
        st.info("Click **Analyse** to fetch data.")
        return

    # --- Fetch & analyse ---
    with st.spinner(f"Fetching data for **{symbol}** …"):
        try:
            result = _fetch_and_analyze(symbol)
        except Exception as exc:
            st.error(f"Failed to fetch data for **{symbol}**: {exc}")
            return

    # Cache result so it survives reruns
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    st.session_state.analysis_cache[cache_key] = result

    ratios: dict = result["ratios"]
    scores: dict = result["scores"]
    fundamentals: dict = result["fundamentals"]
    stats: dict = fundamentals.get("key_stats", {})

    # --- Company header ---
    company_name = stats.get("longName") or stats.get("shortName") or symbol
    sector = stats.get("sector", "N/A")
    industry = stats.get("industry", "N/A")
    employees = stats.get("fullTimeEmployees")
    website = stats.get("website", "")
    market_cap = stats.get("marketCap")

    st.header(company_name)
    header_parts = [sector, industry]
    if employees:
        header_parts.append(f"{employees:,} employees")
    if market_cap:
        header_parts.append(f"Market Cap: {format_large_number(market_cap)}")
    st.caption(" · ".join(p for p in header_parts if p and p != "N/A"))
    if website:
        st.markdown(f"🌐 [{website}]({website})")

    st.markdown("---")

    # --- Score section ---
    st.subheader("Investment Scores")

    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    score_pairs = [
        (g_col1, "composite", "Composite"),
        (g_col2, "stability", "Stability"),
        (g_col3, "quality", "Quality"),
        (g_col4, "valuation", "Valuation"),
    ]
    for col, key, label in score_pairs:
        val = scores.get(key, 0)
        color = score_color(val)
        col.plotly_chart(
            display_score_gauge(val, label),
            use_container_width=True,
            key=f"gauge_{key}_{symbol}",
        )

    st.markdown("---")

    # --- Financial Ratios ---
    st.subheader("Financial Ratios")

    # Build a compact display table per category
    CATEGORIES = {
        "Liquidity": [
            ("Current Ratio", "current_ratio", False),
            ("Quick Ratio", "quick_ratio", False),
            ("Cash Ratio", "cash_ratio", False),
        ],
        "Leverage": [
            ("Debt / Equity", "debt_to_equity", True),
            ("Debt / Assets", "debt_to_assets", True),
            ("Debt / EBITDA", "debt_to_ebitda", True),
            ("Interest Coverage", "interest_coverage", False),
        ],
        "Profitability": [
            ("Gross Margin", "gross_margin", False),
            ("Operating Margin", "operating_margin", False),
            ("Net Margin", "net_margin", False),
            ("ROE", "roe", False),
            ("ROA", "roa", False),
        ],
        "Cash Flow": [
            ("FCF Yield", "fcf_yield", False),
        ],
        "Valuation": [
            ("P/E Ratio", "pe_ratio", True),
            ("P/B Ratio", "pb_ratio", True),
            ("P/S Ratio", "ps_ratio", True),
            ("EV/EBITDA", "ev_ebitda", True),
            ("PEG Ratio", "peg_ratio", True),
        ],
    }

    PCT_KEYS = {"gross_margin", "operating_margin", "net_margin", "roe", "roa", "fcf_yield"}

    cat_names = list(CATEGORIES.keys())
    half = (len(cat_names) + 1) // 2
    left_cats = cat_names[:half]
    right_cats = cat_names[half:]

    def _render_category(cat_name: str) -> None:
        st.markdown(f"**{cat_name}**")
        rows = []
        for label, key, _ in CATEGORIES[cat_name]:
            raw = ratios.get(key)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                val_str = "N/A"
            elif key in PCT_KEYS:
                val_str = format_pct(raw)
            else:
                val_str = format_ratio(raw)
            rows.append({"Metric": label, "Value": val_str})
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        for cn in left_cats:
            _render_category(cn)
    with r_col2:
        for cn in right_cats:
            _render_category(cn)

    st.markdown("---")

    # --- Price chart ---
    st.subheader("Price History (5 Years)")
    price_df: pd.DataFrame = fundamentals.get("price_history", pd.DataFrame())
    if not price_df.empty and "Close" in price_df.columns:
        fig_price = _build_price_figure(price_df, symbol)
        st.plotly_chart(fig_price, use_container_width=True, key=f"price_{symbol}")
    else:
        st.warning("Price data is unavailable for this symbol.")

    st.markdown("---")

    # --- Pre-trade checklist ---
    st.subheader("Pre-Trade Checklist")
    thesis_input = st.text_area(
        "Investment Thesis (optional – improves checklist completeness)",
        height=80,
        key="ca_thesis",
        placeholder="Why are you considering this stock?",
    )
    checklist = ae.pretrade_checklist(ratios, scores, thesis_input)

    passed = sum(1 for i in checklist if i["status"] == "pass")
    total = len(checklist)
    progress_val = passed / total if total else 0.0

    color_label = "green" if passed == total else ("orange" if passed >= total // 2 else "red")
    st.progress(progress_val, text=f"{passed}/{total} items passed")

    display_checklist(checklist)

    st.markdown("---")

    # --- Sentiment Analysis ---
    st.subheader("📰 Sentiment Analysis")
    _render_sentiment_section(symbol)

    st.markdown("---")

    # --- Add to portfolio ---
    st.subheader("Add to Portfolio")
    with st.form("ca_add_to_portfolio"):
        ap_col1, ap_col2, ap_col3 = st.columns(3)
        shares_val = ap_col1.number_input("Shares", min_value=0.001, value=1.0, step=0.001)
        cost_val = ap_col2.number_input("Purchase Price ($)", min_value=0.01, value=100.0)
        sector_val = ap_col3.text_input("Sector", value=sector if sector != "N/A" else "")
        add_btn = st.form_submit_button("➕ Add to Portfolio", type="primary")
        if add_btn:
            portfolio = st.session_state.portfolio
            pm.add_holding(
                portfolio,
                symbol,
                shares_val,
                cost_val,
                sector=sector_val or "Unknown",
                thesis=thesis_input,
            )
            pm.save_portfolio(portfolio)
            st.success(f"Added **{symbol}** ({shares_val} shares @ {format_currency(cost_val)}) to your portfolio.")

    # --- Full analysis (raw) ---
    with st.expander("🔬 View Full Analysis Data", expanded=False):
        st.json(
            {
                "scores": scores,
                "ratios": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in ratios.items()
                    if v is not None
                },
            }
        )
