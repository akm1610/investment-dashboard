"""
src/components/sentiment_analysis.py
--------------------------------------
Sentiment Analysis page – real-time market/news sentiment for a selected ticker.

Sections
--------
A. Ticker input and overall sentiment summary (Bearish / Neutral / Bullish)
B. Sentiment score gauge (0–10 scale, converted to 0–100 for the shared gauge)
C. News source indicator (NewsAPI vs Yahoo Finance fallback)
D. Headlines table with per-headline polarity scores
E. Analyst rating and insider activity summary
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Ensure root-level modules are importable when running as
# ``streamlit run src/app.py`` regardless of the working directory.
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .utils import display_score_gauge


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sentiment_label(score: Optional[float]) -> tuple[str, str]:
    """Return (label, colour) for a polarity score in [-1, +1]."""
    if score is None:
        return "Neutral", "grey"
    if score > 0.05:
        return "Bullish", "green"
    if score < -0.05:
        return "Bearish", "red"
    return "Neutral", "orange"


def _polarity_to_gauge_score(polarity: Optional[float]) -> float:
    """Map a polarity in [-1, +1] to a 0–100 gauge score."""
    if polarity is None:
        return 50.0
    # linear mapping: -1 → 0, 0 → 50, +1 → 100
    return max(0.0, min(100.0, (polarity + 1.0) * 50.0))


def _polarity_bar(polarity: Optional[float]) -> str:
    """Return a small text bar representing polarity."""
    if polarity is None:
        return "–"
    if polarity > 0.05:
        return "🟢 Positive"
    if polarity < -0.05:
        return "🔴 Negative"
    return "🟡 Neutral"


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_sentiment_data(symbol: str) -> dict:
    """Fetch and return all sentiment-related data for *symbol*.

    Results are cached for 15 minutes (900 s) to respect NewsAPI free-tier
    rate limits (100 requests / day).
    """
    try:
        from scoring_engine import (  # type: ignore[import]
            get_news_headlines,
            get_news_sentiment,
        )
        import yfinance as yf

        headlines = get_news_headlines(symbol, max_articles=15)
        news_sentiment = get_news_sentiment(symbol)

        # Analyst rating from yfinance (1 = strong buy, 5 = strong sell)
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

        # Insider net share activity
        insider_net: Optional[float] = None
        try:
            insider_df = yf.Ticker(symbol).insider_transactions
            if insider_df is not None and not insider_df.empty and "Shares" in insider_df.columns:
                insider_net = float(insider_df["Shares"].sum())
        except Exception:  # noqa: BLE001
            pass

        # Determine data source
        news_api_key = os.environ.get("NEWS_API_KEY", "").strip()
        source = "NewsAPI" if (news_api_key and headlines and headlines[0].get("source") != "Yahoo Finance") else "Yahoo Finance (fallback)"

        return {
            "headlines": headlines,
            "news_sentiment": news_sentiment,
            "analyst_rating": analyst_rating,
            "analyst_label": analyst_label,
            "insider_net": insider_net,
            "source": source,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "headlines": [],
            "news_sentiment": None,
            "analyst_rating": None,
            "analyst_label": None,
            "insider_net": None,
            "source": "N/A",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Public page function
# ---------------------------------------------------------------------------


def page_sentiment_analysis() -> None:
    """Render the Sentiment Analysis page."""
    st.title("📰 Sentiment Analysis")
    st.caption(
        "Real-time market sentiment based on news headlines, analyst ratings, "
        "and insider activity for the selected ticker."
    )

    # --- Ticker input ---
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        symbol_raw = st.text_input(
            "Ticker symbol",
            placeholder="e.g. AAPL, MSFT, NVDA",
            key="sa_symbol_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("Analyse", type="primary", use_container_width=True, key="sa_analyse_btn")

    symbol = symbol_raw.upper().strip()

    if not symbol:
        st.info("Enter a ticker symbol above and click **Analyse** to view sentiment data.")
        _render_info_box()
        return

    cache_key = f"sentiment_{symbol}"
    if not analyse_btn and cache_key not in st.session_state.get("analysis_cache", {}):
        st.info("Click **Analyse** to fetch sentiment data.")
        _render_info_box()
        return

    # --- Fetch ---
    with st.spinner(f"Fetching sentiment data for **{symbol}** …"):
        data = _fetch_sentiment_data(symbol)

    # Cache so it survives reruns
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    st.session_state.analysis_cache[cache_key] = data

    if data["error"]:
        st.error(f"Failed to load sentiment data for **{symbol}**: {data['error']}")
        return

    headlines: list = data["headlines"]
    news_sentiment: Optional[float] = data["news_sentiment"]
    analyst_rating: Optional[float] = data["analyst_rating"]
    analyst_label: Optional[str] = data["analyst_label"]
    insider_net: Optional[float] = data["insider_net"]
    source: str = data["source"]

    # --- Section A: Overall Sentiment Summary ---
    st.header(f"Sentiment Overview — {symbol}")

    sentiment_label, sentiment_color = _sentiment_label(news_sentiment)

    overview_col1, overview_col2, overview_col3 = st.columns(3)

    with overview_col1:
        gauge_score = _polarity_to_gauge_score(news_sentiment)
        st.plotly_chart(
            display_score_gauge(gauge_score, "News Sentiment Score"),
            use_container_width=True,
            key=f"sent_gauge_{symbol}",
        )

    with overview_col2:
        st.metric(
            label="Overall Sentiment",
            value=sentiment_label,
            delta=f"{news_sentiment:+.2f} polarity" if news_sentiment is not None else "No data",
        )
        _render_sentiment_badge(sentiment_label, sentiment_color)

        if analyst_rating is not None:
            rating_map = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
            closest = min(rating_map.keys(), key=lambda k: abs(k - analyst_rating))
            label_str = analyst_label or rating_map.get(closest, "N/A")
            st.metric(
                label="Analyst Consensus",
                value=label_str,
                delta=f"{analyst_rating:.1f} / 5.0 (1=Strong Buy)",
            )
        else:
            st.metric(label="Analyst Consensus", value="N/A")

    with overview_col3:
        if insider_net is not None:
            insider_direction = "Net Buying 📈" if insider_net > 0 else "Net Selling 📉"
            st.metric(
                label="Insider Activity",
                value=insider_direction,
                delta=f"{insider_net:+,.0f} shares",
            )
        else:
            st.metric(label="Insider Activity", value="N/A")

        st.caption(f"📡 News source: **{source}**")
        if not os.environ.get("NEWS_API_KEY", "").strip():
            st.caption(
                "💡 Set `NEWS_API_KEY` in your `.env` file to enable live NewsAPI headlines. "
                "Currently using Yahoo Finance news as fallback."
            )

    st.markdown("---")

    # --- Section B: Headlines Table ---
    st.subheader("📋 Recent Headlines & Sentiment")

    if not headlines:
        st.warning(
            f"No headlines found for **{symbol}**. "
            "This may be due to an inactive ticker or missing API key."
        )
    else:
        rows = []
        for article in headlines:
            title = article.get("title", "")
            url = article.get("url", "")
            pub = article.get("published_at", "")
            src = article.get("source", "")
            polarity = article.get("polarity")
            rows.append({
                "Headline": f"[{title}]({url})" if url else title,
                "Source": src,
                "Date": pub[:10] if pub else "—",
                "Sentiment": _polarity_bar(polarity),
                "Score": f"{polarity:+.2f}" if polarity is not None else "—",
            })

        df_headlines = pd.DataFrame(rows)
        st.dataframe(
            df_headlines,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Headline": st.column_config.LinkColumn(
                    "Headline",
                    help="Click to open article",
                    display_text=r"(.+)\((.+)\)" if rows and rows[0]["Headline"].startswith("[") else None,
                ),
            },
        )

        # Polarity distribution chart
        polarities = [a.get("polarity") for a in headlines if a.get("polarity") is not None]
        if polarities:
            st.subheader("📊 Polarity Distribution")
            _render_polarity_chart(polarities, symbol)

    st.markdown("---")

    # --- Section C: Sentiment Summary ---
    st.subheader("🧭 Sentiment Summary")

    total_headlines = len(headlines)
    scored_headlines = [h for h in headlines if h.get("polarity") is not None]
    positive_count = sum(1 for h in scored_headlines if (h.get("polarity") or 0) > 0.05)
    negative_count = sum(1 for h in scored_headlines if (h.get("polarity") or 0) < -0.05)
    neutral_count = len(scored_headlines) - positive_count - negative_count

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Total Headlines", total_headlines)
    s_col2.metric("🟢 Positive", positive_count)
    s_col3.metric("🔴 Negative", negative_count)
    s_col4.metric("🟡 Neutral", neutral_count)

    if news_sentiment is not None:
        summary_parts = []
        if news_sentiment > 0.2:
            summary_parts.append(
                f"Overall market sentiment for **{symbol}** is **strongly bullish** "
                f"(polarity {news_sentiment:+.2f}). News coverage is predominantly positive."
            )
        elif news_sentiment > 0.05:
            summary_parts.append(
                f"Overall market sentiment for **{symbol}** is **mildly bullish** "
                f"(polarity {news_sentiment:+.2f})."
            )
        elif news_sentiment < -0.2:
            summary_parts.append(
                f"Overall market sentiment for **{symbol}** is **strongly bearish** "
                f"(polarity {news_sentiment:+.2f}). News coverage is predominantly negative."
            )
        elif news_sentiment < -0.05:
            summary_parts.append(
                f"Overall market sentiment for **{symbol}** is **mildly bearish** "
                f"(polarity {news_sentiment:+.2f})."
            )
        else:
            summary_parts.append(
                f"Overall market sentiment for **{symbol}** is **neutral** "
                f"(polarity {news_sentiment:+.2f})."
            )

        if analyst_rating is not None:
            if analyst_rating <= 2.0:
                summary_parts.append("Analysts currently rate this stock as a **Buy** or **Strong Buy**.")
            elif analyst_rating <= 2.5:
                summary_parts.append("Analysts lean **Buy** on average.")
            elif analyst_rating <= 3.5:
                summary_parts.append("Analysts rate this stock as a **Hold** on average.")
            else:
                summary_parts.append("Analysts lean **Sell** on this stock.")

        if insider_net is not None:
            if insider_net > 0:
                summary_parts.append("Insiders have been **net buyers** recently, a potentially positive signal.")
            else:
                summary_parts.append("Insiders have been **net sellers** recently.")

        st.info("  \n".join(summary_parts))
    else:
        st.warning(
            f"Insufficient data to compute a news sentiment score for **{symbol}**. "
            "No matching keywords found in available headlines."
        )


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------


def _render_sentiment_badge(label: str, color: str) -> None:
    """Render a coloured HTML badge for the overall sentiment label."""
    color_map = {"green": "#2e7d32", "red": "#c62828", "orange": "#e65100", "grey": "#555"}
    bg = color_map.get(color, "#555")
    st.markdown(
        f'<span style="background:{bg};color:white;padding:4px 12px;'
        f'border-radius:12px;font-weight:600;font-size:14px;">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_polarity_chart(polarities: list[float], symbol: str) -> None:
    """Render a horizontal bar chart of per-headline polarity scores."""
    fig = go.Figure()
    colors = ["#4caf50" if p > 0.05 else "#f44336" if p < -0.05 else "#ff9800" for p in polarities]
    fig.add_trace(
        go.Bar(
            x=polarities,
            y=[f"Article {i+1}" for i in range(len(polarities))],
            orientation="h",
            marker_color=colors,
            hovertemplate="Polarity: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{symbol} – Per-Headline Polarity Scores",
        xaxis_title="Polarity (−1 = Bearish, +1 = Bullish)",
        xaxis_range=[-1.1, 1.1],
        height=max(250, len(polarities) * 25 + 80),
        margin={"l": 80, "r": 20, "t": 50, "b": 40},
        showlegend=False,
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, use_container_width=True, key=f"polarity_chart_{symbol}")


def _render_info_box() -> None:
    """Render an informational box explaining sentiment data sources."""
    with st.expander("ℹ️ How sentiment is calculated", expanded=False):
        st.markdown(
            """
**Sentiment scoring uses three signals:**

1. **News Headline Polarity** – Recent news headlines are fetched (via NewsAPI if a key is
   configured, otherwise from Yahoo Finance) and scored using keyword-based polarity matching.
   Positive keywords (e.g. *beat*, *surge*, *upgrade*) contribute a positive score;
   negative keywords (e.g. *miss*, *decline*, *downgrade*) reduce the score.

2. **Analyst Consensus** – The mean analyst recommendation from Yahoo Finance
   (1 = Strong Buy → 5 = Strong Sell) is displayed as an additional context signal.

3. **Insider Activity** – Net share purchases / sales by company insiders over recent months,
   sourced via Yahoo Finance.

**To enable live NewsAPI headlines:**
- Obtain a free key at [newsapi.org](https://newsapi.org).
- Add `NEWS_API_KEY=your_key` to your `.env` file.
- Without the key, sentiment falls back to Yahoo Finance news headlines.

*Results are cached for 15 minutes to respect API rate limits.*
            """
        )
