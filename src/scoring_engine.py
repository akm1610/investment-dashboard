"""
Intelligent scoring engine for investment recommendations.

Provides context-aware scoring functions that properly differentiate stocks
across the full 0–10 range instead of clustering everything around 4.5–6.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Intelligent Fundamentals Scoring
# ---------------------------------------------------------------------------

def score_fundamentals_intelligent(fundamentals: Dict) -> float:
    """Score fundamentals (0–10) based on quality and growth.

    Priorities:
    1. Profitability (ROE, margins) – rewarded heavily
    2. Margins (gross / operating) – quality indicator
    3. Valuation (PEG ratio logic) – P/E in context of growth
    4. Safety (debt, liquidity) – penalise only when dangerous

    Parameters
    ----------
    fundamentals:
        Dict with keys: roe, gross_margin, operating_margin, pe_ratio,
        earnings_growth, debt_to_equity, current_ratio.  Missing / None
        values are treated as neutral (zero or sensible default).

    Returns
    -------
    float in [0, 10]
    """
    score = 5.0

    # 1. Profitability: ROE
    roe = fundamentals.get("roe") or 0.0
    if roe > 0.20:
        score += 2.5  # Exceptional
    elif roe > 0.15:
        score += 2.0
    elif roe > 0.10:
        score += 1.5
    elif roe > 0.05:
        score += 1.0
    elif roe < 0:
        score -= 1.5  # Negative ROE is a red flag

    # 2. Margins
    gross_margin = fundamentals.get("gross_margin") or 0.0
    operating_margin = fundamentals.get("operating_margin") or 0.0

    if gross_margin > 0.40:
        score += 1.5  # Premium business model
    elif gross_margin > 0.30:
        score += 1.0

    if operating_margin > 0.25:
        score += 1.0  # Highly efficient operations
    elif operating_margin > 0.15:
        score += 0.5

    # 3. Valuation: PEG ratio (P/E in context of growth)
    pe = fundamentals.get("pe_ratio") or 0.0
    growth_rate = fundamentals.get("earnings_growth") or 0.05  # default 5 %

    if pe > 0:
        peg = pe / (growth_rate * 100) if growth_rate > 0 else pe

        if peg < 1.0:
            score += 1.5  # Cheap relative to growth
        elif peg < 1.5:
            score += 1.0  # Fairly valued
        elif peg < 2.0:
            score += 0.5
        elif peg > 3.0:
            score -= 1.0  # Overvalued relative to growth

    # 4. Safety: debt and liquidity
    debt_to_equity = fundamentals.get("debt_to_equity") or 0.0
    current_ratio = fundamentals.get("current_ratio") or 1.0

    if debt_to_equity < 0.5:
        score += 0.5  # Strong balance sheet
    elif debt_to_equity > 3.0:
        score -= 1.0  # High leverage risk

    if current_ratio < 0.5:
        score -= 1.0  # Liquidity concern
    elif current_ratio > 2.0:
        score += 0.5  # Strong liquidity

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# 2. Contextualised Risk Scoring
# ---------------------------------------------------------------------------

def contextualize_risk(
    risk_metrics: Dict,
    fundamentals: Dict,
    technicals: Dict,
) -> float:
    """Score risk (0–10) contextually.

    High volatility is **not** inherently bad when:
    * Fundamentals are strong (high ROE, margins)
    * Growth is positive
    * Technical trend is up

    This rewards growth at appropriate risk levels and avoids
    penalising high-quality growth companies simply for being volatile.

    Parameters
    ----------
    risk_metrics:
        Dict with keys: volatility, sharpe_ratio, max_drawdown.
    fundamentals:
        Dict with key: roe.
    technicals:
        Dict with key: price_vs_sma200.

    Returns
    -------
    float in [0, 10]
    """
    score = 5.0

    volatility = risk_metrics.get("volatility") or 0.20
    sharpe_ratio = risk_metrics.get("sharpe_ratio") or 0.0
    max_drawdown = risk_metrics.get("max_drawdown") or -0.20

    # Base volatility adjustment
    if volatility < 0.15:
        score += 1.5
    elif volatility < 0.25:
        score += 1.0
    elif volatility < 0.40:
        score += 0.0   # Higher but acceptable
    elif volatility < 0.60:
        score -= 1.0
    else:
        score -= 2.0   # Extreme

    # Sharpe ratio: risk-adjusted returns matter
    if sharpe_ratio > 1.5:
        score += 1.5
    elif sharpe_ratio > 1.0:
        score += 1.0
    elif sharpe_ratio > 0.5:
        score += 0.5
    elif sharpe_ratio < 0:
        score -= 1.0

    # Drawdown: deep drawdowns are concerning
    if max_drawdown > -0.15:
        score += 0.5
    elif max_drawdown > -0.30:
        score += 0.0
    elif max_drawdown > -0.50:
        score -= 1.0
    else:
        score -= 2.0

    # Context: strong fundamentals reduce the penalty for high volatility
    roe = fundamentals.get("roe") or 0.0
    if roe > 0.15 and volatility > 0.40:
        score += 1.0  # Quality growth stock — volatility is acceptable

    # Context: uptrend makes volatility less scary
    price_vs_sma200 = technicals.get("price_vs_sma200") or 0.0
    if price_vs_sma200 > 0.05 and volatility > 0.35:
        score += 0.5  # Volatility in uptrend is growth, not pure risk

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# 3. Intelligent ML Scoring
# ---------------------------------------------------------------------------

def score_ml_intelligently(
    ml_prediction: Dict,
    fundamentals: Dict,
    technicals: Dict,
) -> float:
    """Score ML prediction (0–10) with meaningful differentiation.

    Factors:
    1. Model agreement (convergence) — all 4 agree = strong signal
    2. Confidence level — high confidence (>70 %) = more weight
    3. Signal direction — BUY > HOLD > SELL
    4. Alignment with fundamentals — BUY but ROE/margin negative = red flag
    5. Alignment with technicals — BUY but RSI oversold = confirmation

    Parameters
    ----------
    ml_prediction:
        Dict with keys: signal, confidence, model_votes.
    fundamentals:
        Dict with keys: roe, pe_ratio, operating_margin.
    technicals:
        Dict with keys: rsi_14, price_vs_sma200.

    Returns
    -------
    float in [0, 10]
    """
    score = 5.0

    signal = ml_prediction.get("signal") or "HOLD"
    confidence = float(ml_prediction.get("confidence") or 50.0)
    model_votes: Dict = ml_prediction.get("model_votes") or {}

    # Count agreement among constituent models
    buy_votes = sum(1 for v in model_votes.values() if v == "BUY")
    sell_votes = sum(1 for v in model_votes.values() if v == "SELL")
    total_models = len(model_votes) or 1

    if signal == "BUY":
        # Base BUY score from confidence (50 % → +2, 100 % → +4)
        base_score = 6.0 + (confidence / 50.0) * 2.0
        score = min(8.5, base_score)

        # Bonus: model agreement on BUY
        if buy_votes == total_models:
            score += 1.0   # All models agree
        elif buy_votes >= total_models * 0.75:
            score += 0.7   # Strong majority
        elif buy_votes >= total_models * 0.5:
            score += 0.3   # Simple majority
        else:
            score -= 0.5   # Weak consensus

        # Bonus: confidence level
        if confidence > 80:
            score += 0.5
        elif confidence < 55:
            score -= 0.3

    elif signal == "SELL":
        # Base SELL score from confidence
        base_score = 5.0 - (confidence / 50.0) * 2.0
        score = max(1.5, base_score)

        # Penalty: model agreement on SELL
        if sell_votes == total_models:
            score -= 1.0
        elif sell_votes >= total_models * 0.75:
            score -= 0.7
        elif sell_votes >= total_models * 0.5:
            score -= 0.3

        if confidence > 80:
            score -= 0.5

    else:  # HOLD
        if confidence > 70:
            score = 5.5   # Weak confirmation
        elif confidence < 50:
            score = 4.5   # Weak signal
        else:
            score = 5.0   # True neutral

    # Fundamental alignment check
    roe = fundamentals.get("roe") or 0.0
    margin = fundamentals.get("operating_margin") or 0.0

    if signal == "BUY":
        if roe < 0 or margin < 0:
            score -= 1.5   # BUY with negative fundamentals = strong red flag
        elif roe < 0.05:
            score -= 0.7   # BUY with weak fundamentals
        elif roe > 0.15 and margin > 0.15:
            score += 0.5   # BUY confirmed by great fundamentals

    if signal == "SELL":
        if roe > 0.15 and margin > 0.20:
            score += 1.5   # Strong fundamentals override sell signal

    # Technical alignment check
    _rsi = technicals.get("rsi_14")
    rsi = 50.0 if _rsi is None or (isinstance(_rsi, float) and np.isnan(_rsi)) else float(_rsi)
    price_vs_ma = float(technicals.get("price_vs_sma200") or 0.0)

    if signal == "BUY":
        if rsi < 30:
            score += 0.5   # Oversold = good entry point
        elif rsi > 70:
            score -= 0.5   # Overbought = risky entry

        if price_vs_ma > 0.05:
            score += 0.3   # Uptrend confirms BUY
        elif price_vs_ma < -0.10:
            score -= 0.5   # Downtrend conflicts with BUY

    if signal == "SELL":
        if rsi > 70:
            score -= 0.5   # Overbought confirms SELL
        elif rsi < 30:
            score += 0.5   # Oversold conflicts with SELL

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# 4. Technical Scoring
# ---------------------------------------------------------------------------

def score_technicals_intelligent(
    technicals: Dict,
    price_data: pd.DataFrame,
) -> float:
    """Score technicals (0–10) based on momentum, trend, and volume.

    Uses pre-computed ``price_vs_sma200`` from the technicals dict when
    available; falls back to computing the 200-day MA from *price_data*.

    Parameters
    ----------
    technicals:
        Dict with keys: rsi_14, macd_hist (or macd), price_vs_sma200,
        volume_ratio.  Missing / NaN values are treated as neutral.
    price_data:
        OHLCV DataFrame.  Used as fallback for the 200-day MA calculation
        when ``price_vs_sma200`` is not present in *technicals*.

    Returns
    -------
    float in [0, 10]
    """
    score = 5.0

    # RSI: oversold is a buy opportunity; overbought warrants caution
    _rsi = technicals.get("rsi_14")
    rsi = 50.0 if _rsi is None or (isinstance(_rsi, float) and np.isnan(_rsi)) else float(_rsi)

    if 40 <= rsi <= 60:
        score += 1.0   # Neutral zone — steady
    elif rsi < 30:
        score += 2.0   # Oversold — potential bounce
    elif rsi > 70:
        score -= 1.0   # Overbought — caution

    # MACD: prefer histogram (more precise) over raw MACD line
    _macd_hist = technicals.get("macd_hist")
    macd_hist = 0.0 if _macd_hist is None or (isinstance(_macd_hist, float) and np.isnan(_macd_hist)) else float(_macd_hist)
    _macd = technicals.get("macd")
    macd = 0.0 if _macd is None or (isinstance(_macd, float) and np.isnan(_macd)) else float(_macd)
    signal_value = macd_hist if macd_hist != 0.0 else macd

    if signal_value > 0:
        score += 1.0
    elif signal_value < 0:
        score -= 1.0

    # Price vs 200-day MA: use pre-computed ratio when available
    pct_vs_sma200 = technicals.get("price_vs_sma200")
    if pct_vs_sma200 is not None and not (
        isinstance(pct_vs_sma200, float) and np.isnan(pct_vs_sma200)
    ):
        if pct_vs_sma200 > 0:
            score += 1.0
        else:
            score -= 0.5
    elif price_data is not None and len(price_data) > 200:
        price = float(price_data["Close"].iloc[-1])
        ma_200 = float(price_data["Close"].iloc[-200:].mean())
        if price > ma_200:
            score += 1.0
        else:
            score -= 0.5

    # Volume ratio: above-average volume confirms the move
    _vol = technicals.get("volume_ratio")
    volume_ratio = 1.0 if _vol is None or (isinstance(_vol, float) and np.isnan(_vol)) else float(_vol)

    if volume_ratio > 1.2:
        score += 0.5
    elif volume_ratio < 0.7:
        score -= 0.5

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# 5. Sentiment Scoring
# ---------------------------------------------------------------------------

def score_sentiment(ticker: str) -> float:  # noqa: ARG001
    """Score sentiment (0–10) from analyst ratings, news, and insider activity.

    When live sentiment data is unavailable the function returns a neutral 5.0
    so it does not distort the overall score.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (reserved for future live-data integration).

    Returns
    -------
    float in [0, 10]
    """
    score = 5.0

    try:
        analyst_rating: Optional[float] = _get_analyst_rating(ticker)
        if analyst_rating is not None:
            # analyst_rating is 1–5; re-centre around 2.5 → ±1.0 adjustment
            score += (analyst_rating - 2.5) * 0.4

        news_sentiment: Optional[float] = _get_news_sentiment(ticker)
        if news_sentiment is not None:
            # news_sentiment is -1 to +1 → ±2.0 adjustment
            score += news_sentiment * 2.0

        insider_activity: Optional[float] = _get_insider_activity(ticker)
        if insider_activity is not None:
            if insider_activity > 0:
                score += min(1.0, insider_activity / 1000.0)
            elif insider_activity < 0:
                score -= min(0.5, abs(insider_activity) / 2000.0)

    except Exception:  # noqa: BLE001
        pass  # If sentiment unavailable, use neutral

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# 6. ETF Exposure Scoring
# ---------------------------------------------------------------------------

def score_etf_exposure(ticker: str) -> float:  # noqa: ARG001
    """Score ETF exposure (0–10).

    Inclusion in major ETFs is a quality signal — it means institutional
    inclusion criteria have been met.  Falls back to neutral 5.0 when live
    ETF data is unavailable.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (reserved for future live-data integration).

    Returns
    -------
    float in [0, 10]
    """
    score = 5.0

    try:
        etf_inclusion = _get_etf_inclusion(ticker)

        if "SPY" in etf_inclusion or "QQQ" in etf_inclusion:
            score += 1.0

        if "ESG" in str(etf_inclusion).upper():
            score += 0.5

        num_etfs = len(etf_inclusion)
        if num_etfs > 50:
            score += 1.0
        elif num_etfs > 20:
            score += 0.5

    except Exception:  # noqa: BLE001
        pass

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# 7. Distribution Stretching
# ---------------------------------------------------------------------------

def stretch_distribution(raw_score: float, mean: float = 5.5, factor: float = 1.4) -> float:
    """Stretch a raw score so that the output uses the full 0–10 range.

    The transformation ``(x - mean) * factor + mean`` preserves the ordering
    of stocks while spreading them out so that exceptional stocks land near
    9–10 and weak stocks land near 1–2.

    ``mean`` is set to **5.5** (rather than 5.0) because the individual
    scoring functions all start from a baseline of 5.0 but have more upward
    adjustment headroom than downward.  Empirically the unweighted average of
    component scores clusters slightly above 5.0, so centering the stretch at
    5.5 prevents a systematic upward bias in the final output.

    Parameters
    ----------
    raw_score:
        Weighted average score before stretching.
    mean:
        Centre of distribution (default 5.5).
    factor:
        Stretch multiplier (default 1.4).

    Returns
    -------
    float in [0, 10]
    """
    stretched = (raw_score - mean) * factor + mean
    return max(0.0, min(10.0, stretched))


# ---------------------------------------------------------------------------
# 8. Composite Intelligent Score
# ---------------------------------------------------------------------------

def calculate_intelligent_score(
    ticker: str,
    fundamentals: Dict,
    technicals: Dict,
    risk_metrics: Dict,
    ml_prediction: Dict,
    price_data: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate an intelligent, differentiated composite score for a stock.

    Aggregates all component scores using a quality-weighted formula and
    stretches the result to use the full 0–10 range.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (passed to sentiment/ETF scorers).
    fundamentals:
        Dict of fundamental ratios (roe, margins, pe_ratio, …).
    technicals:
        Dict of technical indicators (rsi_14, macd, price_vs_sma200, …).
    risk_metrics:
        Dict of risk measures (volatility, sharpe_ratio, max_drawdown).
    ml_prediction:
        Dict from the ML engine (signal, confidence, model_votes).
    price_data:
        OHLCV DataFrame used for technical and risk calculations.

    Returns
    -------
    Dict with keys: fundamentals, technicals, risk, ml, sentiment, etf,
    raw, final — all floats in [0, 10].
    """
    fund_score = score_fundamentals_intelligent(fundamentals)
    tech_score = score_technicals_intelligent(technicals, price_data)
    risk_score = contextualize_risk(risk_metrics, fundamentals, technicals)
    ml_score = score_ml_intelligently(ml_prediction, fundamentals, technicals)
    sent_score = score_sentiment(ticker)
    etf_score = score_etf_exposure(ticker)

    raw_score = (
        fund_score * 0.40
        + tech_score * 0.25
        + risk_score * 0.15
        + ml_score * 0.12
        + sent_score * 0.05
        + etf_score * 0.03
    )

    final_score = stretch_distribution(raw_score)

    return {
        "fundamentals": fund_score,
        "technicals": tech_score,
        "risk": risk_score,
        "ml": ml_score,
        "sentiment": sent_score,
        "etf": etf_score,
        "raw": raw_score,
        "final": final_score,
    }


# ---------------------------------------------------------------------------
# Private helpers (stubs for future live-data integration)
# ---------------------------------------------------------------------------

def _get_analyst_rating(ticker: str) -> Optional[float]:
    """Return analyst rating (1–5) or None if unavailable.

    Fetches the consensus analyst recommendation from yfinance and maps it to
    the 1–5 scale used internally (1 = strong buy, 5 = strong sell).
    """
    try:
        import yfinance as yf  # lazy import to keep module importable without network

        info = yf.Ticker(ticker).info
        # ``recommendationMean`` is already on a 1–5 scale (1=strong buy, 5=strong sell)
        rating = info.get("recommendationMean")
        if rating is not None and 1.0 <= float(rating) <= 5.0:
            return float(rating)
    except Exception:  # noqa: BLE001
        pass
    return None


_POSITIVE_WORDS: frozenset = frozenset({
    "surge", "surges", "soar", "soars", "beat", "beats", "strong",
    "growth", "record", "profit", "gain", "rally", "upgrade", "upgraded",
    "bullish", "outperform", "exceed", "raised", "positive", "buy",
    "opportunity", "innovative", "expansion", "robust", "solid",
})
_NEGATIVE_WORDS: frozenset = frozenset({
    "fall", "falls", "drop", "drops", "miss", "misses", "weak", "loss",
    "losses", "decline", "cut", "downgrade", "downgraded", "bearish",
    "underperform", "concern", "lawsuit", "warning", "crash", "sell",
    "risk", "volatile", "debt", "layoff", "layoffs", "recall", "fraud",
})


def _keyword_polarity(title: str) -> Optional[float]:
    """Return a polarity score for a headline using keyword matching.

    Returns a float in [-1, 1] if any sentiment keywords are found,
    or None if no keywords match.
    """
    words = set(title.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    if pos + neg > 0:
        return (pos - neg) / (pos + neg)
    return None


def _get_news_sentiment(ticker: str) -> Optional[float]:
    """Return news sentiment score (-1 to +1) or None if unavailable.

    If the ``NEWS_API_KEY`` environment variable is set, fetches live
    headlines from NewsAPI (https://newsapi.org) and applies keyword-based
    polarity scoring.  Falls back to yfinance news headlines when the key is
    absent or the NewsAPI call fails.  The final value is the mean polarity
    clipped to [-1, +1].
    """
    import os
    import requests  # already a transitive dependency

    headlines: list[str] = []

    news_api_key = os.environ.get("NEWS_API_KEY", "").strip()
    if news_api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": news_api_key,
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                for article in resp.json().get("articles", []):
                    title = article.get("title") or ""
                    if title:
                        headlines.append(title)
        except Exception:  # noqa: BLE001
            pass  # fall through to yfinance fallback

    if not headlines:
        try:
            import yfinance as yf  # lazy import

            news = yf.Ticker(ticker).news or []
            for article in news[:20]:  # cap at 20 recent articles
                title = article.get("title") or ""
                # yfinance ≥ 0.2.50 wraps content in a nested dict
                if not title and isinstance(article.get("content"), dict):
                    title = article["content"].get("title") or ""
                if title:
                    headlines.append(title)
        except Exception:  # noqa: BLE001
            pass

    scores: list[float] = []
    for title in headlines:
        polarity = _keyword_polarity(title)
        if polarity is not None:
            scores.append(polarity)

    if scores:
        mean_score = sum(scores) / len(scores)
        return max(-1.0, min(1.0, mean_score))
    return None


def get_news_headlines(ticker: str, max_articles: int = 10) -> list[dict]:
    """Fetch recent news headlines for *ticker* and return sentiment details.

    Returns a list of dicts with keys ``title``, ``url``, ``polarity``,
    and ``source``.  Used by the ``/sentiment/<ticker>`` API endpoint.
    Supports optional NewsAPI integration via the ``NEWS_API_KEY`` env var
    and falls back to yfinance headlines.
    """
    import os
    import requests

    articles: list[dict] = []

    news_api_key = os.environ.get("NEWS_API_KEY", "").strip()
    if news_api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": news_api_key,
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                for item in resp.json().get("articles", []):
                    title = item.get("title") or ""
                    if title:
                        articles.append({
                            "title": title,
                            "url": item.get("url") or "",
                            "published_at": item.get("publishedAt") or "",
                            "source": (item.get("source") or {}).get("name") or "NewsAPI",
                            "polarity": _keyword_polarity(title),
                        })
        except Exception:  # noqa: BLE001
            pass

    if not articles:
        try:
            import yfinance as yf

            news = yf.Ticker(ticker).news or []
            for item in news[:max_articles]:
                title = item.get("title") or ""
                if not title and isinstance(item.get("content"), dict):
                    title = item["content"].get("title") or ""
                link = item.get("link") or item.get("url") or ""
                if not link and isinstance(item.get("content"), dict):
                    link = item["content"].get("canonicalUrl", {}).get("url") or ""
                if title:
                    articles.append({
                        "title": title,
                        "url": link,
                        "published_at": "",
                        "source": "Yahoo Finance",
                        "polarity": _keyword_polarity(title),
                    })
        except Exception:  # noqa: BLE001
            pass

    return articles


def _get_insider_activity(ticker: str) -> Optional[float]:
    """Return net insider share activity or None if unavailable.

    Fetches recent insider transactions via yfinance and returns the net
    number of shares bought (positive) or sold (negative).
    """
    try:
        import yfinance as yf  # lazy import

        insider = yf.Ticker(ticker).insider_transactions
        if insider is None or insider.empty:
            return None
        # Transactions have a ``Shares`` column (positive = buy, negative = sell)
        if "Shares" in insider.columns:
            net_shares = float(insider["Shares"].sum())
            return net_shares
    except Exception:  # noqa: BLE001
        pass
    return None


def _get_etf_inclusion(ticker: str) -> list:
    """Return list of ETF tickers that include this stock, or empty list.

    Checks whether the ticker appears in the holdings of major ETFs (SPY, QQQ,
    IWM) by querying yfinance.  Only SPY/QQQ/IWM are checked to keep the call
    fast.  An exception (e.g. network error) returns an empty list.
    """
    _MAJOR_ETFS = ["SPY", "QQQ", "IWM"]
    found: list[str] = []
    try:
        import yfinance as yf  # lazy import

        ticker_upper = ticker.upper()
        for etf in _MAJOR_ETFS:
            try:
                # ``funds_data.top_holdings`` returns a DataFrame indexed by ticker
                # symbol with at least a ``Weight`` column (yfinance ≥ 0.2.x).
                holdings = yf.Ticker(etf).funds_data.top_holdings
                if holdings is not None and not holdings.empty:
                    symbols = holdings.index.str.upper().tolist()
                    if ticker_upper in symbols:
                        found.append(etf)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return found
