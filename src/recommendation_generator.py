"""
src/recommendation_generator.py
---------------------------------
Live ML-powered recommendation generation for the Risk & Recommendations page.

Classes
-------
RecommendationGenerator
    Fetches real stock data, engineers features, runs ML predictions, and
    filters/ranks results by risk profile.

WatchlistBuilder
    Builds structured watchlists from RecommendationGenerator output,
    including estimated performance metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk profile configuration
# ---------------------------------------------------------------------------

_PROFILE_SCORE_THRESHOLDS: Dict[str, int] = {
    "conservative": 55,
    "moderate": 45,
    "aggressive": 35,
}

_PROFILE_SIGNAL_ALLOW: Dict[str, List[str]] = {
    "conservative": ["BUY", "HOLD"],
    "moderate": ["BUY", "HOLD"],
    "aggressive": ["BUY", "HOLD", "SELL"],
}


# ---------------------------------------------------------------------------
# RecommendationGenerator
# ---------------------------------------------------------------------------


class RecommendationGenerator:
    """
    Generate investment recommendations by combining live market data with
    ML ensemble predictions.

    Parameters
    ----------
    ml_engine : RecommendationEngine
        Loaded (or empty) ML engine from ``src.ml_engine``.
    feature_engineer : FeatureEngineer
        Feature extraction engine from ``src.ml_engine``.
    """

    def __init__(self, ml_engine: Any, feature_engineer: Any) -> None:
        self.ml_engine = ml_engine
        self.feature_engineer = feature_engineer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        tickers: List[str],
        risk_profile: str,
        count: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Generate real recommendations for *tickers* filtered by *risk_profile*.

        Parameters
        ----------
        tickers      : List of ticker symbols to analyse.
        risk_profile : One of 'conservative', 'moderate', 'aggressive'.
        count        : Maximum number of results to return.

        Returns
        -------
        List of recommendation dicts sorted by composite score (descending),
        truncated to *count* items.
        """
        recs: List[Dict[str, Any]] = []
        for ticker in tickers:
            try:
                rec = self._analyse_ticker(ticker, risk_profile)
                if rec is not None:
                    recs.append(rec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s – analysis failed: %s", ticker, exc)
                continue

        return sorted(recs, key=lambda x: x["score"], reverse=True)[:count]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_ticker(
        self, ticker: str, risk_profile: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch data, engineer features, run ML, and return a rec dict or None."""
        from src.data_fetcher import DataFetcher

        fetcher = DataFetcher(ticker)

        # Fetch OHLCV price data
        price_data = fetcher.fetch_stock_data(period="2y")
        if price_data.empty:
            logger.warning("[%s] Empty price data – skipping", ticker)
            return None

        # Fetch company info for fundamentals
        try:
            company_info = fetcher.fetch_company_info()
        except Exception:  # noqa: BLE001
            company_info = {}

        # Engineer technical features
        tech_features = self.feature_engineer.extract_technical_features(price_data)
        if tech_features.empty:
            return None

        # Combine technical + fundamental features into one row
        features = tech_features.tail(1).copy()
        try:
            fund_series = self.feature_engineer.extract_fundamental_features(
                company_info
            )
            for col, val in fund_series.items():
                if col not in features.columns:
                    features[col] = val
        except Exception:  # noqa: BLE001
            pass  # fundamentals optional

        # Run ML prediction
        ml_rec = self.ml_engine.predict(ticker, features=features, horizon="long_term")

        # Compute composite score and signal (with technical fallback)
        signal = self._compute_signal(ml_rec, tech_features)
        score = self._compute_composite_score(ml_rec, tech_features, signal)
        drivers = self._compute_drivers(ml_rec, tech_features)

        # Filter by risk profile
        if not self._matches_risk_profile(score, signal, risk_profile):
            return None

        # Entry / target prices
        current_price = float(price_data["Close"].iloc[-1])
        target_price = self._calculate_target_price(current_price, signal, score)

        # Model vote summary
        votes = ml_rec.get("model_votes", {})
        model_vote_str = (
            f"ML: {len(votes)} model(s) – " + ", ".join(f"{m}={v}" for m, v in votes.items())
            if votes
            else "Technical analysis"
        )

        return {
            "ticker": ticker,
            "score": score,
            "signal": signal,
            "confidence": score,
            "entry_price": round(current_price, 2),
            "target_price": round(target_price, 2),
            "drivers": drivers,
            "model_votes": votes,
            "strength": ml_rec.get("strength", "MODERATE"),
            "model_vote_str": model_vote_str,
        }

    @staticmethod
    def _matches_risk_profile(score: int, signal: str, risk_profile: str) -> bool:
        """Return True if the recommendation fits *risk_profile*."""
        min_score = _PROFILE_SCORE_THRESHOLDS.get(risk_profile, 45)
        allowed_signals = _PROFILE_SIGNAL_ALLOW.get(risk_profile, ["BUY", "HOLD"])
        return score >= min_score and signal in allowed_signals

    @staticmethod
    def _compute_signal(
        ml_rec: Dict[str, Any], tech_features: pd.DataFrame
    ) -> str:
        """
        Determine buy/hold/sell signal.

        Uses the ML engine output when trained models are available; otherwise
        falls back to a simple RSI + MACD + trend composite.
        """
        if ml_rec.get("model_votes"):
            return ml_rec["signal"]

        if tech_features.empty:
            return "HOLD"

        row = tech_features.iloc[-1]
        buy_signals = 0
        sell_signals = 0

        if "rsi_14" in row.index:
            rsi = row["rsi_14"]
            if rsi < 30:
                buy_signals += 2
            elif rsi < 40:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 2
            elif rsi > 60:
                sell_signals += 1

        if "macd_hist" in row.index:
            if row["macd_hist"] > 0:
                buy_signals += 1
            else:
                sell_signals += 1

        if "price_vs_sma200" in row.index:
            pct = row["price_vs_sma200"]
            if pct > 0.05:
                buy_signals += 1
            elif pct < -0.05:
                sell_signals += 1

        if buy_signals >= 2 and buy_signals > sell_signals:
            return "BUY"
        if sell_signals >= 2 and sell_signals > buy_signals:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _compute_composite_score(
        ml_rec: Dict[str, Any], tech_features: pd.DataFrame, signal: str
    ) -> int:
        """
        Compute a 0-100 composite confidence score.

        When trained models are available the ML confidence score is used
        directly.  Without trained models a technical-indicator-based score
        is calculated from RSI, MACD histogram, SMA-200 proximity and ADX.
        """
        if ml_rec.get("model_votes"):
            return int(round(ml_rec["confidence"]))

        score = 50.0
        if tech_features.empty:
            return int(score)

        row = tech_features.iloc[-1]
        adjustments = 0.0

        if "rsi_14" in row.index:
            rsi = row["rsi_14"]
            if rsi < 30:
                adjustments += 15
            elif rsi < 40:
                adjustments += 8
            elif rsi > 70:
                adjustments -= 10
            elif rsi > 60:
                adjustments -= 3

        if "macd_hist" in row.index:
            adjustments += 8 if row["macd_hist"] > 0 else -8

        if "price_vs_sma200" in row.index:
            pct = row["price_vs_sma200"]
            if pct > 0.1:
                adjustments += 8
            elif pct > 0:
                adjustments += 4
            elif pct < -0.1:
                adjustments -= 8
            else:
                adjustments -= 4

        if "adx_14" in row.index and row["adx_14"] > 25:
            if "roc" in row.index:
                adjustments += 5 if row["roc"] > 0 else -5

        if "hist_vol_20" in row.index and not np.isnan(row["hist_vol_20"]):
            if row["hist_vol_20"] > 0.5:
                adjustments -= 5  # very high volatility = penalty

        score = float(np.clip(50 + adjustments, 10, 95))
        return int(round(score))

    @staticmethod
    def _compute_drivers(
        ml_rec: Dict[str, Any], tech_features: pd.DataFrame
    ) -> str:
        """
        Return a human-readable key-drivers string.

        Uses ML engine's ``key_drivers`` list when available; otherwise
        derives drivers from raw technical indicator values.
        """
        if ml_rec.get("key_drivers"):
            return ", ".join(str(d) for d in ml_rec["key_drivers"][:3])

        if tech_features.empty:
            return "Technical signals"

        row = tech_features.iloc[-1]
        parts: List[str] = []

        if "rsi_14" in row.index:
            rsi = row["rsi_14"]
            if rsi < 30:
                parts.append(f"Oversold RSI ({rsi:.0f})")
            elif rsi > 70:
                parts.append(f"Overbought RSI ({rsi:.0f})")
            else:
                parts.append(f"RSI {rsi:.0f}")

        if "macd_hist" in row.index:
            direction = "Bullish" if row["macd_hist"] > 0 else "Bearish"
            parts.append(f"{direction} MACD")

        if "price_vs_sma200" in row.index:
            pct = row["price_vs_sma200"] * 100
            label = "Above" if pct > 0 else "Below"
            parts.append(f"{label} 200-day MA ({pct:+.1f}%)")

        if "revenue_growth" in row.index and not np.isnan(row.get("revenue_growth", float("nan"))):
            val = row["revenue_growth"] * 100
            parts.append(f"Revenue growth {val:+.1f}%")

        return ", ".join(parts) if parts else "Technical signals"

    @staticmethod
    def _calculate_target_price(
        current_price: float, signal: str, score: int
    ) -> float:
        """
        Estimate a target price from entry price, signal direction, and score.

        BUY:  entry × (1 + 5%–20% depending on confidence)
        HOLD: entry × 1.02
        SELL: entry × (1 − 5%–15% depending on confidence)
        """
        normalised = score / 100.0
        if signal == "BUY":
            upside = 0.05 + normalised * 0.15
            return current_price * (1.0 + upside)
        if signal == "SELL":
            downside = 0.05 + normalised * 0.10
            return current_price * (1.0 - downside)
        return current_price * 1.02  # HOLD


# ---------------------------------------------------------------------------
# WatchlistBuilder
# ---------------------------------------------------------------------------


class WatchlistBuilder:
    """
    Build structured watchlist objects from live ML recommendations.

    Each watchlist is a dict matching the schema expected by the
    ``_section_watchlists`` renderer in ``risk_recommendations.py``.
    """

    def build_watchlist(
        self,
        name: str,
        strategy: str,
        description: str,
        tickers: List[str],
        risk_profiles: List[str],
        risk_level: str,
        generator: RecommendationGenerator,
        risk_profile: str,
    ) -> Dict[str, Any]:
        """
        Build a watchlist dict with live holdings and estimated performance.

        Parameters
        ----------
        name         : Display name (e.g. '🤖 AI Growth Leaders').
        strategy     : Strategy label (e.g. 'Growth', 'Income').
        description  : Short description shown in the UI.
        tickers      : Universe of tickers to screen.
        risk_profiles: Profiles this watchlist is suitable for.
        risk_level   : 'LOW', 'MEDIUM', or 'HIGH'.
        generator    : Configured ``RecommendationGenerator`` instance.
        risk_profile : Active user risk profile to use for screening.
        """
        holdings = generator.generate_recommendations(
            tickers, risk_profile, count=8
        )
        performance = self._estimate_performance(holdings)

        return {
            "name": name,
            "strategy": strategy,
            "description": description,
            "risk_level": risk_level,
            "risk_profiles": risk_profiles,
            "holdings_count": len(tickers),
            "performance": performance,
            "holdings": holdings,
        }

    @staticmethod
    def _estimate_performance(holdings: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Estimate win-rate, average return, and Sharpe ratio from holdings.

        When ML models are trained these numbers are driven by model
        confidence; otherwise they are reasonable technical-analysis
        estimates.
        """
        if not holdings:
            return {"win_rate": 0.50, "avg_return": 0.05, "sharpe": 0.80}

        avg_confidence = sum(h["confidence"] for h in holdings) / len(holdings)

        # Win rate scales with average confidence (50% → 0.55, 80% → 0.74)
        win_rate = round(float(np.clip(0.40 + avg_confidence * 0.004, 0.40, 0.85)), 2)

        # Average return estimated from target–entry upside, discounted
        upsides = [
            (h["target_price"] - h["entry_price"]) / h["entry_price"]
            for h in holdings
            if h.get("entry_price", 0) > 0
        ]
        avg_return = round(
            float(np.clip(float(np.mean(upsides)) * 0.7 if upsides else 0.05, 0.01, 0.30)),
            3,
        )

        # Sharpe loosely estimated from win rate
        sharpe = round(float(np.clip(win_rate * 2.0 - 0.30, 0.50, 2.50)), 2)

        return {"win_rate": win_rate, "avg_return": avg_return, "sharpe": sharpe}
