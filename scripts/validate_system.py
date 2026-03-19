#!/usr/bin/env python3
"""
Validation script for investment recommendation system.
Tests the entire pipeline on real stocks with detailed output.
"""

import sys
import os
from typing import Dict, List, Any

# Add repo root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np

from src.data_fetcher import DataFetcher
from src.ml_engine import RecommendationEngine, FeatureEngineer
from src.risk_engine import PortfolioRiskAnalyzer
from src.scoring_engine import (
    score_fundamentals_intelligent,
    contextualize_risk,
    score_ml_meaningfully,
    score_sentiment,
    score_etf_exposure,
    stretch_distribution,
)


class SystemValidator:
    def __init__(self):
        self.ml_engine = RecommendationEngine()
        self.feature_engineer = FeatureEngineer()
        self.risk_analyzer = PortfolioRiskAnalyzer()

    def validate_stock(self, ticker: str) -> Dict[str, Any]:
        """Evaluate a single stock comprehensively."""
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {ticker}")
        print(f"{'=' * 70}\n")

        try:
            # 1. Fetch price data
            print(f"[1/5] Fetching data for {ticker}...")
            fetcher = DataFetcher(ticker)
            price_data = fetcher.fetch_stock_data(period="2y")
            if price_data.empty:
                raise ValueError(f"No price data returned for {ticker}")

            current_price = float(price_data["Close"].iloc[-1])
            print(f"      ✓ Current Price: ${current_price:.2f}")
            print(f"      ✓ Data Points: {len(price_data)} days")

            # 2. Fundamentals Analysis
            print(f"\n[2/6] Analyzing Fundamentals...")
            fundamentals = fetcher.calculate_basic_ratios()
            fundamentals_score = self._score_fundamentals(fundamentals)

            print(f"      Fundamentals Score: {fundamentals_score:.1f}/10")
            for key, value in fundamentals.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"      • {key}: {value:.4f}")
                    else:
                        print(f"      • {key}: {value}")

            # 3. Technical Analysis
            print(f"\n[3/6] Analyzing Technicals...")
            technicals_df = self.feature_engineer.extract_technical_features(price_data)
            technicals = (
                technicals_df.iloc[-1].to_dict() if not technicals_df.empty else {}
            )
            technicals_score = self._score_technicals(technicals, price_data)

            print(f"      Technicals Score: {technicals_score:.1f}/10")
            _TECH_DISPLAY = [
                "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
                "stoch_k", "stoch_d", "adx_14", "atr_14",
                "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
                "volume_ratio",
            ]
            for key in _TECH_DISPLAY:
                value = technicals.get(key)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    print(f"      • {key}: {value:.4f}")

            # 4. Risk Assessment
            print(f"\n[4/6] Analyzing Risk...")
            risk_metrics = self._compute_risk_metrics(price_data)
            risk_score = contextualize_risk(risk_metrics, fundamentals, technicals)

            print(f"      Risk Score: {risk_score:.1f}/10")
            for key, value in risk_metrics.items():
                if isinstance(value, float):
                    print(f"      • {key}: {value:.4f}")
                else:
                    print(f"      • {key}: {value}")

            # 5. ML Prediction
            print(f"\n[5/6] Running ML Models...")
            ml_pred = self.ml_engine.predict(ticker, features=technicals_df)
            ml_score = score_ml_meaningfully(ml_pred, fundamentals)

            ml_confidence = ml_pred.get("confidence", 50.0)
            print(f"      ML Confidence: {ml_confidence:.1f}%")
            print(f"      Signal: {ml_pred.get('signal', 'HOLD')}")
            print(f"      ML Score: {ml_score:.1f}/10")
            model_votes = ml_pred.get("model_votes", {})
            if model_votes:
                print(f"      Model Votes: {model_votes}")

            # 6. Sentiment & ETF
            print(f"\n[6/6] Scoring Sentiment & ETF Exposure...")
            sentiment_score = score_sentiment(ticker)
            etf_score = score_etf_exposure(ticker)
            print(f"      Sentiment Score: {sentiment_score:.1f}/10")
            print(f"      ETF Exposure Score: {etf_score:.1f}/10")

            # 7. Calculate final score using intelligent weighted formula
            raw_score = (
                fundamentals_score * 0.40
                + technicals_score * 0.25
                + risk_score * 0.15
                + ml_score * 0.12
                + sentiment_score * 0.05
                + etf_score * 0.03
            )
            final_score = stretch_distribution(raw_score)

            result = {
                "ticker": ticker,
                "current_price": current_price,
                "final_score": final_score,
                "fundamentals_score": fundamentals_score,
                "technicals_score": technicals_score,
                "risk_score": risk_score,
                "ml_score": ml_score,
                "sentiment_score": sentiment_score,
                "etf_score": etf_score,
                "fundamentals": fundamentals,
                "technicals": technicals,
                "risk_metrics": risk_metrics,
                "ml_prediction": ml_pred,
                "reasons": self._generate_reasons(
                    ticker, fundamentals, technicals, risk_metrics, ml_pred
                ),
            }

            return result

        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            return {
                "ticker": ticker,
                "error": str(e),
                "final_score": 0,
            }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_fundamentals(self, fundamentals: Dict) -> float:
        """Score fundamentals 0-10 using intelligent scoring engine."""
        return score_fundamentals_intelligent(fundamentals)

    def _score_technicals(self, technicals: Dict, price_data: pd.DataFrame) -> float:
        """Score technicals 0-10."""
        score = 5.0

        rsi = technicals.get("rsi_14") or 50.0
        if np.isnan(rsi):
            rsi = 50.0
        if 40 <= rsi <= 60:
            score += 1
        elif rsi < 30:
            score += 2  # Oversold
        elif rsi > 70:
            score -= 1  # Overbought

        macd_hist = technicals.get("macd_hist") or 0.0
        if not np.isnan(macd_hist):
            if macd_hist > 0:
                score += 1
            elif macd_hist < 0:
                score -= 1

        # Price vs 200-day MA
        pct_vs_sma200 = technicals.get("price_vs_sma200")
        if pct_vs_sma200 is not None and not np.isnan(pct_vs_sma200):
            if pct_vs_sma200 > 0:
                score += 1
            else:
                score -= 1

        return max(0.0, min(10.0, score))

    def _compute_risk_metrics(self, price_data: pd.DataFrame) -> Dict:
        """Compute risk metrics from price history."""
        returns = price_data["Close"].pct_change().dropna()

        annual_vol = float(returns.std() * np.sqrt(252))
        annual_return = float(returns.mean() * 252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        # Max drawdown over the full history
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        return {
            "volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "annual_return": annual_return,
        }

    def _generate_reasons(
        self,
        ticker: str,
        fundamentals: Dict,
        technicals: Dict,
        risk_metrics: Dict,
        ml_pred: Dict,
    ) -> List[str]:
        """Generate 3-5 clear reasons for the score."""
        reasons = []

        # Fundamental reason
        pe = fundamentals.get("pe_ratio") or 0.0
        if pe > 0:
            label = "Reasonable valuation" if pe < 25 else "Premium valuation"
            reasons.append(f"P/E ratio of {pe:.1f}: {label}")

        # Technical reason
        rsi = technicals.get("rsi_14") or 50.0
        if np.isnan(rsi):
            rsi = 50.0
        if rsi < 30:
            rsi_label = "Oversold — potential bounce"
        elif rsi > 70:
            rsi_label = "Overbought — caution advised"
        else:
            rsi_label = "Neutral momentum"
        reasons.append(f"RSI(14) at {rsi:.0f}: {rsi_label}")

        # Trend reason
        pct_vs_sma200 = technicals.get("price_vs_sma200")
        if pct_vs_sma200 is not None and not np.isnan(pct_vs_sma200):
            direction = "above" if pct_vs_sma200 > 0 else "below"
            reasons.append(
                f"Price is {abs(pct_vs_sma200 * 100):.1f}% {direction} 200-day MA "
                f"({'uptrend' if pct_vs_sma200 > 0 else 'downtrend'})"
            )

        # Risk reason
        volatility = risk_metrics.get("volatility", 0.20)
        if volatility < 0.15:
            vol_label = "Low"
        elif volatility < 0.30:
            vol_label = "Moderate"
        else:
            vol_label = "High"
        reasons.append(
            f"Annualized volatility {volatility:.1%}: {vol_label} risk"
        )

        # ML reason
        ml_signal = ml_pred.get("signal", "HOLD")
        ml_confidence = ml_pred.get("confidence", 50.0)
        reasons.append(
            f"ML models predict {ml_signal} with {ml_confidence:.0f}% confidence"
        )

        # ROE reason
        roe = fundamentals.get("roe") or 0.0
        if roe > 0:
            roe_label = "Strong" if roe > 0.15 else "Moderate"
            reasons.append(f"ROE of {roe:.1%}: {roe_label} profitability")

        return reasons[:5]

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def compare_to_benchmark(
        self,
        stock_score: float,
        benchmark_name: str = "S&P 500",
        benchmark_score: float = 6.5,
    ) -> str:
        """Compare stock score to benchmark."""
        diff = stock_score - benchmark_score
        if diff > 0.5:
            return f"OUTPERFORMS {benchmark_name} (+{diff:.1f} points)"
        elif diff < -0.5:
            return f"UNDERPERFORMS {benchmark_name} ({diff:.1f} points)"
        else:
            return f"MATCHES {benchmark_name} (±{abs(diff):.1f} points)"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, results: List[Dict]) -> None:
        """Print summary comparison table followed by detailed results."""
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)

        valid = sorted(
            [r for r in results if "error" not in r],
            key=lambda x: x["final_score"],
            reverse=True,
        )
        errors = [r for r in results if "error" in r]

        print(
            f"\n{'Ticker':<12} {'Score':<8} {'Fundamentals':<14} "
            f"{'Technicals':<13} {'Risk':<8} {'ML':<8} {'Sentiment':<11} {'ETF':<6}"
        )
        print("-" * 80)

        for r in valid:
            print(
                f"{r['ticker']:<12} {r['final_score']:<8.1f} "
                f"{r['fundamentals_score']:<14.1f} {r['technicals_score']:<13.1f} "
                f"{r['risk_score']:<8.1f} {r['ml_score']:<8.1f} "
                f"{r.get('sentiment_score', 5.0):<11.1f} {r.get('etf_score', 5.0):<6.1f}"
            )

        if errors:
            print("\nFailed tickers:")
            for r in errors:
                print(f"  {r['ticker']}: {r['error']}")

        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)

        for r in valid:
            self._print_result(r)

    def _print_result(self, result: Dict) -> None:
        """Print detailed result for one stock."""
        print(f"\n{'=' * 70}")
        print(f"{result['ticker']} → Final Score: {result['final_score']:.1f}/10")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"{'=' * 70}")

        # Fundamentals
        print(f"\nFundamentals: {result['fundamentals_score']:.1f}/10")
        fund = result["fundamentals"]
        for key, value in fund.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  ├─ {key}: {value:.4f}")
                else:
                    print(f"  ├─ {key}: {value}")

        # Technicals
        print(f"\nTechnicals: {result['technicals_score']:.1f}/10")
        _DISPLAY_KEYS = [
            ("rsi_14", "RSI(14)"),
            ("rsi_7", "RSI(7)"),
            ("macd", "MACD"),
            ("macd_hist", "MACD Histogram"),
            ("stoch_k", "Stochastic %K"),
            ("stoch_d", "Stochastic %D"),
            ("adx_14", "ADX(14)"),
            ("price_vs_sma20", "Price vs SMA20"),
            ("price_vs_sma50", "Price vs SMA50"),
            ("price_vs_sma200", "Price vs SMA200"),
            ("volume_ratio", "Volume Ratio"),
            ("atr_14", "ATR(14)"),
        ]
        tech = result["technicals"]
        for key, label in _DISPLAY_KEYS:
            value = tech.get(key)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                suffix = "%" if "vs_sma" in key else ""
                display = f"{value * 100:.2f}{suffix}" if "vs_sma" in key else f"{value:.4f}"
                print(f"  ├─ {label}: {display}")

        # Risk
        print(f"\nRisk: {result['risk_score']:.1f}/10")
        rm = result["risk_metrics"]
        print(f"  ├─ Annualized Volatility: {rm['volatility']:.2%}")
        print(f"  ├─ Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
        print(f"  ├─ Max Drawdown: {rm['max_drawdown']:.2%}")
        print(f"  └─ Annual Return (hist): {rm['annual_return']:.2%}")

        # ML
        print(f"\nML Prediction: {result['ml_score']:.1f}/10")
        mp = result["ml_prediction"]
        print(f"  ├─ Signal: {mp.get('signal', 'N/A')}")
        print(f"  ├─ Confidence: {mp.get('confidence', 0):.0f}%")
        print(f"  ├─ Strength: {mp.get('strength', 'N/A')}")
        model_votes = mp.get("model_votes", {})
        if model_votes:
            print(f"  └─ Model Votes: {model_votes}")

        # Sentiment & ETF
        print(f"\nSentiment: {result.get('sentiment_score', 5.0):.1f}/10")
        print(f"ETF Exposure: {result.get('etf_score', 5.0):.1f}/10")

        # Key reasons
        print(f"\nKEY REASONS:")
        for i, reason in enumerate(result["reasons"], 1):
            print(f"  {i}. {reason}")

        # Benchmark comparisons
        print(f"\nBENCHMARK COMPARISON:")
        print(f"  {self.compare_to_benchmark(result['final_score'], 'S&P 500', 6.5)}")
        if ".NS" in result["ticker"]:
            print(f"  {self.compare_to_benchmark(result['final_score'], 'NIFTY 50', 6.3)}")

        print(f"{'=' * 70}")


def main():
    """Run validation on a representative set of US and Indian stocks."""
    validator = SystemValidator()

    tickers = [
        # US Tech
        "AAPL",
        "NVDA",
        "INTC",
        # India (NSE)
        "RELIANCE.NS",
        "INFY.NS",
        "SBIN.NS",
    ]

    results = []
    for ticker in tickers:
        result = validator.validate_stock(ticker)
        results.append(result)

    validator.print_summary(results)


if __name__ == "__main__":
    main()
