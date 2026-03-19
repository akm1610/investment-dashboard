"""
Tests for src/scoring_engine.py
"""

import pytest
from src.scoring_engine import (
    score_fundamentals_intelligent,
    contextualize_risk,
    score_ml_intelligently,
    score_sentiment,
    score_etf_exposure,
    stretch_distribution,
)


# ---------------------------------------------------------------------------
# score_fundamentals_intelligent
# ---------------------------------------------------------------------------

class TestScoreFundamentalsIntelligent:
    def test_returns_float_in_range(self):
        result = score_fundamentals_intelligent({})
        assert 0.0 <= result <= 10.0

    def test_exceptional_roe_boosts_score(self):
        high_roe = score_fundamentals_intelligent({"roe": 0.25})
        low_roe = score_fundamentals_intelligent({"roe": 0.02})
        assert high_roe > low_roe

    def test_negative_roe_lowers_score(self):
        negative = score_fundamentals_intelligent({"roe": -0.10})
        neutral = score_fundamentals_intelligent({"roe": 0.0})
        assert negative < neutral

    def test_high_gross_margin_rewarded(self):
        high_margin = score_fundamentals_intelligent({"gross_margin": 0.50})
        low_margin = score_fundamentals_intelligent({"gross_margin": 0.10})
        assert high_margin > low_margin

    def test_high_operating_margin_rewarded(self):
        high = score_fundamentals_intelligent({"operating_margin": 0.30})
        low = score_fundamentals_intelligent({"operating_margin": 0.05})
        assert high > low

    def test_low_peg_rewarded(self):
        # PE=20, growth=0.40 → PEG=0.5 (cheap)
        good = score_fundamentals_intelligent({"pe_ratio": 20.0, "earnings_growth": 0.40})
        # PE=50, growth=0.05 → PEG=10.0 (expensive)
        bad = score_fundamentals_intelligent({"pe_ratio": 50.0, "earnings_growth": 0.05})
        assert good > bad

    def test_high_debt_penalised(self):
        safe = score_fundamentals_intelligent({"debt_to_equity": 0.3})
        risky = score_fundamentals_intelligent({"debt_to_equity": 4.0})
        assert safe > risky

    def test_low_current_ratio_penalised(self):
        liquid = score_fundamentals_intelligent({"current_ratio": 2.5})
        illiquid = score_fundamentals_intelligent({"current_ratio": 0.4})
        assert liquid > illiquid

    def test_missing_values_neutral(self):
        score = score_fundamentals_intelligent({})
        assert 4.5 <= score <= 5.5  # Should be near neutral baseline of 5

    def test_nvda_profile_scores_high(self):
        """NVIDIA-like profile should score significantly above 5."""
        nvda = {
            "roe": 0.60,
            "gross_margin": 0.73,
            "operating_margin": 0.55,
            "pe_ratio": 35.0,
            "earnings_growth": 0.80,
            "debt_to_equity": 0.40,
            "current_ratio": 4.0,
        }
        score = score_fundamentals_intelligent(nvda)
        assert 7.0 <= score <= 10.0, f"NVDA-like profile scored {score:.2f}, expected 7.0–10.0"

    def test_intc_weak_profile_scores_low(self):
        """INTC-like weak profile (negative ROE, low margins) should score below 5."""
        intc = {
            "roe": -0.05,
            "gross_margin": 0.30,
            "operating_margin": -0.10,
            "pe_ratio": 0.0,
            "debt_to_equity": 1.8,
            "current_ratio": 1.5,
        }
        score = score_fundamentals_intelligent(intc)
        assert 0.0 <= score <= 5.5, f"INTC-like profile scored {score:.2f}, expected 0.0–5.5"

    def test_clamped_to_zero_ten(self):
        # Pathological all-bad inputs should not go below 0
        bad = {
            "roe": -1.0,
            "gross_margin": 0.0,
            "operating_margin": -0.5,
            "pe_ratio": 200.0,
            "earnings_growth": 0.01,
            "debt_to_equity": 10.0,
            "current_ratio": 0.1,
        }
        assert score_fundamentals_intelligent(bad) >= 0.0

        # All-great inputs should not exceed 10
        great = {
            "roe": 0.99,
            "gross_margin": 0.99,
            "operating_margin": 0.99,
            "pe_ratio": 10.0,
            "earnings_growth": 0.99,
            "debt_to_equity": 0.0,
            "current_ratio": 5.0,
        }
        assert score_fundamentals_intelligent(great) <= 10.0


# ---------------------------------------------------------------------------
# contextualize_risk
# ---------------------------------------------------------------------------

class TestContextualizeRisk:
    def test_returns_float_in_range(self):
        result = contextualize_risk({}, {}, {})
        assert 0.0 <= result <= 10.0

    def test_low_volatility_higher_score(self):
        low_vol = contextualize_risk({"volatility": 0.10}, {}, {})
        high_vol = contextualize_risk({"volatility": 0.70}, {}, {})
        assert low_vol > high_vol

    def test_high_sharpe_rewarded(self):
        good = contextualize_risk({"sharpe_ratio": 2.0}, {}, {})
        bad = contextualize_risk({"sharpe_ratio": -0.5}, {}, {})
        assert good > bad

    def test_shallow_drawdown_rewarded(self):
        shallow = contextualize_risk({"max_drawdown": -0.05}, {}, {})
        deep = contextualize_risk({"max_drawdown": -0.60}, {}, {})
        assert shallow > deep

    def test_quality_growth_context_bonus(self):
        """High ROE + high volatility should get a context bonus vs same
        volatility with no quality fundamentals."""
        high_quality = contextualize_risk(
            {"volatility": 0.50}, {"roe": 0.25}, {}
        )
        low_quality = contextualize_risk(
            {"volatility": 0.50}, {"roe": 0.00}, {}
        )
        assert high_quality > low_quality

    def test_uptrend_reduces_volatility_penalty(self):
        """Stock above SMA200 with high vol should score higher than same vol
        below SMA200."""
        uptrend = contextualize_risk(
            {"volatility": 0.45},
            {},
            {"price_vs_sma200": 0.10},
        )
        downtrend = contextualize_risk(
            {"volatility": 0.45},
            {},
            {"price_vs_sma200": -0.10},
        )
        assert uptrend > downtrend

    def test_clamped(self):
        assert contextualize_risk(
            {"volatility": 0.01, "sharpe_ratio": 5.0, "max_drawdown": 0.0},
            {"roe": 0.99},
            {"price_vs_sma200": 0.99},
        ) <= 10.0

        assert contextualize_risk(
            {"volatility": 2.0, "sharpe_ratio": -5.0, "max_drawdown": -1.0},
            {},
            {},
        ) >= 0.0


# ---------------------------------------------------------------------------
# score_ml_intelligently
# ---------------------------------------------------------------------------

class TestScoreMlIntelligently:
    def test_returns_float_in_range(self):
        result = score_ml_intelligently({}, {}, {})
        assert 0.0 <= result <= 10.0

    def test_buy_signal_high_confidence_boosts_score(self):
        buy_high = score_ml_intelligently(
            {"signal": "BUY", "confidence": 90, "model_votes": {"a": "BUY", "b": "BUY", "c": "BUY"}},
            {},
            {},
        )
        hold = score_ml_intelligently({"signal": "HOLD", "confidence": 50}, {}, {})
        assert buy_high > hold

    def test_sell_signal_lowers_score(self):
        sell = score_ml_intelligently(
            {"signal": "SELL", "confidence": 80, "model_votes": {"a": "SELL", "b": "SELL"}},
            {},
            {},
        )
        neutral = score_ml_intelligently({"signal": "HOLD", "confidence": 50}, {}, {})
        assert sell < neutral

    def test_model_agreement_bonus_for_buy(self):
        unanimous = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70,
             "model_votes": {"a": "BUY", "b": "BUY", "c": "BUY", "d": "BUY"}},
            {},
            {},
        )
        split = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70,
             "model_votes": {"a": "BUY", "b": "SELL", "c": "HOLD", "d": "BUY"}},
            {},
            {},
        )
        assert unanimous > split

    def test_buy_with_negative_roe_penalised(self):
        buy_bad = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {"roe": -0.10},
            {},
        )
        buy_good = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {"roe": 0.0},
            {},
        )
        assert buy_bad < buy_good

    def test_sell_with_strong_fundamentals_softened(self):
        sell_strong = score_ml_intelligently(
            {"signal": "SELL", "confidence": 70},
            {"roe": 0.25, "operating_margin": 0.25},
            {},
        )
        sell_weak = score_ml_intelligently(
            {"signal": "SELL", "confidence": 70},
            {"roe": 0.0},
            {},
        )
        assert sell_strong > sell_weak

    def test_neutral_hold_near_5(self):
        score = score_ml_intelligently({"signal": "HOLD", "confidence": 50}, {}, {})
        assert 4.0 <= score <= 6.0

    def test_oversold_rsi_bonus_for_buy(self):
        buy_oversold = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {},
            {"rsi_14": 25},
        )
        buy_neutral_rsi = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {},
            {"rsi_14": 50},
        )
        assert buy_oversold > buy_neutral_rsi

    def test_overbought_rsi_penalty_for_buy(self):
        buy_overbought = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {},
            {"rsi_14": 75},
        )
        buy_neutral_rsi = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {},
            {"rsi_14": 50},
        )
        assert buy_overbought < buy_neutral_rsi

    def test_uptrend_confirms_buy(self):
        buy_uptrend = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {},
            {"price_vs_sma200": 0.10},
        )
        buy_downtrend = score_ml_intelligently(
            {"signal": "BUY", "confidence": 70},
            {},
            {"price_vs_sma200": -0.15},
        )
        assert buy_uptrend > buy_downtrend

    def test_clamped(self):
        assert score_ml_intelligently(
            {"signal": "BUY", "confidence": 100,
             "model_votes": {"a": "BUY", "b": "BUY"}},
            {"roe": 0.50, "operating_margin": 0.50},
            {"rsi_14": 25, "price_vs_sma200": 0.20},
        ) <= 10.0

        assert score_ml_intelligently(
            {"signal": "SELL", "confidence": 100,
             "model_votes": {"a": "SELL", "b": "SELL"}},
            {"roe": -0.50},
            {"rsi_14": 80},
        ) >= 0.0


# ---------------------------------------------------------------------------
# score_sentiment
# ---------------------------------------------------------------------------

class TestScoreSentiment:
    def test_returns_float_in_range(self):
        result = score_sentiment("AAPL")
        assert 0.0 <= result <= 10.0

    def test_returns_neutral_when_no_data(self):
        # Stubs return None so the score stays at baseline 5.0
        result = score_sentiment("AAPL")
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# score_etf_exposure
# ---------------------------------------------------------------------------

class TestScoreEtfExposure:
    def test_returns_float_in_range(self):
        result = score_etf_exposure("AAPL")
        assert 0.0 <= result <= 10.0

    def test_returns_neutral_when_no_data(self):
        result = score_etf_exposure("AAPL")
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# stretch_distribution
# ---------------------------------------------------------------------------

class TestStretchDistribution:
    def test_midpoint_unchanged(self):
        assert stretch_distribution(5.5) == pytest.approx(5.5)

    def test_above_midpoint_stretched_higher(self):
        assert stretch_distribution(7.0) > 7.0

    def test_below_midpoint_stretched_lower(self):
        assert stretch_distribution(4.0) < 4.0

    def test_order_preserved(self):
        scores = [3.0, 4.5, 5.5, 6.5, 8.0]
        stretched = [stretch_distribution(s) for s in scores]
        assert stretched == sorted(stretched)

    def test_clamped_to_zero_ten(self):
        assert stretch_distribution(-100.0) == pytest.approx(0.0)
        assert stretch_distribution(100.0) == pytest.approx(10.0)

    def test_high_scores_use_upper_range(self):
        """Strong stocks (raw ~7-8) should stretch toward 8-9."""
        assert stretch_distribution(7.5) >= 7.5

    def test_low_scores_use_lower_range(self):
        """Weak stocks (raw ~3-4) should stretch toward 1-3."""
        assert stretch_distribution(3.5) <= 3.5

    def test_custom_factor(self):
        """Wider stretch factor should produce more extreme results."""
        narrow = stretch_distribution(7.0, factor=1.0)
        wide = stretch_distribution(7.0, factor=2.0)
        assert wide > narrow

    def test_differentiation_between_stocks(self):
        """The stretch should create meaningful gaps between similar raw scores."""
        raw_gap = 1.0
        raw_a, raw_b = 5.5, 6.5
        stretched_a = stretch_distribution(raw_a)
        stretched_b = stretch_distribution(raw_b)
        stretched_gap = stretched_b - stretched_a
        # Gap should be larger after stretching (factor=1.4 so gap becomes 1.4)
        assert stretched_gap > raw_gap
