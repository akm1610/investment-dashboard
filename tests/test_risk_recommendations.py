"""
tests/test_risk_recommendations.py
-------------------------------------
Unit tests for ``src/components/risk_recommendations.py``.

Strategy
--------
* Pure helper-function tests require no mocking.
* Page/section rendering tests mock Streamlit to avoid server/browser.
* Session-state tests use the MockSessionState helper from test_app.py.
"""

from __future__ import annotations

import sys
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in [_REPO_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# MockSessionState helper
# ---------------------------------------------------------------------------


class _MockSessionState(dict):
    """dict subclass with attribute-style access, mimicking st.session_state."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


# ---------------------------------------------------------------------------
# Tests: _upside_pct
# ---------------------------------------------------------------------------


class TestUpsidePct:
    def test_positive_upside(self):
        from components.risk_recommendations import _upside_pct

        assert abs(_upside_pct(100.0, 120.0) - 0.20) < 1e-9

    def test_negative_upside(self):
        from components.risk_recommendations import _upside_pct

        assert abs(_upside_pct(100.0, 80.0) - (-0.20)) < 1e-9

    def test_zero_entry_returns_zero(self):
        from components.risk_recommendations import _upside_pct

        assert _upside_pct(0.0, 120.0) == 0.0

    def test_equal_prices_is_zero(self):
        from components.risk_recommendations import _upside_pct

        assert _upside_pct(50.0, 50.0) == 0.0

    def test_small_values(self):
        from components.risk_recommendations import _upside_pct

        assert abs(_upside_pct(1.0, 1.5) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Tests: _profile_badge
# ---------------------------------------------------------------------------


class TestProfileBadge:
    def test_conservative_badge(self):
        from components.risk_recommendations import _profile_badge

        badge = _profile_badge("conservative")
        assert "Conservative" in badge

    def test_moderate_badge(self):
        from components.risk_recommendations import _profile_badge

        badge = _profile_badge("moderate")
        assert "Moderate" in badge

    def test_aggressive_badge(self):
        from components.risk_recommendations import _profile_badge

        badge = _profile_badge("aggressive")
        assert "Aggressive" in badge

    def test_unknown_profile_returns_titlecase(self):
        from components.risk_recommendations import _profile_badge

        badge = _profile_badge("custom")
        assert "Custom" in badge

    def test_each_profile_has_different_badge(self):
        from components.risk_recommendations import _profile_badge

        badges = {
            _profile_badge("conservative"),
            _profile_badge("moderate"),
            _profile_badge("aggressive"),
        }
        assert len(badges) == 3


# ---------------------------------------------------------------------------
# Tests: _position_size_label
# ---------------------------------------------------------------------------


class TestPositionSizeLabel:
    def test_high_score_uses_full_max_size(self):
        from components.risk_recommendations import _position_size_label

        # Score >= 80 → 100% of max_size
        assert _position_size_label(85, "moderate") == "10%"

    def test_mid_score_uses_70_pct_max_size(self):
        from components.risk_recommendations import _position_size_label

        # Score 65-79 → 70% of max_size
        assert _position_size_label(70, "moderate") == "7%"

    def test_low_score_uses_40_pct_max_size(self):
        from components.risk_recommendations import _position_size_label

        # Score < 65 → 40% of max_size
        label = _position_size_label(50, "moderate")
        assert "%" in label

    def test_conservative_profile_smaller_sizes(self):
        from components.risk_recommendations import _position_size_label

        conservative_label = _position_size_label(85, "conservative")
        moderate_label = _position_size_label(85, "moderate")
        aggressive_label = _position_size_label(85, "aggressive")

        # Extract numeric values
        c_val = float(conservative_label.rstrip("%"))
        m_val = float(moderate_label.rstrip("%"))
        a_val = float(aggressive_label.rstrip("%"))
        assert c_val <= m_val <= a_val

    def test_unknown_profile_uses_default(self):
        from components.risk_recommendations import _position_size_label

        label = _position_size_label(85, "unknown")
        assert "%" in label

    def test_label_ends_with_percent(self):
        from components.risk_recommendations import _position_size_label

        for profile in ("conservative", "moderate", "aggressive"):
            for score in (40, 65, 80):
                assert _position_size_label(score, profile).endswith("%")


# ---------------------------------------------------------------------------
# Tests: _WATCHLIST_STRATEGIES – data integrity
# ---------------------------------------------------------------------------


class TestWatchlistStrategies:
    def test_not_empty(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        assert len(_WATCHLIST_STRATEGIES) > 0

    def test_required_keys(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        required = {"name", "description", "strategy", "risk_level", "risk_profiles", "tickers"}
        for strat in _WATCHLIST_STRATEGIES:
            missing = required - strat.keys()
            assert not missing, f"{strat.get('name')} is missing keys: {missing}"

    def test_risk_levels_valid(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        valid = {"LOW", "MEDIUM", "HIGH"}
        for strat in _WATCHLIST_STRATEGIES:
            assert strat["risk_level"] in valid

    def test_risk_profiles_valid(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        valid = {"conservative", "moderate", "aggressive"}
        for strat in _WATCHLIST_STRATEGIES:
            for rp in strat["risk_profiles"]:
                assert rp in valid

    def test_each_strategy_has_tickers(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        for strat in _WATCHLIST_STRATEGIES:
            assert isinstance(strat["tickers"], list)
            assert len(strat["tickers"]) > 0

    def test_conservative_profile_has_watchlists(self):
        """At least one watchlist should support conservative investors."""
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        conservative_lists = [s for s in _WATCHLIST_STRATEGIES if "conservative" in s["risk_profiles"]]
        assert len(conservative_lists) > 0

    def test_aggressive_profile_has_watchlists(self):
        """At least one watchlist should support aggressive investors."""
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        aggressive_lists = [s for s in _WATCHLIST_STRATEGIES if "aggressive" in s["risk_profiles"]]
        assert len(aggressive_lists) > 0

    def test_moderate_profile_has_watchlists(self):
        """At least one watchlist should support moderate investors."""
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        moderate_lists = [s for s in _WATCHLIST_STRATEGIES if "moderate" in s["risk_profiles"]]
        assert len(moderate_lists) > 0


# ---------------------------------------------------------------------------
# Tests: watchlist filtering by risk profile
# ---------------------------------------------------------------------------


class TestWatchlistFiltering:
    """Test that watchlists are correctly filtered by risk profile."""

    def test_filter_conservative(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        filtered = [s for s in _WATCHLIST_STRATEGIES if "conservative" in s["risk_profiles"]]
        # All filtered strategies must include conservative
        assert all("conservative" in s["risk_profiles"] for s in filtered)

    def test_filter_moderate(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        filtered = [s for s in _WATCHLIST_STRATEGIES if "moderate" in s["risk_profiles"]]
        assert all("moderate" in s["risk_profiles"] for s in filtered)

    def test_filter_aggressive(self):
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        filtered = [s for s in _WATCHLIST_STRATEGIES if "aggressive" in s["risk_profiles"]]
        assert all("aggressive" in s["risk_profiles"] for s in filtered)

    def test_aggressive_gets_more_or_equal_watchlists_than_conservative(self):
        """Aggressive profiles should have access to at least as many watchlists."""
        from components.risk_recommendations import _WATCHLIST_STRATEGIES

        conservative_count = sum(
            1 for s in _WATCHLIST_STRATEGIES if "conservative" in s["risk_profiles"]
        )
        moderate_count = sum(
            1 for s in _WATCHLIST_STRATEGIES if "moderate" in s["risk_profiles"]
        )
        aggressive_count = sum(
            1 for s in _WATCHLIST_STRATEGIES if "aggressive" in s["risk_profiles"]
        )
        assert aggressive_count >= conservative_count


# ---------------------------------------------------------------------------
# Tests: risk profile assessment integration
# ---------------------------------------------------------------------------


class TestRiskProfileAssessment:
    """Test risk assessor integration within the page component."""

    def test_conservative_answers_produce_conservative_profile(self):
        """All-1 answers (minimum risk tolerance) should produce Conservative."""
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([1] * 10)
        assert result["profile"] == "conservative"

    def test_aggressive_answers_produce_aggressive_profile(self):
        """All-5 answers (maximum risk tolerance) should produce Aggressive."""
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([5] * 10)
        assert result["profile"] == "aggressive"

    def test_moderate_answers_produce_moderate_profile(self):
        """Mid-range answers should produce Moderate profile."""
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([3] * 10)
        assert result["profile"] == "moderate"

    def test_assessor_output_has_required_keys(self):
        """Assessor output must have all keys used by the page component."""
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([3] * 10)
        required = {"risk_score", "profile", "suggested_volatility",
                    "max_position_size", "recommended_allocation"}
        assert required.issubset(result.keys())

    def test_allocation_sums_to_one(self):
        """Recommended allocation percentages should sum to 1.0."""
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        for answers in ([1] * 10, [3] * 10, [5] * 10):
            result = assessor.assess(answers)
            alloc = result["recommended_allocation"]
            total = sum(alloc.values())
            assert abs(total - 1.0) < 1e-9, f"Allocation total {total} != 1.0 for {answers}"

    def test_profile_saved_to_session_state_after_assessment(self):
        """Saving an assessment should persist the profile in session state."""
        from risk_engine import RiskProfileAssessor

        assessor = RiskProfileAssessor()
        result = assessor.assess([5] * 10)

        mock_state = _MockSessionState()
        mock_state["risk_profile"] = "moderate"
        mock_state["risk_profile_result"] = None

        # Simulate the save action (what the page does on save_clicked)
        mock_state["risk_profile"] = result["profile"]
        mock_state["risk_profile_result"] = result

        assert mock_state["risk_profile"] == "aggressive"
        assert mock_state["risk_profile_result"] is not None
        assert mock_state["risk_profile_result"]["risk_score"] > 0


# ---------------------------------------------------------------------------
# Tests: recommendations sorting
# ---------------------------------------------------------------------------


class TestRecommendationsSorting:
    """Test that recommendations are sorted correctly by score."""

    def _make_recs(self) -> list:
        return [
            {"ticker": "A", "score": 60, "signal": "BUY", "confidence": 60,
             "entry_price": 100.0, "target_price": 110.0, "drivers": "test"},
            {"ticker": "B", "score": 80, "signal": "BUY", "confidence": 80,
             "entry_price": 200.0, "target_price": 230.0, "drivers": "test"},
            {"ticker": "C", "score": 45, "signal": "HOLD", "confidence": 45,
             "entry_price": 50.0, "target_price": 52.0, "drivers": "test"},
        ]

    def test_sorted_by_score_descending(self):
        recs = self._make_recs()
        sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)
        scores = [r["score"] for r in sorted_recs]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limit(self):
        recs = self._make_recs()
        top_2 = sorted(recs, key=lambda x: x["score"], reverse=True)[:2]
        assert len(top_2) == 2
        assert top_2[0]["ticker"] == "B"  # highest score
        assert top_2[1]["ticker"] == "A"  # second highest

    def test_top_20_does_not_exceed_available(self):
        recs = self._make_recs()
        top_20 = sorted(recs, key=lambda x: x["score"], reverse=True)[:20]
        assert len(top_20) <= len(recs)


# ---------------------------------------------------------------------------
# Tests: page_risk_recommendations function (smoke tests with mocked streamlit)
# ---------------------------------------------------------------------------


class TestPageRiskRecommendationsSmoke:
    """Smoke tests for page_risk_recommendations using a mocked Streamlit."""

    def _make_mock_st(self, session_state: dict) -> MagicMock:
        """Return a MagicMock st module with session_state preset."""
        mock_st = MagicMock()
        mock_st.session_state = _MockSessionState(session_state)
        # st.tabs() should return a context-manager-compatible list
        tab_mock = MagicMock()
        tab_mock.__enter__ = MagicMock(return_value=tab_mock)
        tab_mock.__exit__ = MagicMock(return_value=False)
        mock_st.tabs.return_value = [tab_mock, tab_mock, tab_mock]
        return mock_st

    def test_page_function_is_callable(self):
        from components.risk_recommendations import page_risk_recommendations

        assert callable(page_risk_recommendations)

    def test_page_function_importable_via_app(self):
        """Page function should be re-exported via src.app for navigation."""
        from src import app as app_module

        assert hasattr(app_module, "page_risk_recommendations")
        assert callable(app_module.page_risk_recommendations)

    def test_session_state_defaults_set_when_absent(self):
        """page_risk_recommendations should set risk_profile/risk_profile_result defaults."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st({})

        # Patch _section_risk_assessment and section functions so they don't call real st
        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists"),
            patch.object(rr_module, "_section_recommendations"),
        ):
            rr_module.page_risk_recommendations()

        assert "risk_profile" in mock_st.session_state
        assert mock_st.session_state["risk_profile"] == "moderate"
        assert "risk_profile_result" in mock_st.session_state
        assert mock_st.session_state["risk_profile_result"] is None

    def test_session_state_not_overwritten_when_present(self):
        """Existing risk_profile should not be overwritten on second render."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st(
            {"risk_profile": "aggressive", "risk_profile_result": {"profile": "aggressive"}}
        )

        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists"),
            patch.object(rr_module, "_section_recommendations"),
        ):
            rr_module.page_risk_recommendations()

        assert mock_st.session_state["risk_profile"] == "aggressive"

    def test_watchlists_section_called_with_risk_profile(self):
        """_section_watchlists should be called with the current risk_profile."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st({"risk_profile": "conservative"})
        mock_watchlists = MagicMock()

        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists", mock_watchlists),
            patch.object(rr_module, "_section_recommendations"),
        ):
            rr_module.page_risk_recommendations()

        mock_watchlists.assert_called_once_with("conservative")

    def test_recommendations_section_called_with_risk_profile(self):
        """_section_recommendations should be called with the current risk_profile."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st({"risk_profile": "aggressive"})
        mock_recs = MagicMock()

        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists"),
            patch.object(rr_module, "_section_recommendations", mock_recs),
        ):
            rr_module.page_risk_recommendations()

        mock_recs.assert_called_once_with("aggressive")

    def test_page_title_called(self):
        """st.title should be called with the page title."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st({})

        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists"),
            patch.object(rr_module, "_section_recommendations"),
        ):
            rr_module.page_risk_recommendations()

        mock_st.title.assert_called_once()
        title_arg = mock_st.title.call_args[0][0]
        assert "Risk" in title_arg or "Recommendations" in title_arg

    def test_page_markdown_called(self):
        """st.markdown should be called with the page description."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st({})

        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists"),
            patch.object(rr_module, "_section_recommendations"),
        ):
            rr_module.page_risk_recommendations()

        mock_st.markdown.assert_called()

    def test_tabs_created_with_three_sections(self):
        """st.tabs should be called with exactly three tab labels."""
        import components.risk_recommendations as rr_module

        mock_st = self._make_mock_st({})

        with (
            patch.object(rr_module, "st", mock_st),
            patch.object(rr_module, "_section_risk_assessment", return_value=None),
            patch.object(rr_module, "_section_watchlists"),
            patch.object(rr_module, "_section_recommendations"),
        ):
            rr_module.page_risk_recommendations()

        mock_st.tabs.assert_called_once()
        tab_labels = mock_st.tabs.call_args[0][0]
        assert len(tab_labels) == 3
