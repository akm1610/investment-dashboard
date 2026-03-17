"""
tests/test_app.py
-----------------
Unit tests for the Streamlit dashboard (src/app.py) and its helper components.

Strategy
--------
* All Streamlit API calls are mocked via ``pytest-mock`` so the tests run
  without a browser/server.
* Tests focus on:
  - Session state initialisation (``_init_session_state``)
  - Utility / formatting helpers in ``src/components/utils.py``
  - Journal helpers (``_entries_to_df``, ``_filter_entries``)
  - Portfolio overview helpers (``_holdings_dataframe``)
"""

from __future__ import annotations

import sys
import os
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup – so pytest can import root-level modules without an install step
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in [_REPO_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_portfolio():
    import portfolio_manager as pm
    return pm._empty_portfolio()


@pytest.fixture
def portfolio_with_entries(empty_portfolio):
    import portfolio_manager as pm
    pm.add_journal_entry(empty_portfolio, "AAPL", "Strong moat", tags=["high", "growth"])
    pm.add_journal_entry(empty_portfolio, "MSFT", "Cloud dominance", tags=["medium"])
    pm.add_journal_entry(empty_portfolio, "AAPL", "Revisited thesis", tags=["low"])
    return empty_portfolio


@pytest.fixture
def sample_alloc_df():
    """A minimal allocation DataFrame matching portfolio_manager output."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "JNJ"],
            "shares": [10.0, 5.0, 8.0],
            "avg_cost": [150.0, 300.0, 160.0],
            "current_price": [180.0, 380.0, 155.0],
            "market_value": [1800.0, 1900.0, 1240.0],
            "cost_basis": [1500.0, 1500.0, 1280.0],
            "unrealized_pnl": [300.0, 400.0, -40.0],
            "pnl_pct": [0.20, 0.267, -0.031],
            "weight": [0.364, 0.384, 0.250],
            "sector": ["Technology", "Technology", "Healthcare"],
        }
    )


# ---------------------------------------------------------------------------
# Tests: utils.format_currency
# ---------------------------------------------------------------------------


class TestFormatCurrency:
    def test_positive_value(self):
        from components.utils import format_currency
        assert format_currency(1234567.89) == "$1,234,567.89"

    def test_zero(self):
        from components.utils import format_currency
        assert format_currency(0.0) == "$0.00"

    def test_negative_value(self):
        from components.utils import format_currency
        result = format_currency(-500.0)
        assert result == "$-500.00"

    def test_none_returns_na(self):
        from components.utils import format_currency
        assert format_currency(None) == "N/A"  # type: ignore[arg-type]

    def test_custom_symbol(self):
        from components.utils import format_currency
        assert format_currency(100.0, symbol="€") == "€100.00"


# ---------------------------------------------------------------------------
# Tests: utils.format_large_number
# ---------------------------------------------------------------------------


class TestFormatLargeNumber:
    def test_billions(self):
        from components.utils import format_large_number
        assert format_large_number(2_500_000_000) == "$2.50B"

    def test_millions(self):
        from components.utils import format_large_number
        assert format_large_number(750_000_000) == "$750.00M"

    def test_thousands(self):
        from components.utils import format_large_number
        assert format_large_number(5_000) == "$5.00K"

    def test_trillions(self):
        from components.utils import format_large_number
        assert format_large_number(3_100_000_000_000) == "$3.10T"

    def test_none_returns_na(self):
        from components.utils import format_large_number
        assert format_large_number(None) == "N/A"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: utils.format_pct
# ---------------------------------------------------------------------------


class TestFormatPct:
    def test_typical(self):
        from components.utils import format_pct
        assert format_pct(0.1234) == "12.3%"

    def test_negative(self):
        from components.utils import format_pct
        assert format_pct(-0.05) == "-5.0%"

    def test_zero(self):
        from components.utils import format_pct
        assert format_pct(0.0) == "0.0%"

    def test_none_returns_na(self):
        from components.utils import format_pct
        assert format_pct(None) == "N/A"

    def test_custom_decimals(self):
        from components.utils import format_pct
        assert format_pct(0.12345, decimals=2) == "12.35%"


# ---------------------------------------------------------------------------
# Tests: utils.format_ratio
# ---------------------------------------------------------------------------


class TestFormatRatio:
    def test_typical(self):
        from components.utils import format_ratio
        assert format_ratio(1.5678) == "1.57"

    def test_none_returns_na(self):
        from components.utils import format_ratio
        assert format_ratio(None) == "N/A"

    def test_custom_decimals(self):
        from components.utils import format_ratio
        assert format_ratio(3.14159, decimals=4) == "3.1416"


# ---------------------------------------------------------------------------
# Tests: utils.get_color_for_value
# ---------------------------------------------------------------------------


class TestGetColorForValue:
    def test_green_when_above_good_min(self):
        from components.utils import get_color_for_value
        assert get_color_for_value(2.0, good_min=1.5, bad_max=1.0) == "green"

    def test_red_when_below_bad_max(self):
        from components.utils import get_color_for_value
        assert get_color_for_value(0.5, good_min=1.5, bad_max=1.0) == "red"

    def test_orange_in_between(self):
        from components.utils import get_color_for_value
        assert get_color_for_value(1.2, good_min=1.5, bad_max=1.0) == "orange"

    def test_inverted_green_when_low(self):
        from components.utils import get_color_for_value
        # Low P/E is good (invert=True)
        assert get_color_for_value(10.0, good_min=35.0, bad_max=15.0, invert=True) == "green"

    def test_inverted_red_when_high(self):
        from components.utils import get_color_for_value
        assert get_color_for_value(40.0, good_min=35.0, bad_max=15.0, invert=True) == "red"

    def test_non_numeric_returns_grey(self):
        from components.utils import get_color_for_value
        assert get_color_for_value(None, good_min=1.0, bad_max=0.5) == "grey"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: utils.score_color
# ---------------------------------------------------------------------------


class TestScoreColor:
    def test_green_high_score(self):
        from components.utils import score_color
        assert score_color(85) == "green"

    def test_orange_mid_score(self):
        from components.utils import score_color
        assert score_color(55) == "orange"

    def test_red_low_score(self):
        from components.utils import score_color
        assert score_color(25) == "red"

    def test_boundary_70_green(self):
        from components.utils import score_color
        assert score_color(70) == "green"

    def test_boundary_40_orange(self):
        from components.utils import score_color
        assert score_color(40) == "orange"


# ---------------------------------------------------------------------------
# Tests: utils.status_emoji
# ---------------------------------------------------------------------------


class TestStatusEmoji:
    def test_pass(self):
        from components.utils import status_emoji
        assert status_emoji("pass") == "✅"

    def test_warn(self):
        from components.utils import status_emoji
        assert status_emoji("warn") == "⚠️"

    def test_fail(self):
        from components.utils import status_emoji
        assert status_emoji("fail") == "❌"

    def test_unknown(self):
        from components.utils import status_emoji
        assert status_emoji("unknown") == "❓"

    def test_case_insensitive(self):
        from components.utils import status_emoji
        assert status_emoji("PASS") == "✅"


# ---------------------------------------------------------------------------
# Tests: utils.display_score_gauge
# ---------------------------------------------------------------------------


class TestDisplayScoreGauge:
    def test_returns_figure(self):
        from components.utils import display_score_gauge
        import plotly.graph_objects as go
        fig = display_score_gauge(75.0, "Test Score")
        assert isinstance(fig, go.Figure)

    def test_clamps_above_100(self):
        from components.utils import display_score_gauge
        fig = display_score_gauge(150.0, "Over")
        # Indicator value should be clamped to 100
        indicator = fig.data[0]
        assert indicator.value == 100.0

    def test_clamps_below_0(self):
        from components.utils import display_score_gauge
        fig = display_score_gauge(-10.0, "Under")
        indicator = fig.data[0]
        assert indicator.value == 0.0


# ---------------------------------------------------------------------------
# Tests: investment_journal._entries_to_df
# ---------------------------------------------------------------------------


class TestEntriesToDf:
    def test_returns_dataframe(self, portfolio_with_entries):
        from components.investment_journal import _entries_to_df
        import portfolio_manager as pm
        entries = pm.get_journal(portfolio_with_entries)
        df = _entries_to_df(entries)
        assert isinstance(df, pd.DataFrame)
        assert "Symbol" in df.columns
        assert "Date" in df.columns

    def test_empty_list_returns_empty_df(self):
        from components.investment_journal import _entries_to_df
        df = _entries_to_df([])
        assert df.empty

    def test_thesis_preview_truncated(self, empty_portfolio):
        import portfolio_manager as pm
        from components.investment_journal import _entries_to_df
        long_thesis = "x" * 200
        pm.add_journal_entry(empty_portfolio, "TSLA", long_thesis)
        entries = pm.get_journal(empty_portfolio)
        df = _entries_to_df(entries)
        preview = df.loc[0, "Thesis Preview"]
        assert len(preview) <= 83  # 80 chars + "…"
        assert preview.endswith("…")

    def test_short_thesis_not_truncated(self, empty_portfolio):
        import portfolio_manager as pm
        from components.investment_journal import _entries_to_df
        short_thesis = "Short thesis"
        pm.add_journal_entry(empty_portfolio, "TSLA", short_thesis)
        entries = pm.get_journal(empty_portfolio)
        df = _entries_to_df(entries)
        assert df.loc[0, "Thesis Preview"] == short_thesis


# ---------------------------------------------------------------------------
# Tests: investment_journal._filter_entries
# ---------------------------------------------------------------------------


class TestFilterEntries:
    def test_filter_by_ticker(self, portfolio_with_entries):
        import portfolio_manager as pm
        from components.investment_journal import _filter_entries
        entries = pm.get_journal(portfolio_with_entries)
        result = _filter_entries(entries, ["AAPL"], None, None, [], "")
        assert all(e["symbol"] == "AAPL" for e in result)
        assert len(result) == 2  # two AAPL entries

    def test_filter_by_date_excludes_future(self, portfolio_with_entries):
        import portfolio_manager as pm
        from components.investment_journal import _filter_entries
        entries = pm.get_journal(portfolio_with_entries)
        # Date range entirely in the future → nothing should match
        result = _filter_entries(
            entries,
            [],
            date(2099, 1, 1),
            date(2099, 12, 31),
            [],
            "",
        )
        assert result == []

    def test_filter_by_date_includes_all_recent(self, portfolio_with_entries):
        import portfolio_manager as pm
        from components.investment_journal import _filter_entries
        entries = pm.get_journal(portfolio_with_entries)
        result = _filter_entries(
            entries,
            [],
            date(2000, 1, 1),
            date(2099, 12, 31),
            [],
            "",
        )
        assert len(result) == 3

    def test_filter_by_conviction(self, portfolio_with_entries):
        import portfolio_manager as pm
        from components.investment_journal import _filter_entries
        entries = pm.get_journal(portfolio_with_entries)
        result = _filter_entries(entries, [], None, None, ["HIGH"], "")
        # Only the AAPL entry with tag "high"
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    def test_full_text_search(self, portfolio_with_entries):
        import portfolio_manager as pm
        from components.investment_journal import _filter_entries
        entries = pm.get_journal(portfolio_with_entries)
        result = _filter_entries(entries, [], None, None, [], "cloud")
        assert len(result) == 1
        assert result[0]["symbol"] == "MSFT"

    def test_empty_filters_returns_all(self, portfolio_with_entries):
        import portfolio_manager as pm
        from components.investment_journal import _filter_entries
        entries = pm.get_journal(portfolio_with_entries)
        result = _filter_entries(entries, [], None, None, [], "")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: portfolio_overview._holdings_dataframe
# ---------------------------------------------------------------------------


class TestHoldingsDataframe:
    def test_returns_dataframe(self, sample_alloc_df):
        from components.portfolio_overview import _holdings_dataframe
        result = _holdings_dataframe(sample_alloc_df)
        assert isinstance(result, pd.DataFrame)

    def test_column_names_renamed(self, sample_alloc_df):
        from components.portfolio_overview import _holdings_dataframe
        result = _holdings_dataframe(sample_alloc_df)
        assert "Symbol" in result.columns
        assert "Market Value" in result.columns
        assert "Unrealized P&L" in result.columns

    def test_monetary_values_formatted(self, sample_alloc_df):
        from components.portfolio_overview import _holdings_dataframe
        result = _holdings_dataframe(sample_alloc_df)
        # Market value for AAPL: 1800.0 → "$1,800.00"
        aapl_row = result[result["Symbol"] == "AAPL"].iloc[0]
        assert aapl_row["Market Value"] == "$1,800.00"

    def test_pnl_pct_formatted_as_percentage(self, sample_alloc_df):
        from components.portfolio_overview import _holdings_dataframe
        result = _holdings_dataframe(sample_alloc_df)
        aapl_row = result[result["Symbol"] == "AAPL"].iloc[0]
        assert "%" in aapl_row["P&L %"]

    def test_empty_alloc_returns_empty(self):
        from components.portfolio_overview import _holdings_dataframe
        result = _holdings_dataframe(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Mock session state helper
# ---------------------------------------------------------------------------


class _MockSessionState(dict):
    """dict subclass that also supports attribute-style get/set access,
    mimicking streamlit's SessionState object in unit tests."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


# ---------------------------------------------------------------------------
# Tests: session state initialisation (mocked Streamlit)
# ---------------------------------------------------------------------------


class TestInitSessionState:
    def test_portfolio_initialised(self, monkeypatch):
        """_init_session_state should set portfolio if missing."""
        from src import app as app_module

        mock_state = _MockSessionState()

        def _fake_load(path: str = "portfolio.json"):
            return {"holdings": {}, "cash": 0.0, "journal": [], "trades": []}

        monkeypatch.setattr(app_module.pm, "load_portfolio", _fake_load)
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert "portfolio" in mock_state
        assert "holdings" in mock_state["portfolio"]

    def test_analysis_cache_initialised(self, monkeypatch):
        from src import app as app_module

        mock_state = _MockSessionState(
            portfolio={"holdings": {}, "cash": 0.0, "journal": [], "trades": []}
        )
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert "analysis_cache" in mock_state
        assert isinstance(mock_state["analysis_cache"], dict)

    def test_concentration_threshold_default(self, monkeypatch):
        from src import app as app_module

        mock_state = _MockSessionState(
            portfolio={"holdings": {}, "cash": 0.0, "journal": [], "trades": []}
        )
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._init_session_state()
        assert mock_state.get("concentration_threshold") == 0.20
