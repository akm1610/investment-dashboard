"""
tests/test_data_fetcher.py
--------------------------
Unit tests for src/data_fetcher.py.

All external network calls are mocked so the tests run fully offline.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetcher import (
    DataFetchError,
    DataFetcher,
    _cache_get,
    _cache_set,
    _clean_dataframe,
    _row_value,
    _to_dataframe,
    _validate_ticker,
    batch_fetch,
    calculate_basic_ratios,
    clear_cache,
    fetch_balance_sheet,
    fetch_cash_flow_data,
    fetch_company_info,
    fetch_financial_statements,
    fetch_income_statement,
    fetch_stock_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PRICES = pd.DataFrame(
    {
        "Open": [150.0, 151.0],
        "High": [155.0, 156.0],
        "Low": [149.0, 150.0],
        "Close": [154.0, 155.0],
        "Volume": [1_000_000, 1_200_000],
    },
    index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
)

SAMPLE_FINANCIALS = pd.DataFrame(
    {"2023-12-31": [100_000, 40_000, 30_000]},
    index=["Total Revenue", "Gross Profit", "Net Income"],
)

SAMPLE_BALANCE = pd.DataFrame(
    {"2023-12-31": [500_000, 200_000, 300_000, 50_000, 100_000]},
    index=[
        "Total Assets",
        "Total Liabilities Net Minority Interest",
        "Stockholders Equity",
        "Total Debt",
        "Total Current Assets",
    ],
)

SAMPLE_CASHFLOW = pd.DataFrame(
    {"2023-12-31": [80_000, -20_000, -10_000]},
    index=["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"],
)

SAMPLE_INFO: Dict[str, Any] = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_000_000_000_000,
    "trailingPE": 28.5,
    "priceToBook": 45.0,
    "debtToEquity": 150.0,
    "returnOnEquity": 1.47,
    "returnOnAssets": 0.28,
    "currentRatio": 1.07,
    "grossMargins": 0.44,
    "operatingMargins": 0.30,
    "trailingEps": 6.11,
}


def _make_yf_ticker_mock() -> MagicMock:
    """Return a MagicMock that mimics a yfinance Ticker object."""
    mock = MagicMock()
    mock.history.return_value = SAMPLE_PRICES.copy()
    mock.financials = SAMPLE_FINANCIALS.copy()
    mock.quarterly_financials = SAMPLE_FINANCIALS.copy()
    mock.balance_sheet = SAMPLE_BALANCE.copy()
    mock.quarterly_balance_sheet = SAMPLE_BALANCE.copy()
    mock.cashflow = SAMPLE_CASHFLOW.copy()
    mock.quarterly_cashflow = SAMPLE_CASHFLOW.copy()
    mock.info = SAMPLE_INFO.copy()
    return mock


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------

class TestValidateTicker:
    def test_uppercase_normalisation(self) -> None:
        assert _validate_ticker("aapl") == "AAPL"

    def test_strips_whitespace(self) -> None:
        assert _validate_ticker("  msft  ") == "MSFT"

    def test_raises_on_empty_string(self) -> None:
        with pytest.raises(ValueError):
            _validate_ticker("")

    def test_raises_on_whitespace_only(self) -> None:
        with pytest.raises(ValueError):
            _validate_ticker("   ")

    def test_raises_on_non_string(self) -> None:
        with pytest.raises(ValueError):
            _validate_ticker(123)  # type: ignore[arg-type]


class TestCleanDataframe:
    def test_drops_all_nan_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [float("nan"), float("nan")]})
        result = _clean_dataframe(df)
        assert "b" not in result.columns

    def test_drops_all_nan_rows(self) -> None:
        df = pd.DataFrame({"a": [1, float("nan")], "b": [2, float("nan")]})
        result = _clean_dataframe(df)
        assert len(result) == 1

    def test_columns_are_strings(self) -> None:
        df = pd.DataFrame({1: [1], 2: [2]})
        result = _clean_dataframe(df)
        assert all(isinstance(c, str) for c in result.columns)


class TestToDataframe:
    def test_passthrough_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert _to_dataframe(df).equals(df)

    def test_converts_dict(self) -> None:
        result = _to_dataframe({"a": [1, 2], "b": [3, 4]})
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_unknown_type_returns_empty(self) -> None:
        result = _to_dataframe(None)
        assert result.empty


class TestRowValue:
    def test_finds_existing_row(self) -> None:
        s = pd.Series({"Total Debt": 50_000.0, "Equity": 300_000.0})
        assert _row_value(s, ["Total Debt"]) == 50_000.0

    def test_case_insensitive(self) -> None:
        s = pd.Series({"total debt": 50_000.0})
        assert _row_value(s, ["Total Debt"]) == 50_000.0

    def test_returns_none_when_missing(self) -> None:
        s = pd.Series({"Revenue": 100_000.0})
        assert _row_value(s, ["Total Debt", "Long Term Debt"]) is None

    def test_uses_first_candidate_found(self) -> None:
        s = pd.Series({"Total Debt": 10.0, "Long Term Debt": 20.0})
        assert _row_value(s, ["Total Debt", "Long Term Debt"]) == 10.0


class TestCache:
    def test_set_and_get(self) -> None:
        clear_cache()
        _cache_set("test_key", 42)
        assert _cache_get("test_key") == 42

    def test_clear_removes_entries(self) -> None:
        _cache_set("key_a", "value_a")
        clear_cache()
        assert _cache_get("key_a") is None

    def test_miss_returns_none(self) -> None:
        clear_cache()
        assert _cache_get("nonexistent") is None


# ---------------------------------------------------------------------------
# DataFetcher tests (all yfinance calls mocked)
# ---------------------------------------------------------------------------

class TestDataFetcherInit:
    def test_ticker_normalised(self) -> None:
        f = DataFetcher("aapl")
        assert f.ticker == "AAPL"

    def test_invalid_ticker_raises(self) -> None:
        with pytest.raises(ValueError):
            DataFetcher("")


@patch("src.data_fetcher.yf.Ticker")
class TestFetchStockData:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_dataframe(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = DataFetcher("AAPL").fetch_stock_data()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_calls_history_with_correct_period(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock_ticker_cls.return_value = mock
        DataFetcher("AAPL").fetch_stock_data(period="1y")
        mock.history.assert_called_once_with(period="1y")

    def test_raises_on_empty_response(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock
        with pytest.raises(DataFetchError):
            DataFetcher("AAPL").fetch_stock_data()

    def test_caches_result(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock_ticker_cls.return_value = mock
        fetcher = DataFetcher("AAPL")
        fetcher.fetch_stock_data()
        fetcher.fetch_stock_data()   # second call – should hit cache
        assert mock.history.call_count == 1


@patch("src.data_fetcher.yf.Ticker")
class TestFetchFinancialStatements:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_six_dataframes(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = DataFetcher("AAPL").fetch_financial_statements()
        assert len(result) == 6
        assert all(isinstance(v, pd.DataFrame) for v in result.values())


@patch("src.data_fetcher.yf.Ticker")
class TestFetchCompanyInfo:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_dict(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = DataFetcher("AAPL").fetch_company_info()
        assert isinstance(result, dict)
        assert result["longName"] == "Apple Inc."

    def test_raises_on_empty_info(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock.info = {}
        mock_ticker_cls.return_value = mock
        with pytest.raises(DataFetchError):
            DataFetcher("AAPL").fetch_company_info()


@patch("src.data_fetcher.yf.Ticker")
class TestFetchCashFlowData:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_annual_by_default(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = DataFetcher("AAPL").fetch_cash_flow_data()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_quarterly_uses_quarterly_attr(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock_ticker_cls.return_value = mock
        DataFetcher("AAPL").fetch_cash_flow_data(quarterly=True)
        # Accessing quarterly_cashflow attribute means it was used
        _ = mock.quarterly_cashflow


@patch("src.data_fetcher.yf.Ticker")
class TestFetchBalanceSheet:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_dataframe(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = DataFetcher("AAPL").fetch_balance_sheet()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


@patch("src.data_fetcher.yf.Ticker")
class TestFetchIncomeStatement:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_dataframe(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = DataFetcher("AAPL").fetch_income_statement()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


@patch("src.data_fetcher.yf.Ticker")
class TestCalculateBasicRatios:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_expected_keys(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        ratios = DataFetcher("AAPL").calculate_basic_ratios()
        expected_keys = {
            "pe_ratio", "pb_ratio", "debt_to_equity",
            "roe", "roa", "current_ratio",
            "gross_margin", "operating_margin", "eps_ttm",
        }
        assert expected_keys == set(ratios.keys())

    def test_ratio_values_match_info(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        ratios = DataFetcher("AAPL").calculate_basic_ratios()
        assert ratios["pe_ratio"] == pytest.approx(28.5)
        assert ratios["pb_ratio"] == pytest.approx(45.0)

    def test_none_for_missing_ratios(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock.info = {"longName": "TestCo"}   # no financial fields
        mock.balance_sheet = pd.DataFrame()
        mock.financials = pd.DataFrame()
        mock_ticker_cls.return_value = mock
        ratios = DataFetcher("AAPL").calculate_basic_ratios()
        assert ratios["pe_ratio"] is None


# ---------------------------------------------------------------------------
# Module-level convenience function smoke tests
# ---------------------------------------------------------------------------

@patch("src.data_fetcher.yf.Ticker")
class TestConvenienceFunctions:
    def setup_method(self) -> None:
        clear_cache()

    def test_fetch_stock_data(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = fetch_stock_data("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_fetch_company_info(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = fetch_company_info("AAPL")
        assert isinstance(result, dict)

    def test_fetch_financial_statements(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = fetch_financial_statements("AAPL")
        assert "annual_income" in result

    def test_fetch_cash_flow_data(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = fetch_cash_flow_data("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_fetch_balance_sheet(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = fetch_balance_sheet("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_fetch_income_statement(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = fetch_income_statement("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_calculate_basic_ratios(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        result = calculate_basic_ratios("AAPL")
        assert "pe_ratio" in result


# ---------------------------------------------------------------------------
# batch_fetch tests
# ---------------------------------------------------------------------------

@patch("src.data_fetcher.yf.Ticker")
class TestBatchFetch:
    def setup_method(self) -> None:
        clear_cache()

    def test_returns_results_for_all_tickers(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        results = batch_fetch(["AAPL", "MSFT"], fetch_type="company_info")
        assert set(results.keys()) == {"AAPL", "MSFT"}

    def test_failed_ticker_stored_as_error(self, mock_ticker_cls: MagicMock) -> None:
        mock = _make_yf_ticker_mock()
        mock.info = {}   # triggers DataFetchError
        mock_ticker_cls.return_value = mock
        results = batch_fetch(["AAPL"], fetch_type="company_info")
        assert isinstance(results["AAPL"], DataFetchError)

    def test_invalid_fetch_type_raises(self, mock_ticker_cls: MagicMock) -> None:
        with pytest.raises(ValueError, match="Unknown fetch_type"):
            batch_fetch(["AAPL"], fetch_type="nonexistent")

    def test_stock_data_fetch_type(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value = _make_yf_ticker_mock()
        results = batch_fetch(["AAPL"], fetch_type="stock_data", period="1y")
        assert isinstance(results["AAPL"], pd.DataFrame)
