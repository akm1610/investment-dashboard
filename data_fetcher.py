"""
data_fetcher.py
---------------
Functions to download fundamental financial data and price history.
Data sources: Yahoo Finance (via yfinance) and SEC EDGAR (XBRL API).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Yahoo Finance helpers
# ---------------------------------------------------------------------------

def get_ticker(symbol: str) -> yf.Ticker:
    """Return a yfinance Ticker object for *symbol*."""
    return yf.Ticker(symbol.upper())


def fetch_price_history(
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    symbol   : Ticker symbol (e.g. 'AAPL').
    period   : Lookback period accepted by yfinance ('1y', '5y', 'max', …).
    interval : Bar size ('1d', '1wk', '1mo').

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume, Dividends,
    Stock Splits] indexed by date, or an empty DataFrame on error.
    """
    try:
        ticker = get_ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning("No price data returned for %s", symbol)
        return df
    except Exception as exc:
        logger.error("fetch_price_history(%s): %s", symbol, exc)
        return pd.DataFrame()


def fetch_income_statement(symbol: str, quarterly: bool = False) -> pd.DataFrame:
    """
    Download the income statement (annual by default).

    Returns a DataFrame where columns are fiscal periods.
    """
    try:
        ticker = get_ticker(symbol)
        if quarterly:
            df = ticker.quarterly_income_stmt
        else:
            df = ticker.income_stmt
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.error("fetch_income_statement(%s): %s", symbol, exc)
        return pd.DataFrame()


def fetch_balance_sheet(symbol: str, quarterly: bool = False) -> pd.DataFrame:
    """Download the balance sheet (annual by default)."""
    try:
        ticker = get_ticker(symbol)
        if quarterly:
            df = ticker.quarterly_balance_sheet
        else:
            df = ticker.balance_sheet
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.error("fetch_balance_sheet(%s): %s", symbol, exc)
        return pd.DataFrame()


def fetch_cash_flow(symbol: str, quarterly: bool = False) -> pd.DataFrame:
    """Download the cash-flow statement (annual by default)."""
    try:
        ticker = get_ticker(symbol)
        if quarterly:
            df = ticker.quarterly_cashflow
        else:
            df = ticker.cashflow
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.error("fetch_cash_flow(%s): %s", symbol, exc)
        return pd.DataFrame()


def fetch_key_stats(symbol: str) -> dict:
    """
    Return a flat dictionary of key statistics / fast info from Yahoo Finance.
    Merges yfinance's ``info`` and ``fast_info`` attributes.
    """
    try:
        ticker = get_ticker(symbol)
        info = ticker.info or {}
        try:
            fast = dict(ticker.fast_info)
        except Exception:
            fast = {}
        return {**fast, **info}
    except Exception as exc:
        logger.error("fetch_key_stats(%s): %s", symbol, exc)
        return {}


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

_EDGAR_HEADERS = {"User-Agent": "investment-dashboard/1.0 (educational use)"}
_EDGAR_BASE = "https://data.sec.gov"


def _get_cik(symbol: str) -> Optional[str]:
    """
    Resolve a stock ticker to its SEC CIK (Central Index Key).
    Returns a zero-padded 10-digit string or None.
    """
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(tickers_url, headers=_EDGAR_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        symbol_upper = symbol.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == symbol_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.error("_get_cik(%s): %s", symbol, exc)
    return None


def fetch_edgar_facts(symbol: str) -> dict:
    """
    Fetch the full XBRL concept facts for a company from SEC EDGAR.

    Returns a nested dict:
    ``{ 'us-gaap': { 'concept': { 'label': ..., 'units': { 'USD': [...] } } } }``
    or an empty dict on failure.
    """
    cik = _get_cik(symbol)
    if cik is None:
        logger.warning("Could not resolve CIK for %s", symbol)
        return {}
    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json().get("facts", {})
    except Exception as exc:
        logger.error("fetch_edgar_facts(%s): %s", symbol, exc)
        return {}


def fetch_edgar_concept(
    symbol: str,
    concept: str,
    taxonomy: str = "us-gaap",
    unit: str = "USD",
) -> pd.DataFrame:
    """
    Retrieve a single XBRL concept time-series from SEC EDGAR.

    Parameters
    ----------
    symbol   : Ticker symbol.
    concept  : XBRL tag, e.g. 'NetIncomeLoss', 'Assets'.
    taxonomy : 'us-gaap' or 'ifrs-full'.
    unit     : Reporting unit, e.g. 'USD', 'shares'.

    Returns
    -------
    DataFrame with columns [end, val, form, filed, accn] sorted by *end*.
    """
    facts = fetch_edgar_facts(symbol)
    try:
        records = facts[taxonomy][concept]["units"][unit]
        df = pd.DataFrame(records)
        df["end"] = pd.to_datetime(df["end"])
        df = df.sort_values("end").reset_index(drop=True)
        return df
    except KeyError:
        logger.warning("Concept %s/%s not found for %s", taxonomy, concept, symbol)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("fetch_edgar_concept(%s, %s): %s", symbol, concept, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Convenience bundler
# ---------------------------------------------------------------------------

def fetch_all_fundamentals(symbol: str) -> dict:
    """
    Bundle all fundamental data for *symbol* into a single dict.

    Keys: 'key_stats', 'income_statement', 'balance_sheet', 'cash_flow',
    'price_history'.
    """
    logger.info("Fetching fundamentals for %s …", symbol)
    return {
        "key_stats": fetch_key_stats(symbol),
        "income_statement": fetch_income_statement(symbol),
        "balance_sheet": fetch_balance_sheet(symbol),
        "cash_flow": fetch_cash_flow(symbol),
        "price_history": fetch_price_history(symbol),
    }
