"""
data_fetcher.py
---------------
Data fetching module for the Long-Term Investment Analysis System.

Retrieves financial statements, stock price history, and fundamental metrics
from Yahoo Finance via the yfinance library.  All public functions return
pandas DataFrames or plain dicts so the rest of the application can work with
them immediately.

Key features
------------
* Retry logic with configurable back-off for transient network errors.
* In-process TTL cache so repeated calls in the same session hit memory, not
  the network.
* Data validation / cleaning helpers that normalise column names and remove
  all-NaN rows.
* Custom ``DataFetchError`` exception for clear error propagation.
* Full type hints and docstrings on every public symbol.
* Structured logging throughout.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Tuple

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


class DataFetchError(Exception):
    """Raised when data cannot be retrieved or is unusable after retries."""


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_RETRY_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAY: float = 1.0   # seconds between retries
DEFAULT_PERIOD: str = "5y"
DEFAULT_BATCH_WORKERS: int = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _retry(
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    delay: float = DEFAULT_RETRY_DELAY,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries *func* up to *attempts* times on *exceptions*.

    Parameters
    ----------
    attempts:
        Maximum number of attempts (including the first call).
    delay:
        Seconds to wait between retries (doubles on each subsequent retry).
    exceptions:
        Tuple of exception types that should trigger a retry.

    Returns
    -------
    Decorator function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception = RuntimeError("No attempts made")
            wait = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < attempts:
                        logger.warning(
                            "%s failed on attempt %d/%d: %s – retrying in %.1fs",
                            func.__name__, attempt, attempts, exc, wait,
                        )
                        time.sleep(wait)
                        wait *= 2
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, attempts, exc,
                        )
            raise DataFetchError(
                f"{func.__name__} failed after {attempts} attempts"
            ) from last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[datetime, Any]] = {}
_CACHE_TTL: timedelta = timedelta(minutes=30)


def _cache_get(key: str) -> Optional[Any]:
    """Return cached value if it exists and has not expired."""
    if key in _CACHE:
        ts, value = _CACHE[key]
        if datetime.now(tz=timezone.utc) - ts < _CACHE_TTL:
            logger.debug("Cache hit: %s", key)
            return value
        del _CACHE[key]
    return None


def _cache_set(key: str, value: Any) -> None:
    """Store *value* in the cache under *key*."""
    _CACHE[key] = (datetime.now(tz=timezone.utc), value)
    logger.debug("Cache set: %s", key)


def clear_cache() -> None:
    """Evict all entries from the in-process cache."""
    _CACHE.clear()
    logger.info("Cache cleared")


# ---------------------------------------------------------------------------
# Data validation / cleaning
# ---------------------------------------------------------------------------

def _validate_ticker(ticker: str) -> str:
    """Normalise *ticker* to uppercase and raise if empty."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return ticker.strip().upper()


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns and rows that are entirely NaN; reset column names.

    Handles yfinance ≥ 0.2.x / 1.x which may return a MultiIndex DataFrame
    with (price_type, ticker) tuple columns.  The first level (price type)
    is retained so downstream code can access columns by name (e.g. "Close").
    """
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c) for c in df.columns]
    return df


def _to_dataframe(data: Any) -> pd.DataFrame:
    """Convert *data* (dict or DataFrame) to a cleaned DataFrame."""
    if isinstance(data, pd.DataFrame):
        return _clean_dataframe(data)
    if isinstance(data, dict):
        return _clean_dataframe(pd.DataFrame(data))
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core DataFetcher class
# ---------------------------------------------------------------------------

class DataFetcher:
    """Encapsulates all data-fetching operations for a single ticker.

    Using the class is optional; the module-level convenience functions
    delegate to a shared instance internally.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. ``"AAPL"``).
    retry_attempts:
        How many times to retry failed API calls.
    retry_delay:
        Initial back-off delay (seconds) between retries.

    Examples
    --------
    >>> fetcher = DataFetcher("MSFT")
    >>> prices = fetcher.fetch_stock_data()
    >>> info = fetcher.fetch_company_info()
    """

    def __init__(
        self,
        ticker: str,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self.ticker: str = _validate_ticker(ticker)
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._yf_ticker: Optional[yf.Ticker] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_yf_ticker(self) -> yf.Ticker:
        if self._yf_ticker is None:
            self._yf_ticker = yf.Ticker(self.ticker)
        return self._yf_ticker

    def _fetch_with_retry(self, func: Callable[[], Any]) -> Any:
        """Call *func* with retry / back-off logic."""
        last_exc: Exception = RuntimeError("No attempts made")
        wait = self._retry_delay
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._retry_attempts:
                    logger.warning(
                        "[%s] Attempt %d/%d failed: %s – retrying in %.1fs",
                        self.ticker, attempt, self._retry_attempts, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= 2
        raise DataFetchError(
            f"[{self.ticker}] Data fetch failed after {self._retry_attempts} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_stock_data(self, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
        """Return historical OHLCV price data for the ticker.

        Parameters
        ----------
        period:
            Lookback window accepted by yfinance (``"1y"``, ``"5y"``,
            ``"max"``, etc.).  Defaults to ``"5y"``.

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume, Dividends, Stock Splits.
            Index is a DatetimeIndex.

        Raises
        ------
        DataFetchError
            If the data cannot be retrieved after all retries.
        """
        cache_key = f"{self.ticker}:stock_data:{period}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Fetching stock price data (period=%s)", self.ticker, period)

        def _fetch() -> pd.DataFrame:
            t = self._get_yf_ticker()
            hist = t.history(period=period)
            if hist.empty:
                raise DataFetchError(
                    f"[{self.ticker}] No price data returned for period={period}"
                )
            return _clean_dataframe(hist)

        result = self._fetch_with_retry(_fetch)
        _cache_set(cache_key, result)
        return result

    def fetch_financial_statements(self) -> Dict[str, pd.DataFrame]:
        """Return annual and quarterly income statements, balance sheets, and
        cash-flow statements.

        Returns
        -------
        dict with keys:
            ``"annual_income"``, ``"quarterly_income"``,
            ``"annual_balance_sheet"``, ``"quarterly_balance_sheet"``,
            ``"annual_cash_flow"``, ``"quarterly_cash_flow"``.

        Raises
        ------
        DataFetchError
            If the data cannot be retrieved after all retries.
        """
        cache_key = f"{self.ticker}:financial_statements"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Fetching financial statements", self.ticker)

        def _fetch() -> Dict[str, pd.DataFrame]:
            t = self._get_yf_ticker()
            return {
                "annual_income": _to_dataframe(t.financials),
                "quarterly_income": _to_dataframe(t.quarterly_financials),
                "annual_balance_sheet": _to_dataframe(t.balance_sheet),
                "quarterly_balance_sheet": _to_dataframe(t.quarterly_balance_sheet),
                "annual_cash_flow": _to_dataframe(t.cashflow),
                "quarterly_cash_flow": _to_dataframe(t.quarterly_cashflow),
            }

        result = self._fetch_with_retry(_fetch)
        _cache_set(cache_key, result)
        return result

    def fetch_company_info(self) -> Dict[str, Any]:
        """Return a dictionary of company metadata and overview fields.

        Typical keys include ``"longName"``, ``"sector"``, ``"industry"``,
        ``"marketCap"``, ``"country"``, ``"website"``, ``"longBusinessSummary"``.

        Returns
        -------
        dict

        Raises
        ------
        DataFetchError
            If the data cannot be retrieved after all retries.
        """
        cache_key = f"{self.ticker}:company_info"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Fetching company info", self.ticker)

        def _fetch() -> Dict[str, Any]:
            t = self._get_yf_ticker()
            info = t.info
            if not info:
                raise DataFetchError(
                    f"[{self.ticker}] Empty company info returned"
                )
            return info

        result = self._fetch_with_retry(_fetch)
        _cache_set(cache_key, result)
        return result

    def fetch_cash_flow_data(self, quarterly: bool = False) -> pd.DataFrame:
        """Return the cash flow statement (operating, investing, financing).

        Parameters
        ----------
        quarterly:
            If ``True``, return quarterly data; otherwise annual (default).

        Returns
        -------
        pd.DataFrame
            Rows are line items; columns are reporting periods.

        Raises
        ------
        DataFetchError
            If the data cannot be retrieved after all retries.
        """
        freq = "quarterly" if quarterly else "annual"
        cache_key = f"{self.ticker}:cash_flow:{freq}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Fetching %s cash flow data", self.ticker, freq)

        def _fetch() -> pd.DataFrame:
            t = self._get_yf_ticker()
            raw = t.quarterly_cashflow if quarterly else t.cashflow
            df = _to_dataframe(raw)
            if df.empty:
                raise DataFetchError(
                    f"[{self.ticker}] No {freq} cash flow data returned"
                )
            return df

        result = self._fetch_with_retry(_fetch)
        _cache_set(cache_key, result)
        return result

    def fetch_balance_sheet(self, quarterly: bool = False) -> pd.DataFrame:
        """Return the balance sheet (assets, liabilities, equity).

        Parameters
        ----------
        quarterly:
            If ``True``, return quarterly data; otherwise annual (default).

        Returns
        -------
        pd.DataFrame
            Rows are line items; columns are reporting periods.

        Raises
        ------
        DataFetchError
            If the data cannot be retrieved after all retries.
        """
        freq = "quarterly" if quarterly else "annual"
        cache_key = f"{self.ticker}:balance_sheet:{freq}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Fetching %s balance sheet", self.ticker, freq)

        def _fetch() -> pd.DataFrame:
            t = self._get_yf_ticker()
            raw = t.quarterly_balance_sheet if quarterly else t.balance_sheet
            df = _to_dataframe(raw)
            if df.empty:
                raise DataFetchError(
                    f"[{self.ticker}] No {freq} balance sheet data returned"
                )
            return df

        result = self._fetch_with_retry(_fetch)
        _cache_set(cache_key, result)
        return result

    def fetch_income_statement(self, quarterly: bool = False) -> pd.DataFrame:
        """Return the income statement (revenue, expenses, net income).

        Parameters
        ----------
        quarterly:
            If ``True``, return quarterly data; otherwise annual (default).

        Returns
        -------
        pd.DataFrame
            Rows are line items; columns are reporting periods.

        Raises
        ------
        DataFetchError
            If the data cannot be retrieved after all retries.
        """
        freq = "quarterly" if quarterly else "annual"
        cache_key = f"{self.ticker}:income_statement:{freq}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Fetching %s income statement", self.ticker, freq)

        def _fetch() -> pd.DataFrame:
            t = self._get_yf_ticker()
            raw = t.quarterly_financials if quarterly else t.financials
            df = _to_dataframe(raw)
            if df.empty:
                raise DataFetchError(
                    f"[{self.ticker}] No {freq} income statement data returned"
                )
            return df

        result = self._fetch_with_retry(_fetch)
        _cache_set(cache_key, result)
        return result

    def calculate_basic_ratios(self) -> Dict[str, Optional[float]]:
        """Compute key fundamental ratios from live market data.

        Ratios calculated
        -----------------
        * **P/E ratio** – trailing price-to-earnings
        * **P/B ratio** – price-to-book
        * **Debt/Equity** – total debt divided by total shareholders equity
        * **ROE** – return on equity (net income / shareholders equity)
        * **ROA** – return on assets (net income / total assets)
        * **Current Ratio** – current assets / current liabilities
        * **Gross Margin** – gross profit / total revenue
        * **Operating Margin** – operating income / total revenue
        * **EPS** – trailing twelve-month earnings per share

        Returns
        -------
        dict
            Keys are ratio names; values are ``float`` or ``None`` when data
            is unavailable.

        Raises
        ------
        DataFetchError
            If the underlying data cannot be retrieved after all retries.
        """
        cache_key = f"{self.ticker}:basic_ratios"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("[%s] Calculating basic financial ratios", self.ticker)

        info = self.fetch_company_info()

        def _safe(key: str) -> Optional[float]:
            val = info.get(key)
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        # Derive Debt/Equity, ROE, ROA from balance sheet when not in info
        debt_to_equity: Optional[float] = _safe("debtToEquity")
        roe: Optional[float] = _safe("returnOnEquity")
        roa: Optional[float] = _safe("returnOnAssets")
        current_ratio: Optional[float] = _safe("currentRatio")

        if any(v is None for v in [debt_to_equity, roe, roa]):
            try:
                bs = self.fetch_balance_sheet()
                inc = self.fetch_income_statement()
                if not bs.empty and not inc.empty:
                    latest_bs = bs.iloc[:, 0]
                    latest_inc = inc.iloc[:, 0]

                    total_debt = _row_value(latest_bs, ["Total Debt", "Long Term Debt"])
                    equity = _row_value(
                        latest_bs,
                        ["Stockholders Equity", "Total Stockholder Equity",
                         "Common Stock Equity"],
                    )
                    total_assets = _row_value(
                        latest_bs, ["Total Assets"]
                    )
                    net_income = _row_value(
                        latest_inc, ["Net Income", "Net Income Common Stockholders"]
                    )

                    if debt_to_equity is None and total_debt is not None and equity:
                        debt_to_equity = total_debt / equity

                    if roe is None and net_income is not None and equity:
                        roe = net_income / equity

                    if roa is None and net_income is not None and total_assets:
                        roa = net_income / total_assets
            except DataFetchError:
                pass

        ratios: Dict[str, Optional[float]] = {
            "pe_ratio": _safe("trailingPE"),
            "pb_ratio": _safe("priceToBook"),
            "debt_to_equity": debt_to_equity,
            "roe": roe,
            "roa": roa,
            "current_ratio": current_ratio,
            "gross_margin": _safe("grossMargins"),
            "operating_margin": _safe("operatingMargins"),
            "eps_ttm": _safe("trailingEps"),
        }

        _cache_set(cache_key, ratios)
        return ratios

    def fetch_all_fundamentals(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Return a flat fundamentals dict suitable for the scoring engine.

        Keys returned
        -------------
        roe, gross_margin, operating_margin, pe_ratio, earnings_growth,
        debt_to_equity, current_ratio.  All values are ``float`` or ``None``.

        Parameters
        ----------
        ticker:
            Ignored; present for API compatibility when callers pass the
            ticker symbol explicitly.

        Returns
        -------
        dict
        """
        ratios = self.calculate_basic_ratios()
        info = self.fetch_company_info()

        earnings_growth: Optional[float] = None
        for key in ("earningsGrowth", "earningsQuarterlyGrowth", "revenueGrowth"):
            val = info.get(key)
            try:
                if val is not None:
                    earnings_growth = float(val)
                    break
            except (TypeError, ValueError):
                pass

        return {
            "roe": ratios.get("roe"),
            "gross_margin": ratios.get("gross_margin"),
            "operating_margin": ratios.get("operating_margin"),
            "pe_ratio": ratios.get("pe_ratio"),
            "earnings_growth": earnings_growth,
            "debt_to_equity": ratios.get("debt_to_equity"),
            "current_ratio": ratios.get("current_ratio"),
        }

    def calculate_technical_indicators(
        self, price_data: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        """Compute technical indicators from OHLCV *price_data*.

        Returns a dict with the scalar values of the most recent bar:

        rsi_14, macd, macd_hist, price_vs_sma200, volume_ratio.

        Missing or insufficient data is returned as ``None``.

        Parameters
        ----------
        price_data:
            OHLCV DataFrame with at least a ``Close`` column and optionally
            a ``Volume`` column.

        Returns
        -------
        dict
        """
        if price_data is None or price_data.empty:
            return {
                "rsi_14": None,
                "macd": None,
                "macd_hist": None,
                "price_vs_sma200": None,
                "volume_ratio": None,
            }

        close = price_data["Close"].dropna()

        def _last(series: pd.Series) -> Optional[float]:
            if series.empty:
                return None
            val = series.iloc[-1]
            try:
                return float(val) if not pd.isna(val) else None
            except (TypeError, ValueError):
                return None

        # RSI-14
        rsi_14: Optional[float] = None
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi_series = 100 - (100 / (1 + rs))
            rsi_14 = _last(rsi_series)

        # MACD (12/26/9)
        macd_val: Optional[float] = None
        macd_hist_val: Optional[float] = None
        if len(close) >= 27:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_val = _last(macd_line)
            macd_hist_val = _last(macd_line - signal_line)

        # Price vs 200-day SMA
        price_vs_sma200: Optional[float] = None
        if len(close) >= 200:
            sma200 = close.rolling(window=200, min_periods=200).mean()
            last_sma200 = _last(sma200)
            last_price = _last(close)
            if last_sma200 and last_price and last_sma200 != 0:
                price_vs_sma200 = (last_price / last_sma200) - 1.0

        # Volume ratio (latest / 20-day average)
        volume_ratio: Optional[float] = None
        if "Volume" in price_data.columns:
            volume = price_data["Volume"].dropna()
            if len(volume) >= 20:
                vol_ma20 = volume.rolling(window=20, min_periods=20).mean()
                last_vol = _last(volume)
                last_ma = _last(vol_ma20)
                if last_ma and last_ma != 0 and last_vol is not None:
                    volume_ratio = last_vol / last_ma

        return {
            "rsi_14": rsi_14,
            "macd": macd_val,
            "macd_hist": macd_hist_val,
            "price_vs_sma200": price_vs_sma200,
            "volume_ratio": volume_ratio,
        }

    def calculate_risk_metrics(
        self, price_data: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        """Compute risk metrics from OHLCV *price_data*.

        Returns a dict with:

        volatility   – annualised standard deviation of daily simple returns.
        sharpe_ratio – annualised Sharpe ratio (assuming 0 % risk-free rate).
        max_drawdown – maximum peak-to-trough drawdown (negative float).

        Parameters
        ----------
        price_data:
            OHLCV DataFrame with at least a ``Close`` column.

        Returns
        -------
        dict
        """
        if price_data is None or price_data.empty:
            return {
                "volatility": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
            }

        close = price_data["Close"].dropna()

        # Annualised volatility
        volatility: Optional[float] = None
        sharpe_ratio: Optional[float] = None
        if len(close) >= 2:
            log_returns = close.pct_change().dropna()
            if not log_returns.empty:
                vol = float(log_returns.std()) * (252 ** 0.5)
                volatility = vol
                mean_return = float(log_returns.mean()) * 252
                if vol and vol != 0:
                    sharpe_ratio = mean_return / vol

        # Maximum drawdown
        max_drawdown: Optional[float] = None
        if len(close) >= 2:
            rolling_max = close.cummax()
            drawdown = (close - rolling_max) / rolling_max.replace(0, float("nan"))
            dd_min = drawdown.min()
            try:
                max_drawdown = float(dd_min) if not pd.isna(dd_min) else None
            except (TypeError, ValueError):
                max_drawdown = None

        return {
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }


# ---------------------------------------------------------------------------
# Small utility used by calculate_basic_ratios
# ---------------------------------------------------------------------------

def _row_value(series: pd.Series, candidates: List[str]) -> Optional[float]:
    """Return the first matching row value from *series* or ``None``."""
    index_lower = {str(i).lower(): i for i in series.index}
    for name in candidates:
        original = index_lower.get(name.lower())
        if original is not None:
            val = series[original]
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

@_retry()
def fetch_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch historical OHLCV price data for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.fetch_stock_data`.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. ``"AAPL"``).
    period:
        Lookback window (``"1y"``, ``"5y"``, ``"max"``, …).

    Returns
    -------
    pd.DataFrame
    """
    return DataFetcher(ticker).fetch_stock_data(period=period)


@_retry()
def fetch_financial_statements(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch all financial statements for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.fetch_financial_statements`.

    Returns
    -------
    dict
        Keys: ``annual_income``, ``quarterly_income``, ``annual_balance_sheet``,
        ``quarterly_balance_sheet``, ``annual_cash_flow``, ``quarterly_cash_flow``.
    """
    return DataFetcher(ticker).fetch_financial_statements()


@_retry()
def fetch_company_info(ticker: str) -> Dict[str, Any]:
    """Fetch company overview and sector info for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.fetch_company_info`.

    Returns
    -------
    dict
    """
    return DataFetcher(ticker).fetch_company_info()


@_retry()
def fetch_cash_flow_data(ticker: str, quarterly: bool = False) -> pd.DataFrame:
    """Fetch cash flow statement for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.fetch_cash_flow_data`.

    Parameters
    ----------
    quarterly:
        Return quarterly data when ``True``; annual otherwise.

    Returns
    -------
    pd.DataFrame
    """
    return DataFetcher(ticker).fetch_cash_flow_data(quarterly=quarterly)


@_retry()
def fetch_balance_sheet(ticker: str, quarterly: bool = False) -> pd.DataFrame:
    """Fetch balance sheet for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.fetch_balance_sheet`.

    Parameters
    ----------
    quarterly:
        Return quarterly data when ``True``; annual otherwise.

    Returns
    -------
    pd.DataFrame
    """
    return DataFetcher(ticker).fetch_balance_sheet(quarterly=quarterly)


@_retry()
def fetch_income_statement(ticker: str, quarterly: bool = False) -> pd.DataFrame:
    """Fetch income statement for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.fetch_income_statement`.

    Parameters
    ----------
    quarterly:
        Return quarterly data when ``True``; annual otherwise.

    Returns
    -------
    pd.DataFrame
    """
    return DataFetcher(ticker).fetch_income_statement(quarterly=quarterly)


@_retry()
def calculate_basic_ratios(ticker: str) -> Dict[str, Optional[float]]:
    """Compute key financial ratios for *ticker*.

    Convenience wrapper around :meth:`DataFetcher.calculate_basic_ratios`.

    Ratios: P/E, P/B, Debt/Equity, ROE, ROA, Current Ratio, Gross Margin,
    Operating Margin, EPS (TTM).

    Returns
    -------
    dict
        Ratio name → ``float`` or ``None``.
    """
    return DataFetcher(ticker).calculate_basic_ratios()


def batch_fetch(
    tickers_list: List[str],
    fetch_type: str = "stock_data",
    period: str = DEFAULT_PERIOD,
    max_workers: int = DEFAULT_BATCH_WORKERS,
) -> Dict[str, Any]:
    """Fetch data for multiple tickers concurrently.

    Parameters
    ----------
    tickers_list:
        List of ticker symbols (e.g. ``["AAPL", "MSFT", "GOOGL"]``).
    fetch_type:
        What to retrieve for each ticker.  One of:
        ``"stock_data"`` (default), ``"company_info"``,
        ``"financial_statements"``, ``"cash_flow"``,
        ``"balance_sheet"``, ``"income_statement"``, ``"ratios"``.
    period:
        Price history period; only used when *fetch_type* is
        ``"stock_data"``.
    max_workers:
        Thread-pool size.  Defaults to ``4``.

    Returns
    -------
    dict
        Mapping of *ticker* → fetched data (DataFrame or dict), with failed
        tickers mapped to the ``DataFetchError`` instance so callers can
        inspect failures individually.

    Examples
    --------
    >>> results = batch_fetch(["AAPL", "MSFT"], fetch_type="ratios")
    >>> results["AAPL"]
    {'pe_ratio': ..., 'pb_ratio': ..., ...}
    """
    _FETCH_DISPATCH: Dict[str, Callable[[DataFetcher], Any]] = {
        "stock_data": lambda f: f.fetch_stock_data(period=period),
        "company_info": lambda f: f.fetch_company_info(),
        "financial_statements": lambda f: f.fetch_financial_statements(),
        "cash_flow": lambda f: f.fetch_cash_flow_data(),
        "balance_sheet": lambda f: f.fetch_balance_sheet(),
        "income_statement": lambda f: f.fetch_income_statement(),
        "ratios": lambda f: f.calculate_basic_ratios(),
    }

    if fetch_type not in _FETCH_DISPATCH:
        raise ValueError(
            f"Unknown fetch_type {fetch_type!r}. "
            f"Choose from: {sorted(_FETCH_DISPATCH)}"
        )

    action = _FETCH_DISPATCH[fetch_type]
    results: Dict[str, Any] = {}

    def _worker(ticker: str) -> Tuple[str, Any]:
        try:
            fetcher = DataFetcher(ticker)
            return ticker, action(fetcher)
        except DataFetchError as exc:
            logger.warning("batch_fetch: failed for %s – %s", ticker, exc)
            return ticker, exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tickers_list}
        for future in as_completed(futures):
            ticker, data = future.result()
            results[ticker] = data

    return results
