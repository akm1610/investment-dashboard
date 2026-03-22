"""
src/ml_engine.py
----------------
Machine-learning engine for the investment recommendation platform.

Architecture
------------
1. FeatureEngineer        – extract/preprocess features from 4 data sources
2. MLModelTrainer         – train LightGBM, XGBoost, LSTM, RandomForest
3. RecommendationEngine   – ensemble voting + dual time-horizon predictions
4. ModelPerformanceTracker– track prediction accuracy over time
5. DataCollector          – aggregate raw data from multiple sources

All external library imports (lightgbm, xgboost, tensorflow, ta) are done
lazily inside the methods that need them so that the module itself can always
be imported, and meaningful errors are raised only when the relevant feature
is actually used.
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BBANDS_PERIOD,
    BBANDS_STD,
    CACHE_TTL_SECONDS,
    EMA_PERIODS,
    ENSEMBLE_WEIGHTS,
    HIST_VOL_PERIOD,
    LGBOOST_PARAMS,
    LONG_TERM_FORWARD_DAYS,
    LONG_TERM_RETURN_THRESHOLD,
    LSTM_PARAMS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    NEWS_LOOKBACK_DAYS,
    OBV_SIGNAL_PERIOD,
    RANDOM_FOREST_PARAMS,
    ROC_PERIOD,
    RSI_PERIOD_LONG,
    RSI_PERIOD_SHORT,
    SHORT_TERM_FORWARD_DAYS,
    SHORT_TERM_RETURN_THRESHOLD,
    SIGNAL_STRENGTH_THRESHOLDS,
    SIGNAL_THRESHOLDS,
    SMA_PERIODS,
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    TECHNICAL_LOOKBACK_DAYS,
    TOP_FEATURES_COUNT,
    TRAIN_TEST_SPLIT_RATIO,
    VOLUME_MA_PERIOD,
    XGBOOST_PARAMS,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Features = pd.DataFrame
Labels = pd.Series
ModelDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float, default: float = np.nan) -> float:
    """Return num / den; return *default* when den is zero or None."""
    try:
        if den is None or den == 0:
            return default
        return num / den
    except Exception:
        return default


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Compute Average True Range."""
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Return (%K, %D) stochastic oscillator."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    k = 100.0 * _safe_div_series(close - lowest_low, highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d


def _safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division of two Series; NaN where den == 0."""
    return num / den.replace(0, np.nan)


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > (-low.diff()).clip(lower=0), 0)
    minus_dm = minus_dm.where(minus_dm > high.diff().clip(lower=0), 0)

    atr_vals = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / atr_vals.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / atr_vals.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(window=period, min_periods=1).mean()


def _kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    result = series.copy()
    for i in range(period, len(series)):
        direction = abs(series.iloc[i] - series.iloc[i - period])
        volatility = (series.iloc[i - period + 1 : i + 1].diff().abs()).sum()
        er = _safe_div(direction, volatility, 0.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result.iloc[i] = result.iloc[i - 1] + sc * (series.iloc[i] - result.iloc[i - 1])
    return result


def _temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_TEST_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-ordered train/test split (no shuffling)."""
    split = int(len(X) * train_ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _generate_labels(
    price_series: pd.Series,
    forward_days: int,
    threshold: float,
) -> pd.Series:
    """
    Generate binary labels:
    1 if forward return > *threshold*, else 0.
    The last *forward_days* rows (no future data available) are dropped.
    """
    forward_return = price_series.shift(-forward_days) / price_series - 1.0
    # Drop NaN *before* comparison so the tail rows are excluded from the output
    forward_return = forward_return.dropna()
    return (forward_return > threshold).astype(int)


# ===========================================================================
# FeatureEngineer
# ===========================================================================


class FeatureEngineer:
    """
    Extract and engineer features from multiple data sources.

    Methods
    -------
    extract_technical_features(price_df)
        Compute 25+ technical indicators from OHLCV data.
    extract_fundamental_features(info)
        Build feature Series from a yfinance info dict or similar mapping.
    extract_sentiment_features(sentiment_data)
        Build feature Series from a sentiment data dict.
    extract_market_features(price_df, market_df)
        Compute volume/liquidity and market-breadth features.
    preprocess_features(features_df)
        Normalise, handle missing data, and select top features.
    """

    # ------------------------------------------------------------------
    # Technical features
    # ------------------------------------------------------------------

    def extract_technical_features(
        self, price_df: pd.DataFrame, days: int = TECHNICAL_LOOKBACK_DAYS
    ) -> pd.DataFrame:
        """
        Extract 25+ technical indicator features from OHLCV price data.

        Parameters
        ----------
        price_df:
            DataFrame with columns Open, High, Low, Close, Volume
            and a DatetimeIndex.  At least 200 rows recommended.
        days:
            How many recent rows to retain in the output.

        Returns
        -------
        DataFrame with one row per trading day, one column per indicator.
        """
        df = price_df.copy()
        # Normalise column names – flatten MultiIndex (e.g. yfinance ≥ 0.2.x
        # may return (price_type, ticker) tuples) before lower-casing.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"price_df is missing columns: {missing}")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        feat = pd.DataFrame(index=df.index)

        # --- Momentum --------------------------------------------------
        feat["rsi_14"] = _rsi(close, RSI_PERIOD_LONG)
        feat["rsi_7"] = _rsi(close, RSI_PERIOD_SHORT)

        # MACD
        ema_fast = _ema(close, MACD_FAST)
        ema_slow = _ema(close, MACD_SLOW)
        macd_line = ema_fast - ema_slow
        macd_signal_line = _ema(macd_line, MACD_SIGNAL)
        feat["macd"] = macd_line
        feat["macd_signal"] = macd_signal_line
        feat["macd_hist"] = macd_line - macd_signal_line

        # Stochastic oscillator
        stoch_k, stoch_d = _stochastic(high, low, close, STOCH_K_PERIOD, STOCH_D_PERIOD)
        feat["stoch_k"] = stoch_k
        feat["stoch_d"] = stoch_d

        # Rate of Change
        feat["roc"] = close.pct_change(ROC_PERIOD) * 100.0

        # Momentum (close - close_N days ago)
        feat[f"momentum_{ROC_PERIOD}"] = close - close.shift(ROC_PERIOD)

        # --- Trend -----------------------------------------------------
        for p in SMA_PERIODS:
            feat[f"sma_{p}"] = close.rolling(window=p, min_periods=1).mean()
        for p in EMA_PERIODS:
            feat[f"ema_{p}"] = _ema(close, p)
        feat["kama_10"] = _kama(close)
        feat["adx_14"] = _adx(high, low, close, ADX_PERIOD)

        # --- Volatility ------------------------------------------------
        sma_20 = feat["sma_20"]
        roll_std = close.rolling(window=BBANDS_PERIOD, min_periods=1).std()
        feat["bb_upper"] = sma_20 + BBANDS_STD * roll_std
        feat["bb_lower"] = sma_20 - BBANDS_STD * roll_std
        feat["bb_width"] = feat["bb_upper"] - feat["bb_lower"]
        feat["bb_pct"] = _safe_div_series(close - feat["bb_lower"], feat["bb_width"])
        feat["atr_14"] = _atr(high, low, close, ATR_PERIOD)
        feat["hist_vol_20"] = close.pct_change().rolling(window=HIST_VOL_PERIOD, min_periods=1).std() * np.sqrt(252)
        feat["std_20"] = close.rolling(window=BBANDS_PERIOD, min_periods=1).std()

        # --- Volume ----------------------------------------------------
        feat["volume_ma_20"] = volume.rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()
        feat["volume_ratio"] = _safe_div_series(volume, feat["volume_ma_20"])
        feat["obv"] = _obv(close, volume)
        feat["obv_signal"] = feat["obv"].rolling(window=OBV_SIGNAL_PERIOD, min_periods=1).mean()
        feat["volume_roc"] = volume.pct_change(ROC_PERIOD) * 100.0
        # Accumulation/Distribution Line
        clv = _safe_div_series((close - low) - (high - close), high - low)
        feat["adl"] = (clv * volume).cumsum()

        # --- Price action ----------------------------------------------
        sma_200 = feat["sma_200"]
        feat["price_vs_sma200"] = _safe_div_series(close, sma_200) - 1.0
        feat["price_vs_bb_pct"] = feat["bb_pct"]

        # Higher High / Lower Low (binary, rolling 10)
        feat["higher_high"] = (
            high.rolling(10, min_periods=1).max() == high
        ).astype(int)
        feat["lower_low"] = (
            low.rolling(10, min_periods=1).min() == low
        ).astype(int)

        # 52-week high/low proximity
        feat["pct_from_52w_high"] = _safe_div_series(
            close, high.rolling(252, min_periods=1).max()
        ) - 1.0
        feat["pct_from_52w_low"] = _safe_div_series(
            close, low.rolling(252, min_periods=1).min()
        ) - 1.0

        # Temporal features
        feat["day_of_week"] = pd.to_datetime(feat.index).dayofweek
        feat["month"] = pd.to_datetime(feat.index).month

        return feat.tail(days).copy()

    # ------------------------------------------------------------------
    # Fundamental features
    # ------------------------------------------------------------------

    def extract_fundamental_features(
        self, info: Dict[str, Any]
    ) -> pd.Series:
        """
        Build a feature Series from a company info dict (e.g. yfinance Ticker.info).

        Covers growth, profitability, valuation, financial-health, and quality
        metrics.  Missing values become NaN – callers should handle them.
        """
        def _get(key: str, default=np.nan) -> float:
            val = info.get(key, default)
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        features: Dict[str, float] = {}

        # Growth
        features["revenue_growth"] = _get("revenueGrowth")
        features["earnings_growth"] = _get("earningsGrowth")
        features["earnings_quarterly_growth"] = _get("earningsQuarterlyGrowth")

        # Profitability
        features["gross_margin"] = _get("grossMargins")
        features["operating_margin"] = _get("operatingMargins")
        features["profit_margin"] = _get("profitMargins")
        features["roe"] = _get("returnOnEquity")
        features["roa"] = _get("returnOnAssets")

        # Valuation
        features["pe_ratio"] = _get("trailingPE")
        features["forward_pe"] = _get("forwardPE")
        features["pb_ratio"] = _get("priceToBook")
        features["ps_ratio"] = _get("priceToSalesTrailing12Months")
        features["peg_ratio"] = _get("pegRatio")
        features["ev_ebitda"] = _get("enterpriseToEbitda")
        features["ev_revenue"] = _get("enterpriseToRevenue")

        # Financial health
        features["debt_to_equity"] = _get("debtToEquity")
        features["current_ratio"] = _get("currentRatio")
        features["quick_ratio"] = _get("quickRatio")
        features["free_cashflow"] = _get("freeCashflow")
        features["operating_cashflow"] = _get("operatingCashflow")
        features["total_debt"] = _get("totalDebt")
        features["cash"] = _get("totalCash")

        # Quality / shareholder returns
        features["payout_ratio"] = _get("payoutRatio")
        features["beta"] = _get("beta")
        features["shares_outstanding"] = _get("sharesOutstanding")
        features["float_shares"] = _get("floatShares")
        features["held_percent_institutions"] = _get("heldPercentInstitutions")
        features["short_ratio"] = _get("shortRatio")

        # Dividend
        features["dividend_yield"] = _get("dividendYield")
        features["five_year_avg_dividend_yield"] = _get("fiveYearAvgDividendYield")

        # Derived: FCF to net income proxy
        fcf = features.get("free_cashflow", np.nan)
        ocf = features.get("operating_cashflow", np.nan)
        features["fcf_to_ocf"] = _safe_div(fcf, ocf)

        return pd.Series(features)

    # ------------------------------------------------------------------
    # Sentiment features
    # ------------------------------------------------------------------

    def extract_sentiment_features(
        self, sentiment_data: Dict[str, Any]
    ) -> pd.Series:
        """
        Build feature Series from a sentiment data dict.

        Expected keys (all optional, missing → NaN):
        - news_sentiment      : aggregate score [-1, +1]
        - news_volume_today   : int
        - news_volume_week    : int
        - news_volume_month   : int
        - positive_ratio      : fraction of positive articles [0, 1]
        - sentiment_trend     : e.g. "improving" → +1, "declining" → -1
        - analyst_rating      : buy fraction [0, 1]
        - analyst_target_price: float
        - current_price       : float (used to compute upside %)
        - reddit_sentiment    : float [-1, +1]
        - twitter_sentiment   : float [-1, +1]
        - stocktwits_sentiment: float [-1, +1]
        - mentions_volume     : int
        - upgrades_30d        : count of analyst upgrades in last 30 days
        - downgrades_30d      : count of analyst downgrades in last 30 days
        """

        def _g(key: str) -> float:
            val = sentiment_data.get(key, np.nan)
            try:
                return float(val) if val is not None else np.nan
            except (TypeError, ValueError):
                return np.nan

        feat: Dict[str, float] = {}

        feat["news_sentiment"] = _g("news_sentiment")
        feat["news_volume_today"] = _g("news_volume_today")
        feat["news_volume_week"] = _g("news_volume_week")
        feat["news_volume_month"] = _g("news_volume_month")
        feat["positive_article_ratio"] = _g("positive_ratio")

        # Encode trend string to numeric
        trend = sentiment_data.get("sentiment_trend", "")
        if isinstance(trend, str):
            if trend.lower() in ("improving", "positive"):
                feat["sentiment_trend"] = 1.0
            elif trend.lower() in ("declining", "negative"):
                feat["sentiment_trend"] = -1.0
            else:
                feat["sentiment_trend"] = 0.0
        else:
            feat["sentiment_trend"] = _g("sentiment_trend")

        feat["analyst_rating"] = _g("analyst_rating")
        analyst_target = _g("analyst_target_price")
        current_price = _g("current_price")
        feat["analyst_upside_pct"] = _safe_div(
            analyst_target - current_price, current_price
        )

        feat["reddit_sentiment"] = _g("reddit_sentiment")
        feat["twitter_sentiment"] = _g("twitter_sentiment")
        feat["stocktwits_sentiment"] = _g("stocktwits_sentiment")
        feat["social_mentions_volume"] = _g("mentions_volume")

        feat["analyst_upgrades_30d"] = _g("upgrades_30d")
        feat["analyst_downgrades_30d"] = _g("downgrades_30d")
        upgrade_count = feat["analyst_upgrades_30d"]
        downgrade_count = feat["analyst_downgrades_30d"]
        total = upgrade_count + downgrade_count
        feat["upgrade_ratio"] = _safe_div(upgrade_count, total)

        # Composite social score (equal-weighted average of available scores)
        social_scores = [
            v
            for k, v in feat.items()
            if k in ("news_sentiment", "reddit_sentiment", "twitter_sentiment", "stocktwits_sentiment")
            and not np.isnan(v)
        ]
        feat["composite_social_sentiment"] = float(np.mean(social_scores)) if social_scores else np.nan

        return pd.Series(feat)

    # ------------------------------------------------------------------
    # Market microstructure features
    # ------------------------------------------------------------------

    def extract_market_features(
        self,
        price_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute market microstructure features (latest snapshot).

        Parameters
        ----------
        price_df:
            OHLCV DataFrame (same format as extract_technical_features).
        market_df:
            Optional OHLCV for a market index (e.g. S&P 500).  Used to
            compute correlation and relative-strength features.
        """
        df = price_df.copy()
        # Flatten MultiIndex columns before lower-casing (yfinance ≥ 0.2.x compat)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        feat: Dict[str, float] = {}

        # Volume
        vol_ma = volume.rolling(VOLUME_MA_PERIOD, min_periods=1).mean()
        feat["volume_vs_avg"] = _safe_div(float(volume.iloc[-1]), float(vol_ma.iloc[-1]))
        feat["volume_trend"] = float(
            volume.tail(5).mean() / vol_ma.iloc[-1] if vol_ma.iloc[-1] != 0 else np.nan
        )

        # Intraday volatility (last row: high - low / close)
        last = df.iloc[-1]
        feat["intraday_volatility"] = _safe_div(
            float(last["high"] - last["low"]), float(last["close"])
        )

        # Price gap vs previous close
        if len(df) > 1:
            prev_close = float(df["close"].iloc[-2])
            today_open = float(df["open"].iloc[-1])
            feat["price_gap"] = _safe_div(today_open - prev_close, prev_close)
        else:
            feat["price_gap"] = np.nan

        # Closing position in daily range (0 = low, 1 = high)
        day_range = float(last["high"] - last["low"])
        feat["close_position"] = _safe_div(float(last["close"] - last["low"]), day_range)

        # VWAP (rolling 20-day proxy: cumulative(price*vol)/cumulative(vol))
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vwap = (typical * volume).rolling(VOLUME_MA_PERIOD, min_periods=1).sum() / volume.rolling(
            VOLUME_MA_PERIOD, min_periods=1
        ).sum().replace(0, np.nan)
        feat["vwap_deviation"] = _safe_div(
            float(close.iloc[-1]) - float(vwap.iloc[-1]), float(vwap.iloc[-1])
        )

        # Market correlation (vs S&P500 proxy)
        if market_df is not None and not market_df.empty:
            mkt = market_df.copy()
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            mkt.columns = [str(c).lower() for c in mkt.columns]
            # Align on common dates
            common = close.index.intersection(mkt.index)
            if len(common) >= 20:
                stock_ret = close.loc[common].pct_change().dropna()
                mkt_ret = mkt["close"].loc[common].pct_change().dropna()
                common2 = stock_ret.index.intersection(mkt_ret.index)
                if len(common2) >= 10:
                    feat["market_correlation"] = float(
                        np.corrcoef(
                            stock_ret.loc[common2].values,
                            mkt_ret.loc[common2].values,
                        )[0, 1]
                    )
                    # Beta (OLS)
                    cov = np.cov(stock_ret.loc[common2].values, mkt_ret.loc[common2].values)
                    feat["beta"] = _safe_div(float(cov[0, 1]), float(cov[1, 1]))
                else:
                    feat["market_correlation"] = np.nan
                    feat["beta"] = np.nan
            else:
                feat["market_correlation"] = np.nan
                feat["beta"] = np.nan
        else:
            feat["market_correlation"] = np.nan
            feat["beta"] = np.nan

        return pd.Series(feat)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_features(
        self, features_df: pd.DataFrame, top_n: int = TOP_FEATURES_COUNT
    ) -> pd.DataFrame:
        """
        Normalise, handle missing data, and (optionally) drop highly-correlated
        features, returning a clean feature matrix.

        Steps
        -----
        1. Forward-fill then backward-fill to handle small gaps.
        2. Drop columns with > 50 % missing values.
        3. Drop rows still containing any NaN.
        4. Remove one column from each pair with |correlation| > 0.95.
        5. Keep at most *top_n* columns (by variance, descending).
        6. Standardise columns using z-score normalisation.

        Parameters
        ----------
        features_df : DataFrame before preprocessing.
        top_n       : Maximum number of features to retain.

        Returns
        -------
        Preprocessed DataFrame (float64 columns, no NaN).
        """
        df = features_df.copy()

        # 1. Fill small gaps
        df = df.ffill().bfill()

        # 2. Drop columns with too many missing values
        missing_frac = df.isna().mean()
        df = df.loc[:, missing_frac <= 0.5]

        # 3. Drop rows with any remaining NaN
        df = df.dropna()

        if df.empty or df.shape[1] == 0:
            return df

        # 4. Remove highly-correlated features
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper_tri.columns if any(upper_tri[col] > 0.95)
        ]
        df = df.drop(columns=to_drop, errors="ignore")

        # 5. Keep top-n by variance
        if df.shape[1] > top_n:
            variances = df.var().sort_values(ascending=False)
            df = df[variances.head(top_n).index]

        # 6. Standardise (z-score)
        mean = df.mean()
        std = df.std().replace(0, 1)
        df = (df - mean) / std

        return df.astype(np.float32)


# ===========================================================================
# MLModelTrainer
# ===========================================================================


class MLModelTrainer:
    """
    Train and manage the four ML models for a given time horizon.

    Attributes
    ----------
    horizon : str
        "short_term" or "long_term".
    models : dict
        Trained model objects keyed by name ("lgb", "xgb", "lstm", "rf").
    feature_names : list[str]
        Column names of the training feature matrix (set after first train).
    """

    def __init__(self, horizon: str = "long_term") -> None:
        if horizon not in ("short_term", "long_term"):
            raise ValueError("horizon must be 'short_term' or 'long_term'")
        self.horizon = horizon
        self.models: ModelDict = {}
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_lgb_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> None:
        """Train a LightGBM gradient-boosting classifier."""
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            ) from exc

        self.feature_names = list(X_train.columns)
        params = {k: v for k, v in LGBOOST_PARAMS.items() if k != "num_boost_round"}
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=LGBOOST_PARAMS["num_boost_round"],
            valid_sets=[dtrain],
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        self.models["lgb"] = model
        logger.info("LightGBM model trained (%s horizon)", self.horizon)

    def train_xgb_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> None:
        """Train an XGBoost classifier."""
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            ) from exc

        self.feature_names = list(X_train.columns)
        params = {k: v for k, v in XGBOOST_PARAMS.items()}
        params["objective"] = "binary:logistic"
        params.pop("eval_metric", None)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        self.models["xgb"] = model
        logger.info("XGBoost model trained (%s horizon)", self.horizon)

    def train_lstm_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train an LSTM neural network.

        Parameters
        ----------
        X_train : 3-D array of shape (samples, timesteps, features).
        y_train : 1-D binary label array.
        """
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow import keras  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "tensorflow is required. Install with: pip install tensorflow"
            ) from exc

        p = LSTM_PARAMS
        if X_train.ndim != 3:
            raise ValueError(
                "X_train for LSTM must be 3-D (samples, timesteps, features), "
                f"got shape {X_train.shape}"
            )

        timesteps = X_train.shape[1]
        n_features = X_train.shape[2]

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(timesteps, n_features)),
                keras.layers.LSTM(p["units_1"], return_sequences=True),
                keras.layers.Dropout(p["dropout"]),
                keras.layers.LSTM(p["units_2"], return_sequences=False),
                keras.layers.Dropout(p["dropout"]),
                keras.layers.Dense(p["dense_units"], activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            X_train,
            y_train,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            validation_split=p["validation_split"],
            verbose=0,
        )
        self.models["lstm"] = model
        logger.info("LSTM model trained (%s horizon)", self.horizon)

    def train_rf_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> None:
        """Train a Random Forest classifier."""
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            ) from exc

        self.feature_names = list(X_train.columns)
        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        model.fit(X_train, y_train)
        self.models["rf"] = model
        logger.info("Random Forest model trained (%s horizon)", self.horizon)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_models(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all available models on the test set.

        Returns a dict like::

            {
                'lgb':      {'accuracy': 0.61, 'precision': 0.60, ...},
                'xgb':      {...},
                'rf':       {...},
                'ensemble': {...},
            }
        """
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        results: Dict[str, Dict[str, float]] = {}
        preds_proba: Dict[str, np.ndarray] = {}

        for name, model in self.models.items():
            if name == "lstm":
                continue  # LSTM needs 3-D input; evaluated separately
            proba = self._predict_proba(name, model, X_test)
            preds_proba[name] = proba
            binary = (proba >= 0.5).astype(int)
            results[name] = {
                "accuracy":  float(accuracy_score(y_test, binary)),
                "precision": float(precision_score(y_test, binary, zero_division=0)),
                "recall":    float(recall_score(y_test, binary, zero_division=0)),
                "f1":        float(f1_score(y_test, binary, zero_division=0)),
            }

        # Ensemble metrics
        if preds_proba:
            weights = ENSEMBLE_WEIGHTS[self.horizon]
            ens_proba = np.zeros(len(y_test), dtype=float)
            total_w = 0.0
            for name, proba in preds_proba.items():
                w = weights.get(name, 0.0)
                ens_proba += w * proba
                total_w += w
            if total_w > 0:
                ens_proba /= total_w
            ens_binary = (ens_proba >= 0.5).astype(int)
            results["ensemble"] = {
                "accuracy":  float(accuracy_score(y_test, ens_binary)),
                "precision": float(precision_score(y_test, ens_binary, zero_division=0)),
                "recall":    float(recall_score(y_test, ens_binary, zero_division=0)),
                "f1":        float(f1_score(y_test, ens_binary, zero_division=0)),
            }

        return results

    def _predict_proba(
        self, name: str, model: Any, X: pd.DataFrame
    ) -> np.ndarray:
        """Return probability array for the positive class."""
        if name == "lgb":
            return model.predict(X)
        if name == "xgb":
            return model.predict_proba(X)[:, 1]
        if name == "rf":
            return model.predict_proba(X)[:, 1]
        raise ValueError(f"Unknown model name: {name}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self, directory: str) -> None:
        """Save all trained models to *directory*."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            if name == "lstm":
                model.save(os.path.join(directory, f"lstm_{self.horizon}.keras"))
            else:
                path = os.path.join(directory, f"{name}_{self.horizon}.pkl")
                with open(path, "wb") as fh:
                    pickle.dump(model, fh)
        # Save feature names
        meta_path = os.path.join(directory, f"meta_{self.horizon}.pkl")
        with open(meta_path, "wb") as fh:
            pickle.dump({"feature_names": self.feature_names, "horizon": self.horizon}, fh)
        logger.info("Models saved to %s", directory)

    def load_models(self, directory: str) -> None:
        """Load pre-trained models from *directory*."""
        meta_path = os.path.join(directory, f"meta_{self.horizon}.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            self.feature_names = meta.get("feature_names", [])

        for name in ("lgb", "xgb", "rf"):
            path = os.path.join(directory, f"{name}_{self.horizon}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    self.models[name] = pickle.load(fh)

        lstm_path = os.path.join(directory, f"lstm_{self.horizon}.keras")
        if os.path.exists(lstm_path):
            try:
                from tensorflow import keras  # type: ignore

                self.models["lstm"] = keras.models.load_model(lstm_path)
            except ImportError:
                logger.warning("tensorflow not available; LSTM model not loaded.")
        logger.info("Models loaded from %s", directory)


# ===========================================================================
# RecommendationEngine
# ===========================================================================


class RecommendationEngine:
    """
    Generate investment recommendations from ML ensemble predictions.

    Usage
    -----
    ::

        engine = RecommendationEngine()
        engine.load_models("models/")

        rec = engine.predict("AAPL", horizon="long_term")
        print(rec["signal"], rec["confidence"])
    """

    def __init__(self) -> None:
        self._trainers: Dict[str, MLModelTrainer] = {}
        self._feature_engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_models(self, directory: str) -> None:
        """Load short-term and long-term model sets from *directory*."""
        for horizon in ("short_term", "long_term"):
            trainer = MLModelTrainer(horizon=horizon)
            trainer.load_models(directory)
            self._trainers[horizon] = trainer
        logger.info("RecommendationEngine: models loaded from %s", directory)

    def save_models(self, directory: str) -> None:
        """Persist all loaded model sets."""
        for trainer in self._trainers.values():
            trainer.save_models(directory)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        ticker: str,
        features: Optional[pd.DataFrame] = None,
        horizon: str = "long_term",
    ) -> Dict[str, Any]:
        """
        Generate a recommendation for *ticker*.

        Parameters
        ----------
        ticker   : Stock ticker symbol.
        features : Pre-computed feature row (1-row DataFrame).  If None the
                   engine will attempt to fetch and engineer features live
                   (requires yfinance and a trained model).
        horizon  : "short_term" or "long_term".

        Returns
        -------
        Recommendation dict with keys: ticker, signal, confidence, strength,
        short_term, long_term, model_votes, feature_importance, key_drivers.
        """
        if horizon not in ("short_term", "long_term"):
            raise ValueError("horizon must be 'short_term' or 'long_term'")

        # Get predictions for both horizons (if models available)
        horizon_results: Dict[str, Dict[str, Any]] = {}
        for h in ("short_term", "long_term"):
            trainer = self._trainers.get(h)
            if trainer and trainer.models and features is not None:
                horizon_results[h] = self._predict_single_horizon(
                    features, trainer, h
                )

        primary = horizon_results.get(horizon, {})
        signal = primary.get("signal", "HOLD")
        confidence = primary.get("confidence", 50.0)

        result: Dict[str, Any] = {
            "ticker": ticker,
            "signal": signal,
            "confidence": round(confidence, 2),
            "strength": self.get_signal_strength(confidence),
            "short_term": horizon_results.get("short_term", {"signal": "HOLD", "confidence": 50.0}),
            "long_term":  horizon_results.get("long_term",  {"signal": "HOLD", "confidence": 50.0}),
            "model_votes": primary.get("model_votes", {}),
            "feature_importance": primary.get("feature_importance", {}),
            "key_drivers": primary.get("key_drivers", []),
        }
        return result

    def _predict_single_horizon(
        self,
        features: pd.DataFrame,
        trainer: MLModelTrainer,
        horizon: str,
    ) -> Dict[str, Any]:
        """Run ensemble prediction for one horizon; return signal + metadata."""
        weights = ENSEMBLE_WEIGHTS[horizon]
        thresholds = SIGNAL_THRESHOLDS[horizon]

        proba_map: Dict[str, float] = {}
        votes: Dict[str, str] = {}

        for name, model in trainer.models.items():
            if name == "lstm":
                continue  # LSTM requires sequence data; skip in live inference
            try:
                proba = float(trainer._predict_proba(name, model, features)[0])
                proba_map[name] = proba
                votes[name] = self._proba_to_signal(proba, thresholds)
            except Exception as exc:
                logger.warning("Model %s prediction failed: %s", name, exc)

        # Weighted ensemble
        total_w = sum(weights.get(n, 0.0) for n in proba_map)
        if total_w == 0:
            final_proba = 0.5
        else:
            final_proba = sum(
                proba_map[n] * weights.get(n, 0.0) for n in proba_map
            ) / total_w

        signal = self._proba_to_signal(final_proba, thresholds)
        confidence = self._proba_to_confidence(final_proba, signal)

        # Feature importance (RF or LGB)
        feat_imp = self._get_feature_importance(trainer, features)
        key_drivers = self._build_key_drivers(feat_imp, features, signal)

        reasoning = self._build_reasoning(signal, feat_imp, key_drivers)

        return {
            "signal":             signal,
            "confidence":         round(confidence, 2),
            "reasoning":          reasoning,
            "model_votes":        votes,
            "model_probas":       {k: round(v, 4) for k, v in proba_map.items()},
            "ensemble_proba":     round(final_proba, 4),
            "feature_importance": feat_imp,
            "key_drivers":        key_drivers,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        tickers: List[str],
        feature_map: Optional[Dict[str, pd.DataFrame]] = None,
        horizon: str = "long_term",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate recommendations for multiple tickers.

        Parameters
        ----------
        tickers     : List of ticker symbols.
        feature_map : Optional mapping ticker → 1-row feature DataFrame.
        horizon     : Time horizon for primary signal.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            feats = feature_map.get(ticker) if feature_map else None
            results[ticker] = self.predict(ticker, features=feats, horizon=horizon)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_signal_strength(confidence: float) -> str:
        """Convert confidence % to a WEAK / MODERATE / STRONG label."""
        if confidence >= SIGNAL_STRENGTH_THRESHOLDS["STRONG"]:
            return "STRONG"
        if confidence >= SIGNAL_STRENGTH_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        return "WEAK"

    @staticmethod
    def _proba_to_signal(proba: float, thresholds: Dict[str, float]) -> str:
        if proba >= thresholds["buy"]:
            return "BUY"
        if proba <= thresholds["sell"]:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _proba_to_confidence(proba: float, signal: str) -> float:
        """Convert raw probability to a 0-100 confidence score."""
        if signal == "BUY":
            return proba * 100.0
        if signal == "SELL":
            return (1.0 - proba) * 100.0
        # HOLD: closer to 50 % = more confident in neutral stance
        return max(0.0, 50.0 - abs(0.5 - proba) * 100.0)

    @staticmethod
    def _get_feature_importance(
        trainer: MLModelTrainer, features: pd.DataFrame
    ) -> Dict[str, float]:
        """Return a feature → importance dict from the best available model."""
        feat_imp: Dict[str, float] = {}
        names = trainer.feature_names or list(features.columns)

        if "lgb" in trainer.models:
            model = trainer.models["lgb"]
            importances = model.feature_importance(importance_type="gain")
            total = importances.sum() or 1.0
            feat_imp = {
                names[i]: round(float(importances[i]) / total, 4)
                for i in range(min(len(names), len(importances)))
            }
        elif "rf" in trainer.models:
            model = trainer.models["rf"]
            importances = model.feature_importances_
            total = importances.sum() or 1.0
            feat_imp = {
                names[i]: round(float(importances[i]) / total, 4)
                for i in range(min(len(names), len(importances)))
            }

        # Return top-10 sorted by importance
        return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10])

    @staticmethod
    def _build_key_drivers(
        feat_imp: Dict[str, float],
        features: pd.DataFrame,
        signal: str,
    ) -> List[str]:
        """Generate human-readable key driver strings."""
        drivers: List[str] = []
        if not feat_imp or features.empty:
            return drivers

        for feat_name in list(feat_imp.keys())[:5]:
            if feat_name not in features.columns:
                continue
            val = float(features[feat_name].iloc[0])
            if "rsi" in feat_name:
                if val < 30:
                    drivers.append(f"Oversold RSI ({val:.1f}) – potential bounce")
                elif val > 70:
                    drivers.append(f"Overbought RSI ({val:.1f}) – caution")
                else:
                    drivers.append(f"Neutral RSI ({val:.1f})")
            elif "sentiment" in feat_name:
                direction = "Positive" if val > 0 else "Negative"
                drivers.append(f"{direction} sentiment score ({val:.2f})")
            elif "pe_ratio" in feat_name and not np.isnan(val):
                drivers.append(f"P/E ratio: {val:.1f}")
            elif "revenue_growth" in feat_name and not np.isnan(val):
                drivers.append(f"Revenue growth: {val*100:.1f}%")
            elif "macd_hist" in feat_name:
                direction = "Bullish" if val > 0 else "Bearish"
                drivers.append(f"{direction} MACD histogram ({val:.4f})")
        return drivers

    @staticmethod
    def _build_reasoning(
        signal: str,
        feat_imp: Dict[str, float],
        key_drivers: List[str],
    ) -> str:
        """Compose a short reasoning string."""
        top_features = ", ".join(list(feat_imp.keys())[:3]) if feat_imp else "N/A"
        driver_str = "; ".join(key_drivers[:2]) if key_drivers else "model consensus"
        return f"{signal} signal driven by {top_features}. {driver_str}."


# ===========================================================================
# ModelPerformanceTracker
# ===========================================================================


class ModelPerformanceTracker:
    """
    Track real-time model performance.

    Predictions and outcomes are stored in-memory as a list of dicts.
    Call ``save`` / ``load`` to persist to disk (pickle).
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        ticker: str,
        prediction: Dict[str, Any],
        actual_return: float,
        horizon: str,
    ) -> None:
        """
        Record a prediction and its realised outcome.

        Parameters
        ----------
        ticker        : Stock ticker.
        prediction    : The dict returned by RecommendationEngine.predict().
        actual_return : Realised return over the prediction horizon (e.g. 0.05 = 5 %).
        horizon       : "short_term" or "long_term".
        """
        signal = prediction.get("signal", "HOLD")
        threshold = (
            SHORT_TERM_RETURN_THRESHOLD
            if horizon == "short_term"
            else LONG_TERM_RETURN_THRESHOLD
        )
        correct = (
            (signal == "BUY"  and actual_return >  threshold)
            or (signal == "SELL" and actual_return < -threshold)
            or (signal == "HOLD" and abs(actual_return) <= threshold)
        )
        record = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "ticker":        ticker,
            "signal":        signal,
            "confidence":    prediction.get("confidence", 50.0),
            "actual_return": actual_return,
            "horizon":       horizon,
            "correct":       correct,
        }
        self._records.append(record)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_accuracy(
        self, horizon: str = "all", days: int = 30
    ) -> Dict[str, float]:
        """
        Accuracy over the last *days* calendar days.

        Parameters
        ----------
        horizon : "short_term", "long_term", or "all".
        days    : Look-back window in calendar days.

        Returns
        -------
        Dict with keys: accuracy, total_predictions, correct_predictions.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        records = [
            r for r in self._records
            if r["timestamp"] >= cutoff
            and (horizon == "all" or r["horizon"] == horizon)
        ]
        if not records:
            return {"accuracy": 0.0, "total_predictions": 0, "correct_predictions": 0}

        correct = sum(1 for r in records if r["correct"])
        return {
            "accuracy":             round(correct / len(records), 4),
            "total_predictions":    len(records),
            "correct_predictions":  correct,
        }

    def get_win_rate(
        self, signal: str = "BUY", horizon: str = "long_term"
    ) -> float:
        """Win rate for a specific signal type (0.0 – 1.0)."""
        filtered = [
            r for r in self._records
            if r["signal"] == signal
            and (horizon == "all" or r["horizon"] == horizon)
        ]
        if not filtered:
            return 0.0
        wins = sum(1 for r in filtered if r["correct"])
        return round(wins / len(filtered), 4)

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a detailed performance metrics dict for the dashboard."""
        if not self._records:
            return {
                "total_predictions": 0,
                "overall_accuracy":  0.0,
                "buy_win_rate":      0.0,
                "sell_win_rate":     0.0,
                "hold_win_rate":     0.0,
                "short_term_accuracy": 0.0,
                "long_term_accuracy":  0.0,
                "recent_30d_accuracy": 0.0,
            }

        total = len(self._records)
        correct = sum(1 for r in self._records if r["correct"])

        return {
            "total_predictions":     total,
            "overall_accuracy":      round(correct / total, 4),
            "buy_win_rate":          self.get_win_rate("BUY", "all"),
            "sell_win_rate":         self.get_win_rate("SELL", "all"),
            "hold_win_rate":         self.get_win_rate("HOLD", "all"),
            "short_term_accuracy":   self.calculate_accuracy("short_term", days=365)["accuracy"],
            "long_term_accuracy":    self.calculate_accuracy("long_term",  days=365)["accuracy"],
            "recent_30d_accuracy":   self.calculate_accuracy("all",        days=30)["accuracy"],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist records to *path* (pickle)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._records, fh)

    def load(self, path: str) -> None:
        """Load records from *path*."""
        if os.path.exists(path):
            with open(path, "rb") as fh:
                self._records = pickle.load(fh)


# ===========================================================================
# DataCollector
# ===========================================================================


class DataCollector:
    """
    Collect and aggregate data from multiple sources for a given ticker.

    All external calls use yfinance.  Sentiment and analyst data are returned
    as empty dicts when no external sentiment API is configured – callers
    should augment these with live data if available.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch all required data for *ticker*.

        Returns
        -------
        Dict with keys:
        - price_history  : pd.DataFrame (OHLCV, 5 years)
        - info           : dict (company fundamentals)
        - sentiment      : dict (placeholder – integrate with news APIs)
        - fetched_at     : ISO timestamp string
        """
        if self._is_cached(ticker):
            return self._cache[ticker]

        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            ) from exc

        yf_ticker = yf.Ticker(ticker)

        # Price history (5 years of daily OHLCV)
        price_history = yf_ticker.history(period="5y", auto_adjust=True)
        # Normalise columns: yfinance ≥ 0.2.x can return a MultiIndex
        # (price_type, ticker) – flatten to plain string names immediately.
        if isinstance(price_history.columns, pd.MultiIndex):
            price_history.columns = price_history.columns.get_level_values(0)
        price_history.columns = [str(c) for c in price_history.columns]

        # Company info
        try:
            info = yf_ticker.info or {}
        except Exception:
            info = {}

        # Placeholder sentiment (extend with real APIs as needed)
        sentiment: Dict[str, Any] = {
            "news_sentiment": np.nan,
            "news_volume_week": np.nan,
            "analyst_rating": info.get("recommendationMean", np.nan),
            "analyst_target_price": info.get("targetMeanPrice", np.nan),
            "current_price": info.get("currentPrice", np.nan),
        }

        result: Dict[str, Any] = {
            "ticker":        ticker,
            "price_history": price_history,
            "info":          info,
            "sentiment":     sentiment,
            "fetched_at":    datetime.now(timezone.utc).isoformat(),
        }
        self._update_cache(ticker, result)
        return result

    def update_cache(self, ticker: str) -> None:
        """Force-refresh cached data for *ticker*."""
        # Invalidate and re-fetch
        self._cache.pop(ticker, None)
        self._cache_timestamps.pop(ticker, None)
        self.fetch_all_data(ticker)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cached(self, ticker: str) -> bool:
        if ticker not in self._cache:
            return False
        age = (datetime.now(timezone.utc) - self._cache_timestamps[ticker]).total_seconds()
        return age < CACHE_TTL_SECONDS

    def _update_cache(self, ticker: str, data: Dict[str, Any]) -> None:
        self._cache[ticker] = data
        self._cache_timestamps[ticker] = datetime.now(timezone.utc)


# ===========================================================================
# Top-level training pipeline helper
# ===========================================================================


def generate_labels(
    price_series: pd.Series,
    horizon: str = "long_term",
    threshold: Optional[float] = None,
) -> pd.Series:
    """
    Generate binary labels for model training.

    Parameters
    ----------
    price_series : Daily close prices.
    horizon      : "short_term" (5-day) or "long_term" (252-day).
    threshold    : Return threshold; defaults to horizon-specific config.
    """
    if horizon == "short_term":
        fwd_days = SHORT_TERM_FORWARD_DAYS
        thresh = SHORT_TERM_RETURN_THRESHOLD if threshold is None else threshold
    else:
        fwd_days = LONG_TERM_FORWARD_DAYS
        thresh = LONG_TERM_RETURN_THRESHOLD if threshold is None else threshold
    return _generate_labels(price_series, fwd_days, thresh)


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_TEST_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Public wrapper for time-ordered train/test split."""
    return _temporal_train_test_split(X, y, train_ratio)
