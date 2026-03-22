"""
algo_trader_ml.py
-----------------
Production-ready ML-based algorithmic trading system with support for stocks,
crypto, and futures trading.

Architecture
------------
1.  TradingConfig         – Central configuration management (dataclass)
2.  DataFetcher           – Multi-source data collection (stocks, crypto, futures)
3.  FeatureEngineer       – Technical indicator calculation and feature engineering
4.  DQNAgent              – Deep Q-Network reinforcement learning agent
5.  PPOAgent              – Proximal Policy Optimization RL agent
6.  LSTMForecaster        – LSTM time-series forecasting model
7.  XGBoostEnsemble       – XGBoost ensemble predictor
8.  ScalpingStrategy      – High-frequency micro-profit strategy
9.  DayTradingStrategy    – Intraday position-holding strategy
10. MeanReversionStrategy – Statistical mean-reversion strategy
11. MomentumStrategy      – Price/volume momentum strategy
12. StrategyComposer      – Combine and weight multiple strategies
13. RiskManager           – Position sizing, stop-loss, drawdown control
14. BacktestEngine        – Realistic backtesting with costs and slippage
15. TradingExecutor       – Paper/live trading integration and audit trail
16. PerformanceMonitor    – Real-time metrics, anomaly detection, alerts
17. AlgoTraderML          – Top-level orchestration class

All heavy optional dependencies (tensorflow, xgboost, ccxt, yfinance) are
imported lazily inside the methods that require them, so this module is always
importable and raises a meaningful ``ImportError`` only when the relevant
feature is actually invoked.

Quick-start example
-------------------
    from algo_trader_ml import AlgoTraderML, TradingConfig

    config = TradingConfig(
        mode="paper",
        symbols=["AAPL", "BTC-USD", "ES=F"],
        max_drawdown=0.10,
    )
    trader = AlgoTraderML(config)
    trader.initialize()
    signals = trader.generate_signals()
    print(signals)

    # Full backtest
    results = trader.backtest(start="2022-01-01", end="2023-12-31")
    trader.monitor.print_summary(results)
"""

from __future__ import annotations

import logging
import math
import random
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Trading-day constant
# ---------------------------------------------------------------------------

_TRADING_DAYS: int = 252
_DEFAULT_RISK_FREE: float = 0.04


# ===========================================================================
# 1. TradingConfig
# ===========================================================================


@dataclass
class TradingConfig:
    """Central configuration for the algo-trading system.

    Parameters
    ----------
    mode:
        ``"paper"`` (default) or ``"live"``.  Paper mode never submits real
        orders; live mode uses the broker adapter in *TradingExecutor*.
    symbols:
        List of ticker symbols to trade.  Stock tickers follow the yfinance
        convention (``"AAPL"``); crypto pairs use the ccxt convention
        (``"BTC/USDT"``); futures use the yfinance suffix (``"ES=F"``).
    asset_types:
        Mapping from symbol to asset class: ``"stock"``, ``"crypto"``, or
        ``"futures"``.  Auto-detected when omitted.
    initial_capital:
        Starting portfolio value in USD.
    max_drawdown:
        Maximum tolerable drawdown as a decimal (e.g. ``0.10`` = 10 %).
        Trading halts when this limit is breached.
    max_position_size:
        Maximum fraction of portfolio allocated to any single position.
    risk_per_trade:
        Fraction of portfolio risked on each trade (fixed-fractional method).
    commission_rate:
        Round-trip commission as a decimal (e.g. ``0.001`` = 0.1 %).
    slippage_rate:
        Market-impact / slippage estimate per trade.
    min_confidence:
        Minimum signal confidence score (0–1) required to execute a trade.
    retrain_interval_days:
        How many calendar days between automatic model retrains.
    lookback_days:
        Number of historical days to fetch for feature engineering.
    timeframe:
        Primary OHLCV bar resolution: ``"1d"``, ``"1h"``, ``"15m"``, etc.
    leverage:
        Leverage multiplier for futures positions (1 = no leverage).
    log_level:
        Python logging level string (``"INFO"``, ``"DEBUG"``, etc.).
    """

    mode: str = "paper"
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "BTC-USD"])
    asset_types: Dict[str, str] = field(default_factory=dict)
    initial_capital: float = 100_000.0
    max_drawdown: float = 0.10
    max_position_size: float = 0.20
    risk_per_trade: float = 0.01
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_confidence: float = 0.55
    retrain_interval_days: int = 7
    lookback_days: int = 365
    timeframe: str = "1d"
    leverage: float = 1.0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Auto-detect asset types for symbols that have no entry
        for sym in self.symbols:
            if sym not in self.asset_types:
                self.asset_types[sym] = _detect_asset_type(sym)


def _detect_asset_type(symbol: str) -> str:
    """Heuristically detect asset class from the symbol string."""
    sym_upper = symbol.upper()
    if sym_upper.endswith("=F"):
        return "futures"
    crypto_suffixes = ("-USD", "/USDT", "/USD", "-USDT", "/BTC", "/ETH")
    if any(sym_upper.endswith(s) for s in crypto_suffixes):
        return "crypto"
    crypto_base = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "MATIC"}
    if sym_upper.split("-")[0] in crypto_base or sym_upper.split("/")[0] in crypto_base:
        return "crypto"
    return "stock"


# ===========================================================================
# 2. DataFetcher
# ===========================================================================


class DataFetcher:
    """Multi-source OHLCV data collector.

    Supports:
    - **Stocks** via ``yfinance``
    - **Crypto** via ``yfinance`` (``BTC-USD``) or ``ccxt`` exchange adapters
    - **Futures** via ``yfinance`` (e.g. ``ES=F``, ``CL=F``)

    Parameters
    ----------
    config:
        Trading configuration.

    Examples
    --------
    >>> fetcher = DataFetcher(config)
    >>> df = fetcher.fetch("AAPL", start="2023-01-01", end="2023-12-31")
    """

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        start:
            ISO-8601 start date string (inclusive).  Defaults to
            ``config.lookback_days`` before today.
        end:
            ISO-8601 end date string (inclusive).  Defaults to today.
        interval:
            Bar interval (``"1d"``, ``"1h"`` …).  Defaults to
            ``config.timeframe``.

        Returns
        -------
        pd.DataFrame
            Columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
            Index: ``pd.DatetimeIndex`` (UTC-aware).

        Raises
        ------
        ValueError
            If data retrieval returns an empty DataFrame.
        """
        interval = interval or self.config.timeframe
        if end is None:
            end_dt = datetime.now(tz=timezone.utc)
        else:
            end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
        if start is None:
            start_dt = end_dt - timedelta(days=self.config.lookback_days)
        else:
            start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()

        asset_type = self.config.asset_types.get(symbol, _detect_asset_type(symbol))
        if asset_type == "crypto" and "/" in symbol:
            df = self._fetch_crypto_ccxt(symbol, start_dt, end_dt, interval)
        else:
            df = self._fetch_yfinance(symbol, start_dt, end_dt, interval)

        df = self._standardize(df)
        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'.")
        return df

    def fetch_all(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for every symbol in ``config.symbols``.

        Returns
        -------
        dict
            Mapping ``symbol -> DataFrame``.  Symbols that fail are logged and
            excluded from the result.
        """
        results: Dict[str, pd.DataFrame] = {}
        for sym in self.config.symbols:
            try:
                results[sym] = self.fetch(sym, start=start, end=end)
                logger.info("Fetched %d rows for %s", len(results[sym]), sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch %s: %s", sym, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Download OHLCV from Yahoo Finance using *yfinance*."""
        try:
            import yfinance as yf  # lazy import
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for stock/crypto/futures data. "
                "Install it with: pip install yfinance"
            ) from exc

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )
        return df

    def _fetch_crypto_ccxt(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Download OHLCV from a crypto exchange via *ccxt*.

        Falls back to yfinance if ccxt is not installed.
        """
        try:
            import ccxt  # lazy import

            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m",
                "1h": "1h", "4h": "4h", "1d": "1d",
            }
            ccxt_interval = interval_map.get(interval, "1d")
            exchange = ccxt.binance({"enableRateLimit": True})
            since = int(start.timestamp() * 1000)
            limit = 1000
            all_ohlcv: List[Any] = []
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, ccxt_interval, since=since, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts >= int(end.timestamp() * 1000):
                    break
                since = last_ts + 1
                if len(ohlcv) < limit:
                    break
            cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
            df = pd.DataFrame(all_ohlcv, columns=cols)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            return df
        except ImportError:
            logger.debug("ccxt not installed; falling back to yfinance for %s", symbol)
            yf_symbol = symbol.replace("/", "-")
            return self._fetch_yfinance(yf_symbol, start, end, interval)

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and index to UTC-aware DatetimeIndex."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        col_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=col_map)
        required = {"Open", "High", "Low", "Close", "Volume"}
        df = df[[c for c in df.columns if c in required]]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
        df = df.dropna(how="all")
        return df


# ===========================================================================
# 3. FeatureEngineer
# ===========================================================================


class FeatureEngineer:
    """Compute technical indicators and derived features from OHLCV data.

    Each ``add_*`` method appends one or more columns in-place and returns the
    DataFrame so calls can be chained::

        fe = FeatureEngineer()
        df = (fe.add_rsi(df)
                .pipe(fe.add_macd)
                .pipe(fe.add_bollinger_bands)
                .pipe(fe.add_atr)
                .pipe(fe.add_volume_features))
    """

    # ---- Simple moving averages ----------------------------------------

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: Tuple[int, ...] = (10, 20, 50, 200)) -> pd.DataFrame:
        """Add simple moving averages for the given *periods*."""
        for p in periods:
            df[f"sma_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: Tuple[int, ...] = (9, 21, 50)) -> pd.DataFrame:
        """Add exponential moving averages for the given *periods*."""
        for p in periods:
            df[f"ema_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
        return df

    # ---- Momentum indicators -------------------------------------------

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Add MACD line, signal line, and histogram."""
        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def add_stochastic(
        df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator (%K and %D)."""
        low_min = df["Low"].rolling(window=k_period, min_periods=1).min()
        high_max = df["High"].rolling(window=k_period, min_periods=1).max()
        denom = (high_max - low_min).replace(0, np.nan)
        df["stoch_k"] = 100 * (df["Close"] - low_min) / denom
        df["stoch_k"] = df["stoch_k"].fillna(50)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period, min_periods=1).mean()
        return df

    @staticmethod
    def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Add Rate of Change (ROC)."""
        df["roc"] = df["Close"].pct_change(periods=period) * 100
        return df

    # ---- Volatility indicators -----------------------------------------

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Add Bollinger Bands (upper, middle, lower) and %B."""
        middle = df["Close"].rolling(window=period, min_periods=1).mean()
        std = df["Close"].rolling(window=period, min_periods=1).std().fillna(0)
        df["bb_middle"] = middle
        df["bb_upper"] = middle + std_dev * std
        df["bb_lower"] = middle - std_dev * std
        band_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / band_width
        df["bb_pct"] = df["bb_pct"].fillna(0.5)
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (ATR) and normalised ATR."""
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period, min_periods=1).mean()
        df["atr_pct"] = df["atr"] / df["Close"].replace(0, np.nan)
        return df

    @staticmethod
    def add_historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add annualised historical volatility."""
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        df["hist_vol"] = log_ret.rolling(window=period, min_periods=2).std() * math.sqrt(
            _TRADING_DAYS
        )
        return df

    # ---- Trend indicators ----------------------------------------------

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (ADX) with +DI and -DI."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Zero out where the opposite direction is larger
        mask = plus_dm < minus_dm
        plus_dm_adj = plus_dm.where(~mask, 0)
        minus_dm_adj = minus_dm.where(mask, 0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=period, min_periods=1).mean()
        atr = atr.replace(0, np.nan)
        plus_di = 100 * plus_dm_adj.rolling(window=period, min_periods=1).mean() / atr
        minus_di = 100 * minus_dm_adj.rolling(window=period, min_periods=1).mean() / atr
        dx_denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / dx_denom
        df["adx"] = dx.rolling(window=period, min_periods=1).mean().fillna(0)
        df["plus_di"] = plus_di.fillna(0)
        df["minus_di"] = minus_di.fillna(0)
        return df

    # ---- Volume indicators ---------------------------------------------

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume (OBV)."""
        direction = np.sign(df["Close"].diff().fillna(0))
        df["obv"] = (direction * df["Volume"]).cumsum()
        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add volume moving average and volume ratio."""
        df["volume_ma"] = df["Volume"].rolling(window=period, min_periods=1).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_ma"].replace(0, np.nan)
        df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
        return df

    # ---- Price-based features ------------------------------------------

    @staticmethod
    def add_returns(df: pd.DataFrame, periods: Tuple[int, ...] = (1, 5, 10, 20)) -> pd.DataFrame:
        """Add log-return features for several *periods*."""
        for p in periods:
            df[f"return_{p}d"] = np.log(
                df["Close"] / df["Close"].shift(p).replace(0, np.nan)
            )
        return df

    @staticmethod
    def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick body/shadow ratios useful for scalping signals."""
        body = (df["Close"] - df["Open"]).abs()
        full_range = (df["High"] - df["Low"]).replace(0, np.nan)
        df["body_ratio"] = body / full_range
        df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / full_range
        df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / full_range
        df["is_bullish"] = (df["Close"] > df["Open"]).astype(float)
        return df

    # ---- All-in-one ----------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all indicator methods and return a feature-rich DataFrame.

        This is the main entry point for the feature-engineering pipeline.
        The returned DataFrame has NaN rows removed.

        Parameters
        ----------
        df:
            OHLCV DataFrame with columns ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume``.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with all technical indicators added.
        """
        if df is None or df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_stochastic(df)
        df = self.add_roc(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_historical_volatility(df)
        df = self.add_adx(df)
        df = self.add_obv(df)
        df = self.add_volume_features(df)
        df = self.add_returns(df)
        df = self.add_candle_features(df)
        # Drop the NaN rows introduced by longer rolling windows
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Return columns that are engineered features (excludes OHLCV)."""
        ohlcv = {"Open", "High", "Low", "Close", "Volume"}
        return [c for c in df.columns if c not in ohlcv]


# ===========================================================================
# 4. DQNAgent (Deep Q-Network)
# ===========================================================================


class DQNAgent:
    """Deep Q-Network reinforcement learning agent for trading.

    State space consists of feature-engineered technical indicators.
    Action space: 0 = HOLD, 1 = BUY, 2 = SELL.

    Parameters
    ----------
    state_size:
        Number of features in the state vector.
    config:
        Trading configuration.
    gamma:
        Discount factor for future rewards.
    epsilon:
        Initial exploration rate (ε-greedy policy).
    epsilon_min:
        Minimum exploration rate.
    epsilon_decay:
        Multiplicative decay applied to *epsilon* after each training step.
    learning_rate:
        Adam optimiser learning rate.
    batch_size:
        Number of transitions sampled per training step.
    memory_size:
        Maximum size of the experience-replay buffer.
    """

    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    N_ACTIONS = 3

    def __init__(
        self,
        state_size: int,
        config: TradingConfig,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        memory_size: int = 10_000,
    ) -> None:
        self.state_size = state_size
        self.config = config
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory: deque = deque(maxlen=memory_size)
        self.model: Any = None
        self.target_model: Any = None
        self._update_target_every = 10
        self._train_step = 0

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        """Build the Q-network using TensorFlow/Keras.

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        """
        try:
            import tensorflow as tf  # lazy import
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for DQNAgent. "
                "Install it with: pip install tensorflow"
            ) from exc

        inp = keras.Input(shape=(self.state_size,), name="state")
        x = keras.layers.Dense(128, activation="relu")(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(self.N_ACTIONS, activation="linear", name="q_values")(x)

        self.model = keras.Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        # Identical architecture for the target network
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        logger.info("DQN model built: state_size=%d", self.state_size)

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using ε-greedy policy.

        Parameters
        ----------
        state:
            1-D feature vector representing the current market state.
        training:
            When ``True``, apply exploration (ε-greedy).  Set to ``False``
            for deterministic inference.

        Returns
        -------
        int
            Action index: 0 = HOLD, 1 = BUY, 2 = SELL.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before act().")
        if training and random.random() < self.epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return int(np.argmax(q_values[0]))

    def replay(self) -> Optional[float]:
        """Sample a mini-batch from memory and train the Q-network.

        Returns
        -------
        float or None
            Training loss, or ``None`` if the buffer is too small.
        """
        if len(self.memory) < self.batch_size:
            return None
        if self.model is None:
            raise RuntimeError("Call build_model() before replay().")

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int32)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        # Double DQN target
        q_next = self.target_model.predict(next_states, verbose=0)
        q_target = self.model.predict(states, verbose=0)
        for i in range(self.batch_size):
            target_val = rewards[i]
            if not dones[i]:
                target_val += self.gamma * np.amax(q_next[i])
            q_target[i][actions[i]] = target_val

        history = self.model.fit(states, q_target, epochs=1, verbose=0)
        loss = float(history.history["loss"][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self._train_step += 1
        if self._train_step % self._update_target_every == 0:
            self.target_model.set_weights(self.model.get_weights())

        return loss

    def save(self, path: str) -> None:
        """Persist the Q-network weights to *path*."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        """Load Q-network weights from *path*."""
        if self.model is None:
            raise RuntimeError("Call build_model() before load().")
        self.model.load_weights(path)
        self.target_model.set_weights(self.model.get_weights())


# ===========================================================================
# 5. PPOAgent (Proximal Policy Optimization)
# ===========================================================================


class PPOAgent:
    """Proximal Policy Optimization agent for trading.

    Uses a shared actor–critic architecture.  The policy (actor) outputs a
    categorical distribution over actions; the critic estimates the value
    function.

    Parameters
    ----------
    state_size:
        Dimension of the state vector.
    config:
        Trading configuration.
    clip_ratio:
        PPO clipping parameter (ε in the paper, typically 0.1–0.3).
    learning_rate:
        Adam learning rate for both actor and critic.
    entropy_coef:
        Coefficient for the entropy bonus (encourages exploration).
    value_coef:
        Coefficient for the critic loss.
    epochs:
        Number of optimisation epochs per PPO update.
    """

    N_ACTIONS = 3  # HOLD, BUY, SELL

    def __init__(
        self,
        state_size: int,
        config: TradingConfig,
        clip_ratio: float = 0.2,
        learning_rate: float = 0.0003,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        epochs: int = 4,
    ) -> None:
        self.state_size = state_size
        self.config = config
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.actor: Any = None
        self.critic: Any = None
        self.optimizer: Any = None
        # Trajectory buffer
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []
        self._dones: List[bool] = []

    def build_model(self) -> None:
        """Build actor and critic networks.

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        """
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for PPOAgent. "
                "Install it with: pip install tensorflow"
            ) from exc

        # Shared body
        state_in = keras.Input(shape=(self.state_size,), name="state")
        shared = keras.layers.Dense(128, activation="relu")(state_in)
        shared = keras.layers.Dense(64, activation="relu")(shared)

        # Actor head
        actor_out = keras.layers.Dense(32, activation="relu")(shared)
        actor_out = keras.layers.Dense(
            self.N_ACTIONS, activation="softmax", name="policy"
        )(actor_out)

        # Critic head
        critic_out = keras.layers.Dense(32, activation="relu")(shared)
        critic_out = keras.layers.Dense(1, activation="linear", name="value")(critic_out)

        self.actor = keras.Model(inputs=state_in, outputs=actor_out)
        self.critic = keras.Model(inputs=state_in, outputs=critic_out)

        try:
            import tensorflow as tf
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        except ImportError:
            pass

        logger.info("PPO model built: state_size=%d", self.state_size)

    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Sample an action from the policy.

        Returns
        -------
        tuple
            ``(action, log_prob, value)`` where *action* is the selected
            action index, *log_prob* is its log-probability, and *value* is
            the critic estimate.
        """
        if self.actor is None:
            raise RuntimeError("Call build_model() before act().")
        import tensorflow as tf

        state_t = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        probs = self.actor(state_t, training=False).numpy()[0]
        value = float(self.critic(state_t, training=False).numpy()[0][0])
        action = int(np.random.choice(self.N_ACTIONS, p=probs))
        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Add one transition to the trajectory buffer."""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._dones.append(done)

    def update(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> Optional[float]:
        """Run PPO update on the collected trajectory.

        Parameters
        ----------
        gamma:
            Discount factor.
        gae_lambda:
            GAE smoothing parameter.

        Returns
        -------
        float or None
            Total loss value, or ``None`` if the buffer is empty.
        """
        if not self._states:
            return None

        import tensorflow as tf

        states = np.array(self._states, dtype=np.float32)
        actions = np.array(self._actions, dtype=np.int32)
        old_log_probs = np.array(self._log_probs, dtype=np.float32)
        rewards = np.array(self._rewards, dtype=np.float32)
        values = np.array(self._values, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        # Generalised Advantage Estimation
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                new_probs = self.actor(states, training=True)
                new_values = tf.squeeze(self.critic(states, training=True), axis=1)
                action_masks = tf.one_hot(actions, self.N_ACTIONS)
                new_log_probs = tf.reduce_sum(
                    action_masks * tf.math.log(new_probs + 1e-10), axis=1
                )
                ratio = tf.exp(new_log_probs - old_log_probs)
                adv_t = tf.convert_to_tensor(advantages, dtype=tf.float32)
                surr1 = ratio * adv_t
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                value_loss = tf.reduce_mean(
                    tf.square(new_values - tf.convert_to_tensor(returns, dtype=tf.float32))
                )
                entropy = -tf.reduce_mean(
                    tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
                )
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
            params = self.actor.trainable_variables + self.critic.trainable_variables
            grads = tape.gradient(loss, params)
            self.optimizer.apply_gradients(zip(grads, params))
            total_loss += float(loss.numpy())

        self._clear_buffer()
        return total_loss / self.epochs

    def _clear_buffer(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._dones.clear()


# ===========================================================================
# 6. LSTMForecaster
# ===========================================================================


class LSTMForecaster:
    """LSTM sequence model for price-direction forecasting.

    Parameters
    ----------
    lookback:
        Number of time steps in each input sequence.
    n_features:
        Number of features per time step.
    config:
        Trading configuration.
    units:
        Tuple of LSTM layer unit counts.
    dropout:
        Dropout rate applied after each LSTM layer.
    epochs:
        Training epochs.
    batch_size:
        Training batch size.
    """

    def __init__(
        self,
        lookback: int,
        n_features: int,
        config: TradingConfig,
        units: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        self.lookback = lookback
        self.n_features = n_features
        self.config = config
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Any = None
        self.scaler: Any = None

    def build_model(self) -> None:
        """Construct the LSTM network.

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        """
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for LSTMForecaster. "
                "Install it with: pip install tensorflow"
            ) from exc

        inp = keras.Input(shape=(self.lookback, self.n_features), name="sequence")
        x = inp
        for i, u in enumerate(self.units):
            return_seqs = i < len(self.units) - 1
            x = keras.layers.LSTM(u, return_sequences=return_seqs)(x)
            x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid", name="direction")(x)

        self.model = keras.Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "LSTM model built: lookback=%d  n_features=%d", self.lookback, self.n_features
        )

    def _make_scaler(self) -> Any:
        """Return a fitted ``MinMaxScaler`` (lazy import)."""
        try:
            from sklearn.preprocessing import MinMaxScaler
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for LSTMForecaster. "
                "Install it with: pip install scikit-learn"
            ) from exc
        return MinMaxScaler()

    def _build_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert a 2-D feature array into 3-D sequences."""
        n = len(X) - self.lookback
        Xs = np.array([X[i : i + self.lookback] for i in range(n)], dtype=np.float32)
        ys = y[self.lookback :] if y is not None else None
        return Xs, ys

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMForecaster":
        """Train the LSTM on feature matrix *X* and binary labels *y*.

        Parameters
        ----------
        X:
            2-D array of shape ``(n_samples, n_features)``.
        y:
            1-D binary label array of length ``n_samples``.

        Returns
        -------
        LSTMForecaster
            Self (for chaining).
        """
        if self.model is None:
            self.build_model()
        self.scaler = self._make_scaler()
        X_scaled = self.scaler.fit_transform(X)
        Xs, ys = self._build_sequences(X_scaled, y)
        self.model.fit(
            Xs,
            ys,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[self._early_stop()],
        )
        return self

    @staticmethod
    def _early_stop() -> Any:
        try:
            from tensorflow import keras
            return keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        except ImportError:
            return None  # unreachable if model is built

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return BUY probability for each sample in *X*.

        Parameters
        ----------
        X:
            2-D feature array.

        Returns
        -------
        np.ndarray
            1-D array of probabilities in [0, 1].
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_scaled = self.scaler.transform(X)
        Xs, _ = self._build_sequences(X_scaled)
        preds = self.model.predict(Xs, verbose=0).flatten()
        # Pad the first *lookback* samples with 0.5 (no-signal)
        padded = np.full(len(X), 0.5)
        padded[self.lookback :] = preds
        return padded


# ===========================================================================
# 7. XGBoostEnsemble
# ===========================================================================


class XGBoostEnsemble:
    """XGBoost-based binary classifier for trade-direction prediction.

    Supports walk-forward retraining and calibrated probability outputs.

    Parameters
    ----------
    config:
        Trading configuration.
    params:
        XGBoost hyper-parameter dictionary.  Uses sensible defaults.
    """

    _DEFAULT_PARAMS: Dict[str, Any] = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 300,
        "eval_metric": "logloss",
        "verbosity": 0,
        "use_label_encoder": False,
        "random_state": 42,
    }

    def __init__(
        self,
        config: TradingConfig,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.params = {**self._DEFAULT_PARAMS, **(params or {})}
        self.model: Any = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostEnsemble":
        """Train the XGBoost classifier.

        Parameters
        ----------
        X:
            2-D feature array.
        y:
            1-D binary label array.

        Returns
        -------
        XGBoostEnsemble
            Self.

        Raises
        ------
        ImportError
            If xgboost is not installed.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostEnsemble. "
                "Install it with: pip install xgboost"
            ) from exc
        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y, verbose=False)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return BUY-class probability for each row in *X*.

        Returns
        -------
        np.ndarray
            1-D array of probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)


# ===========================================================================
# 8–11. Trading Strategies
# ===========================================================================


@dataclass
class TradeSignal:
    """A trading signal produced by a strategy.

    Attributes
    ----------
    symbol:
        Instrument symbol.
    action:
        ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
    confidence:
        Score in [0, 1] indicating model conviction.
    price:
        Suggested entry price.
    stop_loss:
        Suggested stop-loss price (``None`` = not set).
    take_profit:
        Suggested take-profit price (``None`` = not set).
    strategy:
        Name of the generating strategy.
    timestamp:
        UTC timestamp of the signal.
    metadata:
        Arbitrary extra context (indicator values, etc.).
    """

    symbol: str
    action: str
    confidence: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_actionable(self, min_confidence: float) -> bool:
        """Return True if the signal meets the minimum confidence threshold."""
        return self.action != "HOLD" and self.confidence >= min_confidence


class _BaseStrategy:
    """Abstract base for all trading strategies."""

    name: str = "base"

    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    def generate_signal(
        self, symbol: str, df: pd.DataFrame
    ) -> TradeSignal:
        """Generate a TradeSignal for *symbol* from the feature DataFrame *df*.

        Parameters
        ----------
        symbol:
            Instrument ticker.
        df:
            Feature-engineered OHLCV DataFrame (output of
            ``FeatureEngineer.build_features``).

        Returns
        -------
        TradeSignal
        """
        raise NotImplementedError


class ScalpingStrategy(_BaseStrategy):
    """High-frequency micro-profit scalping strategy.

    Looks for very short-term momentum confirmed by volume and candle shape.
    Suitable for 1-minute to 15-minute bars.

    Logic
    -----
    - BUY  when: RSI < 35 AND lower shadow > 60% of full range
                 AND volume ratio > 1.5 AND macd_hist is rising
    - SELL when: RSI > 65 AND upper shadow > 60% of full range
                 AND volume ratio > 1.5 AND macd_hist is falling
    """

    name = "scalping"

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        if df.empty or len(df) < 5:
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=0.0, strategy=self.name)

        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else row
        price = float(row["Close"])

        required = {"rsi", "macd_hist", "volume_ratio", "lower_shadow", "upper_shadow", "atr"}
        if not required.issubset(df.columns):
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=price, strategy=self.name)

        rsi = float(row["rsi"])
        macd_rising = float(row["macd_hist"]) > float(prev["macd_hist"])
        vol_surge = float(row["volume_ratio"]) > 1.5
        lower_shadow = float(row.get("lower_shadow", 0))
        upper_shadow = float(row.get("upper_shadow", 0))
        atr = float(row["atr"])

        if rsi < 35 and lower_shadow > 0.6 and vol_surge and macd_rising:
            confidence = min(0.9, 0.5 + (35 - rsi) / 100 + (lower_shadow - 0.6) * 0.5)
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=price,
                stop_loss=price - atr,
                take_profit=price + 1.5 * atr,
                strategy=self.name,
                metadata={"rsi": rsi, "vol_ratio": float(row["volume_ratio"])},
            )

        if rsi > 65 and upper_shadow > 0.6 and vol_surge and not macd_rising:
            confidence = min(0.9, 0.5 + (rsi - 65) / 100 + (upper_shadow - 0.6) * 0.5)
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                price=price,
                stop_loss=price + atr,
                take_profit=price - 1.5 * atr,
                strategy=self.name,
                metadata={"rsi": rsi, "vol_ratio": float(row["volume_ratio"])},
            )

        return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                           price=price, strategy=self.name)


class DayTradingStrategy(_BaseStrategy):
    """Intraday trend-following strategy.

    Uses EMA crossovers, ADX trend-strength filtering, and MACD confirmation
    to identify directional moves within a single trading session.

    Logic
    -----
    - BUY  when: EMA9 > EMA21 AND EMA21 > EMA50 AND ADX > 25 AND MACD > signal
    - SELL when: EMA9 < EMA21 AND EMA21 < EMA50 AND ADX > 25 AND MACD < signal
    """

    name = "day_trading"

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        if df.empty or len(df) < 50:
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=0.0, strategy=self.name)

        row = df.iloc[-1]
        price = float(row["Close"])

        required = {"ema_9", "ema_21", "ema_50", "adx", "macd", "macd_signal", "atr"}
        if not required.issubset(df.columns):
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=price, strategy=self.name)

        ema9 = float(row["ema_9"])
        ema21 = float(row["ema_21"])
        ema50 = float(row["ema_50"])
        adx = float(row["adx"])
        macd = float(row["macd"])
        macd_sig = float(row["macd_signal"])
        atr = float(row["atr"])

        trending = adx > 25
        bull_align = ema9 > ema21 > ema50
        bear_align = ema9 < ema21 < ema50

        if trending and bull_align and macd > macd_sig:
            confidence = min(0.85, 0.5 + adx / 200 + (ema9 - ema21) / price)
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=float(confidence),
                price=price,
                stop_loss=price - 2 * atr,
                take_profit=price + 3 * atr,
                strategy=self.name,
                metadata={"adx": adx, "ema_alignment": "bullish"},
            )

        if trending and bear_align and macd < macd_sig:
            confidence = min(0.85, 0.5 + adx / 200 + (ema21 - ema9) / price)
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=float(confidence),
                price=price,
                stop_loss=price + 2 * atr,
                take_profit=price - 3 * atr,
                strategy=self.name,
                metadata={"adx": adx, "ema_alignment": "bearish"},
            )

        return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                           price=price, strategy=self.name)


class MeanReversionStrategy(_BaseStrategy):
    """Statistical mean-reversion strategy.

    Trades the assumption that prices revert to their moving average.  Uses
    Bollinger Band extremes and RSI oversold/overbought confirmation.

    Logic
    -----
    - BUY  when: price < lower Bollinger Band AND RSI < 30 AND %B < 0.05
    - SELL when: price > upper Bollinger Band AND RSI > 70 AND %B > 0.95
    """

    name = "mean_reversion"

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        if df.empty or len(df) < 20:
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=0.0, strategy=self.name)

        row = df.iloc[-1]
        price = float(row["Close"])

        required = {"bb_lower", "bb_upper", "bb_pct", "rsi", "atr", "sma_20"}
        if not required.issubset(df.columns):
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=price, strategy=self.name)

        rsi = float(row["rsi"])
        bb_pct = float(row["bb_pct"])
        atr = float(row["atr"])
        sma20 = float(row["sma_20"])

        if price < float(row["bb_lower"]) and rsi < 30 and bb_pct < 0.05:
            confidence = min(0.85, 0.5 + (30 - rsi) / 60 + (0.05 - bb_pct))
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=float(confidence),
                price=price,
                stop_loss=price - 1.5 * atr,
                take_profit=sma20,
                strategy=self.name,
                metadata={"rsi": rsi, "bb_pct": bb_pct},
            )

        if price > float(row["bb_upper"]) and rsi > 70 and bb_pct > 0.95:
            confidence = min(0.85, 0.5 + (rsi - 70) / 60 + (bb_pct - 0.95))
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=float(confidence),
                price=price,
                stop_loss=price + 1.5 * atr,
                take_profit=sma20,
                strategy=self.name,
                metadata={"rsi": rsi, "bb_pct": bb_pct},
            )

        return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                           price=price, strategy=self.name)


class MomentumStrategy(_BaseStrategy):
    """Price and volume momentum strategy.

    Identifies strong directional moves supported by increasing volume.
    Uses ROC, OBV trend, and ADX to filter high-momentum setups.

    Logic
    -----
    - BUY  when: ROC_10 > 3 % AND volume rising AND ADX > 20 AND price > SMA50
    - SELL when: ROC_10 < -3% AND volume rising AND ADX > 20 AND price < SMA50
    """

    name = "momentum"

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        if df.empty or len(df) < 50:
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=0.0, strategy=self.name)

        row = df.iloc[-1]
        prev = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
        price = float(row["Close"])

        required = {"roc", "obv", "adx", "sma_50", "atr", "volume_ratio"}
        if not required.issubset(df.columns):
            return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                               price=price, strategy=self.name)

        roc = float(row["roc"])
        adx = float(row["adx"])
        sma50 = float(row["sma_50"])
        atr = float(row["atr"])
        obv_rising = float(row["obv"]) > float(prev["obv"])
        vol_ratio = float(row["volume_ratio"])

        if roc > 3.0 and obv_rising and adx > 20 and price > sma50 and vol_ratio > 1.2:
            confidence = min(0.88, 0.5 + roc / 20 + adx / 200)
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=float(confidence),
                price=price,
                stop_loss=price - 2 * atr,
                take_profit=price + 4 * atr,
                strategy=self.name,
                metadata={"roc": roc, "adx": adx},
            )

        if roc < -3.0 and not obv_rising and adx > 20 and price < sma50 and vol_ratio > 1.2:
            confidence = min(0.88, 0.5 + abs(roc) / 20 + adx / 200)
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=float(confidence),
                price=price,
                stop_loss=price + 2 * atr,
                take_profit=price - 4 * atr,
                strategy=self.name,
                metadata={"roc": roc, "adx": adx},
            )

        return TradeSignal(symbol=symbol, action="HOLD", confidence=0.0,
                           price=price, strategy=self.name)


class StrategyComposer:
    """Combine signals from multiple strategies using weighted voting.

    Parameters
    ----------
    strategies:
        List of strategy instances.
    weights:
        Optional weight per strategy.  Must match length of *strategies*.
        Defaults to equal weights.
    config:
        Trading configuration.
    """

    def __init__(
        self,
        strategies: List[_BaseStrategy],
        config: TradingConfig,
        weights: Optional[List[float]] = None,
    ) -> None:
        self.strategies = strategies
        self.config = config
        n = len(strategies)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError("len(weights) must match len(strategies).")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        """Aggregate signals from all strategies into a single composite signal.

        Parameters
        ----------
        symbol:
            Instrument ticker.
        df:
            Feature DataFrame.

        Returns
        -------
        TradeSignal
            The composite signal.  Confidence is the weighted average of
            individual confidences; action is determined by majority vote.
        """
        buy_score = 0.0
        sell_score = 0.0
        latest_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
        stop_losses: List[float] = []
        take_profits: List[float] = []

        for strategy, weight in zip(self.strategies, self.weights):
            sig = strategy.generate_signal(symbol, df)
            if sig.action == "BUY":
                buy_score += weight * sig.confidence
                if sig.stop_loss:
                    stop_losses.append(sig.stop_loss)
                if sig.take_profit:
                    take_profits.append(sig.take_profit)
            elif sig.action == "SELL":
                sell_score += weight * sig.confidence
                if sig.stop_loss:
                    stop_losses.append(sig.stop_loss)
                if sig.take_profit:
                    take_profits.append(sig.take_profit)

        if buy_score > sell_score and buy_score >= self.config.min_confidence:
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=buy_score,
                price=latest_price,
                stop_loss=float(np.mean(stop_losses)) if stop_losses else None,
                take_profit=float(np.mean(take_profits)) if take_profits else None,
                strategy="composer",
            )
        if sell_score > buy_score and sell_score >= self.config.min_confidence:
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=sell_score,
                price=latest_price,
                stop_loss=float(np.mean(stop_losses)) if stop_losses else None,
                take_profit=float(np.mean(take_profits)) if take_profits else None,
                strategy="composer",
            )
        return TradeSignal(
            symbol=symbol, action="HOLD", confidence=0.0,
            price=latest_price, strategy="composer"
        )


# ===========================================================================
# 13. RiskManager
# ===========================================================================


class RiskManager:
    """Portfolio and per-trade risk management.

    Implements:
    - **Kelly Criterion** position sizing (fractional Kelly for safety)
    - **Fixed-fractional** position sizing
    - **Volatility-adjusted** sizing (ATR-based)
    - Drawdown monitoring and hard stop
    - Stop-loss / take-profit enforcement

    Parameters
    ----------
    config:
        Trading configuration.
    """

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self._peak_value: float = config.initial_capital
        self._current_value: float = config.initial_capital
        self._drawdown_breached: bool = False

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def kelly_size(
        self,
        win_rate: float,
        win_loss_ratio: float,
        portfolio_value: float,
        price: float,
    ) -> int:
        """Return position size (shares) using fractional Kelly Criterion.

        Parameters
        ----------
        win_rate:
            Historical win rate of the strategy (0–1).
        win_loss_ratio:
            Ratio of average win to average loss (must be > 0).
        portfolio_value:
            Current portfolio equity.
        price:
            Current instrument price.

        Returns
        -------
        int
            Number of shares/units to trade.
        """
        if win_loss_ratio <= 0 or not (0 < win_rate < 1):
            return 0
        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        kelly_fraction = max(0.0, kelly_fraction)
        # Use half-Kelly for safety
        safe_fraction = kelly_fraction * 0.5
        safe_fraction = min(safe_fraction, self.config.max_position_size)
        dollar_amount = portfolio_value * safe_fraction
        return max(0, int(dollar_amount / price))

    def fixed_fractional_size(
        self,
        portfolio_value: float,
        price: float,
        stop_loss_price: Optional[float] = None,
    ) -> int:
        """Return position size using fixed-fractional risk.

        If *stop_loss_price* is provided, sizes the position so the dollar
        risk (``price - stop_loss_price``) equals ``risk_per_trade * equity``.

        Parameters
        ----------
        portfolio_value:
            Current portfolio equity.
        price:
            Entry price.
        stop_loss_price:
            Stop-loss price.  When ``None``, uses ``risk_per_trade`` directly
            as a fraction of equity.

        Returns
        -------
        int
            Number of shares/units to trade.
        """
        max_dollar = portfolio_value * self.config.max_position_size
        if stop_loss_price is not None and stop_loss_price > 0:
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit <= 0:
                return 0
            risk_budget = portfolio_value * self.config.risk_per_trade
            size = int(risk_budget / risk_per_unit)
        else:
            dollar_amount = portfolio_value * self.config.risk_per_trade
            size = int(dollar_amount / price)
        max_size = int(max_dollar / price)
        return min(size, max_size)

    def volatility_adjusted_size(
        self,
        portfolio_value: float,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> int:
        """Position size inversely proportional to recent volatility (ATR).

        Parameters
        ----------
        portfolio_value:
            Current portfolio equity.
        price:
            Entry price.
        atr:
            Average True Range for the instrument.
        atr_multiplier:
            Number of ATRs to use as the risk per trade.

        Returns
        -------
        int
            Number of shares/units.
        """
        if atr <= 0:
            return self.fixed_fractional_size(portfolio_value, price)
        risk_budget = portfolio_value * self.config.risk_per_trade
        stop_distance = atr * atr_multiplier
        size = int(risk_budget / stop_distance)
        max_size = int(portfolio_value * self.config.max_position_size / price)
        return min(size, max_size)

    # ------------------------------------------------------------------
    # Drawdown monitoring
    # ------------------------------------------------------------------

    def update_equity(self, current_value: float) -> None:
        """Update the equity high-water mark and check drawdown limits.

        Parameters
        ----------
        current_value:
            Current portfolio value.
        """
        self._current_value = current_value
        if current_value > self._peak_value:
            self._peak_value = current_value
        dd = self.current_drawdown()
        if dd >= self.config.max_drawdown:
            if not self._drawdown_breached:
                logger.warning(
                    "MAX DRAWDOWN BREACHED: %.1f%% — halting new trades.",
                    dd * 100,
                )
            self._drawdown_breached = True
        else:
            self._drawdown_breached = False

    def current_drawdown(self) -> float:
        """Return current drawdown as a decimal (0 = no drawdown)."""
        if self._peak_value <= 0:
            return 0.0
        return max(0.0, (self._peak_value - self._current_value) / self._peak_value)

    def is_trading_halted(self) -> bool:
        """True when the maximum drawdown limit has been breached."""
        return self._drawdown_breached

    # ------------------------------------------------------------------
    # Trade validation
    # ------------------------------------------------------------------

    def validate_signal(
        self, signal: TradeSignal, portfolio_value: float
    ) -> bool:
        """Return True if the signal passes all risk filters.

        Filters applied:
        1. Drawdown limit not breached.
        2. Signal confidence ≥ ``min_confidence``.
        3. For BUY signals: computed position size > 0.
        """
        if self.is_trading_halted():
            logger.debug("Trade rejected: max drawdown breached.")
            return False
        if not signal.is_actionable(self.config.min_confidence):
            return False
        if signal.action == "BUY":
            size = self.fixed_fractional_size(
                portfolio_value, signal.price, signal.stop_loss
            )
            if size <= 0:
                logger.debug("Trade rejected: position size = 0.")
                return False
        return True


# ===========================================================================
# 14. BacktestEngine
# ===========================================================================


@dataclass
class BacktestResult:
    """Container for backtest performance metrics.

    Attributes
    ----------
    total_return:
        Total percentage return over the backtest period.
    annualised_return:
        Compound annual growth rate.
    sharpe_ratio:
        Risk-adjusted return (annualised).
    sortino_ratio:
        Downside-risk-adjusted return.
    max_drawdown:
        Maximum peak-to-trough drawdown as a decimal.
    win_rate:
        Fraction of closed trades that were profitable.
    profit_factor:
        Ratio of gross profit to gross loss.
    total_trades:
        Total number of completed round-trip trades.
    equity_curve:
        Time series of portfolio value indexed by date.
    trades:
        List of individual trade records.
    """

    total_return: float = 0.0
    annualised_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary dictionary."""
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "annualised_return_pct": round(self.annualised_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 3),
            "total_trades": self.total_trades,
        }


class BacktestEngine:
    """Realistic single-instrument event-driven backtester.

    Features
    --------
    - Commission and slippage modelling
    - Flexible strategy callback interface
    - Walk-forward validation (expanding window)
    - Performance metric computation (Sharpe, Sortino, Calmar, drawdown)
    - Out-of-sample testing

    Parameters
    ----------
    config:
        Trading configuration (uses ``commission_rate`` and ``slippage_rate``).
    feature_engineer:
        ``FeatureEngineer`` instance for on-the-fly feature computation.
    """

    def __init__(
        self,
        config: TradingConfig,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.config = config
        self.fe = feature_engineer or FeatureEngineer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[str, pd.DataFrame], TradeSignal],
        symbol: str = "UNKNOWN",
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """Run a backtest on a single instrument.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame (will be feature-engineered internally).
        strategy_fn:
            Callable ``(symbol, feature_df) -> TradeSignal``.
        symbol:
            Instrument name (for logging).
        initial_capital:
            Starting capital.  Defaults to ``config.initial_capital``.

        Returns
        -------
        BacktestResult
        """
        capital = initial_capital if initial_capital is not None else self.config.initial_capital
        feat_df = self.fe.build_features(df.copy())
        if feat_df.empty:
            logger.warning("No features could be built for %s.", symbol)
            return BacktestResult()

        equity_series: Dict[Any, float] = {}
        trades: List[Dict[str, Any]] = []
        cash = capital
        position = 0
        entry_price = 0.0
        entry_date: Any = None

        for i in range(1, len(feat_df)):
            snapshot = feat_df.iloc[: i + 1]
            signal = strategy_fn(symbol, snapshot)
            current_date = feat_df.index[i]
            price = float(feat_df["Close"].iloc[i])
            effective_price = price * (
                1 + self.config.slippage_rate * (1 if signal.action == "BUY" else -1)
            )

            if signal.action == "BUY" and position == 0 and signal.confidence >= self.config.min_confidence:
                max_units = int(cash * self.config.max_position_size / effective_price)
                if max_units > 0:
                    cost = max_units * effective_price * (1 + self.config.commission_rate)
                    cash -= cost
                    position = max_units
                    entry_price = effective_price
                    entry_date = current_date

            elif signal.action == "SELL" and position > 0:
                proceeds = position * effective_price * (1 - self.config.commission_rate)
                pnl = proceeds - position * entry_price
                cash += proceeds
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": effective_price,
                    "quantity": position,
                    "pnl": pnl,
                    "pnl_pct": pnl / (position * entry_price) if entry_price > 0 else 0,
                })
                position = 0
                entry_price = 0.0

            # Stop-loss / take-profit check for open position
            if position > 0:
                sig_sl = signal.stop_loss
                sig_tp = signal.take_profit
                if sig_sl and price <= sig_sl:
                    proceeds = position * price * (1 - self.config.commission_rate)
                    pnl = proceeds - position * entry_price
                    cash += proceeds
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "quantity": position,
                        "pnl": pnl,
                        "pnl_pct": pnl / (position * entry_price) if entry_price > 0 else 0,
                        "exit_reason": "stop_loss",
                    })
                    position = 0
                    entry_price = 0.0
                elif sig_tp and price >= sig_tp:
                    proceeds = position * price * (1 - self.config.commission_rate)
                    pnl = proceeds - position * entry_price
                    cash += proceeds
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "quantity": position,
                        "pnl": pnl,
                        "pnl_pct": pnl / (position * entry_price) if entry_price > 0 else 0,
                        "exit_reason": "take_profit",
                    })
                    position = 0
                    entry_price = 0.0

            portfolio_value = cash + position * price
            equity_series[current_date] = portfolio_value

        # Close any remaining position at end
        if position > 0:
            last_price = float(feat_df["Close"].iloc[-1])
            cash += position * last_price * (1 - self.config.commission_rate)
            position = 0

        equity_curve = pd.Series(equity_series)
        return self._compute_metrics(equity_curve, trades, capital)

    def walk_forward(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[str, pd.DataFrame], TradeSignal],
        symbol: str = "UNKNOWN",
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> List[BacktestResult]:
        """Walk-forward validation over *n_splits* expanding windows.

        Parameters
        ----------
        df:
            Full OHLCV dataset.
        strategy_fn:
            Strategy callable.
        symbol:
            Instrument name.
        n_splits:
            Number of folds.
        train_ratio:
            Fraction of each fold used for training (the rest is test).

        Returns
        -------
        list of BacktestResult
            One result per fold (out-of-sample portion only).
        """
        results = []
        fold_size = len(df) // n_splits
        for fold in range(n_splits):
            start_idx = 0
            end_idx = (fold + 1) * fold_size
            train_end = start_idx + int((end_idx - start_idx) * train_ratio)
            test_df = df.iloc[train_end:end_idx]
            if len(test_df) < 30:
                continue
            result = self.run(test_df, strategy_fn, symbol)
            results.append(result)
            logger.info(
                "Walk-forward fold %d/%d: return=%.1f%%  sharpe=%.2f  trades=%d",
                fold + 1,
                n_splits,
                result.total_return * 100,
                result.sharpe_ratio,
                result.total_trades,
            )
        return results

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        equity_curve: pd.Series,
        trades: List[Dict[str, Any]],
        initial_capital: float,
    ) -> BacktestResult:
        if equity_curve.empty:
            return BacktestResult()

        final_value = float(equity_curve.iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital

        n_days = len(equity_curve)
        years = n_days / _TRADING_DAYS
        ann_return = (1 + total_return) ** (1 / max(years, 1e-6)) - 1 if years > 0 else 0.0

        daily_returns = equity_curve.pct_change().dropna()
        excess = daily_returns - _DEFAULT_RISK_FREE / _TRADING_DAYS
        sharpe = 0.0
        if len(excess) > 1 and excess.std() > 0:
            sharpe = float(excess.mean() / excess.std() * math.sqrt(_TRADING_DAYS))

        downside = daily_returns[daily_returns < 0]
        sortino = 0.0
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(
                excess.mean() / downside.std() * math.sqrt(_TRADING_DAYS)
            )

        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max.replace(0, np.nan)
        max_dd = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_return=total_return,
            annualised_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades,
        )


# ===========================================================================
# 15. TradingExecutor
# ===========================================================================


@dataclass
class Order:
    """Represents a trading order.

    Attributes
    ----------
    order_id:
        Unique identifier.
    symbol:
        Instrument symbol.
    side:
        ``"BUY"`` or ``"SELL"``.
    quantity:
        Number of units.
    price:
        Order price (``None`` = market order).
    order_type:
        ``"MARKET"`` or ``"LIMIT"``.
    status:
        ``"PENDING"``, ``"FILLED"``, ``"CANCELLED"``, ``"REJECTED"``.
    filled_price:
        Actual fill price after execution.
    timestamp:
        Order creation time.
    """

    order_id: str
    symbol: str
    side: str
    quantity: int
    price: Optional[float] = None
    order_type: str = "MARKET"
    status: str = "PENDING"
    filled_price: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class TradingExecutor:
    """Paper and (stub) live order execution engine.

    In **paper** mode all orders are simulated immediately at the signal price
    adjusted for slippage.  In **live** mode the executor calls the broker
    adapter (pluggable via ``set_broker``).

    Parameters
    ----------
    config:
        Trading configuration.
    risk_manager:
        Risk manager for position-size computation and validation.
    """

    def __init__(
        self,
        config: TradingConfig,
        risk_manager: RiskManager,
    ) -> None:
        self.config = config
        self.risk = risk_manager
        self._broker: Optional[Any] = None
        self._portfolio: Dict[str, int] = {}
        self._cash: float = config.initial_capital
        self._trade_log: List[Dict[str, Any]] = []
        self._order_counter: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_broker(self, broker: Any) -> None:
        """Plug in a live broker adapter (must implement ``submit_order(Order)``)."""
        self._broker = broker

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_signal(self, signal: TradeSignal) -> Optional[Order]:
        """Translate a ``TradeSignal`` into an order and execute it.

        Parameters
        ----------
        signal:
            Trade signal from a strategy.

        Returns
        -------
        Order or None
            Executed order, or ``None`` if the signal was rejected.
        """
        portfolio_value = self._portfolio_value(signal.price)
        if not self.risk.validate_signal(signal, portfolio_value):
            return None

        if signal.action == "BUY":
            size = self.risk.fixed_fractional_size(
                portfolio_value, signal.price, signal.stop_loss
            )
            if size <= 0:
                return None
            return self._submit(signal.symbol, "BUY", size, signal.price)

        if signal.action == "SELL":
            size = self._portfolio.get(signal.symbol, 0)
            if size <= 0:
                logger.debug("No open position to sell for %s.", signal.symbol)
                return None
            return self._submit(signal.symbol, "SELL", size, signal.price)

        return None

    def _submit(
        self, symbol: str, side: str, quantity: int, price: float
    ) -> Order:
        """Create, simulate/send, and log an order."""
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )

        if self.config.mode == "paper" or self._broker is None:
            order = self._paper_fill(order)
        else:
            order = self._live_fill(order)

        self._update_portfolio(order)
        self._log_order(order)
        return order

    def _paper_fill(self, order: Order) -> Order:
        """Simulate an immediate fill at price +/- slippage."""
        slip = self.config.slippage_rate
        fill_price = (order.price or 0.0) * (
            1 + slip if order.side == "BUY" else 1 - slip
        )
        order.filled_price = fill_price
        order.status = "FILLED"
        logger.info(
            "[PAPER] %s %d %s @ %.4f", order.side, order.quantity, order.symbol, fill_price
        )
        return order

    def _live_fill(self, order: Order) -> Order:
        """Submit order to the configured live broker."""
        try:
            result = self._broker.submit_order(order)
            order.filled_price = result.get("filled_price", order.price)
            order.status = result.get("status", "FILLED")
        except Exception as exc:  # noqa: BLE001
            logger.error("Live order failed for %s: %s", order.symbol, exc)
            order.status = "REJECTED"
        return order

    def _update_portfolio(self, order: Order) -> None:
        if order.status != "FILLED" or order.filled_price is None:
            return
        cost = order.quantity * order.filled_price
        comm = cost * self.config.commission_rate
        if order.side == "BUY":
            self._cash -= cost + comm
            self._portfolio[order.symbol] = (
                self._portfolio.get(order.symbol, 0) + order.quantity
            )
        elif order.side == "SELL":
            self._cash += cost - comm
            held = self._portfolio.get(order.symbol, 0)
            self._portfolio[order.symbol] = max(0, held - order.quantity)

    def _log_order(self, order: Order) -> None:
        self._trade_log.append(
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "filled_price": order.filled_price,
                "status": order.status,
                "timestamp": order.timestamp.isoformat(),
            }
        )

    def _portfolio_value(self, current_price: float) -> float:
        """Approximate portfolio value using *current_price* for all positions."""
        position_value = sum(
            qty * current_price for qty in self._portfolio.values()
        )
        return self._cash + position_value

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_trade_log(self) -> pd.DataFrame:
        """Return the full audit trail as a DataFrame."""
        return pd.DataFrame(self._trade_log)

    def get_portfolio(self) -> Dict[str, Any]:
        """Return current positions and cash balance."""
        return {"cash": self._cash, "positions": dict(self._portfolio)}


# ===========================================================================
# 16. PerformanceMonitor
# ===========================================================================


class PerformanceMonitor:
    """Real-time performance tracking, anomaly detection, and alerting.

    Parameters
    ----------
    config:
        Trading configuration.
    alert_callback:
        Optional callable invoked when an alert is raised; receives the alert
        message string.  Defaults to ``logger.warning``.
    """

    def __init__(
        self,
        config: TradingConfig,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self._alert_cb = alert_callback or (lambda msg: logger.warning("ALERT: %s", msg))
        self._equity_history: List[Tuple[datetime, float]] = []
        self._signal_history: List[TradeSignal] = []
        self._anomaly_window = 20

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_equity(self, value: float) -> None:
        """Record the current portfolio equity."""
        self._equity_history.append((datetime.now(tz=timezone.utc), value))

    def record_signal(self, signal: TradeSignal) -> None:
        """Record a generated trade signal."""
        self._signal_history.append(signal)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def current_sharpe(self) -> float:
        """Compute Sharpe ratio from the recorded equity history."""
        if len(self._equity_history) < 5:
            return 0.0
        values = pd.Series([v for _, v in self._equity_history])
        returns = values.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        excess = returns - _DEFAULT_RISK_FREE / _TRADING_DAYS
        return float(excess.mean() / excess.std() * math.sqrt(_TRADING_DAYS))

    def current_drawdown(self) -> float:
        """Return current drawdown from peak equity."""
        if not self._equity_history:
            return 0.0
        values = [v for _, v in self._equity_history]
        peak = max(values)
        current = values[-1]
        return max(0.0, (peak - current) / peak) if peak > 0 else 0.0

    def anomaly_score(self) -> float:
        """Return a z-score indicating how unusual the latest equity change is.

        A value > 2 suggests a significant anomaly.
        """
        if len(self._equity_history) < self._anomaly_window + 1:
            return 0.0
        values = pd.Series([v for _, v in self._equity_history])
        returns = values.pct_change().dropna()
        window = returns.iloc[-self._anomaly_window :]
        if window.std() == 0:
            return 0.0
        latest = returns.iloc[-1]
        z_score = (latest - window.mean()) / window.std()
        return float(abs(z_score))

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def check_alerts(self, risk_manager: RiskManager) -> None:
        """Run all alert checks and invoke the callback for any that trigger."""
        dd = risk_manager.current_drawdown()
        warn_dd = self.config.max_drawdown * 0.75
        if dd >= warn_dd:
            self._alert_cb(
                f"Drawdown warning: current={dd:.1%}  limit={self.config.max_drawdown:.1%}"
            )
        if risk_manager.is_trading_halted():
            self._alert_cb("Trading halted: maximum drawdown exceeded.")
        z = self.anomaly_score()
        if z > 3.0:
            self._alert_cb(f"Equity anomaly detected: z-score={z:.2f}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary of key performance statistics."""
        if not self._equity_history:
            return {}
        initial = self._equity_history[0][1]
        current = self._equity_history[-1][1]
        return {
            "initial_equity": round(initial, 2),
            "current_equity": round(current, 2),
            "total_return_pct": round((current - initial) / initial * 100, 2),
            "sharpe_ratio": round(self.current_sharpe(), 3),
            "current_drawdown_pct": round(self.current_drawdown() * 100, 2),
            "anomaly_score": round(self.anomaly_score(), 3),
            "signals_generated": len(self._signal_history),
        }

    def print_summary(self, result: Optional[BacktestResult] = None) -> None:
        """Print a human-readable performance summary to the logger."""
        data = self.summary()
        if result is not None:
            data.update(result.to_dict())
        lines = ["=" * 60, "  Performance Summary", "=" * 60]
        for key, val in data.items():
            lines.append(f"  {key:<30} {val}")
        lines.append("=" * 60)
        for line in lines:
            logger.info(line)


# ===========================================================================
# 17. AlgoTraderML – Top-level orchestrator
# ===========================================================================


class AlgoTraderML:
    """Top-level orchestration class for the ML-based algorithmic trading system.

    Ties together data fetching, feature engineering, strategy generation,
    risk management, backtesting, trade execution, and performance monitoring.

    Parameters
    ----------
    config:
        ``TradingConfig`` instance.  A default paper-trading config is used
        when omitted.

    Examples
    --------
    Basic paper-trading workflow::

        config = TradingConfig(
            mode="paper",
            symbols=["AAPL", "BTC-USD", "ES=F"],
            max_drawdown=0.10,
        )
        trader = AlgoTraderML(config)
        trader.initialize()
        signals = trader.generate_signals()
        for sig in signals:
            print(sig)
        results = trader.backtest(start="2022-01-01", end="2023-12-31")
        trader.monitor.print_summary(results)
    """

    def __init__(self, config: Optional[TradingConfig] = None) -> None:
        self.config = config or TradingConfig()
        self.fetcher = DataFetcher(self.config)
        self.fe = FeatureEngineer()
        self.risk = RiskManager(self.config)
        self.monitor = PerformanceMonitor(self.config)
        self.executor = TradingExecutor(self.config, self.risk)
        self._strategies: List[_BaseStrategy] = [
            ScalpingStrategy(self.config),
            DayTradingStrategy(self.config),
            MeanReversionStrategy(self.config),
            MomentumStrategy(self.config),
        ]
        self.composer = StrategyComposer(self._strategies, self.config)
        self.backtest_engine = BacktestEngine(self.config, self.fe)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._last_retrain: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        """Fetch historical data for all configured symbols.

        Parameters
        ----------
        start:
            ISO-8601 start date.  Defaults to ``lookback_days`` before today.
        end:
            ISO-8601 end date.  Defaults to today.
        """
        logger.info("Initialising AlgoTraderML for symbols: %s", self.config.symbols)
        self._data_cache = self.fetcher.fetch_all(start=start, end=end)
        logger.info("Data ready for %d symbols.", len(self._data_cache))

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self) -> List[TradeSignal]:
        """Generate composite trading signals for all symbols.

        Returns
        -------
        list of TradeSignal
            Only actionable signals (confidence ≥ ``min_confidence``) with
            action != "HOLD" are returned.
        """
        signals: List[TradeSignal] = []
        for sym, raw_df in self._data_cache.items():
            if raw_df.empty:
                continue
            try:
                feat_df = self.fe.build_features(raw_df.copy())
                if feat_df.empty:
                    continue
                signal = self.composer.generate_signal(sym, feat_df)
                self.monitor.record_signal(signal)
                if signal.is_actionable(self.config.min_confidence):
                    signals.append(signal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Signal generation failed for %s: %s", sym, exc)
        return signals

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbol: Optional[str] = None,
        walk_forward: bool = False,
        n_splits: int = 5,
    ) -> BacktestResult:
        """Run a backtest on the first (or specified) symbol.

        Parameters
        ----------
        start:
            Backtest start date.
        end:
            Backtest end date.
        symbol:
            Symbol to backtest.  Defaults to the first configured symbol.
        walk_forward:
            If ``True``, run walk-forward validation and return the last fold.
        n_splits:
            Number of walk-forward folds.

        Returns
        -------
        BacktestResult
        """
        sym = symbol or self.config.symbols[0]
        try:
            raw_df = self.fetcher.fetch(sym, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not fetch backtest data for %s: %s", sym, exc)
            return BacktestResult()

        strategy_fn = lambda s, df: self.composer.generate_signal(s, df)  # noqa: E731

        if walk_forward:
            fold_results = self.backtest_engine.walk_forward(
                raw_df, strategy_fn, sym, n_splits=n_splits
            )
            return fold_results[-1] if fold_results else BacktestResult()

        result = self.backtest_engine.run(raw_df, strategy_fn, sym)
        logger.info(
            "Backtest [%s]: return=%.1f%%  sharpe=%.2f  max_dd=%.1f%%  trades=%d",
            sym,
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
            result.total_trades,
        )
        return result

    # ------------------------------------------------------------------
    # Live / paper trading loop
    # ------------------------------------------------------------------

    def run_once(self) -> List[Order]:
        """Generate signals and execute them.  Suitable for a scheduled job.

        Returns
        -------
        list of Order
            Executed orders (may be empty if no actionable signals).
        """
        self.monitor.check_alerts(self.risk)
        if self.risk.is_trading_halted():
            logger.warning("Trading halted: max drawdown breached.")
            return []

        signals = self.generate_signals()
        orders: List[Order] = []
        for sig in signals:
            order = self.executor.execute_signal(sig)
            if order and order.status == "FILLED":
                orders.append(order)

        # Update equity tracking
        if self._data_cache:
            approx_price = 0.0
            for df in self._data_cache.values():
                if not df.empty:
                    approx_price = float(df["Close"].iloc[-1])
                    break
            pv = self.executor._portfolio_value(approx_price)
            self.risk.update_equity(pv)
            self.monitor.record_equity(pv)

        return orders

    def run(
        self,
        iterations: int = 1,
        sleep_seconds: float = 0.0,
    ) -> None:
        """Run the trading loop for *iterations* cycles.

        Parameters
        ----------
        iterations:
            Number of trading cycles to execute.  Use a very large number
            (or a ``while True`` loop) for a continuous live-trading daemon.
        sleep_seconds:
            Seconds to sleep between iterations (useful for rate-limit
            compliance in live trading).
        """
        logger.info("Starting trading loop: %d iteration(s).", iterations)
        for i in range(iterations):
            orders = self.run_once()
            logger.info("Iteration %d: %d order(s) executed.", i + 1, len(orders))
            if sleep_seconds > 0 and i < iterations - 1:
                time.sleep(sleep_seconds)

    # ------------------------------------------------------------------
    # Model retraining
    # ------------------------------------------------------------------

    def should_retrain(self) -> bool:
        """Return True if it is time for a periodic model retrain."""
        if self._last_retrain is None:
            return True
        elapsed = (datetime.now(tz=timezone.utc) - self._last_retrain).days
        return elapsed >= self.config.retrain_interval_days

    def train_xgboost(
        self,
        symbol: str,
        forward_days: int = 5,
        return_threshold: float = 0.02,
    ) -> XGBoostEnsemble:
        """Train an XGBoostEnsemble model on historical data for *symbol*.

        Parameters
        ----------
        symbol:
            Symbol to train on.
        forward_days:
            Number of days ahead used to define the label.
        return_threshold:
            Minimum forward return for a BUY label (binary classification).

        Returns
        -------
        XGBoostEnsemble
            Fitted model.
        """
        raw_df = self._data_cache.get(symbol)
        if raw_df is None or raw_df.empty:
            raise ValueError(f"No data cached for '{symbol}'. Call initialize() first.")

        feat_df = self.fe.build_features(raw_df.copy())
        if feat_df.empty:
            raise ValueError("Feature engineering produced an empty DataFrame.")

        feature_cols = self.fe.get_feature_columns(feat_df)
        X = feat_df[feature_cols].values.astype(np.float32)

        fwd_return = feat_df["Close"].pct_change(periods=forward_days).shift(-forward_days)
        y = (fwd_return > return_threshold).astype(int).values
        # Drop rows with NaN labels
        valid = ~np.isnan(fwd_return.values)
        X, y = X[valid], y[valid]

        model = XGBoostEnsemble(self.config)
        model.fit(X, y)
        self._last_retrain = datetime.now(tz=timezone.utc)
        logger.info("XGBoost model trained for %s (%d samples).", symbol, len(y))
        return model


# ===========================================================================
# Convenience factory
# ===========================================================================


def create_trader(
    symbols: Optional[List[str]] = None,
    mode: str = "paper",
    initial_capital: float = 100_000.0,
    max_drawdown: float = 0.10,
    **kwargs: Any,
) -> AlgoTraderML:
    """Create an ``AlgoTraderML`` instance with sensible defaults.

    Parameters
    ----------
    symbols:
        List of ticker symbols.  Defaults to ``["AAPL", "BTC-USD", "ES=F"]``.
    mode:
        ``"paper"`` or ``"live"``.
    initial_capital:
        Starting capital in USD.
    max_drawdown:
        Maximum tolerable drawdown fraction.
    **kwargs:
        Additional keyword arguments forwarded to ``TradingConfig``.

    Returns
    -------
    AlgoTraderML
    """
    cfg = TradingConfig(
        symbols=symbols or ["AAPL", "BTC-USD", "ES=F"],
        mode=mode,
        initial_capital=initial_capital,
        max_drawdown=max_drawdown,
        **kwargs,
    )
    return AlgoTraderML(cfg)


# ===========================================================================
# CLI / example usage
# ===========================================================================


def _demo() -> None:
    """Run a quick demonstration without network calls."""
    import warnings
    warnings.filterwarnings("ignore")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Build synthetic OHLCV data
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )

    # Feature engineering
    fe = FeatureEngineer()
    feat_df = fe.build_features(df.copy())
    logger.info("Features built: %d rows × %d columns", *feat_df.shape)

    # Strategy signals
    config = TradingConfig(symbols=["DEMO"], mode="paper", initial_capital=10_000.0)
    composer = StrategyComposer(
        [
            ScalpingStrategy(config),
            DayTradingStrategy(config),
            MeanReversionStrategy(config),
            MomentumStrategy(config),
        ],
        config,
    )
    signal = composer.generate_signal("DEMO", feat_df)
    logger.info("Composite signal: %s  confidence=%.2f  price=%.2f",
                signal.action, signal.confidence, signal.price)

    # Backtest
    engine = BacktestEngine(config, fe)
    result = engine.run(
        df,
        lambda sym, d: composer.generate_signal(sym, d),
        symbol="DEMO",
    )
    logger.info("Backtest result: %s", result.to_dict())

    # Risk management demo
    risk = RiskManager(config)
    size_kelly = risk.kelly_size(
        win_rate=0.55,
        win_loss_ratio=1.5,
        portfolio_value=10_000.0,
        price=float(df["Close"].iloc[-1]),
    )
    size_ff = risk.fixed_fractional_size(
        portfolio_value=10_000.0,
        price=float(df["Close"].iloc[-1]),
    )
    logger.info("Kelly position size: %d  Fixed-fractional: %d", size_kelly, size_ff)

    monitor = PerformanceMonitor(config)
    monitor.print_summary(result)


if __name__ == "__main__":
    _demo()
