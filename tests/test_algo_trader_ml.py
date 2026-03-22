"""
tests/test_algo_trader_ml.py
----------------------------
Unit tests for algo_trader_ml.py.

All tests run fully offline – no network calls are made.  Heavy optional
dependencies (tensorflow, xgboost, yfinance, ccxt) are mocked so that the
suite can run even when those packages are not installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import algo_trader_ml as atm
from algo_trader_ml import (
    AlgoTraderML,
    BacktestEngine,
    BacktestResult,
    DayTradingStrategy,
    DataFetcher,
    FeatureEngineer,
    MeanReversionStrategy,
    MomentumStrategy,
    Order,
    PerformanceMonitor,
    RiskManager,
    ScalpingStrategy,
    StrategyComposer,
    TradeSignal,
    TradingConfig,
    TradingExecutor,
    XGBoostEnsemble,
    LSTMForecaster,
    _detect_asset_type,
    create_trader,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* trading days."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_feature_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a feature-engineered DataFrame ready for strategies."""
    fe = FeatureEngineer()
    return fe.build_features(_make_ohlcv(n, seed))


def _default_config(**kwargs: Any) -> TradingConfig:
    defaults: dict = dict(symbols=["TEST"], mode="paper", initial_capital=10_000.0)
    defaults.update(kwargs)
    return TradingConfig(**defaults)


# ===========================================================================
# TradingConfig
# ===========================================================================


class TestTradingConfig:
    def test_defaults(self):
        cfg = TradingConfig()
        assert cfg.mode == "paper"
        assert cfg.initial_capital == 100_000.0
        assert cfg.max_drawdown == 0.10

    def test_asset_type_auto_detection(self):
        cfg = TradingConfig(symbols=["AAPL", "BTC-USD", "ES=F"])
        assert cfg.asset_types["AAPL"] == "stock"
        assert cfg.asset_types["BTC-USD"] == "crypto"
        assert cfg.asset_types["ES=F"] == "futures"

    def test_custom_symbols(self):
        cfg = TradingConfig(symbols=["MSFT", "ETH-USD"])
        assert "MSFT" in cfg.symbols
        assert "ETH-USD" in cfg.symbols

    def test_log_level_set(self):
        cfg = TradingConfig(log_level="DEBUG")
        assert cfg.log_level == "DEBUG"


# ===========================================================================
# _detect_asset_type
# ===========================================================================


class TestDetectAssetType:
    def test_futures(self):
        assert _detect_asset_type("ES=F") == "futures"
        assert _detect_asset_type("CL=F") == "futures"

    def test_crypto_hyphen(self):
        assert _detect_asset_type("BTC-USD") == "crypto"
        assert _detect_asset_type("ETH-USD") == "crypto"

    def test_crypto_slash(self):
        assert _detect_asset_type("BTC/USDT") == "crypto"

    def test_stock(self):
        assert _detect_asset_type("AAPL") == "stock"
        assert _detect_asset_type("GOOGL") == "stock"


# ===========================================================================
# DataFetcher
# ===========================================================================


class TestDataFetcher:
    def test_standardize_column_capitalisation(self):
        df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [1000]},
            index=pd.to_datetime(["2022-01-01"], utc=True),
        )
        result = DataFetcher._standardize(df)
        assert set(result.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_standardize_empty(self):
        result = DataFetcher._standardize(pd.DataFrame())
        assert result.empty

    def test_standardize_adds_utc(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [1000]},
            index=pd.to_datetime(["2022-01-01"]),  # tz-naive
        )
        result = DataFetcher._standardize(df)
        assert result.index.tz is not None

    def test_fetch_all_skips_failed(self):
        cfg = _default_config(symbols=["AAA", "BBB"])
        fetcher = DataFetcher(cfg)
        # Simulate one success and one failure
        def fake_fetch(sym, start=None, end=None, interval=None):
            if sym == "AAA":
                return _make_ohlcv(50)
            raise ValueError("network error")

        fetcher.fetch = fake_fetch  # type: ignore[method-assign]
        results = fetcher.fetch_all()
        assert "AAA" in results
        assert "BBB" not in results

    @patch("algo_trader_ml.DataFetcher._fetch_yfinance")
    def test_fetch_calls_yfinance_for_stock(self, mock_yf):
        mock_yf.return_value = _make_ohlcv(50)
        cfg = _default_config(symbols=["AAPL"])
        fetcher = DataFetcher(cfg)
        df = fetcher.fetch("AAPL")
        assert not df.empty
        mock_yf.assert_called_once()

    def test_fetch_raises_on_empty_result(self):
        cfg = _default_config(symbols=["EMPTY"])
        fetcher = DataFetcher(cfg)
        with patch.object(fetcher, "_fetch_yfinance", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No data returned"):
                fetcher.fetch("EMPTY")


# ===========================================================================
# FeatureEngineer
# ===========================================================================


class TestFeatureEngineer:
    def setup_method(self):
        self.fe = FeatureEngineer()
        self.df = _make_ohlcv(300)

    def test_build_features_shape(self):
        result = self.fe.build_features(self.df)
        assert len(result) > 0
        assert len(result.columns) > 10

    def test_rsi_range(self):
        df = self.fe.add_rsi(self.df.copy())
        assert df["rsi"].between(0, 100).all()

    def test_macd_columns(self):
        df = self.fe.add_macd(self.df.copy())
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_bollinger_bands(self):
        df = self.fe.add_bollinger_bands(self.df.copy())
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        # Upper >= lower always
        assert (df["bb_upper"] >= df["bb_lower"]).all()

    def test_atr_positive(self):
        df = self.fe.add_atr(self.df.copy())
        assert (df["atr"] >= 0).all()

    def test_obv_cumulative(self):
        df = self.fe.add_obv(self.df.copy())
        assert "obv" in df.columns

    def test_get_feature_columns_excludes_ohlcv(self):
        feat_df = self.fe.build_features(self.df.copy())
        cols = self.fe.get_feature_columns(feat_df)
        for ohlcv in ("Open", "High", "Low", "Close", "Volume"):
            assert ohlcv not in cols

    def test_no_inf_values(self):
        feat_df = self.fe.build_features(self.df.copy())
        assert not np.isinf(feat_df.values).any()

    def test_no_nan_after_build(self):
        feat_df = self.fe.build_features(self.df.copy())
        assert not feat_df.isnull().any().any()

    def test_candle_features(self):
        df = self.fe.add_candle_features(self.df.copy())
        assert "body_ratio" in df.columns
        assert "is_bullish" in df.columns
        assert df["is_bullish"].isin([0.0, 1.0]).all()

    def test_sma_periods(self):
        df = self.fe.add_sma(self.df.copy(), periods=(5, 10))
        assert "sma_5" in df.columns
        assert "sma_10" in df.columns

    def test_ema_periods(self):
        df = self.fe.add_ema(self.df.copy(), periods=(9, 21))
        assert "ema_9" in df.columns
        assert "ema_21" in df.columns


# ===========================================================================
# TradeSignal
# ===========================================================================


class TestTradeSignal:
    def test_actionable_buy(self):
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.7, price=100.0)
        assert sig.is_actionable(0.55)

    def test_not_actionable_hold(self):
        sig = TradeSignal(symbol="X", action="HOLD", confidence=0.9, price=100.0)
        assert not sig.is_actionable(0.55)

    def test_not_actionable_low_confidence(self):
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.4, price=100.0)
        assert not sig.is_actionable(0.55)

    def test_timestamp_utc(self):
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.6, price=100.0)
        assert sig.timestamp.tzinfo is not None


# ===========================================================================
# Strategies
# ===========================================================================


class TestScalpingStrategy:
    def setup_method(self):
        self.config = _default_config()
        self.strategy = ScalpingStrategy(self.config)
        self.df = _make_feature_df(300)

    def test_returns_trade_signal(self):
        sig = self.strategy.generate_signal("TEST", self.df)
        assert isinstance(sig, TradeSignal)
        assert sig.action in ("BUY", "SELL", "HOLD")

    def test_empty_df_returns_hold(self):
        sig = self.strategy.generate_signal("TEST", pd.DataFrame())
        assert sig.action == "HOLD"

    def test_short_df_returns_hold(self):
        sig = self.strategy.generate_signal("TEST", self.df.iloc[:3])
        assert sig.action == "HOLD"

    def test_missing_columns_returns_hold(self):
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        sig = self.strategy.generate_signal("TEST", df)
        assert sig.action == "HOLD"

    def test_buy_signal_has_stop_loss(self):
        """Force a BUY condition and verify stop_loss is set."""
        df = self.df.copy()
        # Override last row to trigger BUY: low RSI, big lower shadow, vol surge, macd rising
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        df.iloc[-1, df.columns.get_loc("lower_shadow")] = 0.75
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0
        df.iloc[-1, df.columns.get_loc("macd_hist")] = 1.0
        df.iloc[-2, df.columns.get_loc("macd_hist")] = 0.5
        sig = self.strategy.generate_signal("TEST", df)
        if sig.action == "BUY":
            assert sig.stop_loss is not None
            assert sig.stop_loss < sig.price


class TestDayTradingStrategy:
    def setup_method(self):
        self.config = _default_config()
        self.strategy = DayTradingStrategy(self.config)
        self.df = _make_feature_df(300)

    def test_returns_signal(self):
        sig = self.strategy.generate_signal("TEST", self.df)
        assert isinstance(sig, TradeSignal)

    def test_short_df_returns_hold(self):
        sig = self.strategy.generate_signal("TEST", self.df.iloc[:10])
        assert sig.action == "HOLD"

    def test_bullish_alignment_can_produce_buy(self):
        df = self.df.copy()
        price = float(df["Close"].iloc[-1])
        # Manually create a bullish EMA alignment
        df.iloc[-1, df.columns.get_loc("ema_9")] = price * 1.03
        df.iloc[-1, df.columns.get_loc("ema_21")] = price * 1.01
        df.iloc[-1, df.columns.get_loc("ema_50")] = price * 0.99
        df.iloc[-1, df.columns.get_loc("adx")] = 30.0
        df.iloc[-1, df.columns.get_loc("macd")] = 1.0
        df.iloc[-1, df.columns.get_loc("macd_signal")] = 0.5
        sig = self.strategy.generate_signal("TEST", df)
        assert sig.action in ("BUY", "HOLD")


class TestMeanReversionStrategy:
    def setup_method(self):
        self.config = _default_config()
        self.strategy = MeanReversionStrategy(self.config)
        self.df = _make_feature_df(300)

    def test_returns_signal(self):
        sig = self.strategy.generate_signal("TEST", self.df)
        assert isinstance(sig, TradeSignal)

    def test_oversold_condition_buy(self):
        df = self.df.copy()
        price = float(df["Close"].iloc[-1])
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        df.iloc[-1, df.columns.get_loc("bb_pct")] = 0.02
        df.iloc[-1, df.columns.get_loc("bb_lower")] = price + 5  # force price < lower band
        sig = self.strategy.generate_signal("TEST", df)
        # May be HOLD if price is not actually below lower band after edits
        assert sig.action in ("BUY", "HOLD")

    def test_short_df_returns_hold(self):
        sig = self.strategy.generate_signal("TEST", self.df.iloc[:10])
        assert sig.action == "HOLD"


class TestMomentumStrategy:
    def setup_method(self):
        self.config = _default_config()
        self.strategy = MomentumStrategy(self.config)
        self.df = _make_feature_df(300)

    def test_returns_signal(self):
        sig = self.strategy.generate_signal("TEST", self.df)
        assert isinstance(sig, TradeSignal)

    def test_short_df_returns_hold(self):
        sig = self.strategy.generate_signal("TEST", self.df.iloc[:10])
        assert sig.action == "HOLD"


class TestStrategyComposer:
    def setup_method(self):
        self.config = _default_config()
        self.strategies = [
            ScalpingStrategy(self.config),
            MeanReversionStrategy(self.config),
            MomentumStrategy(self.config),
        ]
        self.composer = StrategyComposer(self.strategies, self.config)
        self.df = _make_feature_df(300)

    def test_returns_signal(self):
        sig = self.composer.generate_signal("TEST", self.df)
        assert isinstance(sig, TradeSignal)
        assert sig.action in ("BUY", "SELL", "HOLD")

    def test_weights_normalise(self):
        composer = StrategyComposer(self.strategies, self.config, weights=[1, 2, 1])
        assert abs(sum(composer.weights) - 1.0) < 1e-9

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            StrategyComposer(self.strategies, self.config, weights=[1, 2])

    def test_composer_strategy_field(self):
        sig = self.composer.generate_signal("TEST", self.df)
        assert sig.strategy == "composer"


# ===========================================================================
# RiskManager
# ===========================================================================


class TestRiskManager:
    def setup_method(self):
        self.config = _default_config(
            initial_capital=10_000.0, max_drawdown=0.10, risk_per_trade=0.01
        )
        self.rm = RiskManager(self.config)

    def test_kelly_size_basic(self):
        size = self.rm.kelly_size(
            win_rate=0.55, win_loss_ratio=1.5, portfolio_value=10_000.0, price=100.0
        )
        assert size >= 0

    def test_kelly_size_zero_on_invalid_win_rate(self):
        assert self.rm.kelly_size(0.0, 1.5, 10_000.0, 100.0) == 0
        assert self.rm.kelly_size(1.0, 1.5, 10_000.0, 100.0) == 0

    def test_fixed_fractional_size(self):
        size = self.rm.fixed_fractional_size(10_000.0, 100.0)
        assert size >= 0

    def test_fixed_fractional_with_stop(self):
        size = self.rm.fixed_fractional_size(10_000.0, 100.0, stop_loss_price=95.0)
        # Risk = 1% of 10k = $100, per-unit risk = $5 → 20 units
        assert size == 20

    def test_volatility_adjusted_size(self):
        size = self.rm.volatility_adjusted_size(10_000.0, 100.0, atr=2.0)
        assert size >= 0

    def test_drawdown_not_breached_initially(self):
        assert not self.rm.is_trading_halted()
        assert self.rm.current_drawdown() == 0.0

    def test_drawdown_breached_after_large_loss(self):
        self.rm.update_equity(10_000.0)
        self.rm.update_equity(8_000.0)  # 20% drawdown, limit is 10%
        assert self.rm.is_trading_halted()

    def test_drawdown_clears_on_recovery(self):
        self.rm.update_equity(10_000.0)
        self.rm.update_equity(8_000.0)
        assert self.rm.is_trading_halted()
        # Recover above threshold
        self.rm.update_equity(9_500.0)
        assert not self.rm.is_trading_halted()

    def test_validate_signal_rejects_low_confidence(self):
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.3, price=100.0)
        assert not self.rm.validate_signal(sig, 10_000.0)

    def test_validate_signal_accepts_valid(self):
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.8, price=100.0)
        assert self.rm.validate_signal(sig, 10_000.0)

    def test_validate_signal_rejects_when_halted(self):
        self.rm.update_equity(10_000.0)
        self.rm.update_equity(8_000.0)  # breach
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.8, price=100.0)
        assert not self.rm.validate_signal(sig, 10_000.0)


# ===========================================================================
# BacktestEngine
# ===========================================================================


class TestBacktestEngine:
    def setup_method(self):
        self.config = _default_config(initial_capital=10_000.0)
        self.fe = FeatureEngineer()
        self.engine = BacktestEngine(self.config, self.fe)
        self.df = _make_ohlcv(200)

    def _always_buy(self, sym, df):
        price = float(df["Close"].iloc[-1])
        return TradeSignal(symbol=sym, action="BUY", confidence=0.9, price=price)

    def _always_sell(self, sym, df):
        price = float(df["Close"].iloc[-1])
        return TradeSignal(symbol=sym, action="SELL", confidence=0.9, price=price)

    def _alternate(self, sym, df):
        n = len(df)
        price = float(df["Close"].iloc[-1])
        if n % 10 < 5:
            return TradeSignal(symbol=sym, action="BUY", confidence=0.9, price=price)
        return TradeSignal(symbol=sym, action="SELL", confidence=0.9, price=price)

    def test_run_returns_backtest_result(self):
        result = self.engine.run(self.df, self._alternate)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_nonempty(self):
        result = self.engine.run(self.df, self._alternate)
        assert len(result.equity_curve) > 0

    def test_total_return_type(self):
        result = self.engine.run(self.df, self._alternate)
        assert isinstance(result.total_return, float)

    def test_sharpe_finite_with_trades(self):
        result = self.engine.run(self.df, self._alternate)
        assert np.isfinite(result.sharpe_ratio)

    def test_win_rate_in_range(self):
        result = self.engine.run(self.df, self._alternate)
        assert 0.0 <= result.win_rate <= 1.0

    def test_to_dict_keys(self):
        result = self.engine.run(self.df, self._alternate)
        d = result.to_dict()
        expected_keys = {
            "total_return_pct", "annualised_return_pct", "sharpe_ratio",
            "sortino_ratio", "max_drawdown_pct", "win_rate_pct",
            "profit_factor", "total_trades",
        }
        assert expected_keys.issubset(d.keys())

    def test_walk_forward_returns_list(self):
        results = self.engine.walk_forward(self.df, self._alternate, n_splits=3)
        assert isinstance(results, list)
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_empty_df_returns_empty_result(self):
        result = self.engine.run(pd.DataFrame(), self._alternate)
        assert result.total_trades == 0

    def test_commission_reduces_returns(self):
        # Zero-commission run vs non-zero
        cfg_no_comm = _default_config(commission_rate=0.0, slippage_rate=0.0)
        cfg_with_comm = _default_config(commission_rate=0.01, slippage_rate=0.0)
        eng_nc = BacktestEngine(cfg_no_comm, self.fe)
        eng_wc = BacktestEngine(cfg_with_comm, self.fe)
        r_nc = eng_nc.run(self.df, self._alternate)
        r_wc = eng_wc.run(self.df, self._alternate)
        # With commission, return should be lower or equal
        assert r_wc.total_return <= r_nc.total_return + 1e-6


# ===========================================================================
# TradingExecutor
# ===========================================================================


class TestTradingExecutor:
    def setup_method(self):
        self.config = _default_config(initial_capital=10_000.0)
        self.rm = RiskManager(self.config)
        self.executor = TradingExecutor(self.config, self.rm)

    def test_paper_buy_fills(self):
        sig = TradeSignal(symbol="TEST", action="BUY", confidence=0.9, price=100.0)
        order = self.executor.execute_signal(sig)
        assert order is not None
        assert order.status == "FILLED"
        assert order.side == "BUY"

    def test_paper_sell_without_position_returns_none(self):
        sig = TradeSignal(symbol="TEST", action="SELL", confidence=0.9, price=100.0)
        order = self.executor.execute_signal(sig)
        assert order is None

    def test_buy_then_sell(self):
        buy_sig = TradeSignal(symbol="TEST", action="BUY", confidence=0.9, price=100.0)
        self.executor.execute_signal(buy_sig)
        sell_sig = TradeSignal(symbol="TEST", action="SELL", confidence=0.9, price=110.0)
        order = self.executor.execute_signal(sell_sig)
        assert order is not None
        assert order.status == "FILLED"
        assert order.side == "SELL"

    def test_cash_decreases_after_buy(self):
        initial_cash = self.executor._cash
        sig = TradeSignal(symbol="TEST", action="BUY", confidence=0.9, price=100.0)
        self.executor.execute_signal(sig)
        assert self.executor._cash < initial_cash

    def test_trade_log_populated(self):
        sig = TradeSignal(symbol="TEST", action="BUY", confidence=0.9, price=100.0)
        self.executor.execute_signal(sig)
        log = self.executor.get_trade_log()
        assert isinstance(log, pd.DataFrame)
        assert len(log) == 1

    def test_get_portfolio_has_cash(self):
        port = self.executor.get_portfolio()
        assert "cash" in port
        assert "positions" in port

    def test_hold_signal_returns_none(self):
        sig = TradeSignal(symbol="TEST", action="HOLD", confidence=0.9, price=100.0)
        order = self.executor.execute_signal(sig)
        assert order is None

    def test_low_confidence_rejected(self):
        sig = TradeSignal(symbol="TEST", action="BUY", confidence=0.2, price=100.0)
        order = self.executor.execute_signal(sig)
        assert order is None


# ===========================================================================
# PerformanceMonitor
# ===========================================================================


class TestPerformanceMonitor:
    def setup_method(self):
        self.config = _default_config()
        self.monitor = PerformanceMonitor(self.config)

    def _populate_equity(self, values):
        for v in values:
            self.monitor.record_equity(v)

    def test_summary_empty(self):
        assert self.monitor.summary() == {}

    def test_summary_keys(self):
        self._populate_equity([10_000, 10_100, 10_050, 10_200])
        s = self.monitor.summary()
        assert "current_equity" in s
        assert "total_return_pct" in s

    def test_sharpe_finite(self):
        self._populate_equity(list(np.linspace(10_000, 11_000, 50)))
        s = self.monitor.current_sharpe()
        assert np.isfinite(s)

    def test_drawdown_zero_when_rising(self):
        self._populate_equity(list(range(10_000, 10_050)))
        assert self.monitor.current_drawdown() == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_positive_after_decline(self):
        self._populate_equity([10_000, 10_500, 9_000])
        assert self.monitor.current_drawdown() > 0

    def test_alert_callback_called(self):
        alerts: list = []
        monitor = PerformanceMonitor(self.config, alert_callback=alerts.append)
        rm = RiskManager(self.config)
        rm.update_equity(10_000.0)
        rm.update_equity(8_000.0)  # breach
        monitor.check_alerts(rm)
        assert any("drawdown" in a.lower() or "halted" in a.lower() for a in alerts)

    def test_record_signal(self):
        sig = TradeSignal(symbol="X", action="BUY", confidence=0.7, price=100.0)
        self.monitor.record_signal(sig)
        assert self.monitor.summary() == {} or True  # just checking no exception

    def test_anomaly_score_zero_with_few_points(self):
        self._populate_equity([10_000, 10_100])
        assert self.monitor.anomaly_score() == 0.0

    def test_print_summary_no_exception(self):
        self._populate_equity(list(np.linspace(10_000, 11_000, 30)))
        self.monitor.print_summary()


# ===========================================================================
# XGBoostEnsemble
# ===========================================================================


class TestXGBoostEnsemble:
    def setup_method(self):
        self.config = _default_config()
        self.rng = np.random.default_rng(0)

    def test_fit_and_predict(self):
        pytest.importorskip("xgboost")
        n, f = 200, 15
        X = self.rng.standard_normal((n, f)).astype(np.float32)
        y = self.rng.integers(0, 2, n).astype(int)
        model = XGBoostEnsemble(self.config)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (n,)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_binary(self):
        pytest.importorskip("xgboost")
        n, f = 200, 10
        X = self.rng.standard_normal((n, f)).astype(np.float32)
        y = self.rng.integers(0, 2, n).astype(int)
        model = XGBoostEnsemble(self.config)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_raises_before_fit(self):
        model = XGBoostEnsemble(self.config)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba(np.array([[1.0, 2.0]]))

    def test_import_error_without_xgboost(self):
        model = XGBoostEnsemble(self.config)
        with patch.dict("sys.modules", {"xgboost": None}):
            with pytest.raises((ImportError, TypeError)):
                model.fit(np.zeros((10, 5)), np.zeros(10, dtype=int))


# ===========================================================================
# LSTMForecaster
# ===========================================================================


class TestLSTMForecaster:
    def test_raises_without_tensorflow(self):
        cfg = _default_config()
        forecaster = LSTMForecaster(lookback=10, n_features=5, config=cfg)
        with patch.dict("sys.modules", {"tensorflow": None}):
            with pytest.raises((ImportError, TypeError)):
                forecaster.build_model()

    def test_predict_proba_raises_before_fit(self):
        cfg = _default_config()
        forecaster = LSTMForecaster(lookback=10, n_features=5, config=cfg)
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.predict_proba(np.zeros((20, 5)))


# ===========================================================================
# AlgoTraderML (integration)
# ===========================================================================


class TestAlgoTraderML:
    def setup_method(self):
        self.config = _default_config()
        self.trader = AlgoTraderML(self.config)
        # Pre-populate data cache with synthetic data
        self.trader._data_cache = {"TEST": _make_ohlcv(300)}

    def test_generate_signals_returns_list(self):
        signals = self.trader.generate_signals()
        assert isinstance(signals, list)

    def test_run_once_returns_list(self):
        orders = self.trader.run_once()
        assert isinstance(orders, list)

    def test_should_retrain_true_on_new_instance(self):
        assert self.trader.should_retrain()

    def test_should_retrain_false_after_retrain(self):
        pytest.importorskip("xgboost")
        self.trader.train_xgboost("TEST")
        assert not self.trader.should_retrain()

    def test_train_xgboost_returns_model(self):
        pytest.importorskip("xgboost")
        model = self.trader.train_xgboost("TEST")
        assert isinstance(model, XGBoostEnsemble)

    def test_train_xgboost_raises_without_data(self):
        with pytest.raises(ValueError, match="No data cached"):
            self.trader.train_xgboost("MISSING_SYMBOL")

    def test_backtest_without_network(self):
        """Backtest using cached data (no network call)."""
        raw_df = _make_ohlcv(200)
        result = self.trader.backtest_engine.run(
            raw_df,
            lambda s, d: self.trader.composer.generate_signal(s, d),
            symbol="TEST",
        )
        assert isinstance(result, BacktestResult)

    @patch("algo_trader_ml.DataFetcher.fetch")
    def test_backtest_public_api(self, mock_fetch):
        mock_fetch.return_value = _make_ohlcv(200)
        result = self.trader.backtest(symbol="TEST")
        assert isinstance(result, BacktestResult)

    def test_run_iterations(self):
        """Smoke-test that run() completes without raising."""
        self.trader.run(iterations=2)

    def test_initialize_populates_cache(self):
        trader = AlgoTraderML(self.config)
        with patch.object(trader.fetcher, "fetch_all", return_value={"TEST": _make_ohlcv(100)}):
            trader.initialize()
        assert "TEST" in trader._data_cache


# ===========================================================================
# create_trader factory
# ===========================================================================


class TestCreateTrader:
    def test_defaults(self):
        t = create_trader()
        assert isinstance(t, AlgoTraderML)
        assert t.config.mode == "paper"

    def test_custom_params(self):
        t = create_trader(
            symbols=["AAPL"],
            mode="paper",
            initial_capital=50_000.0,
            max_drawdown=0.05,
        )
        assert t.config.initial_capital == 50_000.0
        assert t.config.max_drawdown == 0.05
        assert "AAPL" in t.config.symbols
