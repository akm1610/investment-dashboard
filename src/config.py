"""
src/config.py
-------------
Central configuration for the ML engine, model hyper-parameters, ensemble
weights, signal thresholds, and training defaults.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------

LGBOOST_PARAMS: dict = {
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.05,
    "num_boost_round": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
}

XGBOOST_PARAMS: dict = {
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
    "eval_metric": "logloss",
    "verbosity": 0,
    "n_jobs": -1,
}

LSTM_PARAMS: dict = {
    "lookback": 60,
    "units_1": 64,
    "units_2": 32,
    "dense_units": 16,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Ensemble weights  (must sum to 1.0 within each horizon)
# ---------------------------------------------------------------------------

ENSEMBLE_WEIGHTS: dict = {
    "short_term": {"lgb": 0.35, "xgb": 0.35, "lstm": 0.20, "rf": 0.10},
    "long_term":  {"lgb": 0.30, "xgb": 0.30, "lstm": 0.25, "rf": 0.15},
}

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------

SIGNAL_THRESHOLDS: dict = {
    "short_term": {"buy": 0.65, "sell": 0.35},
    "long_term":  {"buy": 0.60, "sell": 0.40},
}

# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

SHORT_TERM_RETURN_THRESHOLD: float = 0.01   # 1 % in 1-5 days
LONG_TERM_RETURN_THRESHOLD: float = 0.15    # 15 % in 12 months
SHORT_TERM_FORWARD_DAYS: int = 5
LONG_TERM_FORWARD_DAYS: int = 252           # ~12 months of trading days

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

TRAINING_START_DATE: str = "2019-01-01"
TRAINING_END_DATE: str = "2024-12-31"
TRAIN_TEST_SPLIT_RATIO: float = 0.80

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

TECHNICAL_LOOKBACK_DAYS: int = 365

# Rolling window sizes used by technical indicators
RSI_PERIOD_LONG: int = 14
RSI_PERIOD_SHORT: int = 7
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BBANDS_PERIOD: int = 20
BBANDS_STD: float = 2.0
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
STOCH_K_PERIOD: int = 14
STOCH_D_PERIOD: int = 3
ROC_PERIOD: int = 10
HIST_VOL_PERIOD: int = 20

SMA_PERIODS: list = [20, 50, 200]
EMA_PERIODS: list = [20, 50]

VOLUME_MA_PERIOD: int = 20
OBV_SIGNAL_PERIOD: int = 10

TOP_FEATURES_COUNT: int = 60   # keep top N features after selection

# ---------------------------------------------------------------------------
# Caching / data collection
# ---------------------------------------------------------------------------

NEWS_LOOKBACK_DAYS: int = 30
CACHE_TTL_SECONDS: int = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Signal-strength labels
# ---------------------------------------------------------------------------

SIGNAL_STRENGTH_THRESHOLDS: dict = {
    "STRONG":   80.0,
    "MODERATE": 60.0,
    # < 60 → WEAK
}
