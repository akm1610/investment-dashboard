"""
scripts/train_ml_models.py
---------------------------
Fetch real historical stock data via yfinance, engineer features, and train
the four ML models (LightGBM, XGBoost, Random Forest, LSTM) for both
short-term and long-term horizons.  Trained models are saved to ``models/``.

Usage
-----
Run from the repository root::

    python scripts/train_ml_models.py

Optional flags::

    --tickers AAPL NVDA MSFT AMZN GOOGL   # override default ticker list
    --output  models/                      # override output directory
    --horizon long_term                    # "short_term", "long_term", or "both"
    --skip-lstm                            # skip LSTM (requires TensorFlow)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – allow running from repo root *or* from scripts/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_engine import (  # noqa: E402  (after path setup)
    DataCollector,
    FeatureEngineer,
    MLModelTrainer,
    _generate_labels,
    _temporal_train_test_split,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: List[str] = [
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "BRK-B", "JNJ",
]

# Label generation settings
FORWARD_DAYS_SHORT = 5    # ~1 trading week
FORWARD_DAYS_LONG  = 63   # ~3 trading months
RETURN_THRESHOLD   = 0.02  # 2 % forward return to label as BUY (1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_dataset(
    tickers: List[str],
    forward_days: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch price history for every ticker, compute technical features, and
    attach binary labels.  Returns ``(X, y)`` ready for training.
    """
    collector = DataCollector()
    engineer  = FeatureEngineer()

    frames_X: List[pd.DataFrame] = []
    frames_y: List[pd.Series]    = []

    for ticker in tickers:
        log.info("Fetching data for %s …", ticker)
        try:
            data = collector.fetch_all_data(ticker)
        except Exception as exc:
            log.warning("Skipping %s – data fetch failed: %s", ticker, exc)
            continue

        price_df = data.get("price_history")
        if price_df is None or len(price_df) < 200:
            log.warning("Skipping %s – insufficient price history (%d rows)",
                        ticker, len(price_df) if price_df is not None else 0)
            continue

        # Use all available history for training (not just the tail)
        try:
            feat_df = engineer.extract_technical_features(price_df, days=len(price_df))
        except Exception as exc:
            log.warning("Skipping %s – feature extraction failed: %s", ticker, exc)
            continue

        labels = _generate_labels(price_df["Close"], forward_days, RETURN_THRESHOLD)

        # Align features and labels on the shared index
        common_idx = feat_df.index.intersection(labels.index)
        if len(common_idx) < 100:
            log.warning("Skipping %s – too few aligned rows (%d)", ticker, len(common_idx))
            continue

        frames_X.append(feat_df.loc[common_idx])
        frames_y.append(labels.loc[common_idx])
        log.info("  %s: %d training rows", ticker, len(common_idx))

    if not frames_X:
        raise RuntimeError(
            "No usable training data collected.  "
            "Check your internet connection and ticker list."
        )

    X = pd.concat(frames_X, axis=0).reset_index(drop=True)
    y = pd.concat(frames_y, axis=0).reset_index(drop=True)

    log.info("Total dataset: %d rows, %d features, %.1f%% positive labels",
             len(X), X.shape[1], 100.0 * y.mean())
    return X, y


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_horizon(
    tickers: List[str],
    horizon: str,
    forward_days: int,
    output_dir: str,
    skip_lstm: bool,
) -> None:
    log.info("=" * 60)
    log.info("Training horizon: %s  (forward_days=%d)", horizon, forward_days)
    log.info("=" * 60)

    X, y = _build_dataset(tickers, forward_days)
    engineer = FeatureEngineer()

    # Preprocess (impute, scale, select features)
    try:
        X_proc = engineer.preprocess_features(X)
    except Exception as exc:
        log.warning("preprocess_features failed (%s); using raw features.", exc)
        X_proc = X.fillna(0.0)

    X_train, X_test, y_train, y_test = _temporal_train_test_split(X_proc, y)
    log.info("Train rows: %d  |  Test rows: %d", len(X_train), len(X_test))

    trainer = MLModelTrainer(horizon=horizon)

    # --- LightGBM ---
    log.info("Training LightGBM …")
    try:
        trainer.train_lgb_model(X_train, y_train)
    except Exception as exc:
        log.warning("LightGBM training failed: %s", exc)

    # --- XGBoost ---
    log.info("Training XGBoost …")
    try:
        trainer.train_xgb_model(X_train, y_train)
    except Exception as exc:
        log.warning("XGBoost training failed: %s", exc)

    # --- Random Forest ---
    log.info("Training Random Forest …")
    try:
        trainer.train_rf_model(X_train, y_train)
    except Exception as exc:
        log.warning("Random Forest training failed: %s", exc)

    # --- LSTM ---
    if skip_lstm:
        log.info("Skipping LSTM (--skip-lstm flag set).")
    else:
        log.info("Training LSTM …")
        try:
            # LSTM expects 3-D input: (samples, timesteps=1, features)
            X_lstm = X_train.values.reshape(len(X_train), 1, X_train.shape[1])
            trainer.train_lstm_model(X_lstm, y_train.values)
        except ImportError:
            log.warning("TensorFlow not installed; LSTM skipped.  "
                        "Install with: pip install tensorflow")
        except Exception as exc:
            log.warning("LSTM training failed: %s", exc)

    # --- Evaluation ---
    if trainer.models:
        log.info("Evaluating models on test set …")
        try:
            results = trainer.evaluate_models(X_test, y_test)
            for name, metrics in results.items():
                acc = metrics.get("accuracy", float("nan"))
                log.info("  %-10s  accuracy=%.3f", name, acc)
        except Exception as exc:
            log.warning("Evaluation failed: %s", exc)

    # --- Save ---
    trainer.save_models(output_dir)
    log.info("Models saved to '%s'", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML models on real yfinance data and save to models/."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        metavar="TICKER",
        help="Space-separated list of ticker symbols (default: %(default)s).",
    )
    parser.add_argument(
        "--output", default=os.path.join(_ROOT, "models"),
        metavar="DIR",
        help="Directory to save trained models (default: %(default)s).",
    )
    parser.add_argument(
        "--horizon", choices=["short_term", "long_term", "both"], default="both",
        help="Which time horizon to train (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training (useful when TensorFlow is not installed).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tickers = [t.upper() for t in args.tickers]
    output_dir = args.output
    skip_lstm = args.skip_lstm

    os.makedirs(output_dir, exist_ok=True)
    log.info("Output directory: %s", output_dir)
    log.info("Tickers: %s", tickers)

    horizons = (
        [("short_term", FORWARD_DAYS_SHORT), ("long_term", FORWARD_DAYS_LONG)]
        if args.horizon == "both"
        else [(args.horizon,
               FORWARD_DAYS_SHORT if args.horizon == "short_term" else FORWARD_DAYS_LONG)]
    )

    for horizon, fwd_days in horizons:
        _train_horizon(tickers, horizon, fwd_days, output_dir, skip_lstm)

    log.info("Training complete.  Run the dashboard with: streamlit run src/app.py")


if __name__ == "__main__":
    main()
