#!/usr/bin/env python3
"""
scripts/retrain.py
------------------
Automated ML model retraining pipeline for the investment dashboard.

Usage
-----
    # Retrain on default tickers with 3 years of historical data
    python scripts/retrain.py

    # Custom tickers and period
    python scripts/retrain.py --tickers AAPL MSFT GOOGL NVDA TSLA --period 5y

    # Specify output directory for saved models
    python scripts/retrain.py --model-dir models/production

    # Dry-run: validate data fetch but skip training
    python scripts/retrain.py --dry-run

Environment Variables
---------------------
  MODEL_DIR   Directory to save trained models (default: models/)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
for _path in [_REPO_ROOT, _SRC]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.data_fetcher import DataFetcher
from src.ml_engine import DataCollector, FeatureEngineer, MLModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
    "META", "TSLA", "INTC", "AMD", "JPM",
    "V", "MA", "UNH", "JNJ", "PG",
]
DEFAULT_PERIOD = "3y"
DEFAULT_MODEL_DIR = os.path.join(_REPO_ROOT, "models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_training_data(
    tickers: list[str],
    period: str,
) -> tuple[list, list]:
    """Fetch price and fundamental data for all tickers.

    Returns (features_list, labels_list) ready for training.
    Tickers that fail to fetch are skipped with a warning.
    """
    feat_eng = FeatureEngineer()
    all_features = []
    ticker_status: dict[str, str] = {}

    for ticker in tickers:
        try:
            fetcher = DataFetcher(ticker)
            price_data = fetcher.fetch_stock_data(period=period)
            fundamentals = fetcher.fetch_all_fundamentals(ticker)
            features = feat_eng.extract_features(price_data, fundamentals)
            if features is not None and not features.empty:
                all_features.append((ticker, features))
                ticker_status[ticker] = "✓"
            else:
                ticker_status[ticker] = "⚠ empty"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch data for %s: %s", ticker, exc)
            ticker_status[ticker] = f"✗ {exc}"

    logger.info("Data fetch summary:")
    for ticker, status in ticker_status.items():
        logger.info("  %s: %s", ticker, status)

    return all_features, ticker_status


def _train_models(
    all_features: list,
    model_dir: str,
    dry_run: bool,
) -> None:
    """Train models on the collected features and save to *model_dir*."""
    if not all_features:
        logger.error("No feature data collected – aborting training")
        sys.exit(1)

    logger.info("Collected features from %d tickers", len(all_features))

    if dry_run:
        logger.info("Dry-run mode: skipping model training and saving")
        return

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "short_term"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "long_term"), exist_ok=True)

    for horizon in ("short_term", "long_term"):
        logger.info("Training %s models …", horizon)
        trainer = MLModelTrainer(horizon=horizon)
        trained_count = 0

        for ticker, features in all_features:
            try:
                trainer.train_rf_model(features)
                trained_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipped %s (%s) during training: %s", ticker, horizon, exc)

        if trained_count == 0:
            logger.error("No models trained for horizon %s", horizon)
            continue

        horizon_dir = os.path.join(model_dir, horizon)
        try:
            trainer.save_models(horizon_dir)
            logger.info("Saved %s models to %s", horizon, horizon_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save %s models: %s", horizon, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain investment dashboard ML models with fresh market data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        metavar="TICKER",
        help="Stock tickers to include in the training set (default: %(default)s)",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="yfinance period for historical price data, e.g. 3y, 5y (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        help="Directory to save trained models (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate data without running model training",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    start = time.monotonic()
    ts = datetime.now(tz=timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("Investment Dashboard – Model Retraining Pipeline")
    logger.info("Started at: %s", ts)
    logger.info("Tickers (%d): %s", len(args.tickers), ", ".join(args.tickers))
    logger.info("Period: %s", args.period)
    logger.info("Model dir: %s", args.model_dir)
    logger.info("Dry run: %s", args.dry_run)
    logger.info("=" * 60)

    all_features, _ = _fetch_training_data(args.tickers, args.period)
    _train_models(all_features, args.model_dir, args.dry_run)

    elapsed = time.monotonic() - start
    logger.info("=" * 60)
    logger.info("Retraining complete in %.1f seconds", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
