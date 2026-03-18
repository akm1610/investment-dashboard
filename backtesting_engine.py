"""
backtesting_engine.py
---------------------
Re-exports all public symbols from ``src/backtesting_engine`` so the module
can be imported either way::

    import backtesting_engine                       # root module
    from src.backtesting_engine import BacktestEngine  # package import

See ``src/backtesting_engine.py`` for full documentation.
"""

from src.backtesting_engine import (  # noqa: F401  (re-export)
    STRATEGIES,
    BacktestEngine,
    BacktestVisualizer,
    BenchmarkAnalyzer,
    PerformanceCalculator,
    StrategyAnalyzer,
    TradeTracker,
    macd_strategy,
    mean_reversion_strategy,
    momentum_strategy,
    rsi_strategy,
    walk_forward_test,
)

__all__ = [
    "STRATEGIES",
    "BacktestEngine",
    "BacktestVisualizer",
    "BenchmarkAnalyzer",
    "PerformanceCalculator",
    "StrategyAnalyzer",
    "TradeTracker",
    "macd_strategy",
    "mean_reversion_strategy",
    "momentum_strategy",
    "rsi_strategy",
    "walk_forward_test",
]
