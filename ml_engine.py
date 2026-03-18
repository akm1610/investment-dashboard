"""
ml_engine.py
------------
Re-exports all public symbols from ``src/ml_engine`` so the module can be
imported either way::

    import ml_engine                          # root module
    from src.ml_engine import RecommendationEngine  # package import

See ``src/ml_engine.py`` for full documentation.
"""

from src.ml_engine import (  # noqa: F401  (re-export)
    DataCollector,
    FeatureEngineer,
    MLModelTrainer,
    ModelPerformanceTracker,
    RecommendationEngine,
    generate_labels,
    temporal_train_test_split,
)

__all__ = [
    "DataCollector",
    "FeatureEngineer",
    "MLModelTrainer",
    "ModelPerformanceTracker",
    "RecommendationEngine",
    "generate_labels",
    "temporal_train_test_split",
]
