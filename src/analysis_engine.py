"""
src/analysis_engine.py
----------------------
Re-exports all public symbols from the top-level ``analysis_engine`` module so
the package can be imported as either::

    import analysis_engine as ae          # root module (used by tests)
    from src.analysis_engine import analyze  # package import

See the top-level ``analysis_engine.py`` for full documentation.
"""

from analysis_engine import (  # noqa: F401  (re-export)
    PILLAR_WEIGHTS,
    _row,
    _safe_div,
    analyze,
    compute_ratios,
    compute_scores,
    pretrade_checklist,
)

__all__ = [
    "PILLAR_WEIGHTS",
    "analyze",
    "compute_ratios",
    "compute_scores",
    "pretrade_checklist",
]
