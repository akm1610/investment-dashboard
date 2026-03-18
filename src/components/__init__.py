"""
src/components
--------------
UI component modules for the Investment Dashboard Streamlit app.

Each sub-module corresponds to a full page or a shared helper:

* utils              – formatting helpers and reusable Plotly/Streamlit widgets
* company_analysis   – Page 1: deep-dive into a single company
* portfolio_overview – Page 2: portfolio health, allocation and rebalancing
* pretrade_checklist – Page 3: 7-item decision gate before adding a position
* investment_journal – Page 4: timestamped thesis records
* sidebar            – shared sidebar navigation and settings
"""

from .utils import (
    display_score_gauge,
    display_checklist,
    format_currency,
    get_color_for_value,
    score_color,
    status_emoji,
)

__all__ = [
    "display_score_gauge",
    "display_checklist",
    "format_currency",
    "get_color_for_value",
    "score_color",
    "status_emoji",
]
