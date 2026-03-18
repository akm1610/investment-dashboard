"""
src/components/utils.py
-----------------------
Shared helper functions and reusable Streamlit/Plotly widgets used across
all dashboard pages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_currency(value: float, symbol: str = "$") -> str:
    """Format *value* as a currency string.

    Parameters
    ----------
    value:
        Numeric value to format.
    symbol:
        Currency symbol prefix (default ``"$"``).

    Examples
    --------
    >>> format_currency(1_234_567.89)
    '$1,234,567.89'
    >>> format_currency(0.0)
    '$0.00'
    """
    try:
        return f"{symbol}{value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_large_number(value: float) -> str:
    """Abbreviate large numbers to B/M/K notation.

    Examples
    --------
    >>> format_large_number(2_500_000_000)
    '$2.50B'
    >>> format_large_number(750_000)
    '$750.00K'
    """
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.2f}K"
    return f"${v:.2f}"


def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a decimal ratio as a percentage string.

    Examples
    --------
    >>> format_pct(0.1234)
    '12.3%'
    """
    if value is None:
        return "N/A"
    try:
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_ratio(value: Optional[float], decimals: int = 2) -> str:
    """Format a numeric ratio.

    Examples
    --------
    >>> format_ratio(1.5678)
    '1.57'
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def get_color_for_value(
    value: float,
    good_min: float,
    bad_max: float,
    invert: bool = False,
) -> str:
    """Return a CSS colour string based on how *value* compares to thresholds.

    By default (``invert=False``), higher is better:
    * ``value >= good_min``  → ``"green"``
    * ``value <= bad_max``   → ``"red"``
    * otherwise              → ``"orange"``

    Set ``invert=True`` when *lower* values are desirable (e.g. P/E ratio).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "grey"

    if not invert:
        if v >= good_min:
            return "green"
        if v <= bad_max:
            return "red"
        return "orange"
    else:
        if v <= bad_max:
            return "green"
        if v >= good_min:
            return "red"
        return "orange"


def score_color(score: float) -> str:
    """Map a 0-100 score to a colour name.

    * ``score >= 70`` → ``"green"``
    * ``score >= 40`` → ``"orange"``
    * otherwise       → ``"red"``
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "grey"
    if s >= 70:
        return "green"
    if s >= 40:
        return "orange"
    return "red"


def status_emoji(status: str) -> str:
    """Return an emoji for a checklist item status string."""
    return {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(status.lower(), "❓")


# ---------------------------------------------------------------------------
# Reusable Streamlit/Plotly widgets
# ---------------------------------------------------------------------------


def display_score_gauge(score: float, title: str) -> go.Figure:
    """Build and return a Plotly gauge chart for a score between 0 and 100.

    Parameters
    ----------
    score:
        Numeric score to display (clamped to [0, 100]).
    title:
        Chart title displayed above the gauge needle.

    Returns
    -------
    A :class:`plotly.graph_objects.Figure` ready to pass to
    ``st.plotly_chart()``.
    """
    clamped = max(0.0, min(100.0, float(score)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped,
            title={"text": title, "font": {"size": 16}},
            number={"font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "steelblue", "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ccc",
                "steps": [
                    {"range": [0, 40], "color": "#ff4d4d"},
                    {"range": [40, 70], "color": "#ffc947"},
                    {"range": [70, 100], "color": "#4caf50"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": clamped,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def display_checklist(checklist: List[Dict]) -> None:
    """Render a pre-trade checklist with collapsible status rows.

    Parameters
    ----------
    checklist:
        List of dicts with keys ``item``, ``status`` (pass/warn/fail),
        and ``detail``.
    """
    for item in checklist:
        label = item.get("item", "")
        status = item.get("status", "warn").lower()
        detail = item.get("detail", "")
        emoji = status_emoji(status)

        with st.expander(f"{emoji} {label}", expanded=False):
            color_map = {"pass": "success", "warn": "warning", "fail": "error"}
            fn = getattr(st, color_map.get(status, "info"), st.info)
            fn(detail if detail else "No additional details.")


def display_ratios_table(ratios: Dict[str, float]) -> None:
    """Render a two-column financial ratios table styled with colour coding.

    Parameters
    ----------
    ratios:
        Mapping of ratio names to their values (as returned by
        ``analysis_engine.compute_ratios``).
    """
    CATEGORIES: Dict[str, List[tuple]] = {
        "📊 Liquidity": [
            ("Current Ratio", "current_ratio", 1.5, 1.0, False),
            ("Quick Ratio", "quick_ratio", 1.0, 0.5, False),
            ("Cash Ratio", "cash_ratio", 0.5, 0.2, False),
        ],
        "🏦 Leverage": [
            ("Debt / Equity", "debt_to_equity", 0.5, 2.0, True),
            ("Debt / Assets", "debt_to_assets", 0.3, 0.6, True),
            ("Debt / EBITDA", "debt_to_ebitda", 1.5, 4.0, True),
            ("Interest Coverage", "interest_coverage", 5.0, 1.5, False),
        ],
        "💰 Profitability": [
            ("Gross Margin", "gross_margin", 0.40, 0.15, False),
            ("Operating Margin", "operating_margin", 0.20, 0.05, False),
            ("Net Margin", "net_margin", 0.15, 0.02, False),
            ("ROE", "roe", 0.20, 0.05, False),
            ("ROA", "roa", 0.10, 0.02, False),
        ],
        "💵 Cash Flow": [
            ("FCF Yield", "fcf_yield", 0.05, 0.0, False),
        ],
        "📈 Valuation": [
            ("P/E Ratio", "pe_ratio", 15.0, 35.0, True),
            ("P/B Ratio", "pb_ratio", 1.5, 5.0, True),
            ("P/S Ratio", "ps_ratio", 1.0, 5.0, True),
            ("EV/EBITDA", "ev_ebitda", 8.0, 20.0, True),
            ("PEG Ratio", "peg_ratio", 1.0, 2.0, True),
        ],
    }

    PCT_KEYS = {"gross_margin", "operating_margin", "net_margin", "roe", "roa", "fcf_yield"}

    for category, items in CATEGORIES.items():
        st.markdown(f"**{category}**")
        rows = []
        for label, key, good_threshold, bad_threshold, invert in items:
            raw = ratios.get(key)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                display_val = "N/A"
                color = "grey"
            elif key in PCT_KEYS:
                display_val = format_pct(raw)
                color = get_color_for_value(raw, good_threshold, bad_threshold, invert)
            else:
                display_val = format_ratio(raw)
                color = get_color_for_value(raw, good_threshold, bad_threshold, invert)
            rows.append({"Metric": label, "Value": display_val, "_color": color})

        df = pd.DataFrame(rows)

        _css_map = {
            "green": "color: #2e7d32; font-weight: 600",
            "orange": "color: #e65100; font-weight: 600",
            "red": "color: #c62828; font-weight: 600",
            "grey": "color: #9e9e9e",
        }

        def _style_row(row: pd.Series) -> list:
            # row.name is the integer index into df
            color = df.loc[row.name, "_color"]
            css = _css_map.get(color, "")
            # display_df has 2 columns: "Metric" (unstyled) and "Value" (styled)
            return ["", css]

        display_df = df[["Metric", "Value"]].copy()
        styled = display_df.style.apply(_style_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown("")
