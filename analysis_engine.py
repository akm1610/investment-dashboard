"""
analysis_engine.py
------------------
Compute financial-health scores, quality metrics, and valuation ratios.

The scoring framework uses three pillars:
  1. Financial Stability   (balance-sheet + cash-flow quality)
  2. Business Quality      (profitability + capital efficiency)
  3. Valuation             (price multiples relative to fundamentals)

Each pillar is scored 0-100; a composite score is the weighted average.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights for the composite score
# ---------------------------------------------------------------------------
PILLAR_WEIGHTS = {
    "stability": 0.40,
    "quality": 0.35,
    "valuation": 0.25,
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float, default: float = np.nan) -> float:
    """Return num / den, or *default* when den is zero / None."""
    try:
        if den is None or den == 0:
            return default
        return num / den
    except Exception:
        return default


def _row(df: pd.DataFrame, *keys: str) -> Optional[float]:
    """
    Extract the most-recent (first column) value from a DataFrame
    whose row index may match any of the supplied *keys*.
    Returns None when nothing is found.
    """
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key].iloc[0]
                return float(val) if pd.notna(val) else None
            except Exception:
                continue
    return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Ratio calculators
# ---------------------------------------------------------------------------

def compute_ratios(
    key_stats: dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> dict:
    """
    Derive a comprehensive set of financial ratios.

    Parameters come from ``data_fetcher.fetch_all_fundamentals``.

    Returns a dict with keys like 'current_ratio', 'roe', 'fcf_yield', etc.
    """
    s = key_stats  # shorthand

    # --- Liquidity ---
    current_assets = _row(balance_sheet, "CurrentAssets", "Total Current Assets")
    current_liab = _row(balance_sheet, "CurrentLiabilities", "Total Current Liabilities")
    cash = _row(balance_sheet, "CashAndCashEquivalents", "Cash And Cash Equivalents")
    inventory = _row(balance_sheet, "Inventory") or 0.0

    current_ratio = _safe_div(current_assets, current_liab)
    quick_ratio = _safe_div((current_assets or 0) - inventory, current_liab)
    cash_ratio = _safe_div(cash, current_liab)

    # --- Leverage ---
    total_debt = _row(balance_sheet, "TotalDebt", "Long Term Debt")
    total_equity = _row(balance_sheet, "StockholdersEquity", "Total Stockholders Equity")
    total_assets = _row(balance_sheet, "TotalAssets", "Total Assets")
    ebitda = s.get("ebitda") or _row(income_stmt, "EBITDA")
    net_income = _row(income_stmt, "NetIncome", "Net Income")

    debt_to_equity = _safe_div(total_debt, total_equity)
    debt_to_assets = _safe_div(total_debt, total_assets)
    debt_to_ebitda = _safe_div(total_debt, ebitda)
    interest_expense = _row(income_stmt, "InterestExpense", "Interest Expense")
    ebit = _row(income_stmt, "EBIT", "Operating Income")
    interest_coverage = _safe_div(-(ebit or 0), -(interest_expense or 0))

    # --- Profitability ---
    revenue = _row(income_stmt, "TotalRevenue", "Total Revenue")
    gross_profit = _row(income_stmt, "GrossProfit", "Gross Profit")
    operating_income = _row(income_stmt, "OperatingIncome", "Operating Income", "EBIT")

    gross_margin = _safe_div(gross_profit, revenue)
    operating_margin = _safe_div(operating_income, revenue)
    net_margin = _safe_div(net_income, revenue)
    roe = _safe_div(net_income, total_equity)
    roa = _safe_div(net_income, total_assets)
    roic = s.get("returnOnEquity")  # approximation; use ROIC from key_stats if available

    # --- Cash flow ---
    operating_cf = _row(cash_flow, "OperatingCashFlow", "Total Cash From Operating Activities")
    capex = _row(cash_flow, "CapitalExpenditures", "Capital Expenditures")
    fcf_raw = s.get("freeCashflow")
    if fcf_raw is None and operating_cf is not None and capex is not None:
        fcf_raw = operating_cf + capex  # capex is typically negative in statements

    market_cap = s.get("marketCap")
    fcf_yield = _safe_div(fcf_raw, market_cap)

    # --- Valuation multiples ---
    pe_ratio = s.get("trailingPE") or s.get("forwardPE")
    pb_ratio = s.get("priceToBook")
    ps_ratio = s.get("priceToSalesTrailing12Months")
    ev_ebitda = s.get("enterpriseToEbitda")
    peg_ratio = s.get("pegRatio")

    return {
        # Liquidity
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "cash_ratio": cash_ratio,
        # Leverage
        "debt_to_equity": debt_to_equity,
        "debt_to_assets": debt_to_assets,
        "debt_to_ebitda": debt_to_ebitda,
        "interest_coverage": interest_coverage,
        # Profitability
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roa": roa,
        "roic": roic,
        # Cash flow
        "operating_cf": operating_cf,
        "fcf": fcf_raw,
        "fcf_yield": fcf_yield,
        # Valuation
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "ps_ratio": ps_ratio,
        "ev_ebitda": ev_ebitda,
        "peg_ratio": peg_ratio,
    }


# ---------------------------------------------------------------------------
# Pillar scorers (0 – 100)
# ---------------------------------------------------------------------------

def _score_stability(ratios: dict) -> float:
    """Score financial stability (liquidity + leverage)."""
    score = 50.0  # baseline

    cr = ratios.get("current_ratio")
    if cr is not None:
        if cr >= 2.0:
            score += 10
        elif cr >= 1.5:
            score += 7
        elif cr >= 1.0:
            score += 3
        else:
            score -= 10

    de = ratios.get("debt_to_equity")
    if de is not None:
        if de < 0.5:
            score += 10
        elif de < 1.0:
            score += 5
        elif de < 2.0:
            score += 0
        else:
            score -= 10

    ic = ratios.get("interest_coverage")
    if ic is not None and ic > 0:
        if ic >= 5:
            score += 10
        elif ic >= 3:
            score += 5
        elif ic >= 1.5:
            score += 0
        else:
            score -= 10

    d_ebitda = ratios.get("debt_to_ebitda")
    if d_ebitda is not None and d_ebitda > 0:
        if d_ebitda < 1.5:
            score += 10
        elif d_ebitda < 3.0:
            score += 5
        elif d_ebitda < 5.0:
            score += 0
        else:
            score -= 10

    fcf = ratios.get("fcf")
    if fcf is not None:
        score += 10 if fcf > 0 else -10

    return _clamp(score)


def _score_quality(ratios: dict) -> float:
    """Score business quality (profitability + capital efficiency)."""
    score = 50.0

    gm = ratios.get("gross_margin")
    if gm is not None:
        if gm >= 0.40:
            score += 10
        elif gm >= 0.20:
            score += 5
        else:
            score -= 5

    om = ratios.get("operating_margin")
    if om is not None:
        if om >= 0.20:
            score += 10
        elif om >= 0.10:
            score += 5
        elif om > 0:
            score += 2
        else:
            score -= 10

    nm = ratios.get("net_margin")
    if nm is not None:
        if nm >= 0.15:
            score += 10
        elif nm >= 0.05:
            score += 5
        elif nm > 0:
            score += 2
        else:
            score -= 10

    roe = ratios.get("roe")
    if roe is not None:
        if roe >= 0.20:
            score += 10
        elif roe >= 0.10:
            score += 5
        elif roe > 0:
            score += 2
        else:
            score -= 10

    fcf_yield = ratios.get("fcf_yield")
    if fcf_yield is not None:
        if fcf_yield >= 0.05:
            score += 10
        elif fcf_yield >= 0.02:
            score += 5
        elif fcf_yield > 0:
            score += 2
        else:
            score -= 5

    return _clamp(score)


def _score_valuation(ratios: dict) -> float:
    """
    Score valuation (lower multiples → higher score, but some growth premium
    is acceptable).
    """
    score = 50.0

    pe = ratios.get("pe_ratio")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 15
        elif pe < 25:
            score += 8
        elif pe < 35:
            score += 2
        else:
            score -= 10

    pb = ratios.get("pb_ratio")
    if pb is not None and pb > 0:
        if pb < 1.5:
            score += 10
        elif pb < 3.0:
            score += 5
        elif pb < 5.0:
            score += 0
        else:
            score -= 5

    ev_ebitda = ratios.get("ev_ebitda")
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 8:
            score += 10
        elif ev_ebitda < 15:
            score += 5
        elif ev_ebitda < 25:
            score += 0
        else:
            score -= 10

    peg = ratios.get("peg_ratio")
    if peg is not None and peg > 0:
        if peg < 1.0:
            score += 10
        elif peg < 2.0:
            score += 5
        else:
            score -= 5

    return _clamp(score)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_scores(ratios: dict) -> dict:
    """
    Compute pillar scores and composite score from pre-computed *ratios*.

    Returns a dict::

        {
            'stability': float,   # 0-100
            'quality':   float,   # 0-100
            'valuation': float,   # 0-100
            'composite': float,   # weighted average 0-100
        }
    """
    stability = _score_stability(ratios)
    quality = _score_quality(ratios)
    valuation = _score_valuation(ratios)

    composite = (
        stability * PILLAR_WEIGHTS["stability"]
        + quality * PILLAR_WEIGHTS["quality"]
        + valuation * PILLAR_WEIGHTS["valuation"]
    )

    return {
        "stability": round(stability, 1),
        "quality": round(quality, 1),
        "valuation": round(valuation, 1),
        "composite": round(composite, 1),
    }


# ---------------------------------------------------------------------------
# Pre-trade checklist
# ---------------------------------------------------------------------------

def pretrade_checklist(ratios: dict, scores: dict, thesis: str = "") -> list[dict]:
    """
    Return a list of checklist items with pass/fail status for a proposed trade.

    Each item is a dict::

        {'item': str, 'status': 'pass' | 'fail' | 'warn', 'detail': str}
    """
    items = []

    def _item(label, condition, detail_pass, detail_fail, warn=False):
        if condition is None:
            items.append({"item": label, "status": "warn", "detail": "Data unavailable"})
        elif condition:
            items.append({"item": label, "status": "pass", "detail": detail_pass})
        else:
            status = "warn" if warn else "fail"
            items.append({"item": label, "status": status, "detail": detail_fail})

    cr = ratios.get("current_ratio")
    _item(
        "Adequate liquidity (current ratio ≥ 1.5)",
        None if cr is None else cr >= 1.5,
        f"Current ratio = {cr:.2f}" if cr else "",
        f"Current ratio = {cr:.2f} — below threshold" if cr else "",
    )

    de = ratios.get("debt_to_equity")
    _item(
        "Manageable leverage (D/E < 2.0)",
        None if de is None else de < 2.0,
        f"D/E = {de:.2f}" if de else "",
        f"D/E = {de:.2f} — potentially over-leveraged" if de else "",
    )

    nm = ratios.get("net_margin")
    _item(
        "Positive net margin",
        None if nm is None else nm > 0,
        f"Net margin = {nm:.1%}" if nm else "",
        f"Net margin = {nm:.1%} — company is unprofitable" if nm else "",
    )

    fcf = ratios.get("fcf")
    _item(
        "Positive free cash flow",
        None if fcf is None else fcf > 0,
        "Free cash flow is positive",
        "Free cash flow is negative",
    )

    _item(
        "Quality score ≥ 60",
        None if scores.get("quality") is None else scores["quality"] >= 60,
        f"Quality = {scores.get('quality')}",
        f"Quality = {scores.get('quality')} — below threshold",
        warn=True,
    )

    _item(
        "Composite score ≥ 55",
        None if scores.get("composite") is None else scores["composite"] >= 55,
        f"Composite = {scores.get('composite')}",
        f"Composite = {scores.get('composite')} — below threshold",
        warn=True,
    )

    _item(
        "Investment thesis documented",
        bool(thesis and thesis.strip()),
        "Thesis on record",
        "No investment thesis provided",
    )

    return items


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def analyze(fundamentals: dict) -> dict:
    """
    Full analysis from a fundamentals bundle (output of
    ``data_fetcher.fetch_all_fundamentals``).

    Returns::

        {
            'ratios':  dict,
            'scores':  dict,
            'checklist': list[dict],
        }
    """
    ratios = compute_ratios(
        key_stats=fundamentals.get("key_stats", {}),
        income_stmt=fundamentals.get("income_statement", pd.DataFrame()),
        balance_sheet=fundamentals.get("balance_sheet", pd.DataFrame()),
        cash_flow=fundamentals.get("cash_flow", pd.DataFrame()),
    )
    scores = compute_scores(ratios)
    checklist = pretrade_checklist(ratios, scores)
    return {"ratios": ratios, "scores": scores, "checklist": checklist}
