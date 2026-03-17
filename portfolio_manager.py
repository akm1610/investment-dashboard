"""
portfolio_manager.py
--------------------
Track holdings, compute allocations, detect concentration risk,
and support rebalancing decisions.

Holdings are persisted as a JSON file (``portfolio.json`` by default).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PORTFOLIO_FILE = "portfolio.json"

# ---------------------------------------------------------------------------
# Portfolio I/O
# ---------------------------------------------------------------------------

def _empty_portfolio() -> dict:
    return {
        "holdings": {},   # { symbol: { shares, avg_cost, sector, thesis } }
        "cash": 0.0,
        "journal": [],    # list of journal entries
        "trades": [],     # list of trade records
    }


def load_portfolio(path: str = DEFAULT_PORTFOLIO_FILE) -> dict:
    """Load portfolio from a JSON file; return empty portfolio if absent."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("load_portfolio: %s", exc)
    return _empty_portfolio()


def save_portfolio(portfolio: dict, path: str = DEFAULT_PORTFOLIO_FILE) -> None:
    """Persist the portfolio dict to a JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(portfolio, fh, indent=2, default=str)
    except Exception as exc:
        logger.error("save_portfolio: %s", exc)


# ---------------------------------------------------------------------------
# Holding management
# ---------------------------------------------------------------------------

def add_holding(
    portfolio: dict,
    symbol: str,
    shares: float,
    avg_cost: float,
    sector: str = "Unknown",
    thesis: str = "",
) -> None:
    """
    Add or update a holding in the portfolio.

    If the symbol already exists the position is averaged-in.
    """
    symbol = symbol.upper()
    existing = portfolio["holdings"].get(symbol)
    if existing:
        old_shares = existing["shares"]
        old_cost = existing["avg_cost"]
        new_shares = old_shares + shares
        new_cost = _safe_avg_cost(old_shares, old_cost, shares, avg_cost)
        existing["shares"] = new_shares
        existing["avg_cost"] = new_cost
        if thesis:
            existing["thesis"] = thesis
    else:
        portfolio["holdings"][symbol] = {
            "shares": shares,
            "avg_cost": avg_cost,
            "sector": sector,
            "thesis": thesis,
        }

    portfolio["trades"].append(
        {
            "date": str(date.today()),
            "symbol": symbol,
            "action": "BUY",
            "shares": shares,
            "price": avg_cost,
        }
    )


def remove_holding(
    portfolio: dict,
    symbol: str,
    shares: Optional[float] = None,
    price: Optional[float] = None,
) -> None:
    """
    Reduce or close a position.

    *shares* – how many to sell; None = sell all.
    *price*  – execution price (for trade log).
    """
    symbol = symbol.upper()
    if symbol not in portfolio["holdings"]:
        logger.warning("remove_holding: %s not in portfolio", symbol)
        return

    holding = portfolio["holdings"][symbol]
    shares_to_sell = shares if shares is not None else holding["shares"]

    portfolio["trades"].append(
        {
            "date": str(date.today()),
            "symbol": symbol,
            "action": "SELL",
            "shares": shares_to_sell,
            "price": price,
        }
    )

    remaining = holding["shares"] - shares_to_sell
    if remaining <= 1e-6:
        del portfolio["holdings"][symbol]
    else:
        holding["shares"] = remaining


def _safe_avg_cost(
    old_shares: float,
    old_cost: float,
    new_shares: float,
    new_cost: float,
) -> float:
    total_shares = old_shares + new_shares
    if total_shares == 0:
        return 0.0
    return (old_shares * old_cost + new_shares * new_cost) / total_shares


# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------

def compute_allocation(portfolio: dict, current_prices: dict[str, float]) -> pd.DataFrame:
    """
    Compute current market values and allocation weights.

    Parameters
    ----------
    portfolio      : Portfolio dict from ``load_portfolio``.
    current_prices : { symbol: current_price } mapping.

    Returns
    -------
    DataFrame with columns:
    symbol, shares, avg_cost, current_price, market_value, cost_basis,
    unrealized_pnl, pnl_pct, weight, sector.
    """
    rows = []
    for symbol, h in portfolio["holdings"].items():
        price = current_prices.get(symbol)
        shares = h["shares"]
        avg_cost = h["avg_cost"]
        market_value = shares * price if price else None
        cost_basis = shares * avg_cost
        unreal = (market_value - cost_basis) if market_value is not None else None
        pnl_pct = (unreal / cost_basis) if (unreal is not None and cost_basis != 0) else None
        rows.append(
            {
                "symbol": symbol,
                "shares": shares,
                "avg_cost": avg_cost,
                "current_price": price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unreal,
                "pnl_pct": pnl_pct,
                "sector": h.get("sector", "Unknown"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_value = df["market_value"].sum()
    df["weight"] = df["market_value"].apply(
        lambda v: v / total_value if (total_value and v is not None) else None
    )
    return df


def get_portfolio_summary(
    portfolio: dict, current_prices: dict[str, float]
) -> dict:
    """
    Return high-level portfolio metrics:
    total_value, cost_basis, total_pnl, pnl_pct, cash, num_holdings,
    allocation_df, sector_allocation.
    """
    alloc = compute_allocation(portfolio, current_prices)

    total_value = alloc["market_value"].sum() if not alloc.empty else 0.0
    total_cost = alloc["cost_basis"].sum() if not alloc.empty else 0.0
    total_pnl = total_value - total_cost
    pnl_pct = total_pnl / total_cost if total_cost else 0.0

    sector_alloc: dict = {}
    if not alloc.empty:
        sector_group = alloc.groupby("sector")["market_value"].sum()
        sector_alloc = (sector_group / total_value).to_dict() if total_value else {}

    return {
        "total_value": total_value,
        "cost_basis": total_cost,
        "total_pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "cash": portfolio.get("cash", 0.0),
        "num_holdings": len(portfolio["holdings"]),
        "allocation_df": alloc,
        "sector_allocation": sector_alloc,
    }


# ---------------------------------------------------------------------------
# Concentration & rebalancing alerts
# ---------------------------------------------------------------------------

def concentration_alerts(alloc_df: pd.DataFrame, threshold: float = 0.20) -> list[dict]:
    """
    Flag any position whose weight exceeds *threshold* (default 20%).

    Returns a list of alert dicts.
    """
    alerts = []
    if alloc_df.empty or "weight" not in alloc_df.columns:
        return alerts
    for _, row in alloc_df.iterrows():
        w = row.get("weight")
        if w is not None and w > threshold:
            alerts.append(
                {
                    "type": "CONCENTRATION",
                    "symbol": row["symbol"],
                    "weight": round(w, 4),
                    "message": (
                        f"{row['symbol']} represents {w:.1%} of portfolio "
                        f"(threshold {threshold:.0%})"
                    ),
                }
            )
    return alerts


def rebalancing_suggestions(
    alloc_df: pd.DataFrame,
    target_weights: dict[str, float],
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Compare current weights to *target_weights* and suggest trades.

    Parameters
    ----------
    alloc_df       : Output of ``compute_allocation``.
    target_weights : { symbol: target_weight } (weights should sum to 1).
    tolerance      : Drift beyond which a rebalance is suggested.

    Returns a DataFrame with columns:
    symbol, current_weight, target_weight, drift, action.
    """
    rows = []
    all_symbols = set(alloc_df["symbol"].tolist()) | set(target_weights.keys())
    weight_map = dict(zip(alloc_df["symbol"], alloc_df["weight"]))

    for sym in all_symbols:
        current = weight_map.get(sym, 0.0) or 0.0
        target = target_weights.get(sym, 0.0)
        drift = current - target
        if abs(drift) > tolerance:
            action = "TRIM" if drift > 0 else "ADD"
        else:
            action = "HOLD"
        rows.append(
            {
                "symbol": sym,
                "current_weight": round(current, 4),
                "target_weight": round(target, 4),
                "drift": round(drift, 4),
                "action": action,
            }
        )

    return pd.DataFrame(rows).sort_values("drift", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Investment thesis journal
# ---------------------------------------------------------------------------

def add_journal_entry(
    portfolio: dict,
    symbol: str,
    thesis: str,
    tags: Optional[list[str]] = None,
) -> None:
    """Append a timestamped journal entry to the portfolio."""
    portfolio["journal"].append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol.upper(),
            "thesis": thesis,
            "tags": tags or [],
        }
    )
    # Also update the holding's thesis field if holding exists
    holding = portfolio["holdings"].get(symbol.upper())
    if holding:
        holding["thesis"] = thesis


def get_journal(portfolio: dict, symbol: Optional[str] = None) -> list[dict]:
    """Return journal entries, optionally filtered by *symbol*."""
    entries = portfolio.get("journal", [])
    if symbol:
        entries = [e for e in entries if e["symbol"] == symbol.upper()]
    return entries
