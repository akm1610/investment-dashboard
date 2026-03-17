"""
Tests for portfolio_manager.py
"""

import os
import tempfile

import pandas as pd
import pytest

import portfolio_manager as pm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_portfolio():
    return pm._empty_portfolio()


@pytest.fixture
def portfolio_with_holdings():
    p = pm._empty_portfolio()
    pm.add_holding(p, "AAPL", 10, 150.0, sector="Technology", thesis="iPhone moat")
    pm.add_holding(p, "MSFT", 5, 300.0, sector="Technology", thesis="Cloud growth")
    pm.add_holding(p, "JNJ", 8, 160.0, sector="Healthcare", thesis="Dividend king")
    return p


@pytest.fixture
def current_prices():
    return {"AAPL": 180.0, "MSFT": 380.0, "JNJ": 155.0}


# ---------------------------------------------------------------------------
# Portfolio I/O
# ---------------------------------------------------------------------------

def test_load_portfolio_missing_file():
    p = pm.load_portfolio("/tmp/nonexistent_portfolio_xyz.json")
    assert "holdings" in p
    assert p["holdings"] == {}


def test_save_and_load_portfolio(tmp_path):
    path = str(tmp_path / "port.json")
    p = pm._empty_portfolio()
    pm.add_holding(p, "AAPL", 10, 150.0)
    pm.save_portfolio(p, path)
    loaded = pm.load_portfolio(path)
    assert "AAPL" in loaded["holdings"]
    assert loaded["holdings"]["AAPL"]["shares"] == 10


# ---------------------------------------------------------------------------
# add_holding
# ---------------------------------------------------------------------------

def test_add_new_holding(empty_portfolio):
    pm.add_holding(empty_portfolio, "AAPL", 10, 150.0, sector="Tech")
    assert "AAPL" in empty_portfolio["holdings"]
    h = empty_portfolio["holdings"]["AAPL"]
    assert h["shares"] == 10
    assert h["avg_cost"] == 150.0
    assert h["sector"] == "Tech"


def test_add_holding_normalizes_symbol(empty_portfolio):
    pm.add_holding(empty_portfolio, "aapl", 5, 100.0)
    assert "AAPL" in empty_portfolio["holdings"]


def test_add_holding_averages_cost(empty_portfolio):
    pm.add_holding(empty_portfolio, "AAPL", 10, 100.0)
    pm.add_holding(empty_portfolio, "AAPL", 10, 200.0)
    h = empty_portfolio["holdings"]["AAPL"]
    assert h["shares"] == 20
    assert h["avg_cost"] == pytest.approx(150.0)


def test_add_holding_records_trade(empty_portfolio):
    pm.add_holding(empty_portfolio, "AAPL", 10, 150.0)
    assert len(empty_portfolio["trades"]) == 1
    trade = empty_portfolio["trades"][0]
    assert trade["symbol"] == "AAPL"
    assert trade["action"] == "BUY"


# ---------------------------------------------------------------------------
# remove_holding
# ---------------------------------------------------------------------------

def test_remove_all_shares(portfolio_with_holdings):
    pm.remove_holding(portfolio_with_holdings, "AAPL")
    assert "AAPL" not in portfolio_with_holdings["holdings"]


def test_remove_partial_shares(portfolio_with_holdings):
    pm.remove_holding(portfolio_with_holdings, "AAPL", shares=3)
    assert portfolio_with_holdings["holdings"]["AAPL"]["shares"] == pytest.approx(7.0)


def test_remove_nonexistent_symbol_is_noop(portfolio_with_holdings):
    before = len(portfolio_with_holdings["holdings"])
    pm.remove_holding(portfolio_with_holdings, "TSLA")
    assert len(portfolio_with_holdings["holdings"]) == before


def test_remove_records_sell_trade(portfolio_with_holdings):
    trades_before = len(portfolio_with_holdings["trades"])
    pm.remove_holding(portfolio_with_holdings, "AAPL", price=175.0)
    assert len(portfolio_with_holdings["trades"]) == trades_before + 1
    last_trade = portfolio_with_holdings["trades"][-1]
    assert last_trade["action"] == "SELL"
    assert last_trade["price"] == 175.0


# ---------------------------------------------------------------------------
# compute_allocation
# ---------------------------------------------------------------------------

def test_compute_allocation_returns_dataframe(portfolio_with_holdings, current_prices):
    df = pm.compute_allocation(portfolio_with_holdings, current_prices)
    assert isinstance(df, pd.DataFrame)
    assert set(["symbol", "market_value", "weight"]).issubset(df.columns)


def test_weights_sum_to_one(portfolio_with_holdings, current_prices):
    df = pm.compute_allocation(portfolio_with_holdings, current_prices)
    total = df["weight"].sum()
    assert total == pytest.approx(1.0, abs=1e-6)


def test_market_value_computed_correctly(portfolio_with_holdings, current_prices):
    df = pm.compute_allocation(portfolio_with_holdings, current_prices)
    aapl = df[df["symbol"] == "AAPL"].iloc[0]
    assert aapl["market_value"] == pytest.approx(10 * 180.0)


def test_unrealized_pnl(portfolio_with_holdings, current_prices):
    df = pm.compute_allocation(portfolio_with_holdings, current_prices)
    aapl = df[df["symbol"] == "AAPL"].iloc[0]
    expected = 10 * (180.0 - 150.0)
    assert aapl["unrealized_pnl"] == pytest.approx(expected)


def test_empty_portfolio_allocation():
    df = pm.compute_allocation(pm._empty_portfolio(), {})
    assert df.empty


# ---------------------------------------------------------------------------
# get_portfolio_summary
# ---------------------------------------------------------------------------

def test_portfolio_summary_keys(portfolio_with_holdings, current_prices):
    summary = pm.get_portfolio_summary(portfolio_with_holdings, current_prices)
    for key in ("total_value", "cost_basis", "total_pnl", "pnl_pct",
                "num_holdings", "sector_allocation"):
        assert key in summary


def test_portfolio_summary_total_value(portfolio_with_holdings, current_prices):
    summary = pm.get_portfolio_summary(portfolio_with_holdings, current_prices)
    expected = 10 * 180 + 5 * 380 + 8 * 155
    assert summary["total_value"] == pytest.approx(expected)


def test_sector_allocation_sums_to_one(portfolio_with_holdings, current_prices):
    summary = pm.get_portfolio_summary(portfolio_with_holdings, current_prices)
    total = sum(summary["sector_allocation"].values())
    assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# concentration_alerts
# ---------------------------------------------------------------------------

def test_concentration_alert_raised():
    # One large position > 20%
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "market_value": [9000.0, 1000.0],
        "weight": [0.9, 0.1],
    })
    alerts = pm.concentration_alerts(df, threshold=0.20)
    assert len(alerts) == 1
    assert alerts[0]["symbol"] == "AAPL"


def test_no_concentration_alert():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "market_value": [333.0, 333.0, 334.0],
        "weight": [0.333, 0.333, 0.334],
    })
    # threshold of 0.40 is above each position's ~33% weight — no alerts expected
    alerts = pm.concentration_alerts(df, threshold=0.40)
    assert alerts == []


def test_concentration_alert_empty_df():
    alerts = pm.concentration_alerts(pd.DataFrame())
    assert alerts == []


# ---------------------------------------------------------------------------
# rebalancing_suggestions
# ---------------------------------------------------------------------------

def test_rebalancing_trim_action():
    alloc = pd.DataFrame({
        "symbol": ["AAPL"],
        "weight": [0.50],
        "market_value": [5000.0],
    })
    targets = {"AAPL": 0.25}
    result = pm.rebalancing_suggestions(alloc, targets, tolerance=0.05)
    row = result[result["symbol"] == "AAPL"].iloc[0]
    assert row["action"] == "TRIM"


def test_rebalancing_add_action():
    alloc = pd.DataFrame({
        "symbol": ["AAPL"],
        "weight": [0.10],
        "market_value": [1000.0],
    })
    targets = {"AAPL": 0.25}
    result = pm.rebalancing_suggestions(alloc, targets, tolerance=0.05)
    row = result[result["symbol"] == "AAPL"].iloc[0]
    assert row["action"] == "ADD"


def test_rebalancing_hold_within_tolerance():
    alloc = pd.DataFrame({
        "symbol": ["AAPL"],
        "weight": [0.24],
        "market_value": [2400.0],
    })
    targets = {"AAPL": 0.25}
    result = pm.rebalancing_suggestions(alloc, targets, tolerance=0.05)
    row = result[result["symbol"] == "AAPL"].iloc[0]
    assert row["action"] == "HOLD"


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def test_add_journal_entry(empty_portfolio):
    pm.add_journal_entry(empty_portfolio, "AAPL", "Strong brand moat", tags=["moat", "consumer"])
    journal = pm.get_journal(empty_portfolio)
    assert len(journal) == 1
    assert journal[0]["symbol"] == "AAPL"
    assert "moat" in journal[0]["tags"]


def test_get_journal_filters_by_symbol(empty_portfolio):
    pm.add_journal_entry(empty_portfolio, "AAPL", "Note 1")
    pm.add_journal_entry(empty_portfolio, "MSFT", "Note 2")
    aapl_entries = pm.get_journal(empty_portfolio, "AAPL")
    assert len(aapl_entries) == 1
    assert aapl_entries[0]["symbol"] == "AAPL"


def test_journal_entry_updates_holding_thesis(portfolio_with_holdings):
    pm.add_journal_entry(portfolio_with_holdings, "AAPL", "Updated thesis")
    holding = portfolio_with_holdings["holdings"]["AAPL"]
    assert holding["thesis"] == "Updated thesis"
