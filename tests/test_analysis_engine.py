"""
Tests for analysis_engine.py
"""

import pytest
import pandas as pd
import numpy as np

import analysis_engine as ae


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ratios():
    return {
        "current_ratio": 2.0,
        "quick_ratio": 1.5,
        "cash_ratio": 0.5,
        "debt_to_equity": 0.4,
        "debt_to_assets": 0.2,
        "debt_to_ebitda": 1.2,
        "interest_coverage": 8.0,
        "gross_margin": 0.45,
        "operating_margin": 0.22,
        "net_margin": 0.18,
        "roe": 0.25,
        "roa": 0.12,
        "roic": 0.20,
        "operating_cf": 1_000_000,
        "fcf": 800_000,
        "fcf_yield": 0.04,
        "pe_ratio": 20.0,
        "pb_ratio": 2.5,
        "ps_ratio": 3.0,
        "ev_ebitda": 12.0,
        "peg_ratio": 1.5,
    }


@pytest.fixture
def weak_ratios():
    return {
        "current_ratio": 0.8,
        "debt_to_equity": 3.5,
        "debt_to_ebitda": 6.0,
        "interest_coverage": 1.0,
        "gross_margin": 0.10,
        "operating_margin": -0.05,
        "net_margin": -0.08,
        "roe": -0.10,
        "roa": -0.05,
        "fcf": -500_000,
        "fcf_yield": -0.02,
        "pe_ratio": 60.0,
        "pb_ratio": 8.0,
        "ev_ebitda": 30.0,
        "peg_ratio": 3.5,
    }


# ---------------------------------------------------------------------------
# _safe_div
# ---------------------------------------------------------------------------

def test_safe_div_normal():
    assert ae._safe_div(10, 2) == pytest.approx(5.0)


def test_safe_div_zero_denominator():
    result = ae._safe_div(10, 0)
    assert np.isnan(result)


def test_safe_div_none_denominator():
    result = ae._safe_div(10, None)
    assert np.isnan(result)


def test_safe_div_custom_default():
    assert ae._safe_div(10, 0, default=-1) == -1


# ---------------------------------------------------------------------------
# _row helper
# ---------------------------------------------------------------------------

def test_row_finds_key():
    df = pd.DataFrame({"2023": [100.0]}, index=["TotalRevenue"])
    val = ae._row(df, "TotalRevenue")
    assert val == pytest.approx(100.0)


def test_row_falls_back_to_second_key():
    df = pd.DataFrame({"2023": [200.0]}, index=["Total Revenue"])
    val = ae._row(df, "TotalRevenue", "Total Revenue")
    assert val == pytest.approx(200.0)


def test_row_empty_df():
    assert ae._row(pd.DataFrame(), "TotalRevenue") is None


def test_row_missing_key():
    df = pd.DataFrame({"2023": [100.0]}, index=["SomeOtherKey"])
    assert ae._row(df, "TotalRevenue") is None


# ---------------------------------------------------------------------------
# compute_ratios
# ---------------------------------------------------------------------------

def _make_income_stmt(revenue, gross_profit, operating_income, net_income):
    data = {
        "2023": [revenue, gross_profit, operating_income, net_income]
    }
    return pd.DataFrame(data, index=["TotalRevenue", "GrossProfit", "OperatingIncome", "NetIncome"])


def _make_balance_sheet(current_assets, current_liab, total_debt, equity, total_assets):
    data = {"2023": [current_assets, current_liab, total_debt, equity, total_assets]}
    return pd.DataFrame(
        data,
        index=["CurrentAssets", "CurrentLiabilities", "TotalDebt",
               "StockholdersEquity", "TotalAssets"],
    )


def _make_cash_flow(operating_cf, capex):
    data = {"2023": [operating_cf, capex]}
    return pd.DataFrame(data, index=["OperatingCashFlow", "CapitalExpenditures"])


def test_compute_ratios_returns_expected_keys():
    inc = _make_income_stmt(1000, 400, 200, 150)
    bs = _make_balance_sheet(500, 250, 200, 600, 1000)
    cf = _make_cash_flow(180, -40)
    ratios = ae.compute_ratios({}, inc, bs, cf)
    for key in ("current_ratio", "gross_margin", "net_margin", "roe", "debt_to_equity"):
        assert key in ratios


def test_current_ratio_computed_correctly():
    inc = _make_income_stmt(1000, 400, 200, 150)
    bs = _make_balance_sheet(500, 250, 200, 600, 1000)
    cf = _make_cash_flow(180, -40)
    ratios = ae.compute_ratios({}, inc, bs, cf)
    assert ratios["current_ratio"] == pytest.approx(2.0)


def test_gross_margin_computed_correctly():
    inc = _make_income_stmt(1000, 400, 200, 150)
    bs = _make_balance_sheet(500, 250, 200, 600, 1000)
    cf = _make_cash_flow(180, -40)
    ratios = ae.compute_ratios({}, inc, bs, cf)
    assert ratios["gross_margin"] == pytest.approx(0.4)


def test_compute_ratios_key_stats_valuation():
    key_stats = {"trailingPE": 22.5, "priceToBook": 3.1, "marketCap": 1e9}
    ratios = ae.compute_ratios(key_stats, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert ratios["pe_ratio"] == pytest.approx(22.5)
    assert ratios["pb_ratio"] == pytest.approx(3.1)


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

def test_scores_return_all_keys(sample_ratios):
    scores = ae.compute_scores(sample_ratios)
    for key in ("stability", "quality", "valuation", "composite"):
        assert key in scores


def test_scores_are_in_range(sample_ratios):
    scores = ae.compute_scores(sample_ratios)
    for key in ("stability", "quality", "valuation", "composite"):
        assert 0 <= scores[key] <= 100


def test_good_company_scores_higher_than_weak(sample_ratios, weak_ratios):
    good = ae.compute_scores(sample_ratios)
    bad = ae.compute_scores(weak_ratios)
    assert good["composite"] > bad["composite"]


def test_composite_is_weighted_average(sample_ratios):
    scores = ae.compute_scores(sample_ratios)
    expected = (
        scores["stability"] * ae.PILLAR_WEIGHTS["stability"]
        + scores["quality"] * ae.PILLAR_WEIGHTS["quality"]
        + scores["valuation"] * ae.PILLAR_WEIGHTS["valuation"]
    )
    assert scores["composite"] == pytest.approx(expected, rel=1e-3)


def test_empty_ratios_gives_mid_scores():
    scores = ae.compute_scores({})
    # With all-None ratios, baseline = 50 for each pillar
    assert scores["stability"] == pytest.approx(50.0)
    assert scores["quality"] == pytest.approx(50.0)
    assert scores["valuation"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# pretrade_checklist
# ---------------------------------------------------------------------------

def test_checklist_returns_list(sample_ratios):
    scores = ae.compute_scores(sample_ratios)
    items = ae.pretrade_checklist(sample_ratios, scores, thesis="Strong moat")
    assert isinstance(items, list)
    assert all("item" in i and "status" in i and "detail" in i for i in items)


def test_checklist_passes_thesis_check(sample_ratios):
    scores = ae.compute_scores(sample_ratios)
    items = ae.pretrade_checklist(sample_ratios, scores, thesis="Wide economic moat")
    thesis_items = [i for i in items if "thesis" in i["item"].lower()]
    assert any(i["status"] == "pass" for i in thesis_items)


def test_checklist_fails_thesis_without_text(sample_ratios):
    scores = ae.compute_scores(sample_ratios)
    items = ae.pretrade_checklist(sample_ratios, scores, thesis="")
    thesis_items = [i for i in items if "thesis" in i["item"].lower()]
    assert any(i["status"] == "fail" for i in thesis_items)


def test_checklist_warns_on_none_data():
    scores = ae.compute_scores({})
    items = ae.pretrade_checklist({}, scores)
    warn_items = [i for i in items if i["status"] == "warn"]
    # Several items should be warnings due to missing data
    assert len(warn_items) > 0


def test_checklist_fails_negative_fcf(sample_ratios):
    bad = {**sample_ratios, "fcf": -100_000}
    scores = ae.compute_scores(bad)
    items = ae.pretrade_checklist(bad, scores)
    fcf_items = [i for i in items if "free cash flow" in i["item"].lower()]
    assert any(i["status"] == "fail" for i in fcf_items)


# ---------------------------------------------------------------------------
# analyze convenience function
# ---------------------------------------------------------------------------

def test_analyze_returns_expected_structure():
    result = ae.analyze(
        {
            "key_stats": {"trailingPE": 18.0, "marketCap": 5e9},
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }
    )
    assert "ratios" in result
    assert "scores" in result
    assert "checklist" in result
