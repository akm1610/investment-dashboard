"""
src/risk_engine.py
------------------
Comprehensive risk management engine for the investment dashboard.

Architecture
------------
1. PositionSizer         – Kelly, fixed-fractional, volatility-adjusted,
                           risk-parity, and "smart" sizing methods.
2. RiskMetrics           – Standalone functions for portfolio risk calculations
                           (volatility, VaR, CVaR, Sharpe, Sortino, drawdown,
                           beta, correlation).
3. RiskProfileAssessor   – 10-question questionnaire → Conservative / Moderate
                           / Aggressive profile + asset-allocation suggestion.
4. PortfolioRiskAnalyzer – Concentration, sector, correlation, drawdown, and
                           stress-test analyses.
5. PortfolioHealthMonitor– Real-time alerts, rebalancing suggestions, and a
                           comprehensive risk report.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANNUALISE = 252  # trading days per year
_DEFAULT_RISK_FREE_RATE = 0.04
_KELLY_FRACTION = 0.25  # fractional Kelly for safety

# Concentration alert thresholds
_SINGLE_POSITION_WARN = 0.15
_SINGLE_POSITION_HIGH = 0.20
_TOP5_WARN = 0.60

# Risk-profile bucket boundaries (score 0-100)
_PROFILE_CONSERVATIVE_MAX = 40
_PROFILE_AGGRESSIVE_MIN = 70


# ===========================================================================
# 1. Standalone risk-metric functions
# ===========================================================================


def calculate_portfolio_volatility(returns: pd.Series) -> float:
    """Return annualised portfolio standard deviation."""
    if returns is None or len(returns) < 2:
        return 0.0
    return float(returns.std() * math.sqrt(_ANNUALISE))


def calculate_value_at_risk(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).

    Returns the loss (positive number) that is not exceeded with probability
    *confidence* over a single period.
    """
    if returns is None or len(returns) < 2:
        return 0.0
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def calculate_conditional_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Calculate Conditional VaR / Expected Shortfall.

    Returns the expected loss given that the loss exceeds VaR.
    """
    if returns is None or len(returns) < 2:
        return 0.0
    clean = returns.dropna()
    threshold = np.percentile(clean, (1 - confidence) * 100)
    tail = clean[clean <= threshold]
    if tail.empty:
        return float(-threshold)
    return float(-tail.mean())


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = _DEFAULT_RISK_FREE_RATE
) -> float:
    """Return annualised Sharpe ratio."""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / _ANNUALISE
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(_ANNUALISE))


def calculate_sortino_ratio(
    returns: pd.Series, target_return: float = 0.0
) -> float:
    """Return annualised Sortino ratio (downside deviation denominator)."""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - target_return / _ANNUALISE
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(_ANNUALISE))


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Return maximum peak-to-trough drawdown as a positive fraction."""
    if returns is None or len(returns) < 2:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return float(-drawdowns.min())


def calculate_beta(
    asset_returns: pd.Series, market_returns: pd.Series
) -> float:
    """Return beta of *asset_returns* relative to *market_returns*."""
    if (
        asset_returns is None
        or market_returns is None
        or len(asset_returns) < 2
        or len(market_returns) < 2
    ):
        return 1.0
    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return 1.0
    market_var = aligned["market"].var()
    if market_var == 0:
        return 1.0
    cov = aligned.cov().loc["asset", "market"]
    return float(cov / market_var)


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Return the pairwise Pearson correlation matrix."""
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    return returns_df.corr()


# ===========================================================================
# 2. PositionSizer
# ===========================================================================


class PositionSizer:
    """Calculate optimal position sizes using several risk frameworks."""

    # ------------------------------------------------------------------
    # 2a. Kelly Criterion
    # ------------------------------------------------------------------

    def kelly_sizing(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_size: float,
    ) -> float:
        """
        Calculate dollar position size using the fractional Kelly Criterion.

        Parameters
        ----------
        win_rate  : probability of a winning trade (0-1)
        avg_win   : average dollar gain on a winning trade
        avg_loss  : average dollar loss on a losing trade (positive number)
        account_size : total portfolio value in dollars
        """
        if avg_loss <= 0 or account_size <= 0:
            return 0.0
        b = avg_win / avg_loss  # win/loss ratio
        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p
        kelly_f = (b * p - q) / b
        kelly_f = max(0.0, kelly_f)  # never negative
        fractional = kelly_f * _KELLY_FRACTION
        return float(fractional * account_size)

    # ------------------------------------------------------------------
    # 2b. Fixed-fractional
    # ------------------------------------------------------------------

    def fixed_fractional_sizing(
        self, account_size: float, risk_per_trade: float = 0.02
    ) -> float:
        """Return the dollar amount to risk on a single trade."""
        if account_size <= 0:
            return 0.0
        risk_per_trade = max(0.001, min(0.10, risk_per_trade))
        return float(account_size * risk_per_trade)

    # ------------------------------------------------------------------
    # 2c. 1-2-3 rule by market-cap tier
    # ------------------------------------------------------------------

    def one_two_three_sizing(
        self,
        account_size: float,
        market_cap_tier: str,
    ) -> float:
        """
        Apply the 1-2-3 rule.

        Parameters
        ----------
        market_cap_tier : 'micro', 'mid', or 'large'
        """
        tier_map = {"micro": 0.01, "mid": 0.02, "large": 0.03}
        pct = tier_map.get(market_cap_tier.lower(), 0.02)
        return float(account_size * pct)

    # ------------------------------------------------------------------
    # 2d. Volatility-adjusted
    # ------------------------------------------------------------------

    def volatility_adjusted_sizing(
        self,
        account_size: float,
        stock_volatility: float,
        target_risk: float = 0.02,
    ) -> float:
        """
        Size the position so that the dollar risk equals *target_risk* of
        *account_size* given the stock's daily volatility.
        """
        if stock_volatility <= 0 or account_size <= 0:
            return 0.0
        position_pct = target_risk / stock_volatility
        position_pct = min(position_pct, 0.25)  # hard cap 25 %
        return float(account_size * position_pct)

    # ------------------------------------------------------------------
    # 2e. Risk-parity
    # ------------------------------------------------------------------

    def risk_parity_sizing(
        self,
        stocks: List[str],
        volatilities: List[float],
    ) -> Dict[str, float]:
        """
        Return portfolio weights where each position contributes equal risk.

        Returns a dict ``{ticker: weight}`` that sums to 1.0.
        """
        if not stocks or not volatilities or len(stocks) != len(volatilities):
            return {}
        vols = np.array(volatilities, dtype=float)
        inv_vols = np.where(vols > 0, 1.0 / vols, 0.0)
        total = inv_vols.sum()
        if total == 0:
            weights = np.ones(len(stocks)) / len(stocks)
        else:
            weights = inv_vols / total
        return {ticker: float(w) for ticker, w in zip(stocks, weights)}

    # ------------------------------------------------------------------
    # 2f. Smart / combined sizing
    # ------------------------------------------------------------------

    def suggest_position_size(
        self,
        ticker: str,
        account_size: float,
        confidence_score: float,
        method: str = "smart",
        entry_price: Optional[float] = None,
        stock_volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Recommend a position size.

        Parameters
        ----------
        ticker           : stock symbol
        account_size     : total portfolio in dollars
        confidence_score : model confidence 0-1
        method           : 'smart', 'kelly', 'fixed', 'volatility', 'parity'
        entry_price      : current stock price (optional)
        stock_volatility : annualised daily vol (optional)
        win_rate, avg_win, avg_loss : required for 'kelly' method
        """
        position_pct: float
        method_used: str
        rationale: str

        if method == "kelly" and all(
            v is not None for v in (win_rate, avg_win, avg_loss)
        ):
            dollar_size = self.kelly_sizing(
                win_rate, avg_win, avg_loss, account_size
            )
            position_pct = dollar_size / account_size if account_size > 0 else 0.0
            method_used = "kelly"
            rationale = "Kelly Criterion with 25% safety fraction"

        elif method == "volatility" and stock_volatility is not None:
            dollar_size = self.volatility_adjusted_sizing(
                account_size, stock_volatility
            )
            position_pct = dollar_size / account_size if account_size > 0 else 0.0
            method_used = "volatility_adjusted"
            rationale = (
                "High volatility, conservative sizing"
                if stock_volatility > 0.03
                else "Low volatility, larger position allowed"
            )

        elif method == "smart":
            # Blend volatility-adjusted (if available) with confidence scaling
            base_pct = 0.05  # 5 % base
            if stock_volatility is not None and stock_volatility > 0:
                vol_pct = 0.02 / stock_volatility
                base_pct = min(vol_pct, 0.15)
            # Scale by confidence
            position_pct = base_pct * (0.5 + confidence_score * 0.5)
            position_pct = max(0.01, min(position_pct, 0.20))
            method_used = "smart"
            if stock_volatility:
                rationale = (
                    f"Smart sizing: confidence={confidence_score:.0%}, "
                    f"vol={stock_volatility:.2%}"
                )
            else:
                rationale = f"Smart sizing: confidence={confidence_score:.0%}"

        else:  # fixed fractional default
            position_pct = 0.02  # 2 % risk
            method_used = "fixed_fractional"
            rationale = "Fixed 2% risk per trade"

        notional_value = account_size * position_pct
        shares = (
            int(notional_value / entry_price) if (entry_price and entry_price > 0) else None
        )
        max_loss_dollars = notional_value * 0.02  # 2 % stop-loss

        return {
            "ticker": ticker,
            "position_size": round(position_pct, 4),
            "shares": shares,
            "entry_price": entry_price,
            "notional_value": round(notional_value, 2),
            "max_loss_dollars": round(max_loss_dollars, 2),
            "method_used": method_used,
            "rationale": rationale,
        }


# ===========================================================================
# 3. RiskProfileAssessor
# ===========================================================================


class RiskProfileAssessor:
    """
    Evaluate investor risk tolerance via a 10-question questionnaire.

    Each question is answered on a 1-5 scale (1=lowest, 5=highest tolerance).
    """

    questions: List[str] = [
        "Investment time horizon? (1=<1yr, 5=20+yrs)",
        "How would you react to 20% portfolio loss? (1=panic, 5=hold)",
        "Annual income level? (1=<50k, 5=>250k)",
        "Emergency fund available? (1=none, 5=12+ months)",
        "Investment experience? (1=none, 5=20+ years)",
        "Can you add funds monthly? (1=no, 5=yes, large amounts)",
        "Investment knowledge? (1=none, 5=expert)",
        "Market crash history handling? (1=sold, 5=bought more)",
        "Portfolio concentration comfort? (1=avoid, 5=concentrate)",
        "Leverage comfort? (1=never, 5=yes)",
    ]

    # Profile-specific parameters
    _PROFILES: Dict[str, Dict[str, Any]] = {
        "conservative": {
            "suggested_volatility": 0.08,
            "max_position_size": 0.05,
            "recommended_allocation": {
                "stocks": 0.30,
                "bonds": 0.60,
                "alternatives": 0.10,
            },
        },
        "moderate": {
            "suggested_volatility": 0.15,
            "max_position_size": 0.10,
            "recommended_allocation": {
                "stocks": 0.60,
                "bonds": 0.30,
                "alternatives": 0.10,
            },
        },
        "aggressive": {
            "suggested_volatility": 0.25,
            "max_position_size": 0.20,
            "recommended_allocation": {
                "stocks": 0.85,
                "bonds": 0.05,
                "alternatives": 0.10,
            },
        },
    }

    def assess(self, answers: List[int]) -> Dict[str, Any]:
        """
        Process questionnaire answers and return a risk profile.

        Parameters
        ----------
        answers : list of 10 integers in range 1-5

        Returns
        -------
        dict with keys: risk_score, profile, suggested_volatility,
                        max_position_size, recommended_allocation
        """
        if len(answers) != len(self.questions):
            raise ValueError(
                f"Expected {len(self.questions)} answers, got {len(answers)}"
            )
        for i, a in enumerate(answers):
            if not isinstance(a, (int, float)) or not (1 <= a <= 5):
                raise ValueError(
                    f"Answer {i + 1} must be an integer between 1 and 5, got {a!r}"
                )

        raw_sum = sum(answers)  # 10-50
        # Normalise to 0-100
        risk_score = int(round((raw_sum - 10) / 40 * 100))
        risk_score = max(0, min(100, risk_score))

        if risk_score <= _PROFILE_CONSERVATIVE_MAX:
            profile = "conservative"
        elif risk_score >= _PROFILE_AGGRESSIVE_MIN:
            profile = "aggressive"
        else:
            profile = "moderate"

        params = self._PROFILES[profile]
        return {
            "risk_score": risk_score,
            "profile": profile,
            "suggested_volatility": params["suggested_volatility"],
            "max_position_size": params["max_position_size"],
            "recommended_allocation": dict(params["recommended_allocation"]),
        }

    def validate_portfolio_fit(
        self,
        portfolio: Dict[str, Any],
        answers: Optional[List[int]] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check whether *portfolio* matches the assessed risk profile.

        Parameters
        ----------
        portfolio : dict with at least ``holdings`` (ticker→weight mapping)
                    and optionally ``volatility`` (annualised portfolio vol).
        answers   : raw questionnaire answers (used to derive profile if
                    *profile* not supplied directly).
        profile   : 'conservative', 'moderate', or 'aggressive'
        """
        if profile is None and answers is not None:
            assessed = self.assess(answers)
            profile = assessed["profile"]
        if profile not in self._PROFILES:
            profile = "moderate"

        params = self._PROFILES[profile]
        issues: List[str] = []
        warnings_list: List[str] = []

        holdings: Dict[str, float] = portfolio.get("holdings", {})
        port_vol: Optional[float] = portfolio.get("volatility")
        allocation: Dict[str, float] = portfolio.get("allocation", {})

        # Check single-position sizes
        for ticker, weight in holdings.items():
            if weight > params["max_position_size"]:
                issues.append(
                    f"{ticker} weight {weight:.1%} exceeds max "
                    f"{params['max_position_size']:.1%} for {profile} profile"
                )

        # Check portfolio volatility
        if port_vol is not None and port_vol > params["suggested_volatility"] * 1.25:
            warnings_list.append(
                f"Portfolio volatility {port_vol:.1%} is above "
                f"the suggested {params['suggested_volatility']:.1%} for {profile}"
            )

        # Check stock allocation
        stock_alloc = allocation.get("stocks", None)
        target_stocks = params["recommended_allocation"]["stocks"]
        if stock_alloc is not None and abs(stock_alloc - target_stocks) > 0.15:
            warnings_list.append(
                f"Stock allocation {stock_alloc:.1%} deviates from "
                f"target {target_stocks:.1%} for {profile} profile"
            )

        return {
            "profile": profile,
            "is_aligned": len(issues) == 0,
            "issues": issues,
            "warnings": warnings_list,
        }


# ===========================================================================
# 4. PortfolioRiskAnalyzer
# ===========================================================================


class PortfolioRiskAnalyzer:
    """Comprehensive portfolio risk assessment."""

    # ------------------------------------------------------------------
    # 4a. Concentration
    # ------------------------------------------------------------------

    def analyze_concentration(
        self, holdings: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyse portfolio concentration risks.

        Parameters
        ----------
        holdings : dict of {ticker: portfolio_weight}  (weights 0-1)

        Returns
        -------
        dict with herfindahl_index, single_largest_position, top_5_cumulative,
        alerts, concentration_score
        """
        if not holdings:
            return {
                "herfindahl_index": 0.0,
                "single_largest_position": 0.0,
                "top_5_cumulative": 0.0,
                "alerts": [],
                "concentration_score": 0,
            }

        weights = np.array(list(holdings.values()), dtype=float)
        tickers = list(holdings.keys())

        hhi = float(np.sum(weights**2))
        sorted_indices = np.argsort(-weights)
        largest = float(weights[sorted_indices[0]])
        top5 = float(weights[sorted_indices[:5]].sum())

        alerts = []
        for i, (ticker, w) in enumerate(holdings.items()):
            if w >= _SINGLE_POSITION_HIGH:
                alerts.append(
                    {
                        "position": ticker,
                        "weight": round(w, 4),
                        "severity": "HIGH",
                    }
                )
            elif w >= _SINGLE_POSITION_WARN:
                alerts.append(
                    {
                        "position": ticker,
                        "weight": round(w, 4),
                        "severity": "MEDIUM",
                    }
                )

        # Concentration score 0-100: 0=fully diversified, 100=single stock
        # Normalise HHI: min = 1/n (equal weights), max = 1.0
        n = len(holdings)
        min_hhi = 1.0 / n if n > 0 else 1.0
        if min_hhi >= 1.0:
            concentration_score = 100
        else:
            concentration_score = int(
                round((hhi - min_hhi) / (1.0 - min_hhi) * 100)
            )

        return {
            "herfindahl_index": round(hhi, 4),
            "single_largest_position": round(largest, 4),
            "top_5_cumulative": round(top5, 4),
            "alerts": alerts,
            "concentration_score": max(0, min(100, concentration_score)),
        }

    # ------------------------------------------------------------------
    # 4b. Sector exposure
    # ------------------------------------------------------------------

    def analyze_sector_exposure(
        self, holdings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse sector concentration.

        Parameters
        ----------
        holdings : dict of {ticker: {'weight': float, 'sector': str}}
                   OR dict of {ticker: float} (weight only, sector unknown)
        """
        sector_weights: Dict[str, float] = {}

        for ticker, val in holdings.items():
            if isinstance(val, dict):
                weight = val.get("weight", 0.0)
                sector = val.get("sector", "Unknown")
            else:
                weight = float(val)
                sector = "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        total = sum(sector_weights.values())
        if total > 0:
            sector_pcts = {s: v / total for s, v in sector_weights.items()}
        else:
            sector_pcts = sector_weights

        alerts = []
        for sector, pct in sector_pcts.items():
            if pct > 0.40:
                alerts.append(
                    {
                        "sector": sector,
                        "weight": round(pct, 4),
                        "severity": "HIGH",
                        "message": f"Sector '{sector}' at {pct:.1%} (>40%)",
                    }
                )
            elif pct > 0.25:
                alerts.append(
                    {
                        "sector": sector,
                        "weight": round(pct, 4),
                        "severity": "MEDIUM",
                        "message": f"Sector '{sector}' at {pct:.1%} (>25%)",
                    }
                )

        diversification_score = max(
            0, min(100, int(round((1 - max(sector_pcts.values(), default=0)) * 100)))
        )

        return {
            "sector_weights": {s: round(v, 4) for s, v in sector_pcts.items()},
            "alerts": alerts,
            "diversification_score": diversification_score,
            "num_sectors": len(sector_weights),
        }

    # ------------------------------------------------------------------
    # 4c. Correlation analysis
    # ------------------------------------------------------------------

    def analyze_correlation(
        self, returns_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyse asset correlation and diversification benefit.

        Parameters
        ----------
        returns_df : DataFrame with assets as columns and dates as index
        """
        if returns_df is None or returns_df.empty or returns_df.shape[1] < 2:
            return {
                "correlation_matrix": pd.DataFrame(),
                "avg_correlation": 0.0,
                "highly_correlated_pairs": [],
                "diversification_score": 100,
            }

        corr = returns_df.corr()
        cols = corr.columns.tolist()
        n = len(cols)

        pairs_corr = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_corr.append(corr.iloc[i, j])

        avg_corr = float(np.mean(pairs_corr)) if pairs_corr else 0.0

        high_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                c = corr.iloc[i, j]
                if c > 0.80:
                    high_pairs.append(
                        {
                            "asset1": cols[i],
                            "asset2": cols[j],
                            "correlation": round(float(c), 4),
                            "severity": "HIGH" if c > 0.90 else "MEDIUM",
                        }
                    )

        # diversification_score: higher when avg_corr is low
        div_score = int(round(max(0, min(100, (1 - avg_corr) * 100))))

        return {
            "correlation_matrix": corr,
            "avg_correlation": round(avg_corr, 4),
            "highly_correlated_pairs": high_pairs,
            "diversification_score": div_score,
        }

    # ------------------------------------------------------------------
    # 4d. Drawdown history
    # ------------------------------------------------------------------

    def analyze_drawdown_history(
        self, returns: pd.Series
    ) -> Dict[str, Any]:
        """Return a summary of historical drawdown periods."""
        if returns is None or len(returns) < 2:
            return {
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "num_drawdown_periods": 0,
                "longest_drawdown_days": 0,
                "current_drawdown": 0.0,
            }

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        dd_series = (cumulative - rolling_max) / rolling_max

        max_dd = float(-dd_series.min())
        current_dd = float(-dd_series.iloc[-1])

        # Count distinct drawdown periods (transitions below 0)
        in_dd = dd_series < -1e-8
        transitions = in_dd.astype(int).diff().fillna(0)
        num_periods = int((transitions == 1).sum())

        # Longest consecutive drawdown
        longest = 0
        current = 0
        for v in in_dd:
            if v:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        avg_dd = float(-dd_series[dd_series < -1e-8].mean()) if num_periods else 0.0

        return {
            "max_drawdown": round(max_dd, 4),
            "avg_drawdown": round(avg_dd, 4),
            "num_drawdown_periods": num_periods,
            "longest_drawdown_days": longest,
            "current_drawdown": round(current_dd, 4),
        }

    # ------------------------------------------------------------------
    # 4e. Stress testing
    # ------------------------------------------------------------------

    _STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
        "market_crash": {
            "Technology": -0.40,
            "Financials": -0.45,
            "Consumer Discretionary": -0.35,
            "Energy": -0.30,
            "Healthcare": -0.20,
            "Utilities": -0.15,
            "Consumer Staples": -0.10,
            "Unknown": -0.30,
        },
        "sector_rotation": {
            "Technology": -0.20,
            "Financials": 0.10,
            "Energy": 0.15,
            "Healthcare": 0.05,
            "Consumer Discretionary": -0.10,
            "Consumer Staples": 0.05,
            "Utilities": 0.08,
            "Unknown": -0.05,
        },
        "rate_spike": {
            "Technology": -0.25,
            "Financials": 0.05,
            "Utilities": -0.20,
            "Real Estate": -0.25,
            "Consumer Discretionary": -0.10,
            "Healthcare": -0.05,
            "Energy": 0.10,
            "Unknown": -0.10,
        },
        "recession": {
            "Technology": -0.30,
            "Financials": -0.35,
            "Consumer Discretionary": -0.40,
            "Energy": -0.20,
            "Healthcare": -0.05,
            "Utilities": 0.05,
            "Consumer Staples": 0.00,
            "Unknown": -0.20,
        },
    }

    def stress_test(
        self,
        portfolio: Dict[str, Any],
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Stress test portfolio under different macro scenarios.

        Parameters
        ----------
        portfolio : {ticker: {'weight': float, 'sector': str}} or
                    {ticker: float} (weight only)
        scenarios : list from ['market_crash', 'sector_rotation',
                                'rate_spike', 'recession']
                    (defaults to all four)

        Returns
        -------
        dict of {scenario_name: {'portfolio_loss': float, 'details': [...]}}
        """
        if scenarios is None:
            scenarios = list(self._STRESS_SCENARIOS.keys())

        results: Dict[str, Any] = {}
        for scenario in scenarios:
            if scenario not in self._STRESS_SCENARIOS:
                continue
            shocks = self._STRESS_SCENARIOS[scenario]
            portfolio_return = 0.0
            details = []
            for ticker, val in portfolio.items():
                if isinstance(val, dict):
                    weight = val.get("weight", 0.0)
                    sector = val.get("sector", "Unknown")
                else:
                    weight = float(val)
                    sector = "Unknown"
                shock = shocks.get(sector, shocks.get("Unknown", -0.20))
                position_impact = weight * shock
                portfolio_return += position_impact
                details.append(
                    {
                        "ticker": ticker,
                        "sector": sector,
                        "weight": round(weight, 4),
                        "shock": round(shock, 4),
                        "impact": round(position_impact, 4),
                    }
                )
            results[scenario] = {
                "portfolio_loss": round(portfolio_return, 4),
                "details": details,
            }
        return results


# ===========================================================================
# 5. PortfolioHealthMonitor
# ===========================================================================


class PortfolioHealthMonitor:
    """Monitor portfolio health and generate actionable risk alerts."""

    # Default limits
    DEFAULT_LIMITS: Dict[str, float] = {
        "max_single_position": 0.20,
        "max_sector_weight": 0.40,
        "max_portfolio_volatility": 0.20,
        "max_drawdown": 0.15,
    }

    def __init__(self, limits: Optional[Dict[str, float]] = None):
        self._limits = dict(self.DEFAULT_LIMITS)
        if limits:
            self._limits.update(limits)

    # ------------------------------------------------------------------
    # 5a. Concentration limits
    # ------------------------------------------------------------------

    def check_concentration_limits(
        self,
        portfolio: Dict[str, float],
        limits: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check whether any positions exceed concentration limits.

        Parameters
        ----------
        portfolio : {ticker: weight}
        limits    : {'max_single_position': 0.20, ...}
        """
        effective_limits = dict(self._limits)
        if limits:
            effective_limits.update(limits)
        max_pos = effective_limits.get("max_single_position", 0.20)

        violations = []
        for ticker, weight in portfolio.items():
            if weight > max_pos:
                violations.append(
                    {
                        "ticker": ticker,
                        "weight": round(weight, 4),
                        "limit": max_pos,
                        "excess": round(weight - max_pos, 4),
                    }
                )
        return violations

    # ------------------------------------------------------------------
    # 5b. Volatility limit
    # ------------------------------------------------------------------

    def check_volatility_limits(
        self,
        portfolio: Dict[str, Any],
        max_volatility: Optional[float] = None,
    ) -> bool:
        """
        Return True if portfolio volatility exceeds *max_volatility*.

        Parameters
        ----------
        portfolio    : dict with key 'volatility' (annualised)
        max_volatility : override default limit
        """
        limit = max_volatility if max_volatility is not None else self._limits.get(
            "max_portfolio_volatility", 0.20
        )
        port_vol = portfolio.get("volatility", 0.0) if isinstance(portfolio, dict) else 0.0
        return float(port_vol) > limit

    # ------------------------------------------------------------------
    # 5c. Drawdown limit
    # ------------------------------------------------------------------

    def check_drawdown_limits(
        self,
        returns: pd.Series,
        max_drawdown: Optional[float] = None,
    ) -> bool:
        """Return True if current drawdown exceeds *max_drawdown*."""
        limit = max_drawdown if max_drawdown is not None else self._limits.get(
            "max_drawdown", 0.15
        )
        current_dd = calculate_max_drawdown(returns)
        return current_dd > limit

    # ------------------------------------------------------------------
    # 5d. Alert generation
    # ------------------------------------------------------------------

    def generate_alerts(
        self, portfolio: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable risk alerts for the portfolio.

        Parameters
        ----------
        portfolio : dict with optional keys:
                    - 'holdings'   : {ticker: weight}
                    - 'returns'    : pd.Series of daily returns
                    - 'volatility' : annualised portfolio vol
                    - 'sectors'    : {ticker: sector_name}
        """
        alerts: List[Dict[str, Any]] = []

        holdings: Dict[str, float] = portfolio.get("holdings", {})
        returns: Optional[pd.Series] = portfolio.get("returns")
        port_vol: Optional[float] = portfolio.get("volatility")
        sectors: Dict[str, str] = portfolio.get("sectors", {})

        max_pos = self._limits["max_single_position"]
        max_sector = self._limits["max_sector_weight"]
        max_vol = self._limits["max_portfolio_volatility"]
        max_dd = self._limits["max_drawdown"]

        # Concentration alerts
        for ticker, weight in holdings.items():
            if weight > max_pos:
                excess_pct = int((weight - max_pos) / max_pos * 100)
                alerts.append(
                    {
                        "type": "concentration_warning",
                        "severity": "HIGH" if weight > max_pos * 1.25 else "MEDIUM",
                        "message": (
                            f"{ticker} at {weight:.0%} "
                            f"(limit: {max_pos:.0%})"
                        ),
                        "action": (
                            f"Consider reducing position by "
                            f"{weight - max_pos:.0%}"
                        ),
                    }
                )

        # Sector concentration alerts
        sector_weights: Dict[str, float] = {}
        for ticker, weight in holdings.items():
            sector = sectors.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        for sector, sw in sector_weights.items():
            if sw > max_sector:
                alerts.append(
                    {
                        "type": "sector_concentration",
                        "severity": "HIGH" if sw > max_sector * 1.25 else "MEDIUM",
                        "message": (
                            f"Sector '{sector}' at {sw:.0%} "
                            f"(limit: {max_sector:.0%})"
                        ),
                        "action": (
                            f"Diversify out of {sector} sector; "
                            "consider adding other sectors"
                        ),
                    }
                )

        # Volatility alert
        if port_vol is not None and port_vol > max_vol:
            alerts.append(
                {
                    "type": "volatility_warning",
                    "severity": "HIGH" if port_vol > max_vol * 1.25 else "MEDIUM",
                    "message": (
                        f"Portfolio volatility {port_vol:.1%} exceeds "
                        f"limit {max_vol:.1%}"
                    ),
                    "action": "Add defensive positions or increase cash allocation",
                }
            )

        # Drawdown alert
        if returns is not None and len(returns) >= 2:
            current_dd = calculate_max_drawdown(returns)
            if current_dd > max_dd:
                alerts.append(
                    {
                        "type": "drawdown_warning",
                        "severity": "HIGH" if current_dd > max_dd * 1.5 else "MEDIUM",
                        "message": (
                            f"Max drawdown {current_dd:.1%} exceeds "
                            f"limit {max_dd:.1%}"
                        ),
                        "action": "Review stop-losses and consider hedging",
                    }
                )

        return alerts

    # ------------------------------------------------------------------
    # 5e. Rebalancing suggestions
    # ------------------------------------------------------------------

    def suggest_rebalancing(
        self,
        portfolio: Dict[str, float],
        target_allocation: Dict[str, float],
        min_trade_threshold: float = 0.005,
    ) -> List[Dict[str, Any]]:
        """
        Suggest trades to bring *portfolio* in line with *target_allocation*.

        Parameters
        ----------
        portfolio         : {ticker: current_weight}
        target_allocation : {ticker: target_weight}
        min_trade_threshold : ignore drifts smaller than this fraction
        """
        all_tickers = set(portfolio) | set(target_allocation)
        trades = []
        for ticker in sorted(all_tickers):
            current = portfolio.get(ticker, 0.0)
            target = target_allocation.get(ticker, 0.0)
            drift = target - current
            if abs(drift) >= min_trade_threshold:
                trades.append(
                    {
                        "ticker": ticker,
                        "current_weight": round(current, 4),
                        "target_weight": round(target, 4),
                        "drift": round(drift, 4),
                        "action": "BUY" if drift > 0 else "SELL",
                    }
                )
        return trades

    # ------------------------------------------------------------------
    # 5f. Comprehensive risk report
    # ------------------------------------------------------------------

    def generate_risk_report(
        self, portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.

        Parameters
        ----------
        portfolio : dict with optional keys: holdings, returns, volatility,
                    sectors
        """
        holdings: Dict[str, float] = portfolio.get("holdings", {})
        returns: Optional[pd.Series] = portfolio.get("returns")
        port_vol: Optional[float] = portfolio.get("volatility")

        analyzer = PortfolioRiskAnalyzer()

        # Core metrics
        concentration = analyzer.analyze_concentration(holdings)

        risk_metrics: Dict[str, Any] = {}
        if returns is not None and len(returns) >= 2:
            risk_metrics = {
                "volatility": round(calculate_portfolio_volatility(returns), 4),
                "var_95": round(calculate_value_at_risk(returns), 4),
                "cvar_95": round(calculate_conditional_var(returns), 4),
                "sharpe_ratio": round(calculate_sharpe_ratio(returns), 4),
                "sortino_ratio": round(calculate_sortino_ratio(returns), 4),
                "max_drawdown": round(calculate_max_drawdown(returns), 4),
            }
        elif port_vol is not None:
            risk_metrics = {"volatility": round(port_vol, 4)}

        alerts = self.generate_alerts(portfolio)

        overall_risk_score = _compute_overall_risk_score(
            concentration_score=concentration.get("concentration_score", 0),
            volatility=risk_metrics.get("volatility", port_vol or 0.0),
            max_drawdown=risk_metrics.get("max_drawdown", 0.0),
            num_alerts=len(alerts),
        )

        return {
            "overall_risk_score": overall_risk_score,
            "concentration": concentration,
            "risk_metrics": risk_metrics,
            "alerts": alerts,
            "num_positions": len(holdings),
        }


# ===========================================================================
# Private helpers
# ===========================================================================


def _compute_overall_risk_score(
    concentration_score: float,
    volatility: float,
    max_drawdown: float,
    num_alerts: int,
) -> int:
    """Composite risk score 0-100 (higher = riskier)."""
    vol_score = min(100, int(volatility * 400))  # 25% vol → 100
    dd_score = min(100, int(max_drawdown * 300))  # 33% dd → 100
    alert_score = min(100, num_alerts * 20)
    composite = (
        concentration_score * 0.30
        + vol_score * 0.30
        + dd_score * 0.25
        + alert_score * 0.15
    )
    return max(0, min(100, int(round(composite))))
