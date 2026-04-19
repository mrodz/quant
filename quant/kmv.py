"""
kmv.py — KMV / Expected Default Frequency model
================================================
Implements the Kealhofer-McQuown-Vasicek adaptation of Merton's structural
credit model. Integrates with BondL1 / BondL2 from bonds.py.

Core steps
----------
1.  Infer asset value (V) and asset volatility (σ_V) from observable equity
    market data via the Merton option-pricing relationships (iterative solve).
2.  Compute the KMV default point:  DP = STD + 0.5 × LTD
3.  Compute Distance to Default:    DD = (E[V_T] - DP) / (V × σ_V × √T)
4.  Map DD → Expected Default Frequency via an empirical CDF (or N(-DD) as
    a theoretical proxy when no empirical table is available).
"""

from __future__ import annotations

from enum import Enum

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Callable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import pandas as pd
import datetime

logger = logging.getLogger(__name__)

# ── Input / Output dataclasses ────────────────────────────────────────────────

EMPIRICAL_TABLE = {
    0.0: 0.180,
    0.5: 0.130,
    1.0: 0.070,
    1.5: 0.030,
    2.0: 0.012,
    2.5: 0.005,
    3.0: 0.002,
    3.5: 0.001,
    4.0: 0.0004,
}

@dataclass
class KMVInputs:
    """
    All observable inputs required to run the KMV model for one firm.

    Parameters
    ----------
    equity_value : float
        Current market capitalisation (E), in the same currency units as debt.
    equity_volatility : float
        Annualised equity volatility (σ_E), as a decimal (e.g. 0.35 for 35 %).
    short_term_debt : float
        Book value of debt maturing within one year (STD).
    long_term_debt : float
        Book value of debt maturing beyond one year (LTD).
    risk_free_rate : float
        Continuously compounded risk-free rate, annualised (e.g. 0.05).
    horizon : float
        Credit horizon in years (default 1.0).
    drift : float | None
        Expected annual asset drift μ.  If None, the risk-free rate is used
        (risk-neutral measure); supply the actual asset drift for a real-world
        EDF estimate.
    """
    equity_value:     float
    equity_volatility: float
    short_term_debt:  float
    long_term_debt:   float
    risk_free_rate:   float
    horizon:          float = 1.0
    drift:            Optional[float] = None   # defaults to risk_free_rate

    @property
    def default_point(self) -> float:
        """KMV default point: STD + 50 % of LTD."""
        return self.short_term_debt + 0.5 * self.long_term_debt

    @property
    def total_debt(self) -> float:
        return self.short_term_debt + self.long_term_debt

    @property
    def effective_drift(self) -> float:
        return self.drift if self.drift is not None else self.risk_free_rate


@dataclass
class KMVResult:
    """Output of a KMV solve."""
    # Inferred asset quantities
    asset_value:      float          # V₀
    asset_volatility: float          # σ_V (annualised)

    # Default point
    default_point:    float          # DP = STD + 0.5 × LTD

    # Distance to Default
    dd:               float          # DD in σ-units

    # Expected Default Frequency
    edf:              float          # probability of default in [0, 1]
    edf_pct:          float          # EDF expressed as percentage

    # Inputs echo
    inputs:           KMVInputs

    # Diagnostics
    iterations:       int  = 0
    converged:        bool = True

    def __str__(self) -> str:
        return (
            f"KMV Result\n"
            f"  Asset value       : {self.asset_value:,.2f}\n"
            f"  Asset volatility  : {self.asset_volatility:.4f}  ({self.asset_volatility*100:.2f} %)\n"
            f"  Default point     : {self.default_point:,.2f}  "
            f"(STD {self.inputs.short_term_debt:,.0f} + .5*LTD {0.5*self.inputs.long_term_debt:,.0f})\n"
            f"  Distance to Default: {self.dd:.4f} σ\n"
            f"  EDF               : {self.edf_pct:.4f} %\n"
            f"  Converged         : {self.converged}  ({self.iterations} iters)\n"
        )


# ── Merton helper functions ───────────────────────────────────────────────────

def _d1(V: float, F: float, r: float, sigma: float, T: float) -> float:
    """Merton d₁ (asset value, face value of debt, rate, vol, horizon)."""
    return (math.log(V / F) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(V: float, F: float, r: float, sigma: float, T: float) -> float:
    return _d1(V, F, r, sigma, T) - sigma * math.sqrt(T)


def merton_equity(V: float, F: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes / Merton equity value (call on firm assets).

    E = V·N(d₁) - F·e^{-rT}·N(d₂)
    """
    d1 = _d1(V, F, r, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    return V * norm.cdf(d1) - F * math.exp(-r * T) * norm.cdf(d2)


def merton_equity_delta(V: float, F: float, r: float, sigma: float, T: float) -> float:
    """∂E/∂V = N(d₁) — used in the σ_E ↔ σ_V relationship."""
    return norm.cdf(_d1(V, F, r, sigma, T))


# ── Iterative solve ───────────────────────────────────────────────────────────

def _solve_asset_value_and_vol(
    inputs: KMVInputs,
    tol: float = 1e-8,
    max_iter: int = 1_000,
) -> tuple[float, float, int, bool]:
    """
    Solve for (V, σ_V) given observable (E, σ_E) using the two Merton
    equations:

        E  = V·N(d₁) - F·e^{-rT}·N(d₂)          ... [1]
        σ_E = (V / E) · N(d₁) · σ_V               ... [2]

    Strategy: iterate on V using equation [1] with σ_V updated from [2].
    Converges reliably for a wide range of capital structures.

    Returns (V, σ_V, n_iters, converged).
    """
    E   = inputs.equity_value
    sE  = inputs.equity_volatility
    F   = inputs.default_point           # KMV uses the default point, not total debt
    r   = inputs.risk_free_rate
    T   = inputs.horizon

    # Initial guesses (Ronn-Verma seed)
    V      = E + F
    sigma  = sE * E / V

    for i in range(max_iter):
        d1    = _d1(V, F, r, sigma, T)
        nd1   = norm.cdf(d1)
        d2    = d1 - sigma * math.sqrt(T)

        # Update σ_V from [2]
        sigma_new = sE * E / (V * nd1) if nd1 > 1e-12 else sigma

        # Update V from [1] by inverting the Merton call
        # Solve: merton_equity(V_new, F, r, sigma_new, T) = E
        try:
            lo, hi = F * 1e-4, F * 1e4
            V_new = brentq(
                lambda v: merton_equity(v, F, r, sigma_new, T) - E,
                lo, hi, xtol=tol * 1e-2, maxiter=200,
            )
        except ValueError:
            # Bracket failed — fall back to a Newton step on V
            delta = merton_equity_delta(V, F, r, sigma, T)
            V_new = V - (merton_equity(V, F, r, sigma, T) - E) / max(delta, 1e-12)

        if abs(V_new - V) < tol * V and abs(sigma_new - sigma) < tol:
            return V_new, sigma_new, i + 1, True

        V, sigma = V_new, sigma_new

    logger.warning("KMV solver did not converge after %d iterations.", max_iter)
    return V, sigma, max_iter, False


# ── Distance to Default ───────────────────────────────────────────────────────

def distance_to_default(
    V: float,
    sigma_V: float,
    default_point: float,
    drift: float,
    T: float,
) -> float:
    """
    KMV Distance to Default under the real-world measure:

        DD = (ln(V/DP) + (μ - ½σ²)T) / (σ_V · √T)

    where μ is the asset drift (use r for risk-neutral; use empirical asset
    return for real-world EDF).
    """
    if default_point <= 0 or V <= 0 or sigma_V <= 0 or T <= 0:
        raise ValueError("V, sigma_V, default_point and T must all be positive.")
    ln_ratio = math.log(V / default_point)
    numerator = ln_ratio + (drift - 0.5 * sigma_V ** 2) * T
    denominator = sigma_V * math.sqrt(T)
    return numerator / denominator


# ── EDF mapping ───────────────────────────────────────────────────────────────

def edf_from_dd(
    dd: float,
    empirical_table: Optional[dict[float, float]] = None,
) -> float:
    """
    Map a Distance-to-Default value to an Expected Default Frequency.

    Parameters
    ----------
    dd : float
        Distance to Default in σ-units.
    empirical_table : dict {dd_value → default_probability}, optional
        A monotonically decreasing mapping calibrated to historical default
        data (the Moody's KMV approach).  Values are linearly interpolated.
        If None, the theoretical Gaussian proxy N(-DD) is used.

    Returns
    -------
    float in [0, 1]
    """
    if empirical_table is None:
        # Theoretical proxy: risk-neutral default probability
        return float(norm.cdf(-dd))

    keys   = sorted(empirical_table.keys())
    values = [empirical_table[k] for k in keys]

    if dd <= keys[0]:
        return values[0]
    if dd >= keys[-1]:
        return values[-1]

    # Linear interpolation
    for i in range(len(keys) - 1):
        if keys[i] <= dd <= keys[i + 1]:
            t = (dd - keys[i]) / (keys[i + 1] - keys[i])
            return values[i] + t * (values[i + 1] - values[i])

    return float(norm.cdf(-dd))   # fallback


# ── Public API ────────────────────────────────────────────────────────────────

def run_kmv(
    inputs: KMVInputs,
    empirical_table: Optional[dict[float, float]] = None,
    tol: float = 1e-8,
    max_iter: int = 1_000,
) -> KMVResult:
    """
    Full KMV pipeline: solve for asset value/vol, compute DD, map to EDF.

    Parameters
    ----------
    inputs : KMVInputs
    empirical_table : dict, optional
        Historical DD → default probability table.  See `edf_from_dd`.
    tol : float
        Convergence tolerance for the Merton solver.
    max_iter : int
        Maximum solver iterations.

    Returns
    -------
    KMVResult
    """
    V, sigma_V, n_iter, converged = _solve_asset_value_and_vol(
        inputs, tol=tol, max_iter=max_iter
    )

    dp = inputs.default_point

    dd = distance_to_default(
        V=V,
        sigma_V=sigma_V,
        default_point=dp,
        drift=inputs.effective_drift,
        T=inputs.horizon,
    )

    edf = edf_from_dd(dd, empirical_table)

    return KMVResult(
        asset_value=V,
        asset_volatility=sigma_V,
        default_point=dp,
        dd=dd,
        edf=edf,
        edf_pct=edf * 100,
        inputs=inputs,
        iterations=n_iter,
        converged=converged,
    )


# ── Integration helpers (BondL2 → KMVInputs) ─────────────────────────────────

@dataclass
class _DebtBucket:
    std: float = 0.0
    ltd: float = 0.0
    skipped_no_data: int = 0   # bonds dropped (zero / missing outstanding)
    skipped_matured: int = 0
    
    @property
    def skipped(self) -> int:
        return self.skipped_no_data + self.skipped_matured


class _SkipReason(Enum):
    NO_OUTSTANDING = "no_outstanding"
    MATURED        = "matured"


def _classify_bond(
    bond,
    today: date,
    cutoff_years: float,
) -> tuple[float, float] | _SkipReason:
    """
    Return (std_contribution, ltd_contribution) for a single bond,
    or a _SkipReason if the bond should be excluded.

    No maturity  → conservative LTD.
    Past maturity → SkipReason.MATURED.
    Zero / missing outstanding → SkipReason.NO_OUTSTANDING.
    """
    outstanding = float(getattr(bond, "amount_outstanding", None) or 0.0)
    if outstanding <= 0.0:
        return _SkipReason.NO_OUTSTANDING

    maturity = bond.maturity() if callable(getattr(bond, "maturity", None)) else None
    if maturity is None:
        return 0.0, outstanding

    if isinstance(maturity, pd.Timestamp):
        maturity = maturity.date()
    elif isinstance(maturity, datetime):
        maturity = maturity.date()
    elif isinstance(maturity, date):
        pass
    else:
        raise TypeError(f"Cannot convert {type(maturity)} to date")

    years_to_mat = (maturity - today).days / 365.25

    if years_to_mat < 0.0:
        return _SkipReason.MATURED

    return (outstanding, 0.0) if years_to_mat <= cutoff_years else (0.0, outstanding)

def _aggregate_debt(
    bonds: list,
    short_term_cutoff_years: float = 1.0,
) -> _DebtBucket:
    today  = date.today()
    bucket = _DebtBucket()
    for bond in bonds:
        result = _classify_bond(bond, today, short_term_cutoff_years)
        if result is _SkipReason.NO_OUTSTANDING:
            bucket.skipped_no_data += 1
        elif result is _SkipReason.MATURED:
            bucket.skipped_matured += 1
        else:
            std, ltd = result
            bucket.std += std
            bucket.ltd += ltd
    return bucket


def kmv_inputs_from_bonds(
    equity_value: float,
    equity_volatility: float,
    bonds: list,                    # list[BondL2] — all issuances for one firm
    risk_free_rate: float,
    horizon: float = 1.0,
    drift: Optional[float] = None,
    short_term_cutoff_years: float = 1.0,
    group_by: str = "business_entity",
) -> KMVInputs:
    """
    Build KMVInputs by aggregating debt across **all bonds of one issuer**.

    This is the correct firm-level entry point.  Pass every `BondL2` (or
    compatible object) that belongs to the same issuer — identified by
    `business_entity` or `perm_id` — and the function will:

    1. Classify each bond as short-term (maturity ≤ `short_term_cutoff_years`)
       or long-term.
    2. Sum outstanding amounts into the two buckets.
    3. Return a `KMVInputs` ready for `run_kmv`.

    Parameters
    ----------
    equity_value : float
        Market cap of the issuing firm (from an equities feed).
    equity_volatility : float
        Annualised equity volatility (σ_E).
    bonds : list
        All bond issuances for this firm.  If bonds from multiple issuers
        are accidentally mixed in, pass `group_by` to trigger a consistency
        check (a `ValueError` is raised if more than one issuer is found).
    risk_free_rate : float
    horizon : float
        Credit horizon in years (default 1.0).
    drift : float | None
        Real-world asset drift.  None → use risk_free_rate (risk-neutral).
    short_term_cutoff_years : float
        Maturity threshold that separates STD from LTD (default 1.0 year).
    group_by : str
        Attribute used for the single-issuer consistency check.
        Pass `group_by=""` to skip the check entirely.

    Returns
    -------
    KMVInputs

    Raises
    ------
    ValueError
        If `bonds` is empty, or if multiple distinct issuers are detected.
    """
    if not bonds:
        raise ValueError("bonds list is empty — cannot build KMVInputs.")

    # Consistency check: all bonds should belong to the same issuer
    if group_by:
        issuers = {getattr(b, group_by, None) for b in bonds}
        issuers.discard(None)
        if len(issuers) > 1:
            raise ValueError(
                f"bonds contain {len(issuers)} distinct '{group_by}' values: "
                f"{issuers!r}.  Aggregate per-issuer before calling this function."
            )

    bucket = _aggregate_debt(bonds, short_term_cutoff_years)

    if bucket.skipped:
        logger.warning(
            "%d bond(s) had zero / missing amount_outstanding and were excluded (no data = %d, matured = %d).",
            bucket.skipped,
            bucket.skipped_no_data,
            bucket.skipped_matured
        )

    logger.info(
        "Debt aggregation: STD=%.2f  LTD=%.2f  (%d bonds, %d skipped)",
        bucket.std, bucket.ltd, len(bonds), bucket.skipped,
    )

    return KMVInputs(
        equity_value=equity_value,
        equity_volatility=equity_volatility,
        short_term_debt=bucket.std,
        long_term_debt=bucket.ltd,
        risk_free_rate=risk_free_rate,
        horizon=horizon,
        drift=drift,
    )


def run_kmv_for_issuer(
    equity_value: float,
    equity_volatility: float,
    bonds: list,                    # all BondL2 for one issuer
    risk_free_rate: float,
    horizon: float = 1.0,
    drift: Optional[float] = None,
    short_term_cutoff_years: float = 1.0,
    empirical_table: Optional[dict[float, float]] = None,
    group_by: str = "business_entity",
) -> KMVResult:
    """
    Convenience wrapper: aggregate bonds → build inputs → run KMV in one call.

    Example
    -------
    >>> bonds = client.upgrade_l1_bond(client.list_securities("AAPL"))
    >>> result = run_kmv_for_issuer(
    ...     equity_value=3_000_000,   # $3 tn market cap, same units as AMT_OS
    ...     equity_volatility=0.25,
    ...     bonds=bonds,
    ...     risk_free_rate=0.05,
    ... )
    >>> print(result)
    """
    inputs = kmv_inputs_from_bonds(
        equity_value=equity_value,
        equity_volatility=equity_volatility,
        bonds=bonds,
        risk_free_rate=risk_free_rate,
        horizon=horizon,
        drift=drift,
        short_term_cutoff_years=short_term_cutoff_years,
        group_by=group_by,
    )
    return run_kmv(inputs, empirical_table=empirical_table)
    