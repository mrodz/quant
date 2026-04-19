"""
kmv_timeseries.py — Time-series wrapper for KMVResult
======================================================
Extends kmv.py with KMVTimeSeries: a container for a dated sequence of
KMVResult snapshots that computes investment-grade signals and can feed
directly into kmv_plot.py for visualisation.

Signals computed
----------------
  dd_velocity      : ΔDD / Δday  — rate of change of Distance to Default
  sigma_accel      : Δσ_V / Δday — asset vol acceleration (leads DD)
  edf_change_pct   : percentage change in EDF between adjacent observations
  edf_zscore       : rolling z-score of EDF (how unusual is today's EDF?)
  basis            : observed Z-spread minus EDF-implied fair-value spread
                     (requires z_spreads to be supplied)
  term_structure   : ratio of short-horizon EDF to long-horizon EDF; > 1
                     indicates inversion (near-term stress priced specifically)
  regime           : categorical label — SAFE / WATCH / STRESS / DISTRESS
                     derived from DD thresholds and momentum

Usage
-----
    from kmv import run_kmv, KMVInputs
    from kmv_timeseries import KMVTimeSeries, Snapshot

    snapshots = []
    for dt, eq_val, eq_vol, z_spread in my_data:
        inputs = KMVInputs(equity_value=eq_val, equity_volatility=eq_vol, ...)
        result = run_kmv(inputs)
        snapshots.append(Snapshot(date=dt, result=result, z_spread=z_spread))

    ts = KMVTimeSeries(snapshots)

    print(ts.summary())
    alerts = ts.alerts()
    ts.plot()           # requires kmv_plot.py on the path
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Local imports — kmv.py must be on the path
from quant.kmv import (
    KMVInputs,
    KMVResult,
    run_kmv,
    EMPIRICAL_TABLE,
    edf_from_dd,
)


# ── Regime classification ─────────────────────────────────────────────────────

class Regime(Enum):
    """
    Qualitative credit regime derived from DD level and momentum.

    Thresholds follow the KMV / Moody's Analytics convention:
        DISTRESS  : DD < 1.0  (EDF typically > 7 %)
        STRESS    : DD < 2.0  (EDF typically > 1.2 %)
        WATCH     : DD < 3.5  (EDF typically > 0.1 %)
        SAFE      : DD ≥ 3.5
    """
    SAFE      = "safe"
    WATCH     = "watch"
    STRESS    = "stress"
    DISTRESS  = "distress"

    @staticmethod
    def from_dd(dd: float) -> "Regime":
        if dd < 1.0:
            return Regime.DISTRESS
        if dd < 2.0:
            return Regime.STRESS
        if dd < 3.5:
            return Regime.WATCH
        return Regime.SAFE


# ── Single dated observation ──────────────────────────────────────────────────

@dataclass
class Snapshot:
    """
    One dated KMV observation.

    Parameters
    ----------
    date : date
        Observation date.
    result : KMVResult
        Output of run_kmv() for this date.
    z_spread : float | None
        Observed Z-spread in basis points for the issuer's bond(s) on this
        date.  Required for basis computation; optional otherwise.
    label : str
        Optional free-text tag (e.g. "Q1 2025", "post-earnings").
    """
    date:     date
    result:   KMVResult
    z_spread: Optional[float] = None   # bps
    label:    Optional[str]   = None

    @property
    def dd(self)               -> float: return self.result.dd
    @property
    def edf(self)              -> float: return self.result.edf
    @property
    def edf_pct(self)          -> float: return self.result.edf_pct
    @property
    def asset_value(self)      -> float: return self.result.asset_value
    @property
    def asset_volatility(self) -> float: return self.result.asset_volatility
    @property
    def default_point(self)    -> float: return self.result.default_point
    @property
    def regime(self)           -> Regime: return Regime.from_dd(self.dd)

    def edf_implied_spread_bps(self, lgd: float = 0.60) -> Optional[float]:
        """
        Theoretical fair-value Z-spread implied by EDF:
            s* = EDF × LGD / (1 - EDF)  × 10_000  [bps]

        This is the spread a risk-neutral investor would demand given the
        default probability and an assumed loss given default.

        Parameters
        ----------
        lgd : float
            Loss given default (decimal).  KMV convention ≈ 0.55–0.65.
        """
        edf = self.result.edf
        if edf >= 1.0:
            return None
        spread_decimal = (edf * lgd) / (1.0 - edf)
        return spread_decimal * 10_000   # convert to bps

    def basis_bps(self, lgd: float = 0.60) -> Optional[float]:
        """
        Z-spread basis = observed Z-spread − EDF-implied fair-value spread.

        Positive basis → bond trades cheap to model (potential long opportunity).
        Negative basis → bond trades rich to model (reduce / hedge signal).

        Returns None if z_spread is not supplied.
        """
        if self.z_spread is None:
            return None
        fvs = self.edf_implied_spread_bps(lgd)
        if fvs is None:
            return None
        return self.z_spread - fvs


# ── Derived signal row (one per adjacent pair of snapshots) ──────────────────

@dataclass
class _SignalRow:
    """Internal: computed signals for snapshot[i] relative to snapshot[i-1]."""
    date:              date
    dd:                float
    edf_pct:           float
    asset_volatility:  float
    regime:            Regime
    dd_velocity:       Optional[float]   # Δdd/day
    sigma_accel:       Optional[float]   # Δσ_V/day
    edf_change_pct:    Optional[float]   # % change in EDF
    edf_zscore:        Optional[float]   # rolling z-score
    fv_spread_bps:     Optional[float]   # EDF-implied fair value spread
    basis_bps:         Optional[float]   # observed Z-spread − fv_spread
    z_spread:          Optional[float]   # raw observed Z-spread


# ── Alert definitions ─────────────────────────────────────────────────────────

class AlertLevel(Enum):
    INFO    = "INFO"
    WARNING = "WARNING"
    URGENT  = "URGENT"


@dataclass
class Alert:
    date:        date
    level:       AlertLevel
    signal:      str
    description: str
    value:       float

    def __str__(self) -> str:
        return (
            f"[{self.date}] {self.level.value:8s} | {self.signal:22s} | "
            f"{self.value:+.4f}  — {self.description}"
        )


# ── KMVTimeSeries ─────────────────────────────────────────────────────────────

class KMVTimeSeries:
    """
    Dated sequence of KMVResult snapshots for a single issuer.

    After construction, all derived signals are computed and accessible via
    ``.to_dataframe()``.  Alerts and summary statistics are available via
    ``.alerts()`` and ``.summary()``.

    Parameters
    ----------
    snapshots : sequence of Snapshot
        Must contain at least one entry.  Will be sorted by date ascending.
    lgd : float
        Loss given default used for fair-value spread and basis calculations.
    rolling_window : int
        Window (in observations, not days) for the rolling EDF z-score.
    name : str
        Issuer / instrument label used in display and plot titles.

    Attributes
    ----------
    snapshots : list[Snapshot]
        Sorted snapshots.
    signals : list[_SignalRow]
        One row per snapshot (first row has None for velocity / acceleration).
    """

    def __init__(
        self,
        snapshots: Sequence[Snapshot],
        lgd:            float = 0.60,
        rolling_window: int   = 20,
        name:           str   = "Issuer",
    ) -> None:
        if not snapshots:
            raise ValueError("KMVTimeSeries requires at least one snapshot.")

        self.lgd            = lgd
        self.rolling_window = rolling_window
        self.name           = name

        # Sort by date, warn on duplicates
        self.snapshots: list[Snapshot] = sorted(snapshots, key=lambda s: s.date)
        dates = [s.date for s in self.snapshots]
        if len(dates) != len(set(dates)):
            warnings.warn(
                "KMVTimeSeries: duplicate dates found — later entries will shadow earlier ones in the DataFrame.",
                stacklevel=2,
            )

        self.signals: list[_SignalRow] = self._compute_signals()

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_signals(self) -> list[_SignalRow]:
        snaps = self.snapshots
        n     = len(snaps)

        # Rolling EDF z-score — requires enough history
        edf_pct_vals = np.array([s.edf_pct for s in snaps], dtype=float)
        edf_zscores: list[Optional[float]] = [None] * n
        w = self.rolling_window
        for i in range(w - 1, n):
            window = edf_pct_vals[max(0, i - w + 1): i + 1]
            mu, sigma = window.mean(), window.std(ddof=1)
            edf_zscores[i] = float((edf_pct_vals[i] - mu) / sigma) if sigma > 1e-12 else 0.0

        rows: list[_SignalRow] = []
        for i, snap in enumerate(snaps):
            fvs   = snap.edf_implied_spread_bps(self.lgd)
            basis = snap.basis_bps(self.lgd)

            if i == 0:
                rows.append(_SignalRow(
                    date             = snap.date,
                    dd               = snap.dd,
                    edf_pct          = snap.edf_pct,
                    asset_volatility = snap.asset_volatility,
                    regime           = snap.regime,
                    dd_velocity      = None,
                    sigma_accel      = None,
                    edf_change_pct   = None,
                    edf_zscore       = edf_zscores[0],
                    fv_spread_bps    = fvs,
                    basis_bps        = basis,
                    z_spread         = snap.z_spread,
                ))
                continue

            prev = snaps[i - 1]
            dt   = max((snap.date - prev.date).days, 1)   # guard against zero

            dd_vel     = (snap.dd - prev.dd) / dt
            sig_accel  = (snap.asset_volatility - prev.asset_volatility) / dt
            edf_chg    = (
                ((snap.edf_pct - prev.edf_pct) / prev.edf_pct * 100)
                if prev.edf_pct > 1e-10 else None
            )

            rows.append(_SignalRow(
                date             = snap.date,
                dd               = snap.dd,
                edf_pct          = snap.edf_pct,
                asset_volatility = snap.asset_volatility,
                regime           = snap.regime,
                dd_velocity      = dd_vel,
                sigma_accel      = sig_accel,
                edf_change_pct   = edf_chg,
                edf_zscore       = edf_zscores[i],
                fv_spread_bps    = fvs,
                basis_bps        = basis,
                z_spread         = snap.z_spread,
            ))

        return rows

    # ── Public accessors ──────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return all signals as a tidy DataFrame indexed by date.

        Columns
        -------
        dd, edf_pct, asset_volatility, regime,
        dd_velocity, sigma_accel, edf_change_pct, edf_zscore,
        fv_spread_bps, basis_bps, z_spread
        """
        rows = []
        for r in self.signals:
            rows.append({
                "date":             r.date,
                "dd":               r.dd,
                "edf_pct":          r.edf_pct,
                "asset_volatility": r.asset_volatility,
                "regime":           r.regime.value,
                "dd_velocity":      r.dd_velocity,
                "sigma_accel":      r.sigma_accel,
                "edf_change_pct":   r.edf_change_pct,
                "edf_zscore":       r.edf_zscore,
                "fv_spread_bps":    r.fv_spread_bps,
                "basis_bps":        r.basis_bps,
                "z_spread":         r.z_spread,
            })
        return pd.DataFrame(rows).set_index("date")

    @property
    def latest(self) -> Snapshot:
        """Most recent snapshot."""
        return self.snapshots[-1]

    @property
    def latest_signals(self) -> _SignalRow:
        """Signal row for the most recent snapshot."""
        return self.signals[-1]

    def dd_series(self) -> pd.Series:
        return pd.Series(
            [s.dd for s in self.snapshots],
            index=[s.date for s in self.snapshots],
            name="dd",
        )

    def edf_series(self) -> pd.Series:
        return pd.Series(
            [s.edf_pct for s in self.snapshots],
            index=[s.date for s in self.snapshots],
            name="edf_pct",
        )

    def basis_series(self) -> pd.Series:
        """Z-spread basis (observed − fair-value).  NaN where z_spread is None."""
        return pd.Series(
            [s.basis_bps(self.lgd) for s in self.snapshots],
            index=[s.date for s in self.snapshots],
            name="basis_bps",
        )

    # ── Regime helpers ────────────────────────────────────────────────────────

    def regime_history(self) -> pd.Series:
        """Return regime label at each date."""
        return pd.Series(
            [s.regime.value for s in self.snapshots],
            index=[s.date for s in self.snapshots],
            name="regime",
        )

    def regime_durations(self) -> pd.DataFrame:
        """
        Return a DataFrame of contiguous regime runs with start, end, duration.

        Useful for understanding how long the issuer spent in each regime.
        """
        regimes = self.regime_history()
        runs = []
        prev_r, start = None, None
        for dt, r in regimes.items():
            if r != prev_r:
                if prev_r is not None:
                    runs.append({
                        "regime":   prev_r,
                        "start":    start,
                        "end":      dt,
                        "days":     (dt - start).days,
                        "observations": sum(
                            1 for s in self.snapshots
                            if start <= s.date < dt
                        ),
                    })
                prev_r, start = r, dt
        if prev_r is not None:
            last = self.snapshots[-1].date
            runs.append({
                "regime":   prev_r,
                "start":    start,
                "end":      last,
                "days":     (last - start).days,
                "observations": sum(
                    1 for s in self.snapshots if s.date >= start
                ),
            })
        return pd.DataFrame(runs)

    # ── Alert engine ─────────────────────────────────────────────────────────

    def alerts(
        self,
        dd_velocity_warn:    float = -0.05,   # σ/day
        dd_velocity_urgent:  float = -0.10,
        sigma_accel_warn:    float =  0.002,  # decimal/day
        edf_change_warn:     float =  25.0,   # % change
        edf_zscore_warn:     float =  2.0,
        basis_warn_bps:      float = -50.0,   # bond trading rich to model
        basis_oppty_bps:     float =  80.0,   # bond trading cheap to model
    ) -> list[Alert]:
        """
        Scan the signal history and return a list of investment-relevant alerts.

        Alert types
        -----------
        dd_velocity_drop   : DD falling fast — equity market pricing in risk
                             ahead of bond market.
        sigma_acceleration : Asset vol spiking — leads DD compression.
        edf_spike          : EDF jumped by > edf_change_warn % in one period.
        edf_elevated       : EDF z-score above threshold — unusual for peer group.
        regime_change      : Issuer moved to a worse (or better) regime.
        basis_short        : Bond rich to model — consider reducing / hedging.
        basis_long         : Bond cheap to model — potential entry opportunity.
        """
        result_alerts: list[Alert] = []
        prev_regime: Optional[Regime] = None

        for i, row in enumerate(self.signals):
            snap = self.snapshots[i]

            # ── DD velocity ───────────────────────────────────────────────────
            if row.dd_velocity is not None:
                if row.dd_velocity <= dd_velocity_urgent:
                    result_alerts.append(Alert(
                        date        = row.date,
                        level       = AlertLevel.URGENT,
                        signal      = "dd_velocity_drop",
                        description = (
                            f"DD falling at {row.dd_velocity:.4f} σ/day — "
                            "equity market pricing severe stress ahead of bonds"
                        ),
                        value       = row.dd_velocity,
                    ))
                elif row.dd_velocity <= dd_velocity_warn:
                    result_alerts.append(Alert(
                        date        = row.date,
                        level       = AlertLevel.WARNING,
                        signal      = "dd_velocity_drop",
                        description = (
                            f"DD falling at {row.dd_velocity:.4f} σ/day — "
                            "watch for spread widening"
                        ),
                        value       = row.dd_velocity,
                    ))

            # ── Asset vol acceleration ────────────────────────────────────────
            if row.sigma_accel is not None and row.sigma_accel >= sigma_accel_warn:
                result_alerts.append(Alert(
                    date        = row.date,
                    level       = AlertLevel.WARNING,
                    signal      = "sigma_acceleration",
                    description = (
                        f"Asset vol rising {row.sigma_accel:.5f}/day — "
                        "early warning: DD compression likely to follow"
                    ),
                    value       = row.sigma_accel,
                ))

            # ── EDF spike ────────────────────────────────────────────────────
            if row.edf_change_pct is not None and row.edf_change_pct >= edf_change_warn:
                result_alerts.append(Alert(
                    date        = row.date,
                    level       = AlertLevel.WARNING,
                    signal      = "edf_spike",
                    description = (
                        f"EDF rose {row.edf_change_pct:.1f}% in one period "
                        f"(now {row.edf_pct:.3f}%)"
                    ),
                    value       = row.edf_change_pct,
                ))

            # ── EDF z-score ───────────────────────────────────────────────────
            if row.edf_zscore is not None and row.edf_zscore >= edf_zscore_warn:
                result_alerts.append(Alert(
                    date        = row.date,
                    level       = AlertLevel.WARNING,
                    signal      = "edf_elevated",
                    description = (
                        f"EDF z-score = {row.edf_zscore:.2f} — "
                        "unusually elevated relative to recent history"
                    ),
                    value       = row.edf_zscore,
                ))

            # ── Regime change ────────────────────────────────────────────────
            if prev_regime is not None and snap.regime != prev_regime:
                regimes_ordered = [Regime.SAFE, Regime.WATCH, Regime.STRESS, Regime.DISTRESS]
                worsened = (
                    regimes_ordered.index(snap.regime)
                    > regimes_ordered.index(prev_regime)
                )
                result_alerts.append(Alert(
                    date        = row.date,
                    level       = AlertLevel.URGENT if worsened and snap.regime == Regime.DISTRESS
                                  else AlertLevel.WARNING if worsened
                                  else AlertLevel.INFO,
                    signal      = "regime_change",
                    description = (
                        f"Regime: {prev_regime.value.upper()} → "
                        f"{snap.regime.value.upper()}"
                    ),
                    value       = snap.dd,
                ))

            prev_regime = snap.regime

            # ── Basis signals (require z_spread) ─────────────────────────────
            if row.basis_bps is not None:
                if row.basis_bps <= basis_warn_bps:
                    result_alerts.append(Alert(
                        date        = row.date,
                        level       = AlertLevel.WARNING,
                        signal      = "basis_short",
                        description = (
                            f"Z-spread {row.z_spread:.0f}bps vs model FV "
                            f"{row.fv_spread_bps:.0f}bps — bond trading "
                            f"{abs(row.basis_bps):.0f}bps rich to EDF model "
                            "(consider reducing / hedging)"
                        ),
                        value       = row.basis_bps,
                    ))
                elif row.basis_bps >= basis_oppty_bps:
                    result_alerts.append(Alert(
                        date        = row.date,
                        level       = AlertLevel.INFO,
                        signal      = "basis_long",
                        description = (
                            f"Z-spread {row.z_spread:.0f}bps vs model FV "
                            f"{row.fv_spread_bps:.0f}bps — bond trading "
                            f"{row.basis_bps:.0f}bps cheap to EDF model "
                            "(potential entry opportunity)"
                        ),
                        value       = row.basis_bps,
                    ))

        return result_alerts

    # ── Term structure inversion ───────────────────────────────────────────────

    def term_structure_ratio(
        self,
        short_horizon: float = 0.5,
        long_horizon:  float = 2.0,
        empirical_table: Optional[dict] = None,
    ) -> pd.Series:
        """
        Compute EDF(short_horizon) / EDF(long_horizon) at each date.

        A ratio > 1 indicates term structure inversion — near-term stress
        priced above long-term stress — the highest-urgency signal in the
        KMV toolkit (Moody's Analytics early warning research).

        Both horizons are re-evaluated using the same asset value and
        volatility inferred at each snapshot.
        """
        ratios: dict[date, float] = {}
        table = empirical_table or EMPIRICAL_TABLE

        for snap in self.snapshots:
            r   = snap.result
            inp = r.inputs

            # Re-run DD at short and long horizons without re-solving (reuse V, σ_V)
            def dd_at(T: float) -> float:
                ln_ratio  = math.log(r.asset_value / r.default_point)
                numerator = ln_ratio + (inp.effective_drift - 0.5 * r.asset_volatility**2) * T
                return numerator / (r.asset_volatility * math.sqrt(T))

            dd_short = dd_at(short_horizon)
            dd_long  = dd_at(long_horizon)

            edf_short = edf_from_dd(dd_short, table)
            edf_long  = edf_from_dd(dd_long,  table)

            ratio = edf_short / edf_long if edf_long > 1e-12 else float("inf")
            ratios[snap.date] = ratio

        return pd.Series(ratios, name=f"ts_ratio_{short_horizon}y_{long_horizon}y")

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Print a concise investment summary: current state, trend,
        basis, and key alert count.
        """
        latest  = self.latest
        lsig    = self.latest_signals
        alerts  = self.alerts()
        n_warn  = sum(1 for a in alerts if a.level == AlertLevel.WARNING)
        n_urgent = sum(1 for a in alerts if a.level == AlertLevel.URGENT)
        df      = self.to_dataframe()

        # DD trend: slope of last min(10, n) observations
        dd_window = min(10, len(self.snapshots))
        dd_recent = df["dd"].iloc[-dd_window:]
        dd_slope  = float(np.polyfit(range(dd_window), dd_recent.values, 1)[0])

        basis_str = (
            f"{lsig.basis_bps:+.1f} bps"
            if lsig.basis_bps is not None
            else "N/A (no Z-spread supplied)"
        )

        lines = [
            f"{'═'*60}",
            f"  KMV Time Series  —  {self.name}",
            f"  {len(self.snapshots)} observations  "
            f"{self.snapshots[0].date} → {self.snapshots[-1].date}",
            f"{'─'*60}",
            f"  Latest date       : {latest.date}",
            f"  DD                : {latest.dd:.4f} σ",
            f"  EDF               : {latest.edf_pct:.4f} %",
            f"  Asset vol σ_V     : {latest.asset_volatility*100:.2f} %",
            f"  Regime            : {latest.regime.value.upper()}",
            f"{'─'*60}",
            f"  DD trend (10-obs) : {'↓ falling' if dd_slope < -0.01 else '↑ rising' if dd_slope > 0.01 else '→ stable'}  "
            f"(slope = {dd_slope:.4f} σ/obs)",
            f"  DD velocity       : {lsig.dd_velocity:+.5f} σ/day"
            if lsig.dd_velocity is not None else "  DD velocity       : N/A",
            f"  σ_V acceleration  : {lsig.sigma_accel:+.6f} /day"
            if lsig.sigma_accel is not None else "  σ_V acceleration  : N/A",
            f"  EDF z-score       : {lsig.edf_zscore:+.2f}"
            if lsig.edf_zscore is not None else "  EDF z-score       : N/A",
            f"{'─'*60}",
            f"  FV spread (model) : {lsig.fv_spread_bps:.1f} bps"
            if lsig.fv_spread_bps is not None else "  FV spread         : N/A",
            f"  Observed Z-spread : {latest.z_spread:.1f} bps"
            if latest.z_spread is not None else "  Observed Z-spread : N/A",
            f"  Basis             : {basis_str}",
            f"{'─'*60}",
            f"  Alerts  INFO={sum(1 for a in alerts if a.level==AlertLevel.INFO)}  "
            f"WARNING={n_warn}  URGENT={n_urgent}",
            f"{'═'*60}",
        ]
        return "\n".join(lines)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        title: str | None = None,
        save:  str | None = None,
    ) -> None:
        """
        Four-panel time-series dashboard:
          1. DD over time with regime shading + velocity overlay
          2. EDF (%) — log scale — with z-score colouring
          3. Asset volatility σ_V
          4. Z-spread basis (if z_spreads supplied), else EDF-implied FV spread

        Requires matplotlib (and the same environment as kmv_plot.py).
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError("matplotlib is required for KMVTimeSeries.plot().")

        df    = self.to_dataframe()
        dates = df.index.to_list()
        n     = len(dates)
        title = title or f"KMV Time Series — {self.name}"

        REGIME_COLORS = {
            "safe":     "#D5F5E3",
            "watch":    "#FEF9E7",
            "stress":   "#FDEBD0",
            "distress": "#FADBD8",
        }
        REGIME_BORDERS = {
            "safe":     "#27AE60",
            "watch":    "#F39C12",
            "stress":   "#E67E22",
            "distress": "#C0392B",
        }

        fig = plt.figure(figsize=(14, 10), facecolor="#FAFAFA")
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, color="#1A1A2E")
        gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45,
                                left=0.08, right=0.96, top=0.94, bottom=0.06)

        def _shade_regimes(ax):
            """Shade background by regime for an axis."""
            regime_vals = df["regime"].values
            prev_r, start_idx = regime_vals[0], 0
            for j in range(1, n + 1):
                cur_r = regime_vals[j] if j < n else None
                if cur_r != prev_r:
                    ax.axvspan(
                        dates[start_idx], dates[min(j, n - 1)],
                        color=REGIME_COLORS.get(prev_r, "#FFFFFF"),
                        alpha=0.4, zorder=0,
                    )
                    prev_r, start_idx = cur_r, j

        # ── Panel 1: Distance to Default ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        _shade_regimes(ax1)
        ax1.plot(dates, df["dd"], color="#2C3E50", linewidth=1.8, zorder=3)
        ax1.axhline(3.5, color="#27AE60", linewidth=0.8, linestyle=":", alpha=0.7, label="Safe (3.5σ)")
        ax1.axhline(2.0, color="#E67E22", linewidth=0.8, linestyle=":", alpha=0.7, label="Stress (2.0σ)")
        ax1.axhline(1.0, color="#C0392B", linewidth=0.8, linestyle=":", alpha=0.7, label="Distress (1.0σ)")

        if df["dd_velocity"].notna().any():
            ax1v = ax1.twinx()
            vel  = df["dd_velocity"].fillna(0)
            ax1v.bar(dates, vel, color=["#E74C3C" if v < 0 else "#27AE60" for v in vel],
                     alpha=0.35, width=0.8, zorder=2)
            ax1v.axhline(0, color="#888", linewidth=0.5)
            ax1v.set_ylabel("DD velocity (σ/day)", fontsize=8, color="#888")
            ax1v.tick_params(labelsize=7)

        ax1.set_ylabel("Distance to Default (σ)", fontsize=9, color="#555")
        ax1.set_title("Distance to Default  |  regime shading  |  velocity bars", fontsize=9, pad=4)
        ax1.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax1.tick_params(labelsize=8)
        ax1.spines[["top", "right"]].set_visible(False)

        # ── Panel 2: EDF (log scale) ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        _shade_regimes(ax2)
        edf_vals = df["edf_pct"].clip(lower=1e-4)

        # colour line by z-score if available
        if df["edf_zscore"].notna().any():
            zs = df["edf_zscore"].fillna(0).values
            for j in range(len(dates) - 1):
                col = "#C0392B" if zs[j] > 2 else "#E67E22" if zs[j] > 1 else "#2980B9"
                ax2.semilogy(dates[j:j+2], edf_vals.values[j:j+2], color=col, linewidth=1.8)
        else:
            ax2.semilogy(dates, edf_vals, color="#2980B9", linewidth=1.8)

        ax2.set_ylabel("EDF (%,  log scale)", fontsize=9, color="#555")
        ax2.set_title("Expected Default Frequency  |  colour = EDF z-score  (red > 2σ)", fontsize=9, pad=4)
        ax2.tick_params(labelsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

        # ── Panel 3: Asset volatility ─────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        _shade_regimes(ax3)
        ax3.plot(dates, df["asset_volatility"] * 100, color="#8E44AD", linewidth=1.6)
        if df["sigma_accel"].notna().any():
            ax3v = ax3.twinx()
            accel = df["sigma_accel"].fillna(0) * 1000   # scale to ×10⁻³ for readability
            ax3v.bar(dates, accel,
                     color=["#8E44AD" if v > 0 else "#2ECC71" for v in accel],
                     alpha=0.3, width=0.8)
            ax3v.set_ylabel("σ_V accel (×10⁻³/day)", fontsize=8, color="#888")
            ax3v.tick_params(labelsize=7)
        ax3.set_ylabel("Asset vol σ_V (%)", fontsize=9, color="#555")
        ax3.set_title("Asset volatility  |  acceleration bars", fontsize=9, pad=4)
        ax3.tick_params(labelsize=8)
        ax3.spines[["top", "right"]].set_visible(False)

        # ── Panel 4: Basis or FV spread ───────────────────────────────────────
        ax4 = fig.add_subplot(gs[3])
        has_basis = df["basis_bps"].notna().any()
        if has_basis:
            basis = df["basis_bps"]
            ax4.bar(dates, basis,
                    color=["#E74C3C" if v < 0 else "#27AE60" for v in basis],
                    alpha=0.7, width=0.8)
            ax4.axhline(0, color="#333", linewidth=0.8)
            ax4.axhline(-50, color="#E74C3C", linewidth=0.8, linestyle="--",
                        alpha=0.6, label="Short signal (−50bps)")
            ax4.axhline(80,  color="#27AE60", linewidth=0.8, linestyle="--",
                        alpha=0.6, label="Long oppty (+80bps)")
            ax4.set_ylabel("Basis (bps)", fontsize=9, color="#555")
            ax4.set_title("Z-spread basis  =  observed − EDF fair-value  "
                          "|  green = cheap (long), red = rich (short/hedge)", fontsize=9, pad=4)
            ax4.legend(fontsize=7, framealpha=0.6)
        else:
            if df["fv_spread_bps"].notna().any():
                ax4.plot(dates, df["fv_spread_bps"], color="#1ABC9C", linewidth=1.6)
                ax4.set_ylabel("FV spread (bps)", fontsize=9, color="#555")
                ax4.set_title("EDF-implied fair-value spread", fontsize=9, pad=4)
            else:
                ax4.text(0.5, 0.5, "No Z-spread or FV spread data",
                         transform=ax4.transAxes, ha="center", va="center",
                         fontsize=10, color="#888")

        ax4.tick_params(labelsize=8)
        ax4.spines[["top", "right"]].set_visible(False)

        # ── Regime legend ─────────────────────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color=REGIME_COLORS[r], edgecolor=REGIME_BORDERS[r],
                           linewidth=0.8, label=r.capitalize())
            for r in ["safe", "watch", "stress", "distress"]
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4,
                   fontsize=8, framealpha=0.7, bbox_to_anchor=(0.5, 0.01))

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Saved → {save}")

        plt.show()

    def __len__(self) -> int:
        return len(self.snapshots)

    def __repr__(self) -> str:
        return (
            f"KMVTimeSeries(name={self.name!r}, "
            f"n={len(self)}, "
            f"dates={self.snapshots[0].date}→{self.snapshots[-1].date})"
        )


# ── Factory helpers ────────────────────────────────────────────────────────────

def build_from_panel(
    panel: pd.DataFrame,
    base_inputs: KMVInputs,
    name:        str = "Issuer",
    lgd:         float = 0.60,
    empirical_table: Optional[dict] = None,
) -> KMVTimeSeries:
    """
    Build a KMVTimeSeries from a DataFrame where each row is one date.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain these columns (all other columns are ignored):
          equity_value      : market cap
          equity_volatility : annualised σ_E (decimal)
        Optional columns:
          short_term_debt   : overrides base_inputs if present
          long_term_debt    : overrides base_inputs if present
          z_spread          : observed Z-spread in bps
          label             : free-text tag for each row

        Index must be date-like (date / datetime / str parseable by pd.to_datetime).

    base_inputs : KMVInputs
        Fallback for any field not present in the panel.  Debt figures in
        base_inputs are used when the panel does not supply them (e.g. when
        debt is updated quarterly but equity daily).

    name : str
        Issuer label.

    lgd : float
        Loss given default for fair-value spread / basis computation.

    empirical_table : dict, optional
        DD → EDF mapping.  Defaults to the module-level EMPIRICAL_TABLE.

    Returns
    -------
    KMVTimeSeries
    """
    required = {"equity_value", "equity_volatility"}
    missing  = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    table = empirical_table or EMPIRICAL_TABLE
    snapshots: list[Snapshot] = []

    for idx, row in panel.iterrows():
        obs_date = pd.Timestamp(idx).date() if not isinstance(idx, date) else idx

        inputs = KMVInputs(
            equity_value      = float(row["equity_value"]),
            equity_volatility = float(row["equity_volatility"]),
            short_term_debt   = float(row.get("short_term_debt",  base_inputs.short_term_debt)),
            long_term_debt    = float(row.get("long_term_debt",   base_inputs.long_term_debt)),
            risk_free_rate    = float(row.get("risk_free_rate",   base_inputs.risk_free_rate)),
            horizon           = float(row.get("horizon",          base_inputs.horizon)),
            drift             = float(row["drift"]) if "drift" in row and pd.notna(row.get("drift")) else base_inputs.drift,
        )

        result   = run_kmv(inputs, empirical_table=table)
        z_spread = float(row["z_spread"]) if "z_spread" in row and pd.notna(row.get("z_spread")) else None
        label    = str(row["label"])      if "label"    in row and pd.notna(row.get("label"))    else None

        snapshots.append(Snapshot(date=obs_date, result=result, z_spread=z_spread, label=label))

    return KMVTimeSeries(snapshots, lgd=lgd, name=name)
