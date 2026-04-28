"""
07_validate.py — Three post-backtest validation checks
  1. Lookahead bias check
  2. Signal decay by year
  3. Transaction cost sensitivity
Input:  targets_df.parquet
        results/signal_backtest.csv
Output: results/validation_report.txt
        results/signal_decay.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TARGETS_PATH   = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "targets_df.parquet"
BACKTEST_PATH  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "results" / "signal_backtest.csv"
RESULTS_DIR    = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "results"
SPREAD_COL     = "Option Adjusted Spread Bid"
HORIZON        = 21

COST_SCENARIOS = [0, 5, 10, 25]   # bps round-trip per position

lines = []   # collected output for the report file

def log(s=""):
    print(s)
    lines.append(s)


# ──────────────────────────────────────────────────────────────
# CHECK 1: Lookahead bias
# ──────────────────────────────────────────────────────────────
def check_lookahead(df: pd.DataFrame):
    log("=" * 60)
    log("CHECK 1: LOOKAHEAD BIAS")
    log("=" * 60)

    ric = df["bond_ric"].iloc[0]
    sample = (
        df[df["bond_ric"] == ric]
        .sort_values("Date")
        .head(40)
        [["Date", SPREAD_COL, "fwd_spread_chg"]]
        .copy()
    )

    # Reconstruct what fwd_spread_chg should be
    sample["expected_fwd"] = (
        df[df["bond_ric"] == ric]
        .sort_values("Date")[SPREAD_COL]
        .shift(-HORIZON)
        .values[:40]
        - sample[SPREAD_COL].values
    )
    sample["match"] = np.isclose(
        sample["fwd_spread_chg"], sample["expected_fwd"], atol=0.01, equal_nan=True
    )

    n_valid   = sample["match"].sum()
    n_total   = sample["match"].notna().sum()
    log(f"\n  RIC sampled: {ric}")
    log(f"  Rows checked: {n_total}  |  Correctly forward-looking: {n_valid}")

    if n_valid == n_total:
        log("  PASS — fwd_spread_chg matches spread[t+21] - spread[t] on all rows.")
    else:
        log("  FAIL — mismatch detected. Check 03_targets.py shift logic.")
        log("\n  Sample (first 10 rows):")
        log(sample.head(10).to_string(index=False))

    # Also check: target should have zero correlation with *past* spread changes
    df["past_chg_21d"] = df.groupby("bond_ric")[SPREAD_COL].transform(
        lambda x: x.diff(HORIZON)
    )
    corr = df[["fwd_spread_chg", "past_chg_21d"]].dropna().corr().iloc[0, 1]
    log(f"\n  Correlation(fwd_spread_chg, past_21d_chg): {corr:.4f}")
    if abs(corr) < 0.10:
        log("  PASS — negligible autocorrelation, no obvious lookahead.")
    else:
        log(f"  WARNING — correlation {corr:.3f} is higher than expected.")


# ──────────────────────────────────────────────────────────────
# CHECK 2: Signal decay by year
# ──────────────────────────────────────────────────────────────
def check_signal_decay(backtest: pd.DataFrame):
    log("\n" + "=" * 60)
    log("CHECK 2: SIGNAL DECAY BY YEAR")
    log("=" * 60)

    backtest["year"] = backtest["Date"].dt.year

    yearly_rows = []
    for label, grp in backtest.groupby("label"):
        for year, ygrp in grp.groupby("year"):
            pnl = ygrp["pnl_bps"].dropna()
            if len(pnl) < 20:
                continue
            sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan
            yearly_rows.append({
                "label": label, "year": year,
                "sharpe": round(sharpe, 2),
                "total_pnl": round(pnl.sum(), 1),
                "hit_rate": round((pnl > 0).mean(), 3),
                "n_days": len(pnl),
            })

    yearly = pd.DataFrame(yearly_rows)
    log("\n" + yearly.to_string(index=False))

    # Flag if any label has >50% of total PnL in a single year
    log("\n  PnL concentration check:")
    for label, grp in backtest.groupby("label"):
        total = grp["pnl_bps"].sum()
        by_year = grp.groupby("year")["pnl_bps"].sum()
        max_yr  = by_year.abs().idxmax()
        max_pct = abs(by_year[max_yr] / total) * 100 if total != 0 else 0
        flag = "  <<< CONCENTRATED" if max_pct > 60 else ""
        log(f"    {label:<12}  max year: {max_yr}  ({max_pct:.0f}% of total PnL){flag}")

    return yearly


def plot_signal_decay(backtest: pd.DataFrame, yearly: pd.DataFrame):
    labels = backtest["label"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        grp = backtest[backtest["label"] == label].copy()
        grp["cum_pnl"] = grp["pnl_bps"].cumsum()

        # Shade by year
        years = sorted(grp["Date"].dt.year.unique())
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(years)))
        for yr, col in zip(years, colors):
            mask = grp["Date"].dt.year == yr
            ax.axvspan(grp[mask]["Date"].min(), grp[mask]["Date"].max(),
                       alpha=0.08, color=col)

        ax.plot(grp["Date"], grp["cum_pnl"], lw=1.5, color="#1a5fa8")
        ax.fill_between(grp["Date"],
                        grp["cum_pnl"] - grp["cum_pnl"].cummax(),
                        0, alpha=0.2, color="red")
        ax.axhline(0, color="gray", lw=0.5, linestyle="--")

        # Annotate yearly Sharpe
        yr_data = yearly[yearly["label"] == label]
        for _, row in yr_data.iterrows():
            yr_mid = grp[grp["Date"].dt.year == row["year"]]["Date"].median()
            if pd.notna(yr_mid):
                ypos = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -100
                ax.text(yr_mid, ax.get_ylim()[0],
                        f"SR={row['sharpe']}", fontsize=7,
                        ha="center", va="bottom", color="#333")

        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Cumulative bps")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Signal Decay — Cumulative PnL by Year (shaded)", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "signal_decay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\n  Saved → {out}")


# ──────────────────────────────────────────────────────────────
# CHECK 3: Transaction cost sensitivity
# ──────────────────────────────────────────────────────────────
def check_costs(backtest: pd.DataFrame):
    log("\n" + "=" * 60)
    log("CHECK 3: TRANSACTION COST SENSITIVITY")
    log("=" * 60)
    log(f"\n  Assuming cost is deducted per day (round-trip bps per position)\n")

    rows = []
    for label, grp in backtest.groupby("label"):
        base_total  = grp["pnl_bps"].sum()
        base_sharpe = (grp["pnl_bps"].mean() / grp["pnl_bps"].std() * np.sqrt(252)
                       if grp["pnl_bps"].std() > 0 else np.nan)
        row = {"label": label,
               "gross_sharpe": round(base_sharpe, 3),
               "gross_pnl": round(base_total, 1)}
        for cost in COST_SCENARIOS:
            adj = grp["pnl_bps"] - cost
            s   = adj.mean() / adj.std() * np.sqrt(252) if adj.std() > 0 else np.nan
            row[f"sharpe_{cost}bps"] = round(s, 3)
            row[f"pnl_{cost}bps"]    = round(adj.sum(), 1)
        rows.append(row)

    cost_df = pd.DataFrame(rows)

    # Print Sharpe table
    sharpe_cols = ["label", "gross_sharpe"] + [f"sharpe_{c}bps" for c in COST_SCENARIOS]
    log("  Sharpe after costs:")
    log(cost_df[sharpe_cols].to_string(index=False))

    # Print PnL table
    pnl_cols = ["label", "gross_pnl"] + [f"pnl_{c}bps" for c in COST_SCENARIOS]
    log("\n  Total PnL (bps) after costs:")
    log(cost_df[pnl_cols].to_string(index=False))

    # Break-even cost
    log("\n  Break-even daily cost (Sharpe → 0):")
    for label, grp in backtest.groupby("label"):
        mean_pnl = grp["pnl_bps"].mean()
        log(f"    {label:<12}  ~{mean_pnl:.2f} bps/day  "
            f"({'positive' if mean_pnl > 0 else 'negative'} edge)")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df       = pd.read_parquet(TARGETS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    backtest = pd.read_csv(BACKTEST_PATH, parse_dates=["Date"])

    check_lookahead(df)
    yearly = check_signal_decay(backtest)
    plot_signal_decay(backtest, yearly)
    check_costs(backtest)

    # Save report
    report_path = RESULTS_DIR / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log(f"\nReport saved → {report_path}")