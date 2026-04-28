"""
02_features.py — Cross-sectional feature engineering
Adds sector-relative and time-series features to master_df.
Input:  master_df.parquet
Output: features_df.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "master_df.parquet"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "features_df.parquet"

# Rolling window sizes (trading days)
VOL_WINDOW   = 21   # realized vol lookback
MOM_WINDOW   = 5    # basis momentum
SPREAD_COL   = "Option Adjusted Spread Bid"  # OAS bid column


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Date", "bond_ric"])

    # ------------------------------------------------------------------
    # 1. Sector-relative Distance-to-Default z-score
    #    Identifies bonds that are safer than the sector average but
    #    potentially trade at wider spreads — the "value" signal.
    # ------------------------------------------------------------------
    sector_stats = (
        df.groupby("Date")["dd"]
        .agg(sector_DD_mean="mean", sector_DD_std="std")
        .reset_index()
    )
    df = df.merge(sector_stats, on="Date", how="left")
    df["z_DD"] = (df["dd"] - df["sector_DD_mean"]) / df["sector_DD_std"].replace(0, np.nan)

    # ------------------------------------------------------------------
    # 2. Volatility risk premium  (implied − realized)
    #    High premium → market over-hedging → spreads likely to mean-revert
    # ------------------------------------------------------------------
    df = df.sort_values(["bond_ric", "Date"])
    df["realized_vol_21d"] = (
        df.groupby("bond_ric")[SPREAD_COL]
        .transform(lambda x: x.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252))
    )
    if "30D_A_IM_P" in df.columns:
        df["vol_risk_premium"] = df["30D_A_IM_P"] - df["realized_vol_21d"]

    # ------------------------------------------------------------------
    # 3. Basis momentum  (5-day change in basis_bps)
    #    Narrowing basis often precedes a cash bond rally.
    #    Stale basis (unchanged for 60+ days) is nulled out first —
    #    ~50% of bonds have frozen CDS prints from illiquid periods.
    # ------------------------------------------------------------------
    if "basis_bps" in df.columns:
        # Null stale basis: any bond where basis hasn't moved in 60 days
        df["basis_bps"] = df.groupby("bond_ric")["basis_bps"].transform(
            lambda x: x.where(x.diff().abs().rolling(60, min_periods=1).sum() > 0)
        )
        n_stale = df["basis_bps"].isna().sum()
        print(f"  Nulled {n_stale:,} stale basis_bps rows ({n_stale/len(df)*100:.1f}%)")

        df["basis_momentum_5d"] = df.groupby("bond_ric")["basis_bps"].transform(
            lambda x: x.diff(MOM_WINDOW)
        )
        # Sector-relative basis rank (NaN-excluded cross-sectional percentile)
        df["basis_rank"] = df.groupby("Date")["basis_bps"].rank(pct=True)

    # ------------------------------------------------------------------
    # 4. Vol term structure slope  (90D − 30D implied vol) / 60
    #    Steepening signals an upcoming stress or regime shift
    # ------------------------------------------------------------------
    if {"90D_A_IM_P", "30D_A_IM_P"}.issubset(df.columns):
        df["vol_slope"] = (df["90D_A_IM_P"] - df["30D_A_IM_P"]) / 60.0

    # ------------------------------------------------------------------
    # 5. EDF percentile rank within sector on each date
    #    (edf_spread_divergence removed — zero importance, corrupted by stale basis)
    # ------------------------------------------------------------------
    if "edf_pct" in df.columns:
        df["edf_rank"] = df.groupby("Date")["edf_pct"].rank(pct=True)

    # ------------------------------------------------------------------
    # 6. Sector spread z-score  (bond spread vs. sector average)
    # ------------------------------------------------------------------
    sector_spread = (
        df.groupby("Date")[SPREAD_COL]
        .agg(sector_spread_mean="mean", sector_spread_std="std")
        .reset_index()
    )
    df = df.merge(sector_spread, on="Date", how="left")
    df["z_spread"] = (
        (df[SPREAD_COL] - df["sector_spread_mean"])
        / df["sector_spread_std"].replace(0, np.nan)
    )

    # ------------------------------------------------------------------
    # 7. Spread momentum (5d and 21d)
    # ------------------------------------------------------------------
    for w in [5, 21]:
        df[f"spread_mom_{w}d"] = df.groupby("bond_ric")[SPREAD_COL].transform(
            lambda x: x.diff(w)
        )

    print(f"Feature engineering complete. Shape: {df.shape}")
    new_cols = [c for c in df.columns if c not in pd.read_parquet(INPUT_PATH).columns]
    print(f"New columns added: {new_cols}\n")
    return df


if __name__ == "__main__":
    master = pd.read_parquet(INPUT_PATH)
    features = add_cross_sectional_features(master)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")