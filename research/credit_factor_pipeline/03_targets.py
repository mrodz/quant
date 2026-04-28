"""
03_targets.py — 21-day forward alpha target construction (all four definitions)
Input:  features_df.parquet
Output: targets_df.parquet  (features + 4 target columns)
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "features_df.parquet"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "targets_df.parquet"

HORIZON    = 21           # trading days forward
SPREAD_COL = "Option Adjusted Spread Bid"

# For binary label: fraction of top/bottom to keep
BINARY_THRESHOLD = 0.30


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["bond_ric", "Date"])

    # ------------------------------------------------------------------
    # Raw 21-day forward spread change per bond
    # ------------------------------------------------------------------
    df["fwd_spread_chg"] = df.groupby("bond_ric")[SPREAD_COL].transform(
        lambda x: x.shift(-HORIZON) - x
    )

    # ------------------------------------------------------------------
    # Target 1: Excess spread Δ (vs sector median on that day)
    #   → pure relative value; strips macro beta
    # ------------------------------------------------------------------
    sector_fwd = (
        df.groupby("Date")["fwd_spread_chg"]
        .median()
        .rename("sector_fwd_median")
        .reset_index()
    )
    df = df.merge(sector_fwd, on="Date", how="left")
    df["target_excess_chg"] = df["fwd_spread_chg"] - df["sector_fwd_median"]

    # ------------------------------------------------------------------
    # Target 2: Beta-adjusted Δ (OLS beta to sector mean over trailing 60d)
    #   → most realistic for PM reporting; removes index co-movement
    # ------------------------------------------------------------------
    def rolling_beta(grp: pd.DataFrame, sector_means: pd.Series) -> pd.Series:
        aligned = sector_means.reindex(grp["Date"]).values
        betas = []
        for i in range(len(grp)):
            start = max(0, i - 60)
            y = grp[SPREAD_COL].iloc[start:i+1].values
            x = aligned[start:i+1]
            if len(y) < 10 or np.std(x) < 1e-8:
                betas.append(np.nan)
                continue
            b = np.cov(y, x)[0, 1] / np.var(x)
            betas.append(b)
        return pd.Series(betas, index=grp.index)

    sector_spread_mean = df.groupby("Date")[SPREAD_COL].mean().rename("sector_mean")
    df = df.merge(sector_spread_mean, on="Date", how="left")

    betas = df.groupby("bond_ric", group_keys=False).apply(
        lambda g: rolling_beta(g.reset_index(), sector_spread_mean)
        .values
    )
    df["beta_to_sector"] = np.concatenate(betas)

    sector_fwd_mean = (
        df.groupby("Date")["fwd_spread_chg"]
        .mean()
        .rename("sector_fwd_mean")
        .reset_index()
    )
    df = df.merge(sector_fwd_mean, on="Date", how="left")
    df["target_beta_adj"] = (
        df["fwd_spread_chg"] - df["beta_to_sector"] * df["sector_fwd_mean"]
    )

    # ------------------------------------------------------------------
    # Target 3: Cross-sectional rank [0, 1]
    #   → best for ranking models (LightGBM lambdarank etc.)
    #   → percentile rank of fwd_spread_chg within each date cohort
    # ------------------------------------------------------------------
    df["target_cs_rank"] = df.groupby("Date")["fwd_spread_chg"].rank(pct=True)

    # ------------------------------------------------------------------
    # Target 4: Binary label  (1 = top wideners, 0 = bottom tighteners)
    #   → cleanest XGBoost classification signal
    #   → only bonds in the extreme tails are kept (NaN for middle 40%)
    # ------------------------------------------------------------------
    def binary_label(series: pd.Series, threshold: float) -> pd.Series:
        rank = series.rank(pct=True)
        labels = pd.Series(np.nan, index=series.index)
        labels[rank >= (1 - threshold)] = 1   # top 30% wideners → short candidates
        labels[rank <= threshold]        = 0   # bottom 30% tighteners → long candidates
        return labels

    df["target_binary"] = df.groupby("Date")["fwd_spread_chg"].transform(
        lambda x: binary_label(x, BINARY_THRESHOLD)
    )

    # ------------------------------------------------------------------
    # Drop rows where any target is NaN (end of time series / no fwd data)
    # ------------------------------------------------------------------
    target_cols = ["target_excess_chg", "target_beta_adj", "target_cs_rank", "target_binary"]
    before = len(df)
    df = df.dropna(subset=["fwd_spread_chg"])  # keep rows with valid fwd change
    print(f"Dropped {before - len(df):,} rows with no forward data (end of series)")
    print(f"Remaining rows: {len(df):,}")
    for col in target_cols:
        valid = df[col].notna().sum()
        print(f"  {col}: {valid:,} valid rows ({valid/len(df)*100:.1f}%)")
    print()

    return df


if __name__ == "__main__":
    features = pd.read_parquet(INPUT_PATH)
    targets  = add_targets(features)
    targets.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
