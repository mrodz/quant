"""
01_ingest.py — Master dataframe builder
Reads all CSVs from a folder and concatenates into one sorted master_df.
Handles columns with spaces, mixed dtypes, and prints a data quality report.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_FOLDER = Path(__file__).parent.parent.parent / "data" / "fi_ltm_test"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "master_df.parquet"

# Columns that must be present for downstream scripts to work
REQUIRED_COLS = [
    "Date", "bond_ric",
    "Option Adjusted Spread Bid",
    "dd", "edf_pct", "basis_bps",
    "risk_free_rate", "VIX_Close",
]

# Numeric columns that should never be strings
NUMERIC_COLS = [
    "Option Adjusted Spread Bid", "Z Spread", "Interpolated Government Spread",
    "dd", "edf_pct", "basis_bps", "risk_free_rate", "VIX_Close",
    "30D_A_IM_C", "30D_A_IM_P", "60D_A_IM_C", "60D_A_IM_P", "90D_A_IM_C", "90D_A_IM_P",
    "Modified Duration", "Convexity", "Basis Point Value",
    "Company Market Cap", "Net Debt",
    "Earnings before Interest Taxes Depreciation & Amortization",
    "Price To Book Value Per Share(Time Series Ratio)",
]


def load_master(data_folder: Path = DATA_FOLDER) -> pd.DataFrame:
    all_dfs = []
    files = sorted(data_folder.glob("*.csv"))

    if not files:
        sys.exit(f"No CSV files found in {data_folder.resolve()}")

    print(f"Found {len(files)} CSV files. Loading...")

    failed = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["Date"])
            df["source_file"] = f.stem
            all_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: could not load {f.name} — {e}")
            failed.append(f.name)

    if not all_dfs:
        sys.exit("No files loaded successfully.")

    if failed:
        print(f"\n  {len(failed)} files failed to load: {failed}\n")

    # Concatenate
    master = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values(["Date", "bond_ric"])
        .reset_index(drop=True)
    )

    # Force numeric cols (some may read as object if a cell contains text)
    for col in NUMERIC_COLS:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    # Ensure Date is datetime
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce")
    master = master.dropna(subset=["Date", "bond_ric"])

    print(f"\n{'='*55}")
    print("INGESTION SUMMARY")
    print(f"{'='*55}")
    print(f"  Files loaded:   {len(all_dfs)}")
    print(f"  Total rows:     {len(master):,}")
    print(f"  Unique RICs:    {master['bond_ric'].nunique()}")
    print(f"  Unique dates:   {master['Date'].nunique()}")
    print(f"  Date range:     {master['Date'].min().date()} → {master['Date'].max().date()}")

    # Rows per RIC stats
    ric_counts = master.groupby("bond_ric").size()
    print(f"\n  Rows per RIC:")
    print(f"    min={ric_counts.min()}  median={ric_counts.median():.0f}  max={ric_counts.max()}")

    # Missing value report for key columns
    print(f"\n  Missing values (key columns):")
    check_cols = [c for c in REQUIRED_COLS if c in master.columns]
    for col in check_cols:
        n_missing = master[col].isna().sum()
        pct = n_missing / len(master) * 100
        flag = "  <<<" if pct > 5 else ""
        print(f"    {col:<45} {n_missing:>7,}  ({pct:5.1f}%){flag}")

    # Warn on missing required columns
    missing_required = [c for c in REQUIRED_COLS if c not in master.columns]
    if missing_required:
        print(f"\n  WARNING — required columns not found in data:")
        for c in missing_required:
            print(f"    - {c}")
        print("  Check column names in your CSVs and update REQUIRED_COLS if needed.")

    print(f"\n  All columns present: {sorted(master.columns.tolist())}")
    print(f"{'='*55}\n")

    return master


if __name__ == "__main__":
    master = load_master()
    master.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
