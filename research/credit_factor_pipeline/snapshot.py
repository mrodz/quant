"""
snapshot.py — Single-bond spread signal tool
Usage:
    python snapshot.py my_bond.csv
    python snapshot.py my_bond.csv --model binary        # default
    python snapshot.py my_bond.csv --model cs_rank
    python snapshot.py my_bond.csv --top-drivers 5

Requirements:
    pip install pandas numpy xgboost shap colorama
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from pathlib import Path

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = MAGENTA = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

MODELS_DIR  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "models"
MASTER_PATH = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "master_df.parquet"
SPREAD_COL  = "Option Adjusted Spread Bid"

FEATURE_COLS = [
    "z_DD", "z_spread", "beta_to_sector",
    "vol_risk_premium", "vol_slope", "realized_vol_21d",
    "30D_A_IM_C", "30D_A_IM_P", "60D_A_IM_C", "60D_A_IM_P", "90D_A_IM_C", "90D_A_IM_P",
    "basis_momentum_5d", "basis_rank",
    "spread_mom_5d", "spread_mom_21d",
    "edf_rank",
    "risk_free_rate", "VIX_Close",
    "Modified Duration", "Convexity", "Basis Point Value",
    "Interpolated Government Spread",
    "Price To Book Value Per Share(Time Series Ratio)",
    "Company Market Cap",
]

MODEL_MAP = {
    "binary":   ("target_binary",     "binary"),
    "cs_rank":  ("target_cs_rank",    "reg"),
    "excess":   ("target_excess_chg", "reg"),
    "beta_adj": ("target_beta_adj",   "reg"),
}

VOL_WINDOW = 21
MOM_WINDOW = 5


# ── Rolling feature engineering over full history ─────────────────────────────
def run_feature_engineering(df: pd.DataFrame, sector_ctx: dict) -> pd.DataFrame:
    df = df.copy().sort_values("Date")

    # Time-series features (require full history)
    df["realized_vol_21d"] = (
        df[SPREAD_COL].pct_change()
        .rolling(VOL_WINDOW).std() * np.sqrt(252)
    )
    df["spread_mom_5d"]  = df[SPREAD_COL].diff(5)
    df["spread_mom_21d"] = df[SPREAD_COL].diff(21)

    if "basis_bps" in df.columns:
        # Detect stale CDS feed — if basis hasn't moved in 10 days, null out derived features
        if df["basis_bps"].tail(10).nunique() <= 1:
            print("  Warning: basis_bps appears stale (no change in last 10 rows) — basis features set to NaN")
            df["basis_momentum_5d"] = np.nan
        else:
            df["basis_momentum_5d"] = df["basis_bps"].diff(MOM_WINDOW)

    # Vol features
    if {"30D_A_IM_P", "realized_vol_21d"}.issubset(df.columns):
        df["vol_risk_premium"] = df["30D_A_IM_P"] - df["realized_vol_21d"]
    if {"90D_A_IM_P", "30D_A_IM_P"}.issubset(df.columns):
        df["vol_slope"] = (df["90D_A_IM_P"] - df["30D_A_IM_P"]) / 60.0

    # Cross-sectional features — use sector context from master_df
    if "sector_spread_median" in sector_ctx:
        spread_iqr     = sector_ctx["sector_spread_p75"] - sector_ctx["sector_spread_p25"]
        df["z_spread"] = (df[SPREAD_COL] - sector_ctx["sector_spread_median"]) / (spread_iqr / 1.35 + 1e-8)
    elif "z_spread" not in df.columns:
        df["z_spread"] = df[SPREAD_COL].sub(df[SPREAD_COL].mean()).div(df[SPREAD_COL].std() + 1e-8)

    if "dd" in df.columns:
        df["z_DD"] = df["dd"].sub(df["dd"].mean()).div(df["dd"].std() + 1e-8)
    elif "z_DD" not in df.columns:
        df["z_DD"] = 0.0

    # Rank features (within this bond's own history as proxy)
    if "basis_bps" in df.columns:
        df["basis_rank"] = df["basis_bps"].rank(pct=True)
    if "edf_pct" in df.columns:
        df["edf_rank"] = df["edf_pct"].rank(pct=True)
        spread_rank    = df[SPREAD_COL].rank(pct=True)


    if "beta_to_sector" not in df.columns:
        df["beta_to_sector"] = 1.0

    return df


# ── Sector context ─────────────────────────────────────────────────────────────
def load_sector_context(bond_ric: str, snapshot_date=None) -> dict:
    if not MASTER_PATH.exists():
        return {}
    try:
        master = pd.read_parquet(
            MASTER_PATH,
            columns=["Date", "bond_ric", SPREAD_COL, "dd", "edf_pct", "basis_bps"]
        )
        master["Date"] = pd.to_datetime(master["Date"])
        ref = master[master["Date"] <= pd.to_datetime(snapshot_date)] if snapshot_date else master
        if ref.empty:
            ref = master
        sector_day = ref[ref["Date"] == ref["Date"].max()]
        ctx = {
            "sector_date":          str(ref["Date"].max().date()),
            "sector_n_bonds":       len(sector_day),
            "sector_spread_median": sector_day[SPREAD_COL].median(),
            "sector_spread_p25":    sector_day[SPREAD_COL].quantile(0.25),
            "sector_spread_p75":    sector_day[SPREAD_COL].quantile(0.75),
        }
        bond_row = sector_day[sector_day["bond_ric"] == bond_ric]
        if not bond_row.empty:
            bond_spread = bond_row[SPREAD_COL].iloc[0]
            ctx["bond_spread_rank_pct"]   = (sector_day[SPREAD_COL] <= bond_spread).mean()
            ctx["bond_spread_vs_median"]  = bond_spread - ctx["sector_spread_median"]
        return ctx
    except Exception as e:
        return {"sector_error": str(e)}


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(model_key: str):
    col, task = MODEL_MAP[model_key]
    path = MODELS_DIR / f"model_{col}.json"
    if not path.exists():
        sys.exit(f"Model not found: {path}\nRun 04_model.py first.")
    m = xgb.XGBClassifier() if task == "binary" else xgb.XGBRegressor()
    m.load_model(str(path))
    return m, col, task


# ── SHAP ───────────────────────────────────────────────────────────────────────
def get_shap_drivers(model, X: pd.DataFrame, top_n: int) -> list:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if shap_values.ndim > 1:
        shap_values = shap_values[0]
    return sorted(
        zip(X.columns, shap_values, X.iloc[0].values),
        key=lambda x: abs(x[1]), reverse=True,
    )[:top_n]


# ── Signal ─────────────────────────────────────────────────────────────────────
def interpret_signal(score: float, task: str) -> tuple:
    if task == "binary":
        if score >= 0.65:
            return "WIDEN  (SELL / AVOID)", score * 100, "widen"
        elif score <= 0.35:
            return "TIGHTEN  (BUY)", (1 - score) * 100, "tighten"
        else:
            return "NEUTRAL  (HOLD)", abs(score - 0.5) * 200, "neutral"
    else:
        if score > 5:
            return "WIDEN  (SELL / AVOID)", min(abs(score) / 20 * 100, 95), "widen"
        elif score < -5:
            return "TIGHTEN  (BUY)", min(abs(score) / 20 * 100, 95), "tighten"
        else:
            return "NEUTRAL  (HOLD)", 50.0, "neutral"


# ── Report ─────────────────────────────────────────────────────────────────────
def print_report(bond_ric, snapshot_date, signal, confidence, score,
                 model_key, drivers, sector_ctx, raw_row, direction):
    sep = "─" * 62
    sc  = Fore.GREEN if direction == "tighten" else Fore.RED if direction == "widen" else Fore.YELLOW

    print(f"\n{Style.BRIGHT}{sep}")
    print(f"  BOND SPREAD SIGNAL REPORT")
    print(f"{sep}{Style.RESET_ALL}")
    print(f"  RIC            {Fore.CYAN}{bond_ric}{Style.RESET_ALL}")
    print(f"  Snapshot date  {snapshot_date}")
    print(f"  Model          {model_key}  (21-day horizon)")
    print(sep)

    print(f"\n  {Style.BRIGHT}SIGNAL{Style.RESET_ALL}")
    print(f"  {sc}{Style.BRIGHT}  {signal}{Style.RESET_ALL}")
    print(f"  Confidence     {confidence:.1f}%")
    print(f"  Raw score      {score:.4f}")

    print(f"\n{sep}")
    print(f"  {Style.BRIGHT}KEY INPUTS{Style.RESET_ALL}")
    for col, label in [
        (SPREAD_COL,                                   "OAS bid (bps)"),
        ("dd",                                         "Distance-to-default"),
        ("edf_pct",                                    "EDF percentile"),
        ("basis_bps",                                  "CDS-bond basis (bps)"),
        ("VIX_Close",                                  "VIX"),
        ("30D_A_IM_P",                                 "30D impl. vol (put)"),
        ("Modified Duration",                          "Mod. duration"),
        ("spread_mom_21d",                             "Spread mom 21d (bps)"),
        ("basis_momentum_5d",                          "Basis mom 5d (bps)"),
        ("realized_vol_21d",                           "Realized vol 21d"),
    ]:
        val = raw_row.get(col, np.nan)
        if pd.notna(val):
            print(f"  {label:<32} {val:>10.3f}")

    if sector_ctx and "sector_spread_median" in sector_ctx:
        print(f"\n{sep}")
        print(f"  {Style.BRIGHT}SECTOR CONTEXT  ({sector_ctx.get('sector_date','')}, n={sector_ctx.get('sector_n_bonds','')}){Style.RESET_ALL}")
        print(f"  Sector OAS median          {sector_ctx['sector_spread_median']:>8.1f} bps")
        print(f"  Sector OAS 25th–75th pct   {sector_ctx['sector_spread_p25']:>8.1f} – {sector_ctx['sector_spread_p75']:.1f} bps")
        if "bond_spread_rank_pct" in sector_ctx:
            rp = sector_ctx["bond_spread_rank_pct"] * 100
            vs = sector_ctx["bond_spread_vs_median"]
            rc = Fore.RED if rp > 70 else Fore.GREEN if rp < 30 else Fore.YELLOW
            print(f"  This bond's spread rank    {rc}{rp:>7.1f}th percentile{Style.RESET_ALL}  ({vs:+.1f} bps vs median)")

    print(f"\n{sep}")
    print(f"  {Style.BRIGHT}TOP SPREAD DRIVERS  (SHAP){Style.RESET_ALL}")
    print(f"  {'Feature':<38} {'Value':>8}  {'Impact':>8}")
    print(f"  {'─'*38} {'─'*8}  {'─'*8}")
    for feat, shap_val, feat_val in drivers:
        val_str    = f"{feat_val:.3f}" if pd.notna(feat_val) else "  n/a"
        ic         = Fore.RED if shap_val > 0 else Fore.GREEN
        print(f"  {feat[:37]:<38} {val_str:>8}  {ic}{shap_val:>+.4f}{Style.RESET_ALL}")

    print(f"\n  {Style.DIM}Positive SHAP = pushes toward widening  |  Negative = tightening{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{sep}{Style.RESET_ALL}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bond spread signal — single snapshot")
    parser.add_argument("csv")
    parser.add_argument("--model",       default="binary", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--top-drivers", type=int, default=8)
    parser.add_argument("--ric",         default=None,
                        help="Specific bond_ric to use (default: longest duration)")
    parser.add_argument("--list-bonds",  action="store_true",
                        help="List all bonds in the CSV with their duration and exit")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    if args.list_bonds:
        tmp = pd.read_csv(csv_path, parse_dates=["Date"])
        tmp["Modified Duration"] = pd.to_numeric(tmp.get("Modified Duration"), errors="coerce")
        latest = tmp[tmp["Date"] == tmp["Date"].max()]
        summary = (latest.groupby("bond_ric")["Modified Duration"]
                   .max().sort_values(ascending=False).reset_index())
        print(f"\nBonds in {csv_path.name} as of {tmp['Date'].max().date()}:")
        print(summary.to_string(index=False))
        sys.exit(0)

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    for col in df.columns:
        if col not in ["Date", "bond_ric", "equity_ric", "equity_name",
                       "source_file", "study_start", "study_end", "interval"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    # Select bond: prefer --ric override, else longest-duration bond at latest date
    if args.ric:
        if args.ric not in df["bond_ric"].values:
            sys.exit(f"RIC {args.ric} not found in CSV. Available: {df['bond_ric'].unique().tolist()}")
        bond_ric = args.ric
    elif "bond_ric" in df.columns and "Modified Duration" in df.columns:
        latest   = df[df["Date"] == df["Date"].max()]
        bond_ric = latest.loc[latest["Modified Duration"].idxmax(), "bond_ric"]
        print(f"Auto-selected bond: {bond_ric}  (longest duration = {latest['Modified Duration'].max():.2f}y)")
    else:
        bond_ric = df["bond_ric"].value_counts().idxmax() if "bond_ric" in df.columns else "UNKNOWN"
    df = df[df["bond_ric"] == bond_ric].copy()

    snapshot_date = str(df["Date"].max().date())
    sector_ctx    = load_sector_context(bond_ric, snapshot_date)

    # Run full feature engineering over history, then take last row
    df      = run_feature_engineering(df, sector_ctx)
    last    = df.iloc[[-1]].reset_index(drop=True)
    raw_row = df.iloc[-1]   # for display (has computed rolling cols)

    X = last.reindex(columns=FEATURE_COLS)

    model, col, task = load_model(args.model)

    score = (float(model.predict_proba(X)[0, 1]) if task == "binary"
             else float(model.predict(X)[0]))

    signal, confidence, direction = interpret_signal(score, task)

    X_clean = X.fillna(X.median())
    drivers = get_shap_drivers(model, X_clean, args.top_drivers)

    print_report(bond_ric, snapshot_date, signal, confidence, score,
                 args.model, drivers, sector_ctx, raw_row, direction)


if __name__ == "__main__":
    main()