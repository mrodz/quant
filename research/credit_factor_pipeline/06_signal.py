"""
06_signal.py — Long-short alpha signal construction and backtest
Uses best model (highest IC) or all four; generates daily long/short book.
Input:   targets_df.parquet + models/*.json
Output:  results/signal_backtest.csv
         results/backtest_tearsheet.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

INPUT_PATH  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "targets_df.parquet"
MODELS_DIR  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "models"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "results"; RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FRAC      = 0.70
LONG_SHORT_FRAC = 0.10     # top/bottom 10% each day
SPREAD_COL      = "Option Adjusted Spread Bid"

FEATURE_COLS = [
    "z_DD", "z_spread", "beta_to_sector",
    "vol_risk_premium", "vol_slope", "realized_vol_21d",
    "30D_A_IM_C", "30D_A_IM_P", "60D_A_IM_C", "60D_A_IM_P", "90D_A_IM_C", "90D_A_IM_P",
    "basis_momentum_5d", "basis_rank",
    "spread_mom_5d", "spread_mom_21d",
    "edf_rank", "edf_spread_divergence",
    "risk_free_rate", "VIX_Close",
    "Modified Duration", "Convexity", "Basis Point Value",
    "Interpolated Government Spread",
    "Price To Book Value Per Share(Time Series Ratio)",
    "Company Market Cap",
]

TARGET_CONFIGS = [
    {"col": "target_excess_chg", "task": "reg",    "label": "Excess Δ"},
    {"col": "target_beta_adj",   "task": "reg",    "label": "Beta-adj Δ"},
    {"col": "target_cs_rank",    "task": "reg",    "label": "CS Rank"},
    {"col": "target_binary",     "task": "binary", "label": "Binary"},
]


def load_model(col: str, task: str):
    path = MODELS_DIR / f"model_{col}.json"
    if not path.exists():
        return None
    m = xgb.XGBClassifier() if task == "binary" else xgb.XGBRegressor()
    m.load_model(str(path))
    return m


def get_predictions(df: pd.DataFrame, model, feat_cols: list, task: str) -> pd.Series:
    X = df[feat_cols].values
    if task == "binary":
        return pd.Series(model.predict_proba(X)[:, 1], index=df.index)
    return pd.Series(model.predict(X), index=df.index)


def build_signal(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    col, task, label = config["col"], config["task"], config["label"]
    model = load_model(col, task)
    if model is None:
        return pd.DataFrame()

    sub = df.dropna(subset=[col]).copy()
    sorted_dates = sorted(sub["Date"].unique())
    cutoff = sorted_dates[int(len(sorted_dates) * TRAIN_FRAC)]
    test   = sub[sub["Date"] >= cutoff].copy()

    feat_cols = [c for c in FEATURE_COLS if c in test.columns]
    test["score"] = get_predictions(test, model, feat_cols, task)

    # Daily percentile rank of score
    test["score_rank"] = test.groupby("Date")["score"].rank(pct=True)

    # Assign positions: +1 long (low score = will tighten), -1 short (high score = will widen)
    test["position"] = np.nan
    test.loc[test["score_rank"] <= LONG_SHORT_FRAC,        "position"] =  1   # long leg
    test.loc[test["score_rank"] >= (1 - LONG_SHORT_FRAC),  "position"] = -1   # short leg

    # PnL proxy: position × negative of actual fwd spread change
    # (tightening = positive return for long, widening = positive return for short)
    test["pnl_bps"] = test["position"] * (-test["fwd_spread_chg"])

    # Daily aggregation
    daily = (
        test.dropna(subset=["position", "fwd_spread_chg"])
        .groupby("Date")
        .agg(
            n_long  =("position", lambda x: (x ==  1).sum()),
            n_short =("position", lambda x: (x == -1).sum()),
            pnl_bps =("pnl_bps", "mean"),
        )
        .reset_index()
    )
    daily["target"]     = col
    daily["label"]      = label
    daily["cum_pnl"]    = daily["pnl_bps"].cumsum()
    daily["drawdown"]   = daily["cum_pnl"] - daily["cum_pnl"].cummax()

    return daily


def performance_stats(daily: pd.DataFrame, label: str) -> dict:
    pnl = daily["pnl_bps"].dropna()
    sharpe  = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan
    max_dd  = daily["drawdown"].min()
    hit     = (pnl > 0).mean()
    total   = pnl.sum()
    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "total_pnl_bps": round(total, 1),
        "max_drawdown_bps": round(max_dd, 1),
        "hit_rate": round(hit, 3),
        "n_days": len(pnl),
    }


def plot_tearsheet(all_daily: list, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, daily in zip(axes, all_daily):
        if daily.empty:
            continue
        label = daily["label"].iloc[0]
        ax.plot(daily["Date"], daily["cum_pnl"], lw=1.5, label="Cum PnL (bps)")
        ax.fill_between(daily["Date"], daily["drawdown"], 0, alpha=0.2, color="red", label="Drawdown")
        ax.axhline(0, color="gray", lw=0.5, linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Cumulative bps")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Long-Short Alpha Backtest — Top/Bottom {int(LONG_SHORT_FRAC*100)}%", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    all_daily = []
    stats_list = []

    for cfg in TARGET_CONFIGS:
        print(f"Building signal: {cfg['label']}")
        daily = build_signal(df, cfg)
        if daily.empty:
            print(f"  Skipped (model not found — run 04_model.py first)")
            continue
        all_daily.append(daily)
        stats_list.append(performance_stats(daily, cfg["label"]))

    # Print summary
    if stats_list:
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        stats_df = pd.DataFrame(stats_list)
        print(stats_df.to_string(index=False))

        # Save
        combined = pd.concat(all_daily, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "signal_backtest.csv", index=False)
        stats_df.to_csv(RESULTS_DIR / "backtest_stats.csv", index=False)
        plot_tearsheet(all_daily, RESULTS_DIR / "backtest_tearsheet.png")
