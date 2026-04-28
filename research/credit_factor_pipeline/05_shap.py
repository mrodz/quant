"""
05_shap.py — SHAP feature importance for all four global models
Produces:  results/shap_summary.png  (beeswarm per target)
           results/shap_values.parquet  (raw SHAP values for further analysis)
Input:     targets_df.parquet + models/*.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import shap
import xgboost as xgb

INPUT_PATH   = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "targets_df.parquet"
MODELS_DIR   = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "models"
RESULTS_DIR  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "results"; RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FRAC   = 0.70
SHAP_SAMPLE  = 2000     # random sample from test set (fast; increase if you want)

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
    {"col": "target_excess_chg", "task": "reg",    "label": "Excess spread Δ"},
    {"col": "target_beta_adj",   "task": "reg",    "label": "Beta-adjusted Δ"},
    {"col": "target_cs_rank",    "task": "reg",    "label": "CS percentile rank"},
    {"col": "target_binary",     "task": "binary", "label": "Binary top/bottom 30%"},
]


def get_test_features(df: pd.DataFrame, target_col: str) -> tuple:
    sub = df.dropna(subset=[target_col]).copy()
    sorted_dates = sorted(sub["Date"].unique())
    cutoff = sorted_dates[int(len(sorted_dates) * TRAIN_FRAC)]
    test   = sub[sub["Date"] >= cutoff]
    feat_cols = [c for c in FEATURE_COLS if c in test.columns]
    return test[feat_cols], feat_cols


def run_shap(df: pd.DataFrame):
    fig = plt.figure(figsize=(18, 5 * len(TARGET_CONFIGS)))
    gs  = gridspec.GridSpec(len(TARGET_CONFIGS), 1, hspace=0.5)

    all_shap = {}

    for i, cfg in enumerate(TARGET_CONFIGS):
        col   = cfg["col"]
        task  = cfg["task"]
        label = cfg["label"]
        model_path = MODELS_DIR / f"model_{col}.json"

        if not model_path.exists():
            print(f"Model not found for {col} — skipping. Run 04_model.py first.")
            continue

        print(f"Computing SHAP for: {label}")

        # Load model
        if task == "binary":
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
        model.load_model(str(model_path))

        X_test, feat_cols = get_test_features(df, col)

        # Subsample for speed
        sample_idx = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
        X_sample   = X_test.iloc[sample_idx]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Store raw values
        shap_df = pd.DataFrame(shap_values, columns=feat_cols)
        shap_df["target"] = col
        all_shap[col] = shap_df

        # Beeswarm plot
        ax = fig.add_subplot(gs[i])
        plt.sca(ax)
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feat_cols,
            show=False,
            plot_size=None,
            max_display=12,
        )
        ax.set_title(f"SHAP — {label}", fontsize=13, fontweight="bold", pad=10)

        # Print top 5 features
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = sorted(zip(feat_cols, mean_abs), key=lambda x: -x[1])
        print(f"  Top 5 features:")
        for feat, val in ranked[:5]:
            print(f"    {feat:<30} mean|SHAP| = {val:.4f}")
        print()

    plt.suptitle("SHAP Feature Importance — Global Industrial Bond Model", fontsize=15, y=1.01)
    out_png = RESULTS_DIR / "shap_summary.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_png}")

    # Save raw SHAP values
    if all_shap:
        combined = pd.concat(all_shap.values(), ignore_index=True)
        out_parquet = RESULTS_DIR / "shap_values.parquet"
        combined.to_parquet(out_parquet, index=False)
        print(f"Saved → {out_parquet}")


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    run_shap(df)
