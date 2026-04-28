"""
04_model.py — Global XGBoost model training (one model across all 102 RICs)
Trains on 4 target definitions; evaluates each by IC and hit rate.
Input:  targets_df.parquet
Output: models/  (one .json per target)
        results/model_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

INPUT_PATH   = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "targets_df.parquet"
MODELS_DIR   = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "models"; MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR  = Path(__file__).parent.parent.parent / "data" / "credit_factor_pipeline" / "results"; RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FRAC   = 0.70
N_CV_SPLITS  = 5

# Adjust to columns actually present in your data
FEATURE_COLS = [
    # Structural / solvency
    "z_DD", "z_spread", "beta_to_sector",
    # Volatility
    "vol_risk_premium", "vol_slope", "realized_vol_21d",
    "30D_A_IM_C", "30D_A_IM_P", "60D_A_IM_C", "60D_A_IM_P", "90D_A_IM_C", "90D_A_IM_P",
    # Basis & momentum
    "basis_momentum_5d", "basis_rank",
    "spread_mom_5d", "spread_mom_21d",
    # Default risk
    "edf_rank", "edf_spread_divergence",
    # Macro / rates
    "risk_free_rate", "VIX_Close",
    # Bond metrics
    "Modified Duration", "Convexity", "Basis Point Value",
    "Interpolated Government Spread",
    # Equity / fundamental
    "Price To Book Value Per Share(Time Series Ratio)",
    "Company Market Cap",
]

TARGET_CONFIGS = [
    {"col": "target_excess_chg", "task": "reg",    "label": "Excess spread Δ"},
    {"col": "target_beta_adj",   "task": "reg",    "label": "Beta-adjusted Δ"},
    {"col": "target_cs_rank",    "task": "reg",    "label": "CS percentile rank"},
    {"col": "target_binary",     "task": "binary", "label": "Binary top/bottom 30%"},
]

XGB_PARAMS_REG = dict(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="reg:squarederror", eval_metric="rmse",
    early_stopping_rounds=30, random_state=42, n_jobs=-1,
)
XGB_PARAMS_CLF = dict(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="auc",
    early_stopping_rounds=30, random_state=42, n_jobs=-1,
    scale_pos_weight=1,
)


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and outcomes."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 10:
        return np.nan
    return spearmanr(y_true[valid], y_pred[valid]).correlation


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of times the predicted direction matches actual direction."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    return np.mean(np.sign(y_true[valid]) == np.sign(y_pred[valid]))


def temporal_train_test_split(df: pd.DataFrame, train_frac: float):
    sorted_dates = sorted(df["Date"].unique())
    cutoff_idx   = int(len(sorted_dates) * train_frac)
    cutoff_date  = sorted_dates[cutoff_idx]
    train = df[df["Date"] <  cutoff_date]
    test  = df[df["Date"] >= cutoff_date]
    print(f"  Train: {train['Date'].min().date()} → {train['Date'].max().date()} ({len(train):,} rows)")
    print(f"  Test:  {test['Date'].min().date()}  → {test['Date'].max().date()}  ({len(test):,} rows)")
    return train, test, cutoff_date


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing feature cols (skipped): {missing}")
    return df[available]


def train_target(df: pd.DataFrame, config: dict) -> dict:
    col  = config["col"]
    task = config["task"]
    name = config["label"]
    print(f"\n{'='*60}")
    print(f"Target: {name}  [{col}]")

    # Drop rows without this target
    sub = df.dropna(subset=[col]).copy()
    print(f"  Rows with valid target: {len(sub):,}")

    train_df, test_df, _ = temporal_train_test_split(sub, TRAIN_FRAC)

    X_train = get_features(train_df).values
    y_train = train_df[col].values
    X_test  = get_features(test_df).values
    y_test  = test_df[col].values

    params = XGB_PARAMS_CLF.copy() if task == "binary" else XGB_PARAMS_REG.copy()
    model  = xgb.XGBClassifier(**params) if task == "binary" else xgb.XGBRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Out-of-sample evaluation
    preds = model.predict_proba(X_test)[:, 1] if task == "binary" else model.predict(X_test)

    ic   = information_coefficient(y_test, preds)
    hr   = hit_rate(y_test, preds)
    auc  = roc_auc_score(y_test, preds) if task == "binary" else np.nan

    print(f"  IC (Spearman): {ic:.4f}")
    print(f"  Hit rate:      {hr:.4f}")
    if task == "binary":
        print(f"  ROC-AUC:       {auc:.4f}")

    # Save model
    model_path = MODELS_DIR / f"model_{col}.json"
    model.save_model(str(model_path))

    # Feature importance
    feat_names = get_features(train_df).columns.tolist()
    importance = dict(zip(feat_names, model.feature_importances_))

    return {
        "target": col, "label": name, "task": task,
        "ic": ic, "hit_rate": hr, "auc": auc,
        "best_iteration": model.best_iteration,
        "feature_importance": importance,
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    results = []
    for cfg in TARGET_CONFIGS:
        res = train_target(df, cfg)
        results.append(res)

    # Summary table
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "feature_importance"}
        for r in results
    ])
    summary_path = RESULTS_DIR / "model_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Feature importance per target
    fi_path = RESULTS_DIR / "feature_importance.json"
    fi_data = {
        r["target"]: {k: float(v) for k, v in r["feature_importance"].items()}
        for r in results
    }
    with open(fi_path, "w") as f:
        json.dump(fi_data, f, indent=2)

    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(summary[["label", "ic", "hit_rate", "auc", "best_iteration"]].to_string(index=False))
    print(f"\nSaved → {summary_path}")
    print(f"Saved → {fi_path}")
