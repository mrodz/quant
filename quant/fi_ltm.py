from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from quant import Interval, SessionProvider
from quant.bonds import BondL1
from quant.equities import EquityL1
from quant.kmv import KMVInputs
from quant.kmv_timeseries import build_from_panel
from quant.study import Study, PreparedStudy
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

@dataclass
class _FILtmStudyState:
    common_stock: EquityL1
    bonds: Sequence[BondL1]
    interval: Interval
    start: datetime
    end: datetime
    vix: pd.DataFrame
    risk_free_rate_decimal: float # 0.045 is 4.5%, etc.


class _FILtmStudyImpl(PreparedStudy[_FILtmStudyState, pd.DataFrame]):
    EQUITIES_FIELDS = [
            "BID",
            "ASK",
            "TR.CompanyMarketCap",
            "TR.H.PriceToBVPerShare",
            "TR.F.NetDebt",                                                        
            "TR.F.NetCashFlowOp",
            "TR.F.STDebtCurrPortOfLTDebt",
            "TR.F.DebtLTTot",
            "TR.F.EBITDA",
            "TR.F.TotAssets",
    ]
    
    FI_FIELDS = [
            "TR.ZSPREAD",
            "TR.OPTIONADJUSTEDSPREADBID",
            "TR.MODIFIEDDURATION",
            "TR.CONVEXITY",
            "TR.BASISPOINTVALUE",
            "TR.INTERPOLATEDGOVERNMENTSPREAD",
    ]
    
    def __init__(self, input: _FILtmStudyState) -> None:
        super().__init__()
        self.input = input
    
    def inputs(self) -> _FILtmStudyState:
        return self.input
    
    def run(self, session: SessionProvider) -> pd.DataFrame:
        """
        Expensive, expect ~2 min execution
        """
        
        with session as client:
            print("Pull IV history")
            historical_iv = client.equities.historical_iv(
                self.inputs().common_stock.ric,
                interval=self.inputs().interval,
                start=self.inputs().start,
                end=self.inputs().end,
            )
            
            vix = self.inputs().vix[(self.inputs().start < self.inputs().vix.index) & (self.inputs().vix.index < self.inputs().end)]
            
            print("Pull stock history")
            common_stock_df = client.equities.history_df(self.inputs().common_stock, fields=self.EQUITIES_FIELDS,
                                                        interval=self.inputs().interval,
                                                        start=self.inputs().start,
                                                        end=self.inputs().end,
                                                    )
                
            common_stock_df = common_stock_df.ffill().rename(columns={"BID": "Stock_Bid", "ASK": "Stock_Ask"})
            
            bond_df = pd.concat([
                client.bonds.history_df([*batch], fields=self.FI_FIELDS,
                interval=self.inputs().interval,
                start=self.inputs().start,
                end=self.inputs().end,
                )
                for batch in tqdm(batched(self.inputs().bonds, 3), desc="Pull bond history")
            ])
            
            working_dfs = {}

            for bond_l1 in tqdm(self.inputs().bonds, desc='Build DD σ'):
                bond = bond_df[bond_l1.ric].dropna()
                
                bond = bond.ffill()
                
                working_df = vix.join(common_stock_df).join(bond).join(historical_iv.df).dropna()

                working_df = working_df[~working_df.index.duplicated(keep='first')]

                eq_value = working_df["Company Market Cap"]
                eq_vol = ((working_df["Stock_Bid"] + working_df["Stock_Ask"]) / 2.0).pct_change().rolling(60, min_periods=20).std() * (252 ** 0.5)
                common_idx = eq_value.index.intersection(eq_vol.index)

                panel = pd.DataFrame({
                    "equity_value":      eq_value,
                    "equity_volatility": eq_vol,
                    "short_term_debt":   working_df["Short-Term Debt & Current Portion of Long-Term Debt"],
                    "long_term_debt":    working_df["Debt - Long-Term - Total"],
                    "z_spread":          working_df["Z Spread"].mean()
                }, index=common_idx).dropna()

                base_inputs = KMVInputs(
                    equity_value=1.0, 
                    equity_volatility=0.1, 
                    short_term_debt=1.0, 
                    long_term_debt=1.0, 
                    risk_free_rate=self.inputs().risk_free_rate_decimal
                )
                
                if panel.shape[0] == 0:
                    logger.warning(f"Skipping bad data returned from LSEG for {bond_l1.name}")
                    continue            

                ts = build_from_panel(panel, base_inputs=base_inputs, name=self.inputs().common_stock.name)
                
                working_df_dd = working_df.join(ts.dd_series()).join(ts.basis_series()).join(ts.edf_series()).dropna()
                
                working_dfs[bond_l1.ric] = working_df_dd
            
            ric_to_bond = {bond.ric: bond for bond in self.inputs().bonds}
            
            bond_to_pred_power = {}

            for idx, df in tqdm(working_dfs.items(), desc='Build engineered features'):
                df = df.dropna()
                
                mo_log_move = np.log(df['Z Spread'].shift(-30).clip(lower=5)) - np.log(df['Z Spread'].clip(lower=5).dropna())
                vix_log_move = np.log(df['VIX_Close'].shift(-30).clip(lower=5)) - np.log(df['VIX_Close'].clip(lower=5).dropna())
                df['target_residual'] = mo_log_move - vix_log_move

                # DD Velocity: Is the firm moving toward or away from the default boundary?
                df['dd_velocity_21d'] = df['dd'].diff(21)
                df['dd_velocity_5d'] = df['dd'].diff(5)

                df['term_structure_30_90'] = df['30D_A_IM_P'] - df['90D_A_IM_P']
                df['term_structure_30_60'] = df['30D_A_IM_P'] - df['60D_A_IM_P']

                # Velocity of Volatility
                df['vol_30d_velocity'] = df['30D_A_IM_P'].diff(5) 

                # Relative Volatility (How jumpy is it relative to the average?)
                df['vol_zscore'] = (df['30D_A_IM_P'] - df['30D_A_IM_P'].rolling(252).mean()) / df['30D_A_IM_P'].rolling(252).std()

                # Leverage Ratio: Standard credit anchor
                df['leverage_ratio'] = df['Net Debt'] / df['Earnings before Interest Taxes Depreciation & Amortization'].replace(0, np.nan)
                df['leverage_velocity_63d'] = df['leverage_ratio'].diff(63) # Quarterly change trend

                df['asset_cushion'] = df['Total Assets'] / (df['Short-Term Debt & Current Portion of Long-Term Debt'] + df['Debt - Long-Term - Total']).replace(0, np.nan)

                df['dts'] = df['Modified Duration'] * df['Z Spread']


                # VIX Z-Score: Market panic relative to the last year
                df['vix_rolling_mean'] = df['VIX_Close'].rolling(window=252).mean()
                df['vix_rolling_std'] = df['VIX_Close'].rolling(window=252).std()
                df['vix_zscore'] = (df['VIX_Close'] - df['vix_rolling_mean']) / df['vix_rolling_std'].replace(0, np.nan)

                # Negative Equity Regime Flag; Important for Altria's 2026 data where P/B is negative
                df['is_negative_equity'] = (df['Price To Book Value Per Share(Time Series Ratio)'] < 0).astype(int)

                df['dd_x_neg_equity'] = df['dd'] * df['is_negative_equity']

                # Drop the intermediate rolling columns used for Z-score
                df = df.drop(columns=['vix_rolling_mean', 'vix_rolling_std'])

                # List of finalized features for the model
                feature_cols = [
                    'dd', 'dd_velocity_21d', 'term_structure_30_90', 'term_structure_30_60', 'vol_30d_velocity', 'vol_zscore',
                    'leverage_ratio', 'leverage_velocity_63d', 'asset_cushion',
                    'dts', 'vix_zscore', 'is_negative_equity', 'Convexity', 'dd_x_neg_equity'
                ]
                
                # 1. Target Definition & Data Alignment (Handling the 30-day NA shift)
                # We create the boolean mask first, then align it with our features
                y_raw = (df['target_residual'] > 0) 
                
                # Drop NaNs from features (X) and then reindex the target (y) to match
                X = df[feature_cols].dropna() 
                y = y_raw.reindex(X.index).dropna().astype(int)
                
                # Final check: Ensure X only contains rows that have a valid target
                X = X.reindex(y.index)

                if len(X) <= 5: 
                    logger.warning(f"Skipping iteration: DataFrame is empty or too small after cleaning ({ric_to_bond[idx].name})")
                    continue

                # 2. Time Series Split Setup
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []

                # Helper for Time-Decay Weights (Higher importance for recent 2025-2026 data)
                def get_decay_weights(n, decay_rate=0.005):
                    w = np.exp(decay_rate * np.arange(n))
                    return w / w.mean()

                # 3. Cross-Validation Training Loop
                for train_idx, test_idx in tscv.split(X):
                    # Embargo: Purge the 30 rows prior to the test set to avoid leakage
                    train_idx_purged = train_idx[:-30]
                    
                    X_train, X_test = X.iloc[train_idx_purged], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx_purged], y.iloc[test_idx]
                    
                    # Calculate weights for the training set
                    weights = get_decay_weights(len(X_train))
                    
                    # Define and Fit Model
                    model = XGBClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=20,
                        eval_metric='logloss'
                    )
                    
                    model.fit(X_train, y_train, sample_weight=weights)
                    
                    # Evaluate Directional Hit Rate
                    preds = model.predict(X_test)
                    scores.append(accuracy_score(y_test, preds))

                bond_to_pred_power[idx] = scores

                # 4. Output Results for the current DataFrame
                logger.debug(f"""{ric_to_bond[idx].name}
                             \tValidated Hit Rates (Weighted) across 5 folds: {['{:.2%}'.format(s) for s in scores]}
                             \tAverage Hit Rate: {np.mean(scores):.2%}""")
                
            df_results = pd.DataFrame.from_dict(
                {f'{ric_to_bond[bond].maturity()} - {ric_to_bond[bond].coupon()}': bond_to_pred_power[bond] 
                for bond in working_dfs.keys() 
                if bond in bond_to_pred_power},
                orient='columns'
            )
            df_results.index = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
            
            return df_results

            
class FILtmStudy(Study[_FILtmStudyState, pd.DataFrame]):
    def __init__(self, start: datetime, end: datetime, interval: Interval, vix: pd.DataFrame) -> None:
        self.start = start
        self.end = end
        self.interval = interval
        self.vix = vix
        
    @classmethod
    def name(cls) -> str:
        return "Fixed Income Long Term Maturity Signal Test"
    
    def prepare_args(self, common_stock: EquityL1, bonds: Sequence[BondL1], risk_free_rate_decimal: float) -> _FILtmStudyImpl:
        state = _FILtmStudyState(common_stock, bonds, self.interval, self.start, self.end, self.vix, risk_free_rate_decimal)
        return _FILtmStudyImpl(state)
        
    