import pandas as pd

from quant.kmv_timeseries import KMVTimeSeries

class CreditDirectionInput:
    DEFAULT_COLUMNS = {}
    
    def __init__(self, dd: KMVTimeSeries, df: pd.DataFrame, columns: dict[str, str] = DEFAULT_COLUMNS):
        # σ_V is asset_volatility in KMVResult
        
        
        ...