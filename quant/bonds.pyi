from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Self, Union
import pandas as pd
from quant import Interval

"""
BusinessEntity
PI         RIC 
DocumentTitle        PermID
"""
@dataclass
class BondL1:
    name: str
    ric: str
    perm_id: str
    pi: str
    business_entity: str
    
    @classmethod
    def from_row(cls, row: pd.Series) -> Self:
        ...
        
    def maturity(self) -> Optional[date]:
        ...
        
    def company(self) -> Optional[str]:
        ...
        
    def asset_class(self) -> Optional[str]:
        ...
        
    def upgrade_l1_bond_df(self, l1: BondL1, fields=[]) -> pd.DataFrame:
        ...
        
    def upgrade_l1_bond(self, l1: BondL1 | list[BondL1], fields=[]) -> Sequence[BondL2]:
        ...

@dataclass
class BondL2(BondL1):    
    isin: Optional[str]
    cusip: Optional[str]
    ticker: Optional[str]
    currency: Optional[str]
    issue_date: Optional[date]
    maturity_date: Optional[date]
    coupon_rate: Optional[float]
    coupon_frequency: Optional[str]
    bond_type: Optional[str]
    amount_outstanding: Optional[int]   # AMT_OS / PAR_AMT, in millions

    display_name: Optional[str]
    instrument: Optional[str]
    gv1_text: Optional[str]
    gv2_text: Optional[str]

    # Pricing
    clean_price: Optional[float]        # CLEAN_PRC
    dirty_price: Optional[float]        # DIRTY_PRC
    bid: Optional[float]                # BID
    ask: Optional[float]                # ASK
    mid: Optional[float]                # MID_1
    open_price: Optional[float]         # OPEN_PRC
    hist_close: Optional[float]         # HST_CLOSE
    settle_date: Optional[date]         # SETTLEDATE

    # Yield
    yield_to_maturity: Optional[float]  # YLDTOMAT
    bid_yield: Optional[float]          # BID_YIELD
    ask_yield: Optional[float]          # ASK_YIELD
    mid_yield: Optional[float]          # MID_YLD_1
    hist_close_yield: Optional[float]   # HST_CLSYLD

    # Risk
    duration: Optional[float]           # DURATION
    modified_duration: Optional[float]  # MOD_DURTN
    convexity: Optional[float]          # CONVEXITY
    bpv: Optional[float]                # BPV (basis point value)
    accrued_interest: Optional[float]   # ACCR_INT
    days_to_maturity: Optional[int]     # DAYS_MAT
    accrued_days: Optional[int]         # ACC_DAYS

    # Spreads
    benchmark_spread: Optional[float]   # BMK_SPD
    swap_spread: Optional[float]        # SWAP_SPRD
    oas_bid: Optional[float]            # OAS_BID
    
    @classmethod
    def from_row(cls, row: pd.Series) -> Self:
        ...


@dataclass
class BondHistoryResult:
    df: pd.DataFrame
    bonds: Union[BondL1, Sequence[BondL1]]
    fields: list[str]
    interval: Interval
    start: Optional[Union[date, datetime]] = None
    end: Optional[Union[date, datetime]] = None
    
    @property
    def is_multi(self) -> bool:
        ...
            
    def get(self, bond_or_ric: BondL1 | str, field: str) -> pd.Series:
        """Get a specific field for a specific bond."""
        ...
    
    def __getitem__(self, key: tuple[BondL1 | str, str | list[str]] | str) -> pd.Series | pd.DataFrame:
        """
        result["BID"]                           -> all bonds, BID field (MultiIndex df)
        result[bond, "BID"]                     -> single bond BID series
        result[bond, ["BID", "ASK"]]            -> single bond, multiple fields
        result[bond]                            -> all fields for one bond
        """
        ...
    
    @classmethod
    def from_query_result(cls, bonds: Union[BondL1, Sequence[BondL1]], fields: list[str], interval: Interval, start: Optional[Union[date, datetime]], end: Optional[Union[date, datetime]], df: pd.DataFrame) -> Self:
        ...


class BondsClient:
    def __init__(self, session):
        self._session = session

    def list_securities(self, ticker: str) -> Sequence[BondL1]:
        ...

    def list_securities_df(self, ticker: str) -> pd.DataFrame:
        ...

    def upgrade_l1_bond_df(self, l1: BondL1 | Sequence[BondL1], fields=[]) -> pd.DataFrame:
        ...
        
    def upgrade_l1_bond(self, l1: BondL1 | Sequence[BondL1], fields=[]) -> list[BondL2]:
        ...
        
    def history_df(self, l1: BondL1 | Sequence[BondL1] | str | list[str], fields=[], *, interval: Interval, start: Optional[Union[date, datetime]] = None, end: Optional[Union[date, datetime]] = None) -> pd.DataFrame:
        ...
    
    def history(self, l1: BondL1 | Sequence[BondL1], fields=[], *, interval: Interval, start: Optional[Union[date, datetime]] = None, end: Optional[Union[date, datetime]] = None) -> BondHistoryResult:
        ...