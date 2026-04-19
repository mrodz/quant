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
class EquityL1:
    name: str
    ric: str
    perm_id: str
    pi: str
    business_entity: str

    @classmethod
    def from_row(cls, row: pd.Series) -> Self:
        ...

    def exchange(self) -> Optional[str]:
        ...

    def company(self) -> Optional[str]:
        ...

    def asset_class(self) -> Optional[str]:
        ...

    def upgrade_l1_equity_df(self, l1: "EquityL1", fields=[]) -> pd.DataFrame:
        ...

    def upgrade_l1_equity(self, l1: "EquityL1", fields=[]) -> "EquityL2":
        ...


@dataclass
class EquityL2(EquityL1):
    ticker: Optional[str]

    display_name: Optional[str]
    instrument: Optional[str]

    _exchange: Optional[str]             # EXCH_NAME
    shares_outstanding: Optional[int]   # SHARES_OUT

    # Valuation
    market_cap: Optional[float]         # MKT_CAP

    @classmethod
    def from_row(cls, row: pd.Series) -> Self:
        ...


@dataclass
class EquityHistoryResult:
    df: pd.DataFrame
    equities: Union[EquityL1, Sequence[EquityL1]]
    fields: list[str]
    interval: Interval
    start: Optional[Union[date, datetime]] = None
    end: Optional[Union[date, datetime]] = None

    @property
    def is_multi(self) -> bool:
        ...

    def get(self, equity_or_ric: EquityL1 | str, field: str) -> pd.Series:
        """Get a specific field for a specific equity."""
        ...

    def __getitem__(self, key: tuple[EquityL1 | str, str | list[str]] | str) -> pd.Series | pd.DataFrame:
        """
        result["LAST"]                          -> all equities, LAST field (MultiIndex df)
        result[equity, "LAST"]                  -> single equity LAST series
        result[equity, ["BID", "ASK"]]          -> single equity, multiple fields
        result[equity]                          -> all fields for one equity
        """
        ...

    @classmethod
    def from_query_result(
        cls,
        equities: Union[EquityL1, Sequence[EquityL1]],
        fields: list[str],
        interval: Interval,
        start: Optional[Union[date, datetime]],
        end: Optional[Union[date, datetime]],
        df: pd.DataFrame,
    ) -> Self:
        ...


class EquitiesClient:
    DEFAULT_UPGRADE_FIELDS = [
        "TR.ExchangeTicker", 
        "TR.CommonName", 
        "TR.Ticker", 
        "TR.ExchangeName", 
        "TR.Revenue", 
        "TR.CompanyMarketCapitalization", 
        "TR.NumberofSharesOutstandingActual", 
        "TR.F.ITMShrFulDilComShrOutstTot"
    ]    
    
    def __init__(self, session):
        self._session = session

    def list_securities(self, ticker: str) -> Sequence[EquityL1]:
        ...

    def list_securities_df(self, ticker: str) -> pd.DataFrame:
        ...

    def upgrade_l1_equity_df(self, l1: EquityL1 | Sequence[EquityL1], fields=DEFAULT_UPGRADE_FIELDS) -> pd.DataFrame:
        ...

    def upgrade_l1_equity(self, l1: EquityL1 | Sequence[EquityL1], fields=DEFAULT_UPGRADE_FIELDS) -> list[EquityL2]:
        ...

    def history_df(
        self,
        l1: EquityL1 | Sequence[EquityL1],
        fields=[],
        *,
        interval: Interval,
        start: Optional[Union[date, datetime]] = None,
        end: Optional[Union[date, datetime]] = None,
    ) -> pd.DataFrame:
        ...

    def history(
        self,
        l1: EquityL1 | Sequence[EquityL1],
        fields=[],
        *,
        interval: Interval,
        start: Optional[Union[date, datetime]] = None,
        end: Optional[Union[date, datetime]] = None,
    ) -> EquityHistoryResult:
        ...