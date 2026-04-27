from __future__ import annotations

from collections.abc import Sequence
import lseg.data as ld
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional, Callable, Self, Union, cast
import logging
import pandas as pd
from quant import Interval, QuantException, SessionNotOpenError
from quant.bonds import BondL1

logger = logging.getLogger(__name__)

"""
BusinessEntity
PI         RIC
DocumentTitle        PermID
"""

@dataclass
class EquityL1:
    name: Optional[str]
    ric: str
    perm_id: str
    pi: Optional[str]
    business_entity: Optional[str]

    @classmethod
    def from_row(cls, row: pd.Series) -> Self:
        return cls(
            name=row["DocumentTitle"],
            ric=row["RIC"],
            perm_id=row["PermID"],
            pi=row["PI"],
            business_entity=row["BusinessEntity"],
        )

    def exchange(self) -> Optional[str]:
        """Best-effort exchange extraction from the RIC suffix (e.g. 'L' from 'VOD.L')."""
        parts = self.ric.split(".")
        return parts[-1] if len(parts) > 1 else None

    def company(self) -> Optional[str]:
        if self.name is None:
            return None
        
        parts = self.name.split(",")
        if not parts:
            return None
        maybe_company = parts[0].strip()
        return maybe_company if maybe_company else None

    def asset_class(self) -> Optional[str]:
        if self.name is None:
            return None
        
        parts = self.name.split(",")
        if len(parts) < 2:
            return None
        maybe_asset_class = parts[1].strip()
        return maybe_asset_class if maybe_asset_class else None

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
    def _from_row(cls, l1: EquityL1, row: pd.Series) -> Self:
        def _date(val) -> Optional[date]:
            if pd.isna(val):
                return None
            if isinstance(val, date):
                return val
            try:
                return pd.Timestamp(val).date()
            except Exception:
                return None

        def _float(val) -> Optional[float]:
            try:
                return float(val) if pd.notna(val) else None
            except (TypeError, ValueError):
                return None

        def _int(val) -> Optional[int]:
            try:
                return int(val) if pd.notna(val) else None
            except (TypeError, ValueError):
                return None

        def _str(val) -> Optional[str]:
            if pd.isna(val):
                return None
            s = str(val).strip()
            return s if s else None

        return cls(
            name=l1.name,
            ric=l1.ric,
            perm_id=l1.perm_id,
            pi=l1.pi,
            business_entity=l1.business_entity,
            ticker=_str(row["Exchange Ticker"]),
            display_name=_str(row["Company Common Name"]),
            instrument=_str(row["Instrument"]),
            _exchange=_str(row["Exchange Name"]),
            shares_outstanding=_int(row["Diluted Shares - Total"]),
            market_cap=_int(row["Company Market Capitalization"]),
        )

    def company(self) -> Optional[str]:
        return self.display_name if self.display_name is not None else super().company()

    def exchange(self) -> Optional[str]:
        return self._exchange if self._exchange is not None else super().exchange()


# ── Exceptions ────────────────────────────────────────────────────────────────


class EquitiesClientError(QuantException):
    """Base error for all EquitiesClient failures."""


class EquityNotFoundError(EquitiesClientError):
    """Raised when a ticker/RIC cannot be located."""

    def __init__(self, ric: str) -> None:
        super().__init__(f"Equity not found: {ric!r}")
        self.ric = ric





@dataclass
class HistoricalIV:    
    df: pd.DataFrame
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        return cls(df.drop(columns=["IMP_VOLT"]))
    
    @property    
    def c30(self) -> pd.Series[float]:
        return self.df["30D_A_IM_C"]
    
    @property    
    def p30(self) -> pd.Series[float]:
        return self.df["30D_A_IM_P"]
    @property    
    def c60(self) -> pd.Series[float]:
        return self.df["60D_A_IM_C"]
    @property    
    def p60(self) -> pd.Series[float]:
        return self.df["60D_A_IM_P"]
    @property    
    def c90(self) -> pd.Series[float]:
        return self.df["90D_A_IM_C"]
    @property    
    def p90(self) -> pd.Series[float]:
        return self.df["90D_A_IM_P"] 
    


# ── EquityHistoryResult ───────────────────────────────────────────────────────


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
        return isinstance(self.equities, list)

    def get(self, equity_or_ric: EquityL1 | str, field: str) -> pd.Series:
        """Get a specific field for a specific equity."""
        ric = equity_or_ric if isinstance(equity_or_ric, str) else equity_or_ric.ric
        if self.is_multi:
            if ric not in self.df.columns.get_level_values(0):
                raise KeyError(f"{ric!r} not found; available: {self.df.columns.get_level_values(0).unique().tolist()}")
            if field not in self.df[ric].columns:
                raise KeyError(f"{field!r} was not requested; available: {self.df[ric].columns.tolist()}")
            return self.df[ric][field]
        if field not in self.df.columns:
            raise KeyError(f"{field!r} was not requested; available: {self.df.columns.tolist()}")
        return self.df[field]

    def __getitem__(self, key: tuple[EquityL1 | str, str | list[str]] | str) -> pd.Series | pd.DataFrame:
        """
        result["LAST"]                          -> all equities, LAST field (MultiIndex df)
        result[equity, "LAST"]                  -> single equity LAST series
        result[equity, ["BID", "ASK"]]          -> single equity, multiple fields
        result[equity]                          -> all fields for one equity
        """
        if isinstance(key, tuple):
            equity_or_ric, fields = key
            ric = equity_or_ric.ric if hasattr(equity_or_ric, "ric") else equity_or_ric
            if isinstance(fields, list):
                if self.is_multi:
                    if ric not in self.df.columns.get_level_values(0):
                        raise KeyError(f"{ric!r} not found; available: {self.df.columns.get_level_values(0).unique().tolist()}")
                    return self.df[ric].reindex(columns=fields)
                return self.df.reindex(columns=fields)
            return self.get(equity_or_ric, fields)
        if isinstance(key, str):
            field = key
            if self.is_multi:
                if field not in self.df.columns.get_level_values(1):
                    raise KeyError(f"{field!r} was not requested; available: {self.df.columns.get_level_values(1).unique().tolist()}")
                return self.df.xs(field, axis=1, level=1)
            if field not in self.df.columns:
                raise KeyError(f"{field!r} was not requested; available: {self.df.columns.tolist()}")
            return self.df[[field]]
        # assume equity object
        if self.is_multi:
            if key.ric not in self.df.columns.get_level_values(0):
                raise KeyError(f"{key.ric!r} not found; available: {self.df.columns.get_level_values(0).unique().tolist()}")
            return self.df[key.ric]
        return self.df
    
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
        return cls(
            df=df,
            equities=equities,
            fields=fields,
            interval=interval,
            start=start,
            end=end,
        )


# ── EquitiesClient ────────────────────────────────────────────────────────────


class EquitiesClient:
    DEFAULT_UPGRADE_FIELDS = [
        "DocumentTitle",
        "TR.ExchangeTicker", 
        "TR.CommonName", 
        "TR.Ticker", 
        "TR.ExchangeName", 
        "TR.Revenue", 
        "TR.CompanyMarketCapitalization", 
        "TR.NumberofSharesOutstandingActual", 
        "TR.F.ITMShrFulDilComShrOutstTot",
        "TR.RIC",
        
    ]    
    """
    High-level interface for LSEG equity data.

    Not intended to be instantiated directly — obtain via SessionProvider:

        with SessionProvider("config/prod.json") as client:
            equities = client.equities.list_securities("AAPL")
    """

    def __init__(self, is_active: Callable[[], bool]) -> None:
        self.__is_active = is_active

    def list_securities_df(self, ticker: str) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("list_securities_df")

        return ld.discovery.search(
            view=ld.discovery.Views.EQUITY_QUOTES,
            query=ticker,
            top=20,
        ).dropna()

    def list_securities(self, ticker: str) -> Sequence[EquityL1]:
        if not self.__is_active():
            raise SessionNotOpenError("list_securities")

        df = self.list_securities_df(ticker)
        return [EquityL1.from_row(row) for _, row in df.iterrows()]


    @staticmethod
    def __gen_accept_rics_l1(l1: EquityL1 | Sequence[EquityL1] | str | Sequence[str]) -> list[str]:
        if isinstance(l1, str):
            return [l1]
        elif not isinstance(l1, Sequence):
            return [l1.ric]

        return [ric if isinstance(ric, str) else ric.ric for ric in l1]
    
    
    def __gen_init_l1(self, l1: EquityL1 | Sequence[EquityL1] | str | Sequence[str]) -> list[EquityL1]:
        assert self.__is_active()
        
        if not isinstance(l1, str):
            if not isinstance(l1, Sequence):
                return [l1]
            if isinstance(l1, Sequence):
                if all([not isinstance(equity, str) for equity in l1]):
                    return cast(list[EquityL1], l1)
            
            rics = cast(list[str], l1)
        else:
            rics = [l1]
            
        for ric in rics:
            """
            BusinessEntity
            PI         RIC
            DocumentTitle        PermID
            """
            RIC = ld.discovery.SymbolTypes.RIC
            OA_PERM_ID = ld.discovery.SymbolTypes.OA_PERM_ID
            # OA_PERM_ID = ld.discovery.SymbolTypes.NAM
            
            # ric='NVDA.O', perm_id='55839263858', pi='747620',
            
            df = ld.discovery.convert_symbols(symbols=rics, from_symbol_type=RIC, to_symbol_types=[OA_PERM_ID, ])
            
            result = []
            
            for ric in rics:
                l1_i = EquityL1(name=None, ric=ric, perm_id=df.loc[ric, "IssuerOAPermID"], pi=None, business_entity=None)
                result.append(l1_i)
            
            return result
        
    
    def bonds_of_equity(self, l1: EquityL1 | str) -> list[BondL1]:
        if not self.__is_active():
            raise SessionNotOpenError("bonds_of_equity")
                
        if isinstance(l1, str):
            l1 = self.__gen_init_l1(l1)[0]
        
        df = ld.discovery.search(
            view = ld.discovery.Views.GOV_CORP_INSTRUMENTS,
            filter = f"ParentOAPermID eq '{l1.perm_id}' and IsActive eq true and not(AssetStatus in ('MAT'))",
            select = "DocumentTitle, RIC, PermID, PI, BusinessEntity",
            top = 10000)
        
        
        # print('@@@', df.to_string())
        # assert df["RIC"].isna().sum() == 0
        
        return [BondL1.from_row(row) for _, row in df.iterrows() if row["RIC"] is not pd.NA]
        
            
    def upgrade_l1_equity_df(self, l1: EquityL1 | Sequence[EquityL1] | str | Sequence[str], fields: list[str]=DEFAULT_UPGRADE_FIELDS) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("upgrade_l1_equity_df")

        universe = self.__gen_accept_rics_l1(l1)
        
        f = set(fields)
        f_s = set(self.DEFAULT_UPGRADE_FIELDS)
        
        return ld.get_data(universe=universe, fields=list(f | f_s))

    def upgrade_l1_equity(self, l1: EquityL1 | Sequence[EquityL1] | str | Sequence[str], fields=DEFAULT_UPGRADE_FIELDS) -> list[EquityL2]:
        if not self.__is_active():
            raise SessionNotOpenError("upgrade_l1_equity")

        l1 = self.__gen_init_l1(l1)

        df = self.upgrade_l1_equity_df(l1, fields)

        if isinstance(l1, list):
            result = []
            for equity in l1:
                result.append(
                    EquityL2._from_row(equity, df.loc[df["Instrument"] == equity.ric].iloc[0])
                )
            return result
        else:
            return [EquityL2._from_row(l1, row) for _, row in df.iterrows()]

    def history_df(
        self,
        l1: EquityL1 | Sequence[EquityL1] | str | Sequence[str],
        fields=[],
        *,
        interval: Interval,
        start: Optional[Union[date, datetime]] = None,
        end: Optional[Union[date, datetime]] = None,
    ) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("history_df")

        universe = [eq if isinstance(eq, str) else eq.ric for eq in l1] if isinstance(l1, list) else [l1 if isinstance(l1, str) else l1.ric]
        return ld.get_history(universe=universe, fields=fields, start=start, end=end, interval=interval.value).dropna(how='all', axis=0)

    def history(
        self,
        l1: EquityL1 | Sequence[EquityL1],
        fields=[],
        *,
        interval: Interval,
        start: Optional[Union[date, datetime]] = None,
        end: Optional[Union[date, datetime]] = None,
    ) -> EquityHistoryResult:
        if not self.__is_active():
            raise SessionNotOpenError("history")

        if len(fields) == 0:
            raise ValueError("did not request any fields, effectively a no-op")

        df = self.history_df(l1, fields=fields, interval=interval, start=start, end=end)
        return EquityHistoryResult.from_query_result(l1, fields, interval, start, end, df)
    
    def historical_iv(
        self,
        l1: EquityL1 | str,
        *,
        interval: Interval,
        start: Optional[Union[date, datetime]] = None,
        end: Optional[Union[date, datetime]] = None,
    ) -> HistoricalIV:
        if not self.__is_active():
            raise SessionNotOpenError("historical_iv")
        
        l1_ric = l1 if isinstance(l1, str) else l1.ric
        
        if '.' in l1_ric:
            ric_parts = l1_ric.split('.')
            assert len(ric_parts) == 2
            l1_ric = ric_parts[0]
        
        vol_ric = f"{l1_ric}ATMIV.U"
        
        df = ld.get_history(universe=[vol_ric], interval=interval.value, start=start, end=end)
        
        return HistoricalIV.from_df(df)