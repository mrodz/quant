from collections.abc import Sequence
import lseg.data as ld
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Callable, Self, Union
import logging
import pandas as pd
from datetime import datetime
from quant import Interval, QuantException, SessionNotOpenError

logger = logging.getLogger(__name__)

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
    def from_row(cls, row: pd.Series, *, ric: Optional[str] = None) -> Self:
        return cls(
            name=row["DocumentTitle"],
            ric=ric if ric is not None else row["RIC"],
            perm_id=row["PermID"],
            pi=row["PI"],
            business_entity=row["BusinessEntity"],
        )
        
    def maturity(self) -> Optional[date]:
        parts = self.name.split()
        if not parts:
            return None
        
        maybe_date_part = parts[-1]
        
        try:
            return datetime.strptime(maybe_date_part, "%d-%b-%Y").date()
        except ValueError:
            return None
        
    def coupon(self) -> Optional[float]:
        parts = self.name.split()
        if not parts:
            return None
        
        maybe_coupon_part = parts[-2]
        
        try:
            return float(maybe_coupon_part)
        except ValueError:
            return None
        
    
    def company(self) -> Optional[str]:
        parts = self.name.split(',')
        if not parts:
            return None
        
        maybe_company_part = parts[0].strip()
        
        if not maybe_company_part:
            return None
        
        return maybe_company_part


    def asset_class(self) -> Optional[str]:
        parts = self.name.split(',')
        if not parts:
            return None
        
        if len(parts) < 2:
            return None
        
        maybe_asset_class_part = parts[1].strip()
        
        if not maybe_asset_class_part:
            return None
        
        return maybe_asset_class_part


@dataclass
class BondL2(BondL1):
    # Identity / issuance
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
    def _from_row(cls, l1: BondL1, row: pd.Series) -> Self:
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
            # BondL1 fields — adapt to your actual L1 source fields
            name=l1.name,
            ric=l1.ric,
            perm_id=l1.perm_id,
            pi=l1.pi,
            business_entity=l1.business_entity,
            # Identity
            isin=_str(row.get("OFFCL_CODE")),
            cusip=_str(row.get("OFFC_CODE2")),
            ticker=_str(row.get("TICKER")),
            currency=_str(row.get("CURRENCY")),
            issue_date=_date(row.get("ISSUE_DATE")),
            maturity_date=_date(row.get("MATUR_DATE")),
            coupon_rate=_float(row.get("COUPN_RATE")),
            coupon_frequency=_str(row.get("PAY_FREQ")),
            bond_type=_str(row.get("BOND_TYPE")),
            amount_outstanding=_int(row.get("AMT_OS")),
        
            instrument=_str(row.get("Instrument")),
            display_name=_str(row.get("DSPLY_NAME")),
            gv1_text=_str(row.get("GV1_TEXT")),
            gv2_text=_str(row.get("GV2_TEXT")),

            # Pricing
            clean_price=_float(row.get("CLEAN_PRC")),
            dirty_price=_float(row.get("DIRTY_PRC")),
            bid=_float(row.get("BID")),
            ask=_float(row.get("ASK")),
            mid=_float(row.get("MID_1")),
            open_price=_float(row.get("OPEN_PRC")),
            hist_close=_float(row.get("HST_CLOSE")),
            settle_date=_date(row.get("SETTLEDATE")),

            # Yield
            yield_to_maturity=_float(row.get("YLDTOMAT")),
            bid_yield=_float(row.get("BID_YIELD")),
            ask_yield=_float(row.get("ASK_YIELD")),
            mid_yield=_float(row.get("MID_YLD_1")),
            hist_close_yield=_float(row.get("HST_CLSYLD")),

            # Risk
            duration=_float(row.get("DURATION")),
            modified_duration=_float(row.get("MOD_DURTN")),
            convexity=_float(row.get("CONVEXITY")),
            bpv=_float(row.get("BPV")),
            accrued_interest=_float(row.get("ACCR_INT")),
            days_to_maturity=_int(row.get("DAYS_MAT")),
            accrued_days=_int(row.get("ACC_DAYS")),

            # Spreads
            benchmark_spread=_float(row.get("BMK_SPD")),
            swap_spread=_float(row.get("SWAP_SPRD")),
            oas_bid=_float(row.get("OAS_BID")),
        )
        
    def company(self) -> Optional[str]:
        return self.display_name if self.dirty_price is not None else super().company()
    
    def maturity(self) -> date | None:
        return self.maturity_date if self.maturity_date is not None else super().maturity()
 
 
# ── Exceptions ───────────────────────────────────────────────────────────────
 
 
class BondsClientError(QuantException):
    """Base error for all BondsClient failures."""
 
 
class BondNotFoundError(BondsClientError):
    """Raised when an ISIN cannot be located."""
 
    def __init__(self, isin: str) -> None:
        super().__init__(f"Bond not found: {isin!r}")
        self.isin = isin
 
 
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
        return isinstance(self.bonds, list)
    
    def _ric(self, bond_or_ric: BondL1 | str) -> str:
        return bond_or_ric.ric if hasattr(bond_or_ric, "ric") else bond_or_ric

    def get(self, bond_or_ric: BondL1 | str, field: str) -> pd.Series:
        """Get a specific field for a specific bond."""
        ric = self._ric(bond_or_ric)
        if self.is_multi:
            if ric not in self.df.columns.get_level_values(0):
                raise KeyError(f"{ric!r} not found; available: {self.df.columns.get_level_values(0).unique().tolist()}")
            if field not in self.df.columns.get_level_values(1):
                raise KeyError(f"{field!r} not in columns; available: {self.df.columns.get_level_values(1).unique().tolist()}")
            return self.df[ric][field]
        # single-bond flat df
        if field not in self.df.columns:
            raise KeyError(f"{field!r} not in columns; available: {self.df.columns.tolist()}")
        return self.df[field]

    def __getitem__(
        self,
        key: tuple[BondL1 | str, str | list[str]] | BondL1 | str,
    ) -> pd.Series | pd.DataFrame:
        # ── tuple: (bond_or_ric, field | [fields]) ──────────────────────────────
        if isinstance(key, tuple):
            bond_or_ric, fields = key
            ric = self._ric(bond_or_ric)
            if self.is_multi:
                if ric not in self.df.columns.get_level_values(0):
                    raise KeyError(f"{ric!r} not found; available: {self.df.columns.get_level_values(0).unique().tolist()}")
                bond_df = self.df[ric]  # flat df for this one bond
            else:
                bond_df = self.df
            if isinstance(fields, list):
                missing = [f for f in fields if f not in bond_df.columns]
                if missing:
                    raise KeyError(f"{missing} not in columns; available: {bond_df.columns.tolist()}")
                return bond_df[fields]
            return self.get(bond_or_ric, fields)  # single field → Series

        # ── bare string: result["BID"] → all bonds, that field ──────────────────
        # ── bare string: result["BID"] → all bonds, that field ──────────────────
        if isinstance(key, str):
            field = key
            if self.is_multi:
                if not isinstance(self.df.columns, pd.MultiIndex):
                    raise KeyError(
                        f"DataFrame columns are not a MultiIndex; columns: {self.df.columns.tolist()}"
                    )
                available = self.df.columns.get_level_values(1).unique().tolist()
                if field not in available:
                    raise KeyError(f"{field!r} not requested; available: {available}")
                return self.df.xs(field, axis=1, level=1)
            if field not in self.df.columns:
                raise KeyError(f"{field!r} not requested; available: {self.df.columns.tolist()}")
            return self.df[[field]]

        # ── bare BondL1: result[bond] → all fields for that bond ─────────────────
        ric = self._ric(key)
        if self.is_multi:
            if ric not in self.df.columns.get_level_values(0):
                raise KeyError(f"{ric!r} not found; available: {self.df.columns.get_level_values(0).unique().tolist()}")
            return self.df[ric]
        return self.df
    
    @classmethod
    def from_query_result(cls, bonds, fields, interval, start, end, df):
        if not isinstance(df.columns, pd.MultiIndex):
            if len(fields) == 1:
                # Single field, one or more bonds — columns are RICs
                df.columns = pd.MultiIndex.from_arrays(
                    [df.columns, [fields[0]] * len(df.columns)],
                    names=["RIC", "field"],
                )
            else:
                # Fallback: flat columns with multiple fields — shouldn't normally
                # happen, but wrap defensively
                pass
        return cls(df=df, bonds=bonds, fields=fields, interval=interval, start=start, end=end)
 
def _require_isin(isin: str) -> str:
    isin = isin.strip().upper()
    if len(isin) != 12:
        raise ValueError(f"ISIN must be 12 characters, got {len(isin)!r}: {isin!r}")
    return isin
 
 
def _parse_date(value) -> date:
    """Coerce a string, datetime, or date to a date object."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"Cannot convert {type(value)} to date")
 
 
# ── BondsClient ──────────────────────────────────────────────────────────────
 
 
class BondsClient:
    """
    High-level interface for LSEG bond data.
 
    Not intended to be instantiated directly — obtain via SessionProvider:
 
        with SessionProvider("config/prod.json") as client:
            bond = client.get("US912828ZL32")
    """
 
    def __init__(self, is_active: Callable[[], bool]) -> None:
        self.__is_active = is_active
 
    def list_securities_df(self, ticker: str) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("list_securities_df")
        
        return ld.discovery.search(
            view=ld.discovery.Views.GOV_CORP_INSTRUMENTS,
            query=ticker,
            top=20
        ).dropna()
        
    def securities_from_equity_ric(self, ric: str) -> list[BondL1]:
        if not self.__is_active():
            raise SessionNotOpenError("securities_from_equity_ric")

        df = ld.get_data(
            universe=ric,
            fields=["DocumentTitle", "RIC", "PermID", "PI", "BusinessEntity"],
            parameters={"SType": "bonds"},
        )

        if df is None or df.empty:
            return []

        return [BondL1.from_row(row) for _, row in df.iterrows()]
    
    
    def list_securities(self, ticker: str) -> list[BondL1]:
        if not self.__is_active():
            raise SessionNotOpenError("list_securities")
        
        df = self.list_securities_df(ticker)
        return [BondL1.from_row(row) for _, row in df.iterrows()]
    
 
    # ── Search ────────────────────────────────────────────────────────────────
    def upgrade_l1_bond_df(self, l1: BondL1 | list[BondL1], fields=[]) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("upgrade_l1_bond_df")
        
        universe = [bond.ric for bond in l1] if isinstance(l1, list) else [l1.ric]
        universe = list(filter(lambda ric: not pd.isna(ric), universe))
        return ld.get_data(universe=universe, fields=fields)
    
    def upgrade_l1_bond(self, l1: BondL1 | list[BondL1], fields=[]) -> Sequence[BondL2]:
        df = self.upgrade_l1_bond_df(l1, fields)
         
        if isinstance(l1, list):
            if "Instrument" not in df.columns:
                return []

            result = []

            for bond in l1:
                print(bond)
                print("#" * 25)
                for x in df["Instrument"]:
                    print(x)
                print("\n\n")

                rows = df.loc[df["Instrument"] == bond.ric]
                if rows.empty:
                    continue
                result.append(BondL2._from_row(bond, rows.iloc[0]))

            return result
        else:
            return [BondL2._from_row(l1, row) for _, row in df.iterrows()]
        
        
    def history_df(self, l1: BondL1 | Sequence[BondL1] | str | list[str], fields=[], *, interval: Interval, start: Optional[Union[date, datetime]] = None, end: Optional[Union[date, datetime]] = None) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("pricing_df")
        
        universe = [bond if isinstance(bond, str) else bond.ric for bond in l1] if isinstance(l1, list) else [l1 if isinstance(l1, str) else l1.ric]
        
        return ld.get_history(universe=universe, fields=fields, start=start, end=end, interval=interval.value).dropna(how='all', axis=0)
    
    def history(self, l1: BondL1 | Sequence[BondL1], fields=[], *, interval: Interval, start: Optional[Union[date, datetime]] = None, end: Optional[Union[date, datetime]] = None) -> BondHistoryResult:
        if not self.__is_active():
            raise SessionNotOpenError("history")
        
        if len(fields) == 0:
            raise ValueError("did not request any fields, effectively a no-op")
        
        df = self.history_df(l1, fields=fields, interval=interval, start=start, end=end)
        
        result = BondHistoryResult.from_query_result(l1, fields, interval, start, end, df)
        
        return result
        
        
        