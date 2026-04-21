from enum import Enum
from typing import Callable
import lseg.data as ld
import pandas as pd
from .series_group import SeriesGroup, SeriesGroupStack, Axis
from . import kmv
from . import kmv_timeseries
from . import credit_direction

class Interval(str, Enum):
    TICK = "tick"
    TAS = "tas"
    TAQ = "taq"
    MINUTE = "minute"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_10 = "10min"
    MIN_30 = "30min"
    MIN_60 = "60min"
    HOURLY = "hourly"
    HOUR_1 = "1h"
    DAILY = "daily"
    DAY_1 = "1d"
    DAY_1_UPPER = "1D"
    DAY_7_UPPER = "7D"
    DAY_7 = "7d"
    WEEKLY = "weekly"
    WEEK_1 = "1W"
    MONTHLY = "monthly"
    MONTH_1 = "1M"
    MONTH_3 = "3M"
    MONTH_6 = "6M"
    YEARLY = "yearly"
    YEAR_1 = "1Y"

class QuantException(Exception):
    """Base error for all library failures."""

class SessionNotOpenError(QuantException):
    """Raised when an operation is attempted without an active session."""

from quant.bonds import BondsClient
from quant.equities import EquitiesClient
from quant.options import OptionsClient, OptionChainExchangeHost, OptionsChainResult, OptionChainRICFormatter

class Client:
    def __init__(self, is_active: Callable[[], bool]):
        self.bonds = BondsClient(is_active)
        self.equities = EquitiesClient(is_active)
        self.options = OptionsClient(is_active)


class SessionProvider:
    def __init__(self, config_path):
        self.__config_path = config_path
        self.__open = False

    def __enter__(self):
        self.__session = ld.open_session(config_name=self.__config_path)
        self.__open = True
        return Client(self.open)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.__session.close()
        self.__open = False
        
    def open(self) -> bool:
        return self.__open
