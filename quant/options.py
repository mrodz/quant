from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Callable, Optional, Union
from datetime import date, datetime

import lseg.data as ld
import pandas as pd

from quant import SessionNotOpenError, Interval
from quant.equities import EquityL1

import time
from tqdm import tqdm

class OptionChainExchangeHost(Enum):
    NYSE = 'N'
    NASDAQ = 'O'
    LONDON = 'L'
    TOKYO = 'T'
    NYMEX = 'U'
    CBOT = 'CBT'
    CME = 'CME'


class OptionChainRICFormatter(ABC):
    def format(self, ric: str, exchange: OptionChainExchangeHost | str) -> str:
        exchange_code = exchange if isinstance(exchange, str) else exchange.value
        return f"0#{ric}*.{exchange_code}"
        

class OptionsChainResult:
    def __init__(self, chain: ld.discovery.Chain) -> None:
        self.constituents = chain.constituents
        self.summary_links = chain.summary_links
        self.name = chain.name

class OptionsClient:
    DEFAULT_CHAIN_RIC_FORMATTER = OptionChainRICFormatter()
    
    def __init__(self, is_active: Callable[[], bool]):
        self.__is_active = is_active
        
    def chain(self, underlying: str | EquityL1, exchange: OptionChainExchangeHost | str, formatter: OptionChainRICFormatter = DEFAULT_CHAIN_RIC_FORMATTER) -> OptionsChainResult:
        if not self.__is_active():
            raise SessionNotOpenError("chain")
        
        underlying_ric = underlying if isinstance(underlying, str) else underlying.ric
        option_ric = formatter.format(underlying_ric, exchange)
        
        chain = ld.discovery.Chain(option_ric)
        
        result = OptionsChainResult(chain)
        
        return result
    
    def chain_history_df(self, chain: OptionsChainResult, fields=[], *, interval: Interval, start: Optional[Union[date, datetime]] = None, end: Optional[Union[date, datetime]] = None, batch = True) -> pd.DataFrame:
        if not self.__is_active():
            raise SessionNotOpenError("chain_history")
        
        if not batch:
            return ld.get_history(universe=chain.constituents, fields=fields, start=start, end=end, interval=interval.value).dropna(how='all', axis=0)
        
        constituents = chain.constituents
        batch_size = 3
        all_batches = []
        
        # Rate limit: 5 requests per second = 0.2s delay between calls
        rate_limit_delay = 0.2 

        # Iterate through constituents in steps of 3
        for i in tqdm(range(0, len(constituents), batch_size), desc="Fetching Option History"):
            batch_universe = constituents[i : i + batch_size]
            
            try:
                # Record start time to ensure precise rate limiting
                start_time = time.time()
                
                # Request historical data for the current batch
                df_batch = ld.get_history(
                    universe=batch_universe, 
                    fields=fields, 
                    start=start, 
                    end=end, 
                    interval=interval.value
                )
                
                if not df_batch.empty:
                    all_batches.append(df_batch)
                    
                # Calculate how long the request took and sleep if necessary
                elapsed = time.time() - start_time
                if elapsed < rate_limit_delay:
                    time.sleep(rate_limit_delay - elapsed)
                        
            except Exception as e:
                print(f"\nError fetching batch starting at index {i}: {e}")
                # Still sleep on error to maintain rate limit integrity
                time.sleep(rate_limit_delay)

        # Concatenate all batches into a single DataFrame
        if not all_batches:
            return pd.DataFrame()

        return pd.concat(all_batches).dropna(how='all', axis=0)