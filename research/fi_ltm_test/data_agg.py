from __future__ import annotations

import lseg.data as ld

from pathlib import Path
from datetime import datetime
import hashlib
import pickle
import quant
from quant.fi_ltm import ParquetBondDfCache
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

CONFIG_PATH = Path(__file__).parent.parent.parent / "lseg-data.config.json"
VIX_PATH = Path(__file__).parent.parent.parent / "data" / "vix.csv"
STORE_PATH = Path(__file__).parent.parent.parent / "data" / "fi_ltm_test"

STUDY_END = datetime(2026, 4, 24)

RISK_FREE_RATE = 0.045

def main():
    vix = pd.read_csv(VIX_PATH, sep='\t', parse_dates=True, index_col=0).rename(columns={"Close": "VIX_Close"})
    print("Loaded ^VIX data")
    session = quant.SessionProvider(CONFIG_PATH)
    
    bond_cache = ParquetBondDfCache(STORE_PATH / "bond_cache")

    with session as client:
        all_us_industrials_equities = ld.discovery.Screener('U(IN(Equity(active,public,primary))/*UNV:Public*/), IN(TR.HQCountryCode,"US"), IN(TR.GICSSectorCode,"20")')
        print("Discovering US Industrials equities...")
        all_us_industrials_equities_rics = list(all_us_industrials_equities)
        print(f"Discovered {len(all_us_industrials_equities_rics)} US Industrials equities")
        rics_hash = hashlib.md5(str(sorted(all_us_industrials_equities_rics)).encode()).hexdigest()[:8]
        equities_cache = STORE_PATH / f"equities_cache_{rics_hash}.pkl"
        if equities_cache.exists():
            with open(equities_cache, "rb") as f:
                equities = pickle.load(f)
            print(f"Loaded {len(equities)} US Industrials equities (from cache)")
        else:
            equities = client.equities.upgrade_l1_equity(all_us_industrials_equities_rics)
            with open(equities_cache, "wb") as f:
                pickle.dump(equities, f)
            print(f"Loaded {len(all_us_industrials_equities_rics)} US Industrials equities")

        for equity in tqdm(equities):
            try:
                bonds_l1 = client.equities.bonds_of_equity(equity)
                if len(bonds_l1) == 0:
                    continue

                bonds = client.bonds.upgrade_l1_bond(bonds_l1)
                if len(bonds) == 0:
                    continue

                start = min([bond.issue_date for bond in bonds if bond.issue_date is not None])

                study = quant.study.fi_ltm.FILtmStudy(datetime.combine(start, datetime.min.time()), STUDY_END, quant.Interval.DAILY, vix)

                ready_to_run = study.prepare_args(equity, bonds, RISK_FREE_RATE, bond_cache=bond_cache)

                result = ready_to_run.run_client(client)

                df = result.to_dataframe()
                if not df.empty:
                    df.to_csv(STORE_PATH / f'{equity.ric}.csv')
            except Exception as e:
                print(f"Skipping {equity.ric}: {e}")

if __name__ == "__main__":
    main()