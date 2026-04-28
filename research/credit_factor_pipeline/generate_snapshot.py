"""
generate_snapshot.py — Generate a snapshot CSV for one equity RIC
Usage:
    python generate_snapshot.py MO
    python generate_snapshot.py MO --out /path/to/output.csv
"""

from __future__ import annotations
import argparse
import warnings
warnings.filterwarnings(action="ignore")

from pathlib import Path
from datetime import datetime

import quant
import pandas as pd

CONFIG_PATH    = Path(__file__).parent.parent.parent / "lseg-data.config.json"
VIX_PATH       = Path(__file__).parent.parent.parent / "data" / "vix.csv"

STUDY_END      = datetime(2026, 4, 24)
RISK_FREE_RATE = 0.045


def generate(equity_ric: str, out_path: Path):
    vix = pd.read_csv(VIX_PATH, sep="\t", parse_dates=True, index_col=0).rename(
        columns={"Close": "VIX_Close"}
    )

    session = quant.SessionProvider(CONFIG_PATH)

    with session as client:
        stocks = client.equities.list_securities(equity_ric)
        equity = next((s for s in stocks if s.ric == equity_ric), None)

        if equity is None:
            if not stocks:
                raise SystemExit(f"No equity found for: {equity_ric}")
            equity = stocks[0]
            print(f"Using equity: {equity.ric} — {equity.name}")

        bonds = client.bonds.list_securities(equity.company() if equity.name is not None else equity.ric)

        if not bonds:
            raise SystemExit(f"No bonds found for: {equity.company() if equity.name is not None else equity.ric}")

        print(f"\nFound {len(bonds)} bond(s) for {equity.name}:")
        print(f"  {'RIC':<20}  {'Name'}")
        print(f"  {'─'*20}  {'─'*45}")
        for b in sorted(bonds, key=lambda x: getattr(x, 'name', '') or ''):
            print(f"  {b.ric:<20}  {getattr(b, 'name', 'n/a')}")
        print()

        study = quant.study.fi_ltm.FILtmStudy(
            datetime(1997, 1, 18),
            STUDY_END,
            quant.Interval.DAILY,
            vix,
        )

        result = study.prepare_args(equity, bonds, RISK_FREE_RATE).run(session)
        df     = result.to_dataframe()

        if df.empty:
            raise SystemExit(f"Study returned empty dataframe for {equity_ric}")

        if "Date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "Date"})

        print(f"Saved {len(df):,} rows ({df['Date'].min()} → {df['Date'].max()}) → {out_path}")
        df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate snapshot CSV for one equity RIC")
    parser.add_argument("ric",  help="Equity RIC, e.g. MO or WSC.OQ")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: <ric>.csv in current directory)")
    args = parser.parse_args()

    out = args.out or Path(f"{args.ric.replace('.', '_')}.csv")
    generate(args.ric, out)


if __name__ == "__main__":
    main()