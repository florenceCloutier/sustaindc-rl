"""Build SustainDC carbon-intensity files (data/CarbonIntensity/<BA>_<YEAR>_NG_&_avgCI.csv) from EIA-930 data.

Source: EIA's Hourly Electric Grid Monitor six-month BALANCE files (public, no account), e.g.
    https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv
Year Y needs (Y, Jan_Jun), (Y, Jul_Dec) and (Y-1, Jul_Dec): the first UTC hours of Jan 1 are Dec 31 local time.

Rule (it reproduces the repo's 2022 CISO and ERCO files exactly): raw hourly net generation by fuel (not the imputed or
adjusted columns), indexed by the UTC time at the END of the hour; missing values -> 0 and negative values (battery
charging and pumping, which EIA reports in "Other" and "Hydro", solar at night) -> 0; avg_CI = generation-weighted
mean of lifecycle emission factors (g CO2/kWh). Battery discharge therefore counts as "Other" (230 g/kWh), as in 2022.
Hours EIA has no row for stay empty; utils/managers.CI_Manager fills them with the year's mean.
The simulator's location files are copies under the location's name: CA = CISO, TX = ERCO (utils/utils_cf.obtain_paths).

Usage: python scripts/build_ci_eia930.py RAW_DIR OUT_DIR YEAR BA [BA ...]
    e.g. python scripts/build_ci_eia930.py raw data/CarbonIntensity 2023 CISO ERCO
"""
import sys
import pandas as pd

RAW, OUT, YEAR, BAS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
FUELS = {'WND': 'Wind', 'SUN': 'Solar', 'WAT': 'Hydropower and Pumped Storage', 'OIL': 'All Petroleum Products',
         'NG': 'Natural Gas', 'COL': 'Coal', 'NUC': 'Nuclear', 'OTH': 'Other Fuel Sources'}
ALIASES = {'CISO': 'CA', 'ERCO': 'TX', 'AZPS': 'AZ', 'NYIS': 'NY'}  # simulator location names
EF = {'WND': 11, 'SUN': 41, 'WAT': 24, 'OIL': 650, 'NG': 490, 'COL': 820, 'NUC': 12, 'OTH': 230}
files = [f'{RAW}/EIA930_BALANCE_{y}_{h}.csv' for y, h in ((YEAR - 1, 'Jul_Dec'), (YEAR, 'Jan_Jun'), (YEAR, 'Jul_Dec'))]
frames = []
for f in files:
    try:
        frames.append(pd.read_csv(f, thousands=',', low_memory=False))
    except FileNotFoundError:
        print(f'  (missing {f}: the first UTC hours of Jan 1 will be empty)')
d = pd.concat(frames)
d = d[d['Balancing Authority'].isin(BAS)].copy()
d['ts'] = pd.to_datetime(d['UTC Time at End of Hour'], format='%m/%d/%Y %I:%M:%S %p', utc=True)
index = pd.date_range(f'{YEAR}-01-01 00:00', f'{YEAR}-12-31 23:00', freq='h', tz='UTC', name='timestamp')
for ba in BAS:
    x = d[d['Balancing Authority'] == ba].drop_duplicates('ts').set_index('ts').reindex(index)
    present = x['Balancing Authority'].notna()
    g = pd.DataFrame({k: x[f'Net Generation (MW) from {v}'] for k, v in FUELS.items()}, index=index)
    g = g.fillna(0).clip(lower=0).where(present)  # rows EIA does not have stay NaN
    g['avg_CI'] = sum(g[k] * EF[k] for k in EF) / g[list(FUELS)].sum(axis=1)
    path = f'{OUT}/{ba}_{YEAR}_NG_&_avgCI.csv'
    g.to_csv(path, date_format='%Y-%m-%d %H:%M:%S+00:00')
    print(f'{path}: {len(g)} rows, {int((~present).sum())} hours without EIA data')
    if ba in ALIASES:
        alias = f'{OUT}/{ALIASES[ba]}_{YEAR}_NG_&_avgCI.csv'
        g.to_csv(alias, date_format='%Y-%m-%d %H:%M:%S+00:00')
        print(f'{alias}: copy for the simulator location')
