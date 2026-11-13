"""Convert DCcluster-Opt (HPE sustain-cluster, MIT licence) data into SustainDC's file formats.

Source: a clone of https://github.com/HewlettPackard/sustain-cluster (default ../sustain-cluster).

Writes, per region and year:
  data/CarbonIntensity/DCC_<REGION>_<YEAR>_NG_&_avgCI.csv   hourly CI (Electricity Maps, direct gCO2/kWh), time-stamped
                                                            in the region's *local standard time* (so CI_Manager does
                                                            not shift it: row k = local standard hour k, like the EPW)
  data/Weather/DCC_<REGION>_<YEAR>.epw                     EPW file (Open-Meteo dry bulb, RH, pressure from elevation,
                                                            wind, cloud cover; irradiance 0 -> no PV for these regions),
                                                            rows in local standard time
and once:
  data/Jobs/alibaba2020_jobs.npz                           the Alibaba 2020 GPU job trace as plain arrays (no pickle)

Sites then use location 'dcc:<REGION>[:<YEAR>]' (see utils.utils_cf.obtain_paths).

The source series are hourly in UTC (CI) or local wall-clock time with DST (weather). Both are converted to UTC, then to
the region's fixed standard offset; the neighbouring years' files fill the hours that fall across the year boundary.

Run from the repo root:
    python scripts/import_dccluster_data.py --src ../sustain-cluster --years 2023
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
import zoneinfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def region_tz(src, region, year):
    with open(os.path.join(src, 'data', 'weather', region, f'{year}.json')) as f:
        return zoneinfo.ZoneInfo(json.load(f)['timezone'])


def standard_offset_hours(tz, year):
    """UTC offset without daylight saving (EPW and SustainDC use local standard time)."""
    t = dt.datetime(year, 1, 15, 12, tzinfo=tz)
    return (t.utcoffset() - t.dst()).total_seconds() / 3600


def local_standard_index(year, offset_h):
    """UTC timestamps of the hours 00:00 Jan 1 .. 23:00 Dec 31 of `year` in local standard time (8760 rows)."""
    local = pd.date_range(f'{year}-01-01', periods=8760, freq='h')
    return local - pd.Timedelta(hours=offset_h)  # naive UTC


def load_ci_utc(src, region, years):
    frames = []
    for y in years:
        for path in glob.glob(os.path.join(src, 'data', 'carbon_intensity', region, str(y), '*.csv')):
            df = pd.read_csv(path)
            frames.append(pd.Series(df['Carbon Intensity gCO₂eq/kWh (direct)'].values,
                                    index=pd.to_datetime(df['Datetime (UTC)'])))
    s = pd.concat(frames).sort_index()
    return s[~s.index.duplicated()]


def load_weather_utc(src, region, years):
    frames, meta = [], None
    for y in years:
        path = os.path.join(src, 'data', 'weather', region, f'{y}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        meta = meta or d
        h = d['hourly']
        local = pd.to_datetime(h['time']).tz_localize(d['timezone'], ambiguous='NaT', nonexistent='NaT')
        df = pd.DataFrame({'temp': h['temperature_2m'], 'rh': h['relative_humidity_2m'],
                           'cloud': h.get('cloudcover', [np.nan] * len(local)),
                           'wind_kmh': h.get('windspeed_10m', [np.nan] * len(local))}, index=local)
        df = df[df.index.notna()]
        df.index = df.index.tz_convert('UTC').tz_localize(None)
        frames.append(df.astype(float))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated()], meta


def on_index(series_or_df, idx):
    """Values at the (naive UTC) hours `idx`, linearly interpolated over gaps, edges held."""
    full = series_or_df.reindex(series_or_df.index.union(idx)).interpolate(method='time').ffill().bfill()
    return full.reindex(idx)


def write_ci(out_dir, name, year, offset_h, values):
    tz = dt.timezone(dt.timedelta(hours=offset_h))
    stamps = pd.date_range(f'{year}-01-01', periods=8760, freq='h', tz=tz)
    pd.DataFrame({'timestamp': stamps.map(lambda t: t.isoformat(sep=' ')), 'avg_CI': np.round(values, 3)}).to_csv(
        os.path.join(out_dir, f'{name}_NG_&_avgCI.csv'), index=False)


def dew_point(t, rh):
    a, b = 17.62, 243.12
    g = np.log(np.clip(rh, 1, 100) / 100) + a * t / (b + t)
    return b * g / (a - g)


def write_epw(out_dir, name, region, year, offset_h, meta, w):
    elev = float(meta.get('elevation', 0.0))
    pressure = 101325.0 * (1 - 2.25577e-5 * elev) ** 5.25588
    header = [
        f"LOCATION,{region},-,-,DCcluster-Opt/Open-Meteo,000000,{meta['latitude']:.4f},{meta['longitude']:.4f},"
        f"{offset_h:.1f},{elev:.1f}",
        'DESIGN CONDITIONS,0', 'TYPICAL/EXTREME PERIODS,0', 'GROUND TEMPERATURES,0',
        'HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0',
        f'COMMENTS 1,Converted from DCcluster-Opt (HPE sustain-cluster) Open-Meteo data, year {year}, local standard time',
        'COMMENTS 2,Irradiance fields are 0 (not in the source data)',
        'DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31',
    ]
    local = pd.date_range(f'{year}-01-01', periods=8760, freq='h')
    temp, rh = w['temp'].values, np.clip(w['rh'].values, 0, 100)
    dew = dew_point(temp, rh)
    cloud = np.nan_to_num(w['cloud'].values, nan=50.0) / 10
    wind = np.nan_to_num(w['wind_kmh'].values, nan=0.0) / 3.6
    rows = []
    for k, t in enumerate(local):
        fields = [t.year, t.month, t.day, t.hour + 1, 60, '?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9',
                  f'{temp[k]:.1f}', f'{dew[k]:.1f}', f'{rh[k]:.0f}', f'{pressure:.0f}',
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     # radiation and illuminance (unknown)
                  0, f'{wind[k]:.1f}', f'{cloud[k]:.0f}', f'{cloud[k]:.0f}', 9999, 99999, 9, 999999999,
                  0, 0.0, 0, 88, 0.0, 0, 1.0]
        rows.append(','.join(str(x) for x in fields))
    with open(os.path.join(out_dir, f'{name}.epw'), 'w') as f:
        f.write('\n'.join(header + rows) + '\n')


def convert_jobs(src, out_path):
    from utils.safe_unpickle import safe_load
    pkl = os.path.join(src, 'data', 'workload', 'alibaba_2020_dataset', 'result_df_full_year_2020.pkl')
    if not os.path.exists(pkl):
        raise FileNotFoundError(f'{pkl} missing: unzip result_df_full_year_2020.zip in that directory first')
    df = safe_load(pkl)  # refuses any non numpy/pandas/datetime global
    t0 = pd.Timestamp('2020-01-01', tz='UTC')
    steps = ((df['interval_15m'] - t0) / pd.Timedelta(minutes=15)).astype(int).values
    arrival, cols = [], []
    for k, m in zip(steps, df['tasks_matrix']):
        if len(m):
            a = np.asarray(m)[:, 4:8].astype(float)  # duration_min, cpu_usage %, gpu_wrk_util %, avg_mem GB
            arrival.append(np.full(len(a), k, dtype=np.int32))
            cols.append(a)
    arrival = np.concatenate(arrival)
    a = np.concatenate(cols)
    task_scale = 5.0  # DCcluster-Opt's extract_tasks_from_row: requests = 5 x the trace's per-task usage
    np.savez_compressed(out_path, arrival_step=arrival, duration_min=a[:, 0].astype(np.float32),
                        cores=(task_scale * a[:, 1] / 100).astype(np.float32),
                        gpus=(task_scale * a[:, 2] / 100).astype(np.float32),
                        mem_gb=(task_scale * a[:, 3]).astype(np.float32),
                        step_minutes=np.int32(15), trace_start_utc=np.array('2020-01-01T00:00Z'))
    return len(arrival)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', default=os.path.join(os.path.dirname(REPO), 'sustain-cluster'))
    p.add_argument('--years', default='2023', help='comma-separated (2021-2024; non-leap years recommended)')
    p.add_argument('--regions', default='', help='comma-separated (default: every region with an integer offset)')
    p.add_argument('--skip_jobs', action='store_true')
    args = p.parse_args()

    regions = args.regions.split(',') if args.regions else sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(args.src, 'data', 'weather', '*')) if os.path.isdir(d))
    ci_dir, wea_dir = os.path.join(REPO, 'data', 'CarbonIntensity'), os.path.join(REPO, 'data', 'Weather')
    for year in (int(y) for y in args.years.split(',')):
        for region in regions:
            tz = region_tz(args.src, region, year)
            off = standard_offset_hours(tz, year)
            if off != int(off):
                print(f'skip {region}: non-integer UTC offset {off}')
                continue
            idx = local_standard_index(year, off)
            years = [year - 1, year, year + 1]
            ci = on_index(load_ci_utc(args.src, region, years), idx)
            weather, meta = load_weather_utc(args.src, region, years)
            weather = on_index(weather, idx)
            name = f'DCC_{region}_{year}'
            write_ci(ci_dir, name, year, off, ci.values)
            write_epw(wea_dir, name, region, year, off, meta, weather)
            print(f'{name}: UTC{off:+.0f}, CI mean {ci.mean():.0f} sd {ci.std():.0f} g/kWh, '
                  f'temp {weather["temp"].min():.1f}..{weather["temp"].max():.1f} C')
    if not args.skip_jobs:
        os.makedirs(os.path.join(REPO, 'data', 'Jobs'), exist_ok=True)
        out = os.path.join(REPO, 'data', 'Jobs', 'alibaba2020_jobs.npz')
        print(f'{convert_jobs(args.src, out)} jobs -> {out}')


if __name__ == '__main__':
    main()
