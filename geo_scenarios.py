"""Large-N site lists for GeoJobsDC: the DCcluster-Opt regions x site variants.

A network of N sites is built by going round the regions (site k is in region k mod R, so every region gets N / R
sites, +-1) and drawing each site's variant with a seeded generator:
    capacity_mw         0.5, 1 or 2 MW (IT)
    dc_config_file      three cooling plants (utils/dc_config.json; dc_config_efficient.json: chiller COP 6, cooling
                        tower approach 3 C; dc_config_legacy.json: COP 4, approach 5 C), so sites in the same region
                        differ in cooling efficiency
    battery_hours       battery energy = hours x capacity x 1.2 (peak IT + cooling), 0.25, 1 or 2 h
    population_weight   job demand at the site (origins), capacity x lognormal(0, 0.3)
Sites in one region share CI and weather; the variants decide how close their indices are. With `homogeneous=True`
every site gets the default variant (1 MW, dc_config.json, 1 h battery, weight 1): exact ties within a region, the
hardest case for screening.

Names:
    dccluster5, us4, eu_asia   the fixed sets of geo_hardness.py (one default site per region)
    dcc15                      every imported region once (default variant)
    n<N>[h][s<seed>]           N sites as above, e.g. n50, n200s3, n20h (homogeneous)

    from geo_scenarios import site_configs
    sites = site_configs('n50', year=2023)          # list of GeoJobsDC site dicts
"""
import glob
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
FIXED_SETS = {
    'dccluster5': ['US-CAL-CISO', 'DE-LU', 'CL-SIC', 'SG', 'AU-NSW'],      # DCcluster-Opt's evaluation cluster
    'us4': ['US-CAL-CISO', 'US-NY-NYIS', 'US-TEX-ERCO', 'US-MIDA-PJM'],
    'eu_asia': ['DE-LU', 'GB', 'JP-TK', 'KR', 'SG'],
}
CAPACITIES_MW = (0.5, 1.0, 2.0)
DC_CONFIGS = ('dc_config.json', 'dc_config_efficient.json', 'dc_config_legacy.json')
BATTERY_HOURS = (0.25, 1.0, 2.0)
PEAK_FACTOR = 1.2  # peak total power / IT capacity, for battery sizing


def dcc_regions(year=2023):
    """Regions with both CI and weather imported for `year` (scripts/import_dccluster_data.py), sorted."""
    ci = {re.match(r'DCC_(.+)_(\d{4})_NG', os.path.basename(f)).group(1)
          for f in glob.glob(os.path.join(REPO, 'data', 'CarbonIntensity', f'DCC_*_{year}_NG_&_avgCI.csv'))}
    wea = {os.path.basename(f)[4:-len(f'_{year}.epw')]
           for f in glob.glob(os.path.join(REPO, 'data', 'Weather', f'DCC_*_{year}.epw'))}
    regions = sorted(ci & wea)
    if not regions:
        raise FileNotFoundError(f'No DCcluster-Opt data for {year}: run scripts/import_dccluster_data.py --years {year}')
    return regions


def default_site(region, year):
    return {'location': f'dcc:{region}:{year}', 'name': region}


def make_sites(n, seed=0, year=2023, regions=None, homogeneous=False):
    """N site dicts, round-robin over `regions` (default: all imported for `year`), variants drawn with `seed`."""
    regions = list(regions or dcc_regions(year))
    rng = np.random.default_rng(seed)
    sites = []
    for k in range(n):
        reg = regions[k % len(regions)]
        site = default_site(reg, year)
        site['name'] = f'{reg}#{k // len(regions)}'
        if homogeneous:
            cap, cfg, hours, w = 1.0, DC_CONFIGS[0], 1.0, 1.0
        else:
            cap = float(rng.choice(CAPACITIES_MW))
            cfg = str(rng.choice(DC_CONFIGS))
            hours = float(rng.choice(BATTERY_HOURS))
            w = cap * float(rng.lognormal(0.0, 0.3))
        site.update({'datacenter_capacity_mw': cap, 'dc_config_file': cfg,
                     'max_bat_cap_Mw': hours * cap * PEAK_FACTOR, 'population_weight': w})
        sites.append(site)
    return sites


def site_configs(name, year=2023):
    """Site dicts for a scenario name (module docstring)."""
    if name in FIXED_SETS:
        return [default_site(r, year) for r in FIXED_SETS[name]]
    if name == 'dcc15':
        return [default_site(r, year) for r in dcc_regions(year)]
    m = re.fullmatch(r'n(\d+)(h?)(?:s(\d+))?', name)
    if not m:
        raise ValueError(f'Unknown scenario {name!r}: {list(FIXED_SETS)}, dcc15 or n<N>[h][s<seed>]')
    return make_sites(int(m.group(1)), seed=int(m.group(3) or 0), year=year, homogeneous=bool(m.group(2)))


def describe(sites):
    """One line per site: name, capacity, cooling config, battery MWh, population weight."""
    return [f"{s.get('name', s['location']):>16} {s.get('datacenter_capacity_mw', 1.0):4.1f} MW "
            f"{s.get('dc_config_file', 'dc_config.json'):>20} bat {s.get('max_bat_cap_Mw') or 0:4.2f} MWh "
            f"w {s.get('population_weight', 1.0):.2f}" for s in sites]
