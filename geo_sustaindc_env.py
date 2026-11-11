"""Geo-distributed SustainDC: N data centers in different locations, plus a dispatcher that routes migratable work.

Each site is a full SustainDC (its own location -> weather and carbon intensity, capacity, cooling config, battery,
PV and its own load-shifting / cooling / battery agents). On top of them, a dispatcher decides every step where the
*migratable* work goes.

Workload model ("local + migratable pool"):
  * every site keeps its own workload trace (in its local time, so diurnal peaks are naturally offset by time zone);
  * the non-shiftable part of a site's work (latency-bound, interactive) always runs at that site;
  * of its flexible (deferrable) tasks, a fraction `migratable_fraction` is put in a shared pool each step (batch
    work that can run anywhere); the rest stays local;
  * the dispatcher splits the pool over the sites (weights on the simplex, so splitting is native); each site's
    load-shifting agent then decides, as before, whether to run its flexible tasks now or defer them in its queue.
  * migration is free (no transfer energy or latency cost).
`migratable_fraction = 0` gives N independent SustainDCs (the no-migration reference).

Units: a site task is 1% of that site's capacity for one 15-minute step. Sites can have different capacities, so the
pool is counted in *units* of 0.01 MW for 15 minutes (a site task = capacity_MW units). Routed units are converted
back to whole site tasks with a per-site carry of the fractional remainder (bounded by one task per site; it is
reported in the info as `carry_units`).

Clock: all sites step together. The episode start is drawn as in SustainDC, in the local standard time of the first
site (the reference); every other site starts at the same instant in its own local standard time (offsets from the
EPW weather files). With a single site, GeoSustainDC reproduces SustainDC exactly for the same seed.

Interface (geo_dispatch_env.py wraps it as a gymnasium env for a learned dispatcher):
    obs = geo.reset()
    geo.route(weights)        # optional: route first, so per-site controllers see the work routed to them
    obs, reward, terminated, truncated, info = geo.step({'dispatch': weights (N,) or None, 'sites': [site_actions] * N})
  * obs['dispatcher']: per-site features (see DISPATCH_FEATURES) flattened, + the pool size;
    obs['sites'][i]: the usual SustainDC observation dict of site i.
  * reward: sum over sites of the CO2 team reward (utils/reward_creator.co2_team_reward).
  * info['sites'][i]: site i's step info ('__common__'); info['geo']: routing details.
"""
import math

import numpy as np

from sustaindc_env import SustainDC
from utils import reward_creator
from utils.managers import epw_utc_offset
from utils.utils_cf import obtain_paths

NOOP = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
STEP_HOURS = 0.25
UNIT_MW = 0.01  # pool unit: 0.01 MW for one step (= one task of a 1 MW data center)

# Applied to every site unless overridden (top-level config keys, then the site's own keys, take precedence).
GEO_SITE_DEFAULTS = {
    'month': 6,
    'days_per_episode': 7,
    'flexible_load': 0.6,
    'agents': ['agent_ls', 'agent_dc', 'agent_bat'],
    # The default SustainDC rewards normalize energy with a module-level history shared by every env in the
    # process, which would mix the sites; the CO2 team reward has no shared state.
    'ls_reward': 'co2_team_reward', 'dc_reward': 'co2_team_reward', 'bat_reward': 'co2_team_reward',
}

GEO_DEFAULT_CONFIG = {
    # One dict per site with any SustainDC config keys; at least 'location'.
    'sites': [{'location': 'ca'}, {'location': 'ny'}, {'location': 'wa'}],
    # Fraction of each site's flexible tasks that join the shared pool every step (0 = no migration).
    'migratable_fraction': 1.0,
}

DISPATCH_FEATURES = ('ci_now', 'ci_next4h', 'ambient', 'workload', 'headroom', 'queue', 'battery_soc',
                     'hour_sin', 'hour_cos', 'capacity')


class GeoSustainDC:
    def __init__(self, config=None):
        config = dict(config or {})
        sites = config.pop('sites', GEO_DEFAULT_CONFIG['sites'])
        self.migratable_fraction = float(config.pop('migratable_fraction', GEO_DEFAULT_CONFIG['migratable_fraction']))
        if not 0.0 <= self.migratable_fraction <= 1.0:
            raise ValueError(f"migratable_fraction must be in [0, 1]; got {self.migratable_fraction!r}")
        if not sites:
            raise ValueError("GeoSustainDC needs at least one site")
        self.site_configs = [{**GEO_SITE_DEFAULTS, **config, **site} for site in sites]
        self.sites = [SustainDC(cfg) for cfg in self.site_configs]
        self.n_sites = len(self.sites)
        self.names = [cfg.get('name', cfg['location']) for cfg in self.site_configs]
        self.capacity_mw = np.array([float(s.datacenter_capacity_mw) for s in self.sites])
        self.units_per_task = self.capacity_mw / UNIT_MW / 100  # pool units per site task

        # Local standard time offsets (hours) relative to the first site.
        offsets = [epw_utc_offset(obtain_paths(s.location)[1]) for s in self.sites]
        self.utc_offsets = np.array(offsets)
        self.hour_shift = [o - offsets[0] for o in offsets]
        for shift in self.hour_shift:
            if shift != int(shift):
                raise ValueError(f"Non-integer time offset between sites ({self.hour_shift}) is not supported")

        self.n_features = len(DISPATCH_FEATURES)
        self.dispatcher_obs_dim = self.n_sites * self.n_features + 1

    # -----------------------------------------------------------------------------------------------------------------
    def reset(self, init_day=None, init_hour=None):
        """Start all sites at the same instant. init_day/init_hour: in the first site's local time (drawn at random,
        exactly as SustainDC.reset does, when not given)."""
        if init_day is None or init_hour is None:
            init_day, init_hour = self.sites[0].draw_start()
        starts = [self._local_start(init_day, init_hour, shift) for shift in self.hour_shift]
        if min(d for d, _ in starts) < 0:  # a site would start on the previous year's last day: start one day later
            init_day += 1
            starts = [self._local_start(init_day, init_hour, shift) for shift in self.hour_shift]
        self.start = (init_day, init_hour)
        self.carry = np.zeros(self.n_sites)
        self._carry_before = np.zeros(self.n_sites)
        self._routed = None
        site_obs = [s.reset(init_day=d, init_hour=h) for s, (d, h) in zip(self.sites, starts)]
        self._update_pool()
        return {'dispatcher': self.dispatcher_obs(), 'sites': site_obs}

    @staticmethod
    def _local_start(day, hour, shift):
        hour = hour + int(shift)
        return day + hour // 24, hour % 24

    # -----------------------------------------------------------------------------------------------------------------
    def _update_pool(self):
        """Split each site's current work into non-shiftable, local flexible and migratable tasks."""
        split = np.array([s.ls_env.task_split() for s in self.sites])  # (N, 2): non-shiftable, shiftable
        self.non_shiftable = split[:, 0]
        self.migratable = np.floor(split[:, 1] * self.migratable_fraction + 1e-9).astype(int)
        self.local_flexible = split[:, 1] - self.migratable
        self.pool_units = float(np.sum(self.migratable * self.units_per_task))

    def local_weights(self):
        """Dispatch weights that send every site's migratable work back to it (no migration)."""
        if self.pool_units <= 0:
            return self.capacity_mw / self.capacity_mw.sum()
        return self.migratable * self.units_per_task / self.pool_units

    def headroom_units(self, cap=90):
        """Pool units each site can take this step before reaching `cap` % utilization (the load-shifting cap)."""
        free = np.maximum(cap - self.non_shiftable - self.local_flexible, 0)
        return free * self.units_per_task

    def site_state(self, i):
        """Raw (unnormalized) state of site i at the step about to be simulated."""
        s = self.sites[i]
        t = s.ci_m.time_step
        ci = s.ci_m.carbon_smooth
        hour = s.t_m.hour
        return {
            'ci_now': float(ci[t]),                                    # gCO2/kWh
            'ci_next4h': float(np.mean(ci[t:t + 16])),                 # gCO2/kWh
            'ambient': float(s.dc_env.ambient_temp),                   # C
            'workload': float(s.ls_env.workload),                      # local utilization (0-1)
            'headroom': float(self.headroom_units()[i] / self.units_per_task[i] / 100),  # fraction of capacity
            'queue': len(s.ls_env.tasks_queue) / s.ls_env.queue_max_len,
            'battery_soc': float(s.bat_env.get_battery_soc()),
            'hour_sin': math.sin(2 * math.pi * hour / 24),
            'hour_cos': math.cos(2 * math.pi * hour / 24),
            'capacity': float(self.capacity_mw[i]),                    # MW
        }

    def dispatcher_obs(self):
        """Per-site features (DISPATCH_FEATURES, roughly scaled to [-1, 1]) + pool size (fraction of fleet capacity)."""
        scale = {'ci_now': 1000.0, 'ci_next4h': 1000.0, 'ambient': 40.0, 'capacity': float(self.capacity_mw.max())}
        feats = []
        for i in range(self.n_sites):
            st = self.site_state(i)
            feats += [st[k] / scale.get(k, 1.0) for k in DISPATCH_FEATURES]
        total_units = float(np.sum(100 * self.units_per_task))
        feats.append(self.pool_units / total_units)
        return np.asarray(feats, dtype=np.float32)

    # -----------------------------------------------------------------------------------------------------------------
    def _validate_weights(self, weights):
        if weights is None:
            return self.local_weights()
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (self.n_sites,) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(f"dispatch weights must be {self.n_sites} finite non-negative numbers, got {weights!r}")
        total = w.sum()
        return w / total if total > 0 else self.local_weights()

    def route(self, weights):
        """Split this step's pool over the sites (weights on the simplex; None = no migration) and hand each site its
        shiftable tasks. Sites update their observations, so per-site controllers called after route() see the work
        they will run. Can be called again before step() to re-route (it starts again from the same carry)."""
        weights = self._validate_weights(weights)
        alloc_units = self.pool_units * weights
        x = alloc_units / self.units_per_task + self._carry_before
        routed = np.floor(x + 1e-9).astype(int)
        self.carry = x - routed
        for i, s in enumerate(self.sites):
            s.set_routed_shiftable_tasks(self.local_flexible[i] + routed[i])
        self._routed = {
            'weights': weights, 'pool_units': self.pool_units, 'routed_tasks': routed,
            'migrated_out_units': self.migratable * self.units_per_task, 'routed_in_units': routed * self.units_per_task,
            'carry_units': self.carry * self.units_per_task,
        }
        return self._routed

    def step(self, action):
        """action: {'dispatch': weights or None, 'sites': [site action dict] * N (default: do nothing)}.
        A 'dispatch' entry (re-)routes the pool; without one, the routing of the last route() call this step is used,
        or no migration if there was none."""
        if action.get('dispatch') is not None or self._routed is None:
            self.route(action.get('dispatch'))
        geo_info, self._routed = self._routed, None
        site_actions = action.get('sites') or [dict(NOOP)] * self.n_sites
        if len(site_actions) != self.n_sites:
            raise ValueError(f"Expected {self.n_sites} site action dicts, got {len(site_actions)}")

        site_obs, site_info, reward, truncated = [], [], 0.0, False
        for s, a in zip(self.sites, site_actions):
            obs, _, _, trunc, info = s.step(a)
            common = info['__common__']
            site_obs.append(obs)
            site_info.append(common)
            reward += reward_creator.co2_team_reward(common)
            truncated |= bool(trunc['__all__'])

        self._carry_before = self.carry.copy()
        self._update_pool()
        obs = {'dispatcher': self.dispatcher_obs(), 'sites': site_obs}
        return obs, reward, False, truncated, {'sites': site_info, 'geo': geo_info}
