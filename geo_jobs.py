"""Job-level geo-distributed SustainDC: real AI jobs (Alibaba 2020 GPU trace, via DCcluster-Opt) dispatched over N data
centers, each a full SustainDC (IT + cooling physics, cooling and battery agents).

Why jobs (and not the utilization pool of geo_sustaindc_env.py): a job keeps what makes dispatch hard.
  * commitment: once started, a job runs at its site until it ends (non-preemptive, never moved);
  * indivisibility: a job runs at one site;
  * per-job deadlines: deadline = arrival + sla_multiplier x duration (DCcluster-Opt's rule, default 1.5), so a job
    can wait at most (sla_multiplier - 1) x its duration before starting.

Workload (one episode):
  * jobs are the trace's jobs arriving in the episode's UTC window (trace day-of-year = data day-of-year), replicated
    (with +-20% size jitter on the copies, as DCcluster-Opt does) so that the mean fleet utilization is `target_util`;
  * each job's origin site is drawn with DCcluster-Opt's rule: population_weight x (1.0 during local 8:00-20:00,
    0.3 otherwise), so each site's demand follows its own business hours;
  * a `rigid_fraction` of jobs (latency-bound) must start at their origin on arrival;
  * size: IT power with DCcluster-Opt's sizing (20 W/core, 500 W/GPU, 2.5 W/GB), as a fraction of the site's IT
    capacity: a job of P watts occupies P / (capacity_MW x 1e6) of the site's CPU load (the DC model then computes
    the real IT + cooling power);
  * arrivals stop `drain_steps` before the end, so every job's deadline falls inside the episode and every policy has
    to run the same work (CO2 is directly comparable).

Each 15-minute step:
  1. new jobs join `pending`;
  2. the dispatcher gives, for each pending job (in earliest-latest-start order), a site to start at now, or -1 (wait);
  3. a start is accepted if the site's committed load + size <= cap; a rigid job may only start at its origin;
  4. pending jobs at their latest start are force-started (origin first, else the site with most room), if room;
  5. each site's utilization = sum over running jobs of size x (fraction of this step it runs): a 40-min job of 2%
     adds 2%, 2%, 1.33%; the site's DC model runs with it (SustainDC.set_external_utilization).

Usage:
    geo = GeoJobsDC({'sites': [{'location': 'dcc:US-CAL-CISO'}, {'location': 'dcc:DE-LU'}], 'target_util': 0.5})
    geo.reset()
    while True:
        view = geo.pending_view()
        _, reward, _, truncated, info = geo.step({'assign': sites_for(view), 'sites': None})
        if truncated: break
    geo.episode_metrics()
"""
import math
import os

import numpy as np

from geo_sustaindc_env import NOOP, STEP_HOURS, GeoSustainDC
from utils import reward_creator

JOBS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Jobs', 'alibaba2020_jobs.npz')
W_PER_CORE, W_PER_GPU, W_PER_GB = 20.0, 500.0, 2.5  # DCcluster-Opt's resource power sizing
STEPS_PER_DAY = 96

JOB_DEFAULTS = {
    'target_util': 0.5,        # mean fleet utilization the replicated trace is scaled to
    'sla_multiplier': 1.5,     # deadline = arrival + sla_multiplier x duration
    'rigid_fraction': 0.0,     # jobs that must start at their origin on arrival
    'cap': 1.0,                # max committed load per site (fraction of capacity)
    'jobs_file': JOBS_FILE,
}

_TRACE = {}


def load_trace(path=JOBS_FILE):
    if path not in _TRACE:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} missing: run scripts/import_dccluster_data.py first')
        z = np.load(path)
        power = W_PER_CORE * z['cores'] + W_PER_GPU * z['gpus'] + W_PER_GB * z['mem_gb']
        dur_steps = z['duration_min'].astype(np.float64) / 15.0
        n_steps = int(z['arrival_step'].max()) + 1
        _TRACE[path] = {
            'arrival': z['arrival_step'].astype(np.int64), 'dur_steps': dur_steps, 'power_w': power.astype(np.float64),
            'n_steps': n_steps,
            'mean_running_w': float(np.sum(power * dur_steps) / n_steps),  # mean power of the jobs running at a time
            'max_dur_steps': float(dur_steps.max()),
        }
    return _TRACE[path]


class GeoJobsDC(GeoSustainDC):
    """GeoSustainDC with job-level work. Site configs as in GeoSustainDC; extra site key 'population_weight'
    (default 1). Job options: see JOB_DEFAULTS (top-level config keys)."""

    def __init__(self, config=None):
        config = dict(config or {})
        self.job_cfg = {k: config.pop(k, v) for k, v in JOB_DEFAULTS.items()}
        sites = config.get('sites') or []
        self.population_weight = np.array([float(s.pop('population_weight', 1.0)) if isinstance(s, dict) else 1.0
                                           for s in sites] or [1.0])
        config['migratable_fraction'] = 0.0  # the pool model is unused in job mode
        super().__init__(config)
        if len(self.population_weight) != self.n_sites:
            self.population_weight = np.ones(self.n_sites)
        self.trace = load_trace(self.job_cfg['jobs_file'])
        self.cap_w = self.capacity_mw * 1e6
        self.cap = float(self.job_cfg['cap'])
        sla = float(self.job_cfg['sla_multiplier'])
        if sla < 1.0:
            raise ValueError(f'sla_multiplier must be >= 1, got {sla}')
        self.drain_steps = int(math.ceil(sla * self.trace['max_dur_steps'])) + 1
        self.horizon = int(self.site_configs[0]['days_per_episode']) * STEPS_PER_DAY
        if self.horizon <= self.drain_steps:
            raise ValueError(f'Episode ({self.horizon} steps) shorter than the drain period ({self.drain_steps})')

    # -----------------------------------------------------------------------------------------------------------------
    def reset(self, init_day=None, init_hour=None):
        obs = super().reset(init_day, init_hour)
        day0, hour0 = self.start
        # UTC step of the episode start (the first site's clock is its local standard time)
        self.utc0 = int(day0 * STEPS_PER_DAY + hour0 * 4 - self.utc_offsets[0] * 4)
        self._make_jobs()
        self.t = 0
        self.status = np.zeros(self.n_jobs, dtype=np.int8)   # 0 not arrived, 1 pending, 2 running, 3 done
        self.site = np.full(self.n_jobs, -1, dtype=np.int64)
        self.start_step = np.full(self.n_jobs, -1, dtype=np.int64)
        self.end_step = np.full(self.n_jobs, np.inf)
        self.forced = np.zeros(self.n_jobs, dtype=bool)
        self._arrive()
        self.totals = {'co2_kg': 0.0, 'energy_mwh': 0.0, 'it_mwh': 0.0, 'hvac_mwh': 0.0, 'water_l': 0.0}
        self.util_history = []
        return obs

    def _make_jobs(self):
        """Jobs arriving in this episode: trace window, replicated to target_util, origins drawn, deadlines set."""
        tr, cfg = self.trace, self.job_cfg
        rng = np.random.default_rng(np.random.randint(2**31 - 1))  # seeded through np.random (episode seed)
        n_arr = self.horizon - self.drain_steps
        rel = (tr['arrival'] - self.utc0) % tr['n_steps']
        sel = np.flatnonzero(rel < n_arr)
        factor = cfg['target_util'] * float(self.cap_w.sum()) / tr['mean_running_w']
        copies = np.floor(factor).astype(int) + (rng.random(len(sel)) < factor - np.floor(factor))
        idx = np.repeat(sel, copies)
        first = np.r_[True, idx[1:] != idx[:-1]]  # the first copy of each trace job keeps its size
        jitter = np.where(first, 1.0, rng.uniform(0.8, 1.2, len(idx)))
        order = np.argsort(rel[idx], kind='stable')
        idx, jitter = idx[order], jitter[order]
        self.n_jobs = len(idx)
        self.arrival = rel[idx].astype(np.int64)
        self.dur = tr['dur_steps'][idx]
        self.power_w = tr['power_w'][idx] * jitter
        self.deadline = self.arrival + cfg['sla_multiplier'] * self.dur
        self.latest_start = self.deadline - self.dur
        self.rigid = rng.random(self.n_jobs) < cfg['rigid_fraction']
        # origin: population weight x business-hours activity in each site's local standard time
        utc_hour = ((self.utc0 + self.arrival) // 4) % 24
        local_hour = (utc_hour[:, None] + self.utc_offsets[None, :].astype(int)) % 24
        w = self.population_weight[None, :] * np.where((local_hour >= 8) & (local_hour < 20), 1.0, 0.3)
        cdf = np.cumsum(w / w.sum(1, keepdims=True), axis=1)
        self.origin = np.minimum((rng.random(self.n_jobs)[:, None] > cdf).sum(1), self.n_sites - 1)
        self.size = self.power_w[:, None] / self.cap_w[None, :]  # (jobs, sites): fraction of each site's capacity
        self._arr_ptr = 0

    def _arrive(self):
        j0 = self._arr_ptr
        j1 = int(np.searchsorted(self.arrival, self.t, side='right'))
        self.status[j0:j1] = 1
        self._arr_ptr = j1

    # -----------------------------------------------------------------------------------------------------------------
    def committed_load(self):
        """Per-site load committed by running jobs during the current step (capacity check)."""
        run = np.flatnonzero(self.status == 2)
        load = np.zeros(self.n_sites)
        np.add.at(load, self.site[run], self.size[run, self.site[run]])
        return load

    def pending_jobs(self):
        """Indices of pending jobs, earliest latest-start first."""
        p = np.flatnonzero(self.status == 1)
        return p[np.argsort(self.latest_start[p], kind='stable')]

    def pending_view(self):
        """What a dispatcher sees at the current step (all per-job arrays are aligned with 'jobs')."""
        p = self.pending_jobs()
        return {'t': self.t, 'horizon': self.horizon, 'jobs': p, 'arrival': self.arrival[p], 'dur': self.dur[p],
                'latest_start': self.latest_start[p], 'deadline': self.deadline[p], 'power_w': self.power_w[p],
                'size': self.size[p], 'origin': self.origin[p], 'rigid': self.rigid[p],
                'load': self.committed_load(), 'cap': self.cap}

    def _try_start(self, j, i, load):
        if i < 0 or i >= self.n_sites or (self.rigid[j] and i != self.origin[j]):
            return False
        if load[i] + self.size[j, i] > self.cap + 1e-12:
            return False
        load[i] += self.size[j, i]
        self.status[j], self.site[j], self.start_step[j] = 2, i, self.t
        self.end_step[j] = self.t + self.dur[j]
        return True

    def schedule(self, assign):
        """Apply the dispatcher's decisions for the current step. `assign`: site per job of pending_view()['jobs']
        (-1 = wait), or a dict {job index: site}. Then force-start jobs at their latest start."""
        p = self.pending_jobs()
        load = self.committed_load()
        if assign is not None:
            if isinstance(assign, dict):
                items = [(j, s) for j, s in assign.items()]
            else:
                assign = np.asarray(assign, dtype=np.int64).reshape(-1)
                if len(assign) != len(p):
                    raise ValueError(f'assign has {len(assign)} entries for {len(p)} pending jobs')
                items = zip(p, assign)
            for j, i in items:
                if self.status[j] == 1:
                    self._try_start(int(j), int(i), load)
        # rigid jobs start at their origin on arrival if they can (no waiting by choice)
        for j in p[(self.status[p] == 1) & self.rigid[p]]:
            self._try_start(int(j), int(self.origin[j]), load)
        due = p[(self.status[p] == 1) & (self.latest_start[p] < self.t + 1)]
        for j in due:
            if self._try_start(int(j), int(self.origin[j]), load):
                self.forced[j] = True
                continue
            if not self.rigid[j]:
                for i in np.argsort(load - self.size[j]):
                    if self._try_start(int(j), int(i), load):
                        self.forced[j] = True
                        break

    def utilization(self):
        """Per-site utilization of the current step: running jobs x the fraction of the step they run."""
        run = np.flatnonzero(self.status == 2)
        frac = np.clip(self.end_step[run] - self.t, 0.0, 1.0)
        u = np.zeros(self.n_sites)
        np.add.at(u, self.site[run], self.size[run, self.site[run]] * frac)
        return u

    # -----------------------------------------------------------------------------------------------------------------
    def step(self, action):
        """action: {'assign': per pending job site or -1 (see schedule), 'sites': [site action dict] * N or None}."""
        self.schedule(action.get('assign'))
        u = self.utilization()
        self.util_history.append(u)
        site_actions = action.get('sites') or [dict(NOOP)] * self.n_sites
        site_obs, site_info, reward, truncated = [], [], 0.0, False
        for s, a, ui in zip(self.sites, site_actions, u):
            s.set_external_utilization(ui)
            obs, _, _, trunc, info = s.step(a)
            common = info['__common__']
            site_obs.append(obs)
            site_info.append(common)
            reward += reward_creator.co2_team_reward(common)
            truncated |= bool(trunc['__all__'])
            self.totals['co2_kg'] += common['bat_CO2_footprint'] / 1e3
            self.totals['energy_mwh'] += common['bat_total_energy_with_battery_KWh'] / 1e3
            self.totals['it_mwh'] += common['dc_ITE_total_power_kW'] * STEP_HOURS / 1e3
            self.totals['hvac_mwh'] += common['dc_HVAC_total_power_kW'] * STEP_HOURS / 1e3
            self.totals['water_l'] += common['dc_water_usage']
        self.t += 1
        self.status[(self.status == 2) & (self.end_step <= self.t)] = 3
        self._arrive()
        truncated |= self.t >= self.horizon
        info = {'sites': site_info, 'utilization': u}
        return {'sites': site_obs}, reward, False, truncated, info

    def episode_metrics(self):
        started = self.status >= 2
        late = started & (self.end_step > self.deadline + 1e-9)
        job_mwh = self.power_w * self.dur * STEP_HOURS / 1e6
        moved = started & (self.site != self.origin)
        m = dict(self.totals)
        m.update({
            'jobs': int(self.n_jobs), 'jobs_unstarted': int((~started).sum()),
            'job_mwh': float(job_mwh.sum()), 'job_mwh_migrated': float(job_mwh[moved].sum()),
            'sla_violations': int(late.sum()), 'sla_violation_rate': float(late.sum() / max(started.sum(), 1)),
            'late_job_hours': float(np.sum((self.end_step[late] - self.deadline[late]) * STEP_HOURS)),
            'forced_starts': int(self.forced.sum()),
            'mean_wait_h': float(np.mean((self.start_step[started] - self.arrival[started]) * STEP_HOURS)) if started.any() else 0.0,
            'mean_util': np.mean(self.util_history, axis=0).tolist() if self.util_history else [],
            'share': (np.bincount(self.site[started], weights=job_mwh[started], minlength=self.n_sites)
                      / max(job_mwh[started].sum(), 1e-12)).tolist(),
        })
        return m
