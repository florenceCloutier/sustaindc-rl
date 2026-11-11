"""Heuristic dispatchers for GeoSustainDC (geo_sustaindc_env.py) and the GeoPolicy that pairs one with per-site
controllers.

A dispatcher returns the weights that split this step's migratable pool over the sites. All of them use only the
current state (no forecast):
    local       no migration: every site keeps its own migratable work (the reference)
    capacity    split in proportion to site capacity
    random      Dirichlet(1) weights (seeded per episode), a sanity check
    greedy_ci   fill sites in order of lowest current carbon intensity, each up to the load-shifting cap (90%)
    merit       merit order on the marginal CO2 of one more unit of work at each site: carbon intensity x marginal
                IT + cooling energy at the site's current load, weather and setpoint (so a hot site with a high PUE
                ranks lower). Filled in chunks of 10% of a site's capacity, re-pricing each chunk (the marginal
                energy rises with load). This is the index policy of the restless-bandit view (arms = sites,
                index = marginal CO2 per unit of work), filled greedily under the "all work is served" budget.
When the pool exceeds the free capacity of all sites, the rest is spread in proportion to capacity (it then goes
to the sites' queues).

Per-site controllers are any eval_compare.py policy spec (noop, rule:ls,dc,bat, oracle:..., mpc:..., harl:...);
one independent instance per site. Note: controllers that look at the site's workload (LS rule/oracle) see the
site's own trace, not the work routed to it.
"""
import numpy as np

from baseline_policies import dc_power

DISPATCHERS = ('local', 'capacity', 'random', 'greedy_ci', 'merit')


class Dispatcher:
    def __init__(self, kind):
        if kind not in DISPATCHERS:
            raise ValueError(f"Unknown dispatcher {kind!r}; choose from {DISPATCHERS}")
        self.kind = kind

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def weights(self, geo):
        if self.kind == 'local' or geo.pool_units <= 0:
            return geo.local_weights()
        if self.kind == 'capacity':
            return geo.capacity_mw / geo.capacity_mw.sum()
        if self.kind == 'random':
            return self.rng.dirichlet(np.ones(geo.n_sites))
        if self.kind == 'greedy_ci':
            ci = [geo.site_state(i)['ci_now'] for i in range(geo.n_sites)]
            return merit_fill(geo, lambda i, units: ci[i])
        return merit_fill(geo, marginal_co2_index(geo))


def marginal_co2_index(geo):
    """index(i, units) -> gCO2 per pool unit of the next unit of work at site i, after `units` already routed there."""
    cache = {}

    def index(i, units):
        key = (i, round(units, 6))
        if key not in cache:
            s = geo.sites[i]
            u = (geo.non_shiftable[i] + geo.local_flexible[i]) / 100 + units / (100 * geo.units_per_task[i])
            lo, hi = max(min(u, 1.0) - 0.025, 0.0), min(max(u, 0.0) + 0.025, 1.0)
            dc = s.dc_env
            p_lo = sum(dc_power(s, lo, dc.raw_curr_stpt, dc.ambient_temp, dc.wet_bulb))
            p_hi = sum(dc_power(s, hi, dc.raw_curr_stpt, dc.ambient_temp, dc.wet_bulb))
            kwh_per_unit = (p_hi - p_lo) / (hi - lo) / 1e3 * 0.25 / (100 * geo.units_per_task[i])
            cache[key] = kwh_per_unit * float(s.ci_m.carbon_smooth[s.ci_m.time_step])
        return cache[key]
    return index


def merit_fill(geo, index, chunk_frac=0.1, cap=90):
    """Greedy fill: give the next chunk of the pool to the site with the lowest index at its current fill level."""
    pool = geo.pool_units
    room = geo.headroom_units(cap)
    chunk = chunk_frac * 100 * geo.units_per_task
    alloc = np.zeros(geo.n_sites)
    remaining = pool
    while remaining > 1e-9:
        open_sites = [i for i in range(geo.n_sites) if room[i] - alloc[i] > 1e-9]
        if not open_sites:
            break
        i = min(open_sites, key=lambda j: index(j, alloc[j]))
        amount = min(remaining, room[i] - alloc[i], chunk[i])
        alloc[i] += amount
        remaining -= amount
    if remaining > 1e-9:
        alloc += remaining * geo.capacity_mw / geo.capacity_mw.sum()
    return alloc / pool


class GeoPolicy:
    """A dispatcher + one controller per site. reset(seed, geo) at the start of each episode, act(geo) -> action.

    The dispatcher decides every `dispatch_every` steps (default 4 = hourly) and its weights are held in between
    (applied to each step's pool). `local` (no migration) is not a decision and is recomputed every step. The pool is
    routed before the site controllers act, so they see the work routed to their site."""

    def __init__(self, dispatcher='local', local_spec='noop', dispatch_every=4):
        self.dispatcher = dispatcher if hasattr(dispatcher, 'weights') else make_dispatcher(dispatcher)
        self.local_spec = local_spec
        self.dispatch_every = int(dispatch_every)
        self.local = None

    def reset(self, seed, geo):
        from eval_compare import make_policy
        if self.local is None or len(self.local) != geo.n_sites:
            self.local = [make_policy(self.local_spec)[1] for _ in range(geo.n_sites)]
        self.dispatcher.reset(seed)
        for k, p in enumerate(self.local):
            p.reset(seed + 1000 * k)
        self.n_steps = 0
        self.held = None

    def act(self, geo):
        if self.held is None or self.n_steps % self.dispatch_every == 0 or getattr(self.dispatcher, 'kind', '') == 'local':
            self.held = self.dispatcher.weights(geo)
        self.n_steps += 1
        geo.route(self.held)
        return {'dispatch': None, 'sites': [p.act(s) for p, s in zip(self.local, geo.sites)]}


class SB3Dispatcher:
    """A dispatcher trained with train_geo_dispatcher.py: <run_dir> holds model.zip, vecnormalize.pkl, config.json."""
    kind = 'sb3'

    def __init__(self, run_dir, model_name='model'):
        import json, os, pickle
        from stable_baselines3 import PPO
        with open(os.path.join(run_dir, 'config.json')) as f:
            self.config = json.load(f)
        self.model = PPO.load(os.path.join(run_dir, f'{model_name}.zip'), device='cpu')
        norm_path = os.path.join(run_dir, f'{model_name}_vecnormalize.pkl')
        self.obs_rms, self.clip_obs = None, 10.0
        if os.path.exists(norm_path):
            with open(norm_path, 'rb') as f:
                vn = pickle.load(f)
            if vn.norm_obs:
                self.obs_rms, self.clip_obs, self.eps = vn.obs_rms, vn.clip_obs, vn.epsilon

    def reset(self, seed):
        pass

    def weights(self, geo):
        from geo_dispatch_env import action_to_weights
        obs = geo.dispatcher_obs()
        if self.obs_rms is not None:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps), -self.clip_obs, self.clip_obs)
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        return action_to_weights(action, self.config['action_temperature'])


def make_dispatcher(spec):
    """'local', 'capacity', 'random', 'greedy_ci', 'merit' or 'sb3:<run_dir>[:<model>]'."""
    kind, _, rest = spec.partition(':')
    if kind == 'sb3':
        run_dir, _, model = rest.partition(':')
        return SB3Dispatcher(run_dir, model or 'model')
    return Dispatcher(spec)
