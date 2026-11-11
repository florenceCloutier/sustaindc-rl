"""Gymnasium env for learning the dispatcher of GeoSustainDC (geo_sustaindc_env.py) with the per-site controllers
frozen (hierarchical setup: hourly dispatcher over 15-minute site controllers).

* One env step = one dispatch decision, held for `dispatch_every` simulator steps (default 4 = one hour). Within
  the interval the same weights split each 15-minute step's pool; the site controllers act every 15 minutes and see
  the work routed to them.
* Observation: GeoSustainDC.dispatcher_obs() at the decision time (per-site CI now / next 4 h, ambient temperature,
  local workload, headroom, queue, battery SoC, local hour, capacity, + pool size).
* Action: Box(-1, 1, (N,)); weights = softmax(action_temperature * action), so a policy can split the pool or (with
  saturated actions) send it all to one site.
* Reward: sum over the interval and the sites of the CO2 team reward (utils/reward_creator.co2_team_reward: CO2 in
  units of 100 kg, dropped work charged at 2x). Episodes end by truncation (time limit).
* reset(seed=s) seeds `random` / `np.random` (where all of the simulator's randomness is drawn), so episodes are
  reproducible and comparable with eval_geo.py.

Config: GeoSustainDC keys (sites, migratable_fraction, month, days_per_episode, flexible_load, ...) plus
    local            per-site controller spec, eval_compare.py syntax (default 'rule:ls,dc,bat')
    dispatch_every   simulator steps per decision (default 4)
    action_temperature  softmax temperature of the action (default 5)
"""
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from geo_sustaindc_env import GeoSustainDC

DISPATCH_ENV_DEFAULTS = {'local': 'rule:ls,dc,bat', 'dispatch_every': 4, 'action_temperature': 5.0}


def action_to_weights(action, temperature):
    z = temperature * np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    e = np.exp(z - z.max())
    return e / e.sum()


class _HeldDispatcher:
    """Dispatcher used inside GeoPolicy: returns the weights the agent chose for the current interval."""
    kind = 'agent'

    def __init__(self):
        self.current = None

    def reset(self, seed):
        pass

    def weights(self, geo):
        return self.current


class GeoDispatchEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None):
        from geo_baselines import GeoPolicy
        config = dict(config or {})
        opts = {k: config.pop(k, v) for k, v in DISPATCH_ENV_DEFAULTS.items()}
        self.local_spec = opts['local']
        self.dispatch_every = int(opts['dispatch_every'])
        self.temperature = float(opts['action_temperature'])
        self.geo = GeoSustainDC(config)
        self.n_sites = self.geo.n_sites
        self.observation_space = spaces.Box(-10.0, 10.0, (self.geo.dispatcher_obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_sites,), np.float32)
        self._agent = _HeldDispatcher()
        # dispatch_every=1 inside the policy: the env itself holds the action for the whole interval
        self.policy = GeoPolicy(self._agent, self.local_spec, dispatch_every=1)
        self._seed_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
        episode_seed = int(seed) if seed is not None else int(self._seed_rng.integers(2**31 - 1))
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        self.geo.reset()
        self.policy.reset(episode_seed, self.geo)
        return self.geo.dispatcher_obs(), {'episode_seed': episode_seed}

    def step(self, action):
        weights = action_to_weights(action, self.temperature)
        self._agent.current = weights
        reward, truncated = 0.0, False
        co2_kg = dropped = migrated_units = 0.0
        for _ in range(self.dispatch_every):
            _, r, _, truncated, info = self.geo.step(self.policy.act(self.geo))
            reward += r
            co2_kg += sum(i['bat_CO2_footprint'] for i in info['sites']) / 1e3
            dropped += sum(i['ls_tasks_dropped'] for i in info['sites'])
            g = info['geo']
            migrated_units += float(np.sum(np.maximum(g['routed_in_units'] - g['migrated_out_units'], 0)))
            if truncated:
                break
        obs = self.geo.dispatcher_obs()
        return obs, float(reward), False, bool(truncated), {
            'weights': weights, 'co2_kg': co2_kg, 'tasks_dropped': dropped, 'migrated_units': migrated_units}
