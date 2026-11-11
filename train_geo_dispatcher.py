"""Train the GeoSustainDC dispatcher (geo_dispatch_env.py) with PPO, per-site controllers frozen.

Run from the repo root, e.g.:
    python train_geo_dispatcher.py --sites ca,ny,wa --local rule:ls,dc,bat --n_envs 8 --total_timesteps 200000

Every --eval_freq decisions the deterministic policy is scored on the same --n_eval episodes (seeds eval_seed + k),
on CO2 (kg) summed over the episode; the lowest-CO2 model is saved as best_model.zip (+ its VecNormalize stats).
The run directory then works as a dispatcher in eval_geo.py: --dispatch sb3:<run_dir>:best_model
"""
import argparse
import datetime
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from geo_dispatch_env import GeoDispatchEnv


def env_config_from_args(args):
    caps = [float(c) for c in args.capacities.split(',')] if args.capacities else None
    sites = []
    for k, loc in enumerate(args.sites.split(',')):
        site = {'location': loc}
        if caps:
            site['datacenter_capacity_mw'] = caps[k]
        sites.append(site)
    return {'sites': sites, 'month': args.month, 'days_per_episode': args.days, 'flexible_load': args.flexible_load,
            'migratable_fraction': args.migratable_fraction, 'local': args.local,
            'dispatch_every': args.dispatch_every, 'action_temperature': args.action_temperature}


def save(model, vec_norm, run_dir, name):
    model.save(os.path.join(run_dir, f'{name}.zip'))
    vec_norm.save(os.path.join(run_dir, f'{name}_vecnormalize.pkl'))


class FixedSeedCO2Eval(BaseCallback):
    def __init__(self, env_config, run_dir, eval_freq, n_eval, eval_seed):
        super().__init__()
        self.env = GeoDispatchEnv(dict(env_config))
        self.run_dir, self.eval_freq, self.n_eval, self.eval_seed = run_dir, eval_freq, n_eval, eval_seed
        self.best = np.inf
        self.history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq:
            return True
        vec_norm = self.model.get_vec_normalize_env()
        co2s = []
        for k in range(self.n_eval):
            obs, _ = self.env.reset(seed=self.eval_seed + k)
            co2, done = 0.0, False
            while not done:
                action, _ = self.model.predict(vec_norm.normalize_obs(obs), deterministic=True)
                obs, _, term, trunc, info = self.env.step(action)
                co2 += info['co2_kg']
                done = term or trunc
            co2s.append(co2)
        mean = float(np.mean(co2s))
        self.logger.record('eval/co2_kg', mean)
        self.history.append({'timesteps': int(self.num_timesteps), 'co2_kg': mean, 'per_episode': co2s})
        with open(os.path.join(self.run_dir, 'eval_history.json'), 'w') as f:
            json.dump(self.history, f, indent=1)
        print(f"[eval] {self.num_timesteps} decisions: CO2 {mean:.1f} kg {'(best)' if mean < self.best else ''}",
              flush=True)
        if mean < self.best:
            self.best = mean
            save(self.model, vec_norm, self.run_dir, 'best_model')
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sites', default='ca,ny,wa')
    p.add_argument('--capacities', default='')
    p.add_argument('--month', type=int, default=6)
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--flexible_load', type=float, default=0.6)
    p.add_argument('--migratable_fraction', type=float, default=1.0)
    p.add_argument('--local', default='rule:ls,dc,bat', help='per-site controller spec (eval_compare.py syntax)')
    p.add_argument('--dispatch_every', type=int, default=4)
    p.add_argument('--action_temperature', type=float, default=5.0)
    p.add_argument('--n_envs', type=int, default=8)
    p.add_argument('--total_timesteps', type=int, default=200_000, help='dispatch decisions')
    p.add_argument('--n_steps', type=int, default=168, help='PPO rollout length per env (168 = one 7-day episode)')
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--ent_coef', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--eval_freq', type=int, default=5000, help='in calls per env (decisions / n_envs)')
    p.add_argument('--n_eval', type=int, default=3)
    p.add_argument('--eval_seed', type=int, default=1000)
    p.add_argument('--run_dir', default='')
    args = p.parse_args()

    env_config = env_config_from_args(args)
    run_dir = args.run_dir or os.path.join('results', 'geo_dispatcher',
                                           datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + f'-s{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'args': vars(args), 'env_config': env_config,
                   'action_temperature': args.action_temperature}, f, indent=1)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = vec_cls([lambda: GeoDispatchEnv(dict(env_config)) for _ in range(args.n_envs)])
    venv.seed(args.seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)
    model = PPO('MlpPolicy', venv, n_steps=args.n_steps, batch_size=min(256, args.n_steps * args.n_envs),
                learning_rate=args.learning_rate, ent_coef=args.ent_coef, gamma=0.99, seed=args.seed, verbose=1,
                tensorboard_log=os.path.join(run_dir, 'tb'))
    callback = FixedSeedCO2Eval(env_config, run_dir, args.eval_freq, args.n_eval, args.eval_seed)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    save(model, venv, run_dir, 'model')
    print(f"Saved {run_dir} (best eval CO2 {callback.best:.1f} kg)")


if __name__ == '__main__':
    main()
