"""Checks for step 6 baselines B2 (geo_milp.py) and B3 (geo_rl_jobs.py). Run from the repo root:
    python tests/verify_baselines.py [--cap_mw 0.02] [--time_limit 300]
~10 minutes. B1 (geo_mpc.py) is checked in tests/verify_screening.py (Lagrangian / LP consistency).

B2 on a tiny instance (2 sites of 20 kW, 1 day):
  * the MILP solves, its LP relaxation is <= its value, and the fluid LP bound (geo_lp_oracle geo_time, battery +
    cooling envelope) is <= the MILP's CO2 under the true CI (both are relaxations of the job-level problem);
  * replaying the schedule in the simulator: every job starts, CO2 close to the MILP's model CO2 (the energy lines
    are a lower envelope of the simulator's model, and the replay rounds the battery to its 3 actions);
  * the two-stage scenario version solves.
B3: one PPO iteration on a small network runs, and the trained router plugs into run_queue_episode.
"""
import argparse
import os
import random
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import numpy as np

results = []


def check(name, cond, detail=''):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cap_mw', type=float, default=0.02)
    p.add_argument('--time_limit', type=float, default=300)
    args = p.parse_args()
    from geo_index import IndexPolicy, run_index_episode
    from geo_job_baselines import JobRouter, run_queue_episode
    from geo_lp_oracle import solve_bound
    from geo_milp import MILPReplay, solve_job_milp, tiny_geo

    print('B2: exact job-level MILP (tiny instance)')
    geo = tiny_geo(['US-CAL-CISO', 'DE-LU'], cap_mw=args.cap_mw, days=1)
    sol = solve_job_milp(geo, seed=1, cooling='exact', time_limit=args.time_limit)
    check('MILP solves', sol['status'] in (0, 1) and 'co2_model_kg' in sol,
          f"status {sol['status']}, {sol['n_jobs']} jobs, {sol['n_binaries']} binaries, gap {sol['mip_gap']:.2e}, "
          f"{sol['solve_s']:.0f}s")
    rel = solve_job_milp(geo, seed=1, cooling='exact', relax=True, time_limit=args.time_limit)
    check('LP relaxation <= MILP', rel['objective_g'] <= sol['objective_g'] * (1 + 1e-6),
          f"{rel['objective_g'] / 1e3:.2f} vs {sol['objective_g'] / 1e3:.2f} kg-eq")
    random.seed(1)
    np.random.seed(1)
    geo.reset()
    bound = solve_bound(geo, 'geo_time', verbose=False, site_flex='battery+cooling')
    check('fluid bound <= MILP CO2', bound['co2_kg'] <= sol['co2_model_kg'] * 1.01,
          f"bound {bound['co2_kg']:.2f} kg, MILP {sol['co2_model_kg']:.2f} kg, overflow {sol['overflow']:.3f} jobs")
    rep = run_index_episode(geo, MILPReplay(sol), seed=1)
    check('replay: every job starts, CO2 near the model', rep['jobs_unstarted'] == 0
          and abs(rep['co2_kg'] / sol['co2_model_kg'] - 1) < 0.1,
          f"sim {rep['co2_kg']:.2f} kg vs model {sol['co2_model_kg']:.2f}, sla {rep['sla_violations']}")
    others = {k: run_queue_episode(geo, JobRouter(k), 1)['co2_kg'] for k in ('origin', 'merit')}
    others['index'] = run_index_episode(geo, IndexPolicy(), 1)['co2_kg']
    print('     simulated CO2 (kg): ' + ', '.join(f'{k} {v:.2f}' for k, v in {**others, 'milp_replay': rep['co2_kg']}.items()))
    saa = solve_job_milp(geo, seed=1, cooling='exact', scenarios=3, sigma24=0.1, time_limit=args.time_limit)
    check('two-stage scenario MILP solves', saa['status'] in (0, 1) and 'co2_model_kg' in saa,
          f"{saa['n_binaries']} binaries, {saa['solve_s']:.0f}s, CO2 at true CI {saa.get('co2_model_kg', np.nan):.2f} kg")

    print('B3: PPO router smoke')
    import torch
    from geo_jobs import GeoJobsDC
    from geo_rl_jobs import JobRoutingEnv, PPORouter, RouterNet, collect, ppo_update
    from geo_scenarios import site_configs
    g5 = GeoJobsDC({'sites': site_configs('dccluster5'), 'month': 6, 'days_per_episode': 1, 'target_util': 0.5,
                    'routing': 'site_queue'})
    env = JobRoutingEnv(g5, obs_hours=6)
    net = RouterNet(env.obs_dim, env.job_dim, g5.n_sites, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    ep = collect(env, net, seed=3)
    info = ppo_update(net, opt, [ep], epochs=1)
    check('collect + update', len(ep['act']) > 0 and np.isfinite(info['pg']),
          f"{len(ep['rew'])} steps, {len(ep['act'])} routed jobs, co2 {ep['co2_kg']:.0f} kg")
    cfg = {'obs_dim': env.obs_dim, 'job_dim': env.job_dim, 'n_sites': g5.n_sites, 'hidden': 64, 'obs_hours': 6}
    m = run_queue_episode(g5, PPORouter(net, cfg), 3)
    routed = int((g5.site[:g5._arr_ptr] >= 0).sum())
    check('PPORouter runs in run_queue_episode and routes every arrival', routed == g5._arr_ptr,
          f"co2 {m['co2_kg']:.0f} kg, {m['jobs_unstarted']} jobs unstarted (untrained router)")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
