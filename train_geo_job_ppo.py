"""Train the B3 job router (geo_rl_jobs.py) with PPO on a GeoJobsDC scenario; log the learning curve and wall clock.

Run from the repo root, e.g.
    python train_geo_job_ppo.py --scenario dccluster5 --days 1 --iters 3 --episodes 1       # smoke
    python train_geo_job_ppo.py --scenario n20 --days 2 --iters 300 --episodes 2 --sigma24 0.1
Every --eval_every iterations the greedy policy runs on --n_eval fixed seeds (eval_seed + k, distinct from the
training seeds); the lowest eval CO2 + lateness is saved as best.pt. The run directory then works as a method in
geo_scaling.py: --methods ppo:<run_dir>
"""
import argparse
import datetime
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import torch


def main():
    from geo_jobs import GeoJobsDC
    from geo_rl_jobs import JobRoutingEnv, PPORouter, RouterNet, collect, ppo_update
    from geo_scenarios import site_configs
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', default='dccluster5')
    p.add_argument('--days', type=int, default=1)
    p.add_argument('--month', type=int, default=6)
    p.add_argument('--util', type=float, default=0.5)
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--sigma24', type=float, default=0.0)
    p.add_argument('--rho', type=float, default=0.3)
    p.add_argument('--obs_hours', type=int, default=12)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--episodes', type=int, default=2, help='episodes per PPO update')
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--n_eval', type=int, default=2)
    p.add_argument('--eval_seed', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--out_dir', default='')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    run_dir = args.out_dir or os.path.join('results', 'geo_job_ppo', f"{args.scenario}_{datetime.datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    geo = GeoJobsDC({'sites': site_configs(args.scenario, year=args.year), 'month': args.month,
                     'days_per_episode': args.days, 'target_util': args.util, 'routing': 'site_queue'})
    env = JobRoutingEnv(geo, args.obs_hours, args.sigma24, args.rho, fc_seed=0)
    net = RouterNet(env.obs_dim, env.job_dim, geo.n_sites, args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    cfg = {**vars(args), 'obs_dim': env.obs_dim, 'job_dim': env.job_dim, 'n_sites': geo.n_sites,
           'n_params': sum(q.numel() for q in net.parameters())}
    json.dump(cfg, open(os.path.join(run_dir, 'config.json'), 'w'), indent=1)
    print(f"{geo.n_sites} sites, obs {env.obs_dim}, job feats {env.job_dim}, {cfg['n_params']} parameters -> {run_dir}",
          flush=True)

    def evaluate():
        router = PPORouter(net, cfg)
        net.eval()
        scores = []
        for k in range(args.n_eval):
            router_env = JobRoutingEnv(geo, args.obs_hours, args.sigma24, args.rho, fc_seed=0)
            o = router_env.reset(args.eval_seed + k)
            done, co2, late = False, 0.0, 0.0
            while not done:
                if len(o['arr']):
                    with torch.no_grad():
                        a = net.logits(torch.from_numpy(o['glob'])[None], torch.from_numpy(o['jobs']),
                                       torch.from_numpy(o['mask'])).argmax(1).numpy()
                else:
                    a = []
                o, _, done, info = router_env.step(a)
                co2 += info['co2_kg']
                late += info['late']
            scores.append((co2, late, geo.episode_metrics()['sla_violations']))
        net.train()
        return np.mean(scores, axis=0)

    curve, best, t_start, seed = [], np.inf, time.time(), args.seed * 100000
    for it in range(1, args.iters + 1):
        t0 = time.time()
        batch = []
        for _ in range(args.episodes):
            seed += 1
            batch.append(collect(env, net, seed))
        t_collect = time.time() - t0
        info = ppo_update(net, opt, batch)
        row = {'iter': it, 'wall_s': time.time() - t_start, 'collect_s': t_collect, 'update_s': time.time() - t0 - t_collect,
               'train_co2_kg': float(np.mean([b['co2_kg'] for b in batch])),
               'train_late': float(np.mean([b['late'] for b in batch])),
               'jobs_per_episode': float(np.mean([len(b['act']) for b in batch])), **info}
        if it % args.eval_every == 0 or it == args.iters:
            co2, late, sla = evaluate()
            row.update({'eval_co2_kg': float(co2), 'eval_late': float(late), 'eval_sla': float(sla)})
            score = co2 + env.late_kg * late
            if score < best:
                best = score
                torch.save(net.state_dict(), os.path.join(run_dir, 'best.pt'))
        curve.append(row)
        torch.save(net.state_dict(), os.path.join(run_dir, 'last.pt'))
        json.dump(curve, open(os.path.join(run_dir, 'curve.json'), 'w'), indent=1)
        print(f"it {it:4d} {row['wall_s']:7.0f}s train co2 {row['train_co2_kg']:9.1f} late {row['train_late']:.3f}"
              + (f" | eval co2 {row['eval_co2_kg']:9.1f} late {row['eval_late']:.3f} sla {row['eval_sla']:.0f}"
                 if 'eval_co2_kg' in row else ''), flush=True)


if __name__ == '__main__':
    main()
