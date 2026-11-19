"""Oracle ladder: where does price-coordinated routing (geo_dual.DualPolicy) lose to the perfect-foresight bound?

DualOraclePolicy adds, one at a time, information or effort a real dispatcher would not have:
    sigma24 = 0        perfect CI forecasts (instead of the noisy nominal forecast)
    oracle_h = H       the bundles are the TRUE arrivals of the next H hours (each job with its own release time,
                       SiteLP 'release'), instead of the past hour's arrivals standing in for the next hour's
    replan_every = k   coordinate every k steps (default 4 = hourly)
    window_h           the sites' planning window (default 24 h)

Run the ladder on the dccluster5 episodes of geo_scaling/cmp_n5_v1 (same seeds, so the same bounds):
    python geo_ladder.py [--seeds 1,2,3] [--out <json>]
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np

from geo_dual import DualPolicy

ROOT = '/network/scratch/f/florence.cloutier/sustaindc-rl_results/geo_scaling/cmp_n5_v1'


class DualOraclePolicy(DualPolicy):
    def __init__(self, oracle_h=0, replan_every=4, **kw):
        super().__init__(**kw)
        self.oracle_h, self.replan_every = float(oracle_h), int(replan_every)

    def _bundles(self, geo):
        if not self.oracle_h:
            return super()._bundles(geo)
        t = geo.t
        lo, hi = np.searchsorted(geo.arrival, [t, t + int(round(4 * self.oracle_h))])
        jobs = np.arange(lo, hi)
        jobs = jobs[~geo.rigid[jobs] & (geo.site[jobs] < 0)]          # not yet routed
        slack = geo.latest_start - geo.arrival
        nb = len(self.edges) + 1
        hour = (geo.arrival[jobs] - t) // 4                          # 0 = arrives within the coming hour
        groups = {}
        for h in np.unique(hour):
            for b in range(nb):
                j = jobs[(hour == h) & (self._bundle_of(slack[jobs]) == b)]
                if len(j):
                    # one bundle per (release hour, slack class): id b for the coming hour (the one routing uses),
                    # b + nb * h for later hours (planned for, so they claim capacity, but not routed yet)
                    groups[int(b + nb * h)] = {'jobs': j, 'slack': slack[j], 'release': geo.arrival[j] - t,
                                               'work': float(np.sum(geo.power_w[j] * geo.dur[j]))}
        return groups

    def _replan(self, geo):
        if not self.oracle_h:
            return super()._replan(geo)
        import time as _time
        t = geo.t
        self.stats['replans'] += 1
        self.acc[:] = 0.0
        self.curves = {}
        groups = self._bundles(geo)
        nb = len(self.edges) + 1
        if not groups:
            self.target = None
            return
        t0 = _time.time()
        bids, phi, lps, xs, info = self.coordinate(geo, groups)
        for k, v in (('dw_rounds', info['rounds']), ('dw_gap', info['gap']), ('dw_site_s', info['site_s']),
                     ('dw_master_s', info['master_s']), ('dw_total_s', _time.time() - t0), ('dw_unrouted', info['unrouted'])):
            self.stats[k].append(v)
        self.work = {b: groups[b]['work'] for b in bids if b < nb}
        self.target = np.zeros((geo.n_sites, nb))
        for k, b in enumerate(bids):
            if b < nb:                                                 # only the coming hour's shares route jobs
                self.target[:, b] = phi[:, k] * groups[b]['work']
                for i in range(geo.n_sites):
                    self.curves[(i, b)] = True
        for i, (lp, x) in enumerate(zip(lps, xs)):
            if x is not None:
                self.plans[i] = self._make_plan(lp, x[:lp.c_obj.size], t)

    def coordinate(self, geo, groups):
        """With many bundles (true arrivals over many hours) Dantzig-Wolfe needs far more rounds than its cap, so the
        oracle steps solve the joint LP directly: the same optimum when Dantzig-Wolfe converges, and the ladder
        measures the value of information, not of the decomposition."""
        from scipy.optimize import linprog
        if not self.oracle_h:
            return super().coordinate(geo, groups)
        lps, bids, c, A_ub, b_ub, A_eq, b_eq, bounds, iphi, off = self.joint_lp(geo, groups)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status != 0:
            raise RuntimeError(f'joint LP failed: {res.message}')
        xs = [np.r_[res.x[off[i]:off[i + 1]], res.x[iphi[i]]] for i in range(geo.n_sites)]
        return bids, res.x[iphi], lps, xs, {'rounds': 1, 'gap': 0.0, 'site_s': 0.0, 'master_s': 0.0, 'unrouted': 0.0}

    def act(self, geo):
        if geo.t % 4 != 0 and geo.t % self.replan_every == 0:
            self._t = geo.t
            self._replan(geo)
        return super().act(geo)                                       # replans itself at t % 4 == 0


STEPS = [
    ('0 dual as now', dict(sigma24=0.1)),
    ('1 + perfect CI forecast', dict(sigma24=0.0)),
    ('2 + true next-hour arrivals', dict(sigma24=0.0, oracle_h=1)),
    ('3 + true arrivals, next 24 h', dict(sigma24=0.0, oracle_h=24)),
    ('4 + re-plan every 15 min', dict(sigma24=0.0, oracle_h=24, replan_every=1)),
    ('5 + 48 h window', dict(sigma24=0.0, oracle_h=24, replan_every=1, window_h=48)),
]


def main():
    from geo_index import run_index_episode
    from geo_jobs import GeoJobsDC
    from geo_scenarios import site_configs
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1,2,3')
    p.add_argument('--steps', default='0,1,2,3,4,5')
    p.add_argument('--out', default='')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    geo = GeoJobsDC({'sites': site_configs('dccluster5'), 'month': 6, 'days_per_episode': 1, 'target_util': 0.5,
                     'routing': 'site_queue'})
    ref = {s: json.load(open(f'{ROOT}/case_dccluster5_m6_u0.5_n0.1_s{s}.json')) for s in seeds}
    rows = []
    for k in (int(x) for x in args.steps.split(',')):
        name, kw = STEPS[k]
        caps = []
        for s in seeds:
            base, bound = ref[s]['methods']['origin']['co2_kg'], ref[s]['bound']['co2_kg']
            t0 = time.time()
            m = run_index_episode(geo, DualOraclePolicy(fc_seed=s, admission='latest', **kw), s)
            cap = (base - m['co2_kg']) / (base - bound)
            caps.append(cap)
            rows.append({'step': name, 'seed': s, 'co2_kg': m['co2_kg'], 'captured': cap, 'late': m['sla_violations'],
                         'seconds': time.time() - t0})
            print(f"{name:<32} seed {s}: captured {cap:6.1%}, CO2 {m['co2_kg']:8.1f} kg, late {m['sla_violations']}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        print(f"{name:<32} MEAN captured {np.mean(caps):6.1%}", flush=True)
    if args.out:
        json.dump(rows, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
