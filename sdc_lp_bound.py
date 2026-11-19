"""Perfect-foresight lower bound on an episode's co2_work_adjusted for the single data center, and a policy that
replays the bound's plan (`lp_replay`), both on the evaluation protocol of eval_compare.py / scripts/eval_protocol.sh.

The bound is a linear program over the whole episode with the true exogenous inputs (workload, weather, carbon
intensity). Every trajectory the simulator can produce, under any policy, is feasible in it, and its objective
is the metric itself, so its optimum is a lower bound on the co2_work_adjusted of every policy in the paired
table (learned, MPC or rule-based) on the same episode.

Per step t, with ns_t / sh_t the non-shiftable / shiftable tasks the env derives from the submitted workload w_t
(ceil / floor, flexible_workload_ratio and queue_max_len read from the env), one task = 1 % utilization:

    queue        Q_{t+1} = Q_t + d_t - p_t,  0 <= Q <= queue_max_len,  0 <= d_t <= sh_t,  0 <= p_t <= cap_t
                 d = tasks deferred, p = tasks run from the queue (regular and overdue), cap_t = max(0, 90 - ns_t - sh_t)
    utilization  u_t = (ns_t + sh_t - d_t + p_t) / 100
    energy       z_t >= a_tj + b_tj u_t          lower convex hull of  min_{s in S} (IT + HVAC energy)(u, s, weather_t)
    battery      s_{t+1} = s_t + eff_c ch_t - dis_t / eff_d,  soc_min C <= s <= soc_max C,  s_0 = soc_min C,
                 0 <= ch_t, dis_t <= P dt,  dis_t <= z_t            (grid energy g_t = z_t + ch_t - dis_t >= 0)
    work charge  y >= sum_t w_t - sum_t u_t,  y >= 0
    minimize     sum_t CI_t g_t / 1e3  +  kappa y      (kg; kappa as in sdc_eval.metrics: the marginal energy of one
                                                        util-step at the episode's mean load and ambient x mean CI)

Why every simulator trajectory is feasible (the relaxations):
* queue: task ages, the forced processing of overdue tasks, the all-or-nothing actions and integrality are dropped;
  deferring and processing may happen in the same step. "Dropped" tasks (deferred into a full queue) run in the step
  itself in envs/carbon_ls.py, so they need no variable: d counts only the tasks actually added to the queue.
* energy: the simulator's utilization is always a whole number of tasks, and the hull is built over every whole
  number of tasks the step allows, so it lies below the true energy at every reachable point. S is every CRAC setpoint
  the cooling action can reach (reachable_setpoints); the +-1 C per step rate limit is ignored.
* battery: charging and discharging are continuous and the charging taper above taper_start is ignored.
* the metric's max(unexecuted work, 0) is exact through y.
The CO2 and the work charge use the same quantities as sdc_eval.metrics (bat_avg_CI, ls_original_workload,
outside_temp, the final wet bulb), read from a do-nothing run of the same episode (the inputs do not depend on the
actions; eval_compare checks that with its fingerprints).

Solver: scipy's HiGHS (dual simplex; about 30k rows for a 30-day episode, a few seconds).

Usage (from the repo root), on the protocol of scripts/eval_protocol.sh:
    python sdc_lp_bound.py --location tx --month 6 --days_per_episode 30 --pv_capacity_kWp 0 \\
        --n_episodes 10 --seed 1000 --out_dir results/lp_bound/tx
writes bound.json and bound.csv (per episode: noop, bound, % vs noop) and prints the paired % change vs noop with its
95% CI (t distribution, as eval_compare.paired_report).

Replay policy, for eval_compare.py:  lp_replay[:<params>]   e.g. lp_replay=lp_s1
solves the episode's LP at reset (perfect foresight, like mpc_joint's oracle) and rounds its plan to actions every
step: defer if d_t >= defer_frac x sh_t, else process if p_t >= process_tasks, else nothing; charge / discharge when
that flow exceeds bat_frac x the battery's per-step power limit; cooling moves toward the hull's best setpoint at the
planned utilization. An achievable number in the same table (the bound is below it). Parameters: defer_frac (0.5),
process_tasks (1), bat_frac (0.1).

Only PV off (pv_config.capacity_kWp = 0) is supported, as in the protocol.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import envs.datacenter as DataCenter
from baseline_policies import NOOP, setpoint_action
from sdc_eval.constants import STEP_HOURS

MAX_UTIL_TASKS = 90  # envs/carbon_ls.py never schedules queued work above 90 % utilization


# ---------------------------------------------------------------------------------------------------------------------
# Inputs of one episode
# ---------------------------------------------------------------------------------------------------------------------
def reachable_setpoints(initial, lo, hi, max_step=10):
    """Every CRAC setpoint the cooling action can reach: whole-degree moves (the action's +-1 C, accelerated to +-2,
    +-3, ... by repeats) from the initial setpoint, clipped to [lo, hi] (envs/dc_gym.py). Values rounded to 1e-6."""
    seen, frontier = set(), {round(float(initial), 6)}
    while frontier:
        seen |= frontier
        nxt = set()
        for x in frontier:
            for k in range(-max_step, max_step + 1):
                y = round(min(max(x + k, lo), hi), 6)
                if y not in seen:
                    nxt.add(y)
        frontier = nxt
    return sorted(seen)


def exo_traces(sdc):
    """Exogenous inputs of the episode the env was just reset to, step by step, from a do-nothing run on a copy.

    Returns a dict of arrays (one entry per step): w (submitted workload, what the ls env schedules), amb / wb (the
    weather the DC model uses this step), ci (g/kWh, the battery env's CI = bat_avg_CI), and the scalars the metric's
    work charge uses (sdc_eval.metrics.EpisodeAccumulator.finalize): kappa_kg (kg per util-step of unexecuted work)
    and noop (the do-nothing episode's metrics). The RNG states and the reward history are restored afterwards, so
    calling this does not change the live episode.
    """
    from sdc_eval.metrics import EpisodeAccumulator, marginal_kwh_per_util_step
    import utils.reward_creator as rc
    states = random.getstate(), np.random.get_state()
    history = list(rc.energy_history)
    sim = copy.deepcopy(sdc)
    sim.calculate_reward = lambda params: (0.0, 0.0, 0.0)
    acc = EpisodeAccumulator()
    w, amb, wb, ci = [], [], [], []
    try:
        while True:
            w.append(float(sim.ls_env.workload))
            amb.append(float(sim.dc_env.ambient_temp))
            wb.append(float(sim.dc_env.wet_bulb))
            _, _, _, truncs, info = sim.step(dict(NOOP))
            i = info['agent_ls']
            ci.append(float(i['bat_avg_CI']))
            acc.step(i)
            if truncs['__all__']:
                break
        noop = acc.finalize(sim)
        # The metric's charge: marginal energy at the mean submitted load and mean (post-step) ambient, final wet bulb.
        kappa = marginal_kwh_per_util_step(sim, acc.m['work_submitted'] / acc.steps, acc.amb_sum / acc.steps) \
            * (acc.ci_sum / acc.steps) / 1e3
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        rc.energy_history.clear()
        rc.energy_history.extend(history)
    return {'w': np.array(w), 'amb': np.array(amb), 'wb': np.array(wb), 'ci': np.array(ci),
            'kappa_kg': float(kappa), 'noop': noop}


class EnergyModel:
    """IT + HVAC energy (kWh per step) of the DC at whole-task utilizations and every reachable setpoint, with the
    same functions as envs/dc_gym.py. The IT part does not depend on the weather and is tabulated once."""

    def __init__(self, sdc):
        dc = sdc.dc_env
        self.cfg = dc.DC_Config
        self.setpoints = reachable_setpoints(dc.initial_stpt, dc.min_temp, dc.max_temp)
        self.u = np.arange(101) / 100
        self.it, self.ret = {}, {}
        for s in self.setpoints:
            for k in range(101):
                cpu, fan, outlet = dc.dc.compute_datacenter_IT_load_outlet_temp([k] * self.cfg.NUM_RACKS, s)
                self.it[k, s] = sum(cpu) + sum(fan)
                self.ret[k, s] = DataCenter.calculate_avg_CRAC_return_temp(self.cfg.RACK_RETURN_APPROACH_TEMP_LIST,
                                                                            outlet)
        self._cache = {}

    def table(self, amb, wb):
        """(energy kWh (101,), best setpoint (101,)) at each whole-task utilization: min over the setpoints."""
        key = (float(amb), float(wb))
        if key not in self._cache:
            e = np.empty((len(self.setpoints), 101))
            for j, s in enumerate(self.setpoints):
                for k in range(101):
                    it = self.it[k, s]
                    _, ct_fan, _, chiller, _, _ = DataCenter.calculate_HVAC_power(s, self.ret[k, s], amb, it, self.cfg,
                                                                                  wet_bulb_temp=wb)
                    e[j, k] = (it + ct_fan + chiller) / 1e3 * STEP_HOURS
            best = e.argmin(axis=0)
            self._cache[key] = (e[best, np.arange(101)], np.asarray(self.setpoints)[best])
        return self._cache[key]


def lower_hull_lines(x, y):
    """Lines (a, b) of the lower convex hull of the points (x, y), x increasing: max_j a_j + b_j x <= y at every
    point, with equality at the hull's vertices."""
    hull = []
    for p in zip(x, y):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:  # hull[-1] is not below the chord
                hull.pop()
            else:
                break
        hull.append(p)
    if len(hull) == 1:
        return [(hull[0][1], 0.0)]
    lines = []
    for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
        b = (y2 - y1) / (x2 - x1)
        lines.append((y1 - b * x1, b))
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# The LP
# ---------------------------------------------------------------------------------------------------------------------
def build_and_solve(sdc, exo=None, energy=None, verbose=False):
    """Solve the episode's LP from the env right after reset(). Returns a dict: bound_kg (the lower bound on
    co2_work_adjusted, kg), the plan (d, p, ch, dis, u, setpoint per step), the noop metrics and solver info."""
    t0 = time.time()
    if getattr(sdc.bat_env, 'pv_dispatch', False) or float(np.max(sdc.pv_m.pv_kw)) > 0:
        raise NotImplementedError('sdc_lp_bound supports PV off only (pv_capacity_kWp = 0)')
    exo = exo or exo_traces(sdc)
    energy = energy or EnergyModel(sdc)
    ls, bat = sdc.ls_env, sdc.bat_env.battery
    T = len(exo['w'])
    ns = np.array([int(math.ceil(w * ls.non_shiftable_tasks_percentage * 100)) for w in exo['w']])
    sh = np.array([int(math.floor(w * ls.shiftable_tasks_percentage * 100)) for w in exo['w']])
    cap = np.maximum(MAX_UTIL_TASKS - ns - sh, 0)
    q0, qmax = float(len(ls.tasks_queue)), float(ls.queue_max_len)
    C, P = float(bat.capacity), float(bat.power_mw) * STEP_HOURS * 1e3  # MWh; kWh per step
    s_lo, s_hi, s0 = bat.soc_min * C * 1e3, bat.soc_max * C * 1e3, float(bat.current_load) * 1e3  # kWh
    ec, ed = bat.eff_charge, bat.eff_discharge

    # Variable layout: per step [d, p, Q_next, z, ch, dis, S_next], then y.
    nv = 7
    D, PR, Q, Z, CH, DIS, S = range(nv)
    n = nv * T + 1
    Y = n - 1
    idx = lambda t, k: nv * t + k

    c = np.zeros(n)
    lb, ub = np.zeros(n), np.full(n, np.inf)
    for t in range(T):
        c[idx(t, Z)] = c[idx(t, CH)] = exo['ci'][t] / 1e3
        c[idx(t, DIS)] = -exo['ci'][t] / 1e3
        ub[idx(t, D)] = sh[t]
        ub[idx(t, PR)] = cap[t]
        ub[idx(t, Q)] = qmax
        ub[idx(t, CH)] = ub[idx(t, DIS)] = P
        lb[idx(t, S)], ub[idx(t, S)] = s_lo, s_hi
    c[Y] = exo['kappa_kg']

    eq_r, eq_c, eq_v, beq = [], [], [], []
    ub_r, ub_c, ub_v, bub = [], [], [], []
    setpoint_tables = []
    row_eq = row_ub = 0
    for t in range(T):
        # Q_{t+1} - Q_t - d_t + p_t = 0
        eq_r += [row_eq] * 3; eq_c += [idx(t, Q), idx(t, D), idx(t, PR)]; eq_v += [1.0, -1.0, 1.0]
        if t > 0:
            eq_r.append(row_eq); eq_c.append(idx(t - 1, Q)); eq_v.append(-1.0)
        beq.append(q0 if t == 0 else 0.0); row_eq += 1
        # S_{t+1} - S_t - ec ch_t + dis_t / ed = 0
        eq_r += [row_eq] * 3; eq_c += [idx(t, S), idx(t, CH), idx(t, DIS)]; eq_v += [1.0, -ec, 1.0 / ed]
        if t > 0:
            eq_r.append(row_eq); eq_c.append(idx(t - 1, S)); eq_v.append(-1.0)
        beq.append(s0 if t == 0 else 0.0); row_eq += 1
        # energy hull over the step's reachable whole-task utilizations: a + b u <= z, u = (ns + sh - d + p) / 100
        e, best_sp = energy.table(exo['amb'][t], exo['wb'][t])
        setpoint_tables.append(best_sp)
        k_lo, k_hi = ns[t], min(ns[t] + sh[t] + cap[t], 100)
        for a, b in lower_hull_lines(energy.u[k_lo:k_hi + 1], e[k_lo:k_hi + 1]):
            # -z + (b / 100) (p - d) <= -a - b (ns + sh) / 100
            ub_r += [row_ub] * 3; ub_c += [idx(t, Z), idx(t, PR), idx(t, D)]; ub_v += [-1.0, b / 100, -b / 100]
            bub.append(-a - b * (ns[t] + sh[t]) / 100); row_ub += 1
        # dis_t - z_t <= 0
        ub_r += [row_ub] * 2; ub_c += [idx(t, DIS), idx(t, Z)]; ub_v += [1.0, -1.0]
        bub.append(0.0); row_ub += 1
    # -y - sum (p - d) / 100 <= -(sum w - sum (ns + sh) / 100)
    ub_r += [row_ub] * (2 * T + 1)
    ub_c += [Y] + [idx(t, PR) for t in range(T)] + [idx(t, D) for t in range(T)]
    ub_v += [-1.0] + [-0.01] * T + [0.01] * T
    bub.append(-(exo['w'].sum() - (ns + sh).sum() / 100)); row_ub += 1

    A_eq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(row_eq, n))
    A_ub = sp.csr_matrix((ub_v, (ub_r, ub_c)), shape=(row_ub, n))
    t_build = time.time() - t0
    res = linprog(c, A_ub=A_ub, b_ub=np.array(bub), A_eq=A_eq, b_eq=np.array(beq), bounds=np.c_[lb, ub],
                  method='highs-ds')
    if res.status != 0:
        raise RuntimeError(f'LP not solved: {res.message}')
    x = res.x
    get = lambda k: np.array([x[idx(t, k)] for t in range(T)])
    d, p = get(D), get(PR)
    u = (ns + sh - d + p) / 100
    k_u = np.clip(np.rint(u * 100).astype(int), 0, 100)
    out = {
        'bound_kg': float(res.fun), 'steps': T, 'rows': row_eq + row_ub, 'cols': n,
        'build_s': t_build, 'solve_s': time.time() - t0 - t_build, 'status': res.message,
        'unexecuted_util_steps': float(x[Y]), 'kappa_kg': exo['kappa_kg'],
        'co2_kg': float(res.fun - exo['kappa_kg'] * x[Y]),
        'plan': {'d': d, 'p': p, 'ch': get(CH), 'dis': get(DIS), 'u': u, 'sh': sh,
                 'setpoint': np.array([tab[k] for tab, k in zip(setpoint_tables, k_u)])},
        'noop': exo['noop'], 'setpoints': energy.setpoints,
    }
    if verbose:
        print(f"  LP: {T} steps, {out['rows']} rows, build {t_build:.1f}s, solve {out['solve_s']:.1f}s, "
              f"bound {out['bound_kg']:.1f} kg (noop {exo['noop']['co2_work_adjusted']:.1f})", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Replay policy
# ---------------------------------------------------------------------------------------------------------------------
class LPReplayPolicy:
    """Solves the episode's LP at reset (perfect foresight) and rounds its plan to actions (see the module doc)."""
    include_dc_setpoint = False
    include_pv_obs = False
    train_config = {}

    def __init__(self, defer_frac=0.5, process_tasks=1.0, bat_frac=0.1, verbose=False):
        self.defer_frac, self.process_tasks, self.bat_frac = float(defer_frac), float(process_tasks), float(bat_frac)
        self.verbose = verbose
        self.plan, self.k = None, 0

    def reset(self, seed):
        self.plan, self.k = None, 0  # solved on the first act(), which sees the env right after reset

    def act(self, sdc):
        if self.plan is None:
            self.solution = build_and_solve(sdc, verbose=self.verbose)
            self.plan = self.solution['plan']
            self.limit = float(sdc.bat_env.battery.power_mw) * STEP_HOURS * 1e3
        k, plan = min(self.k, len(self.plan['d']) - 1), self.plan
        self.k += 1
        if plan['sh'][k] > 0 and plan['d'][k] >= self.defer_frac * plan['sh'][k]:
            ls = 0
        elif plan['p'][k] >= self.process_tasks:
            ls = 2
        else:
            ls = 1
        ch, dis, thr = plan['ch'][k], plan['dis'][k], self.bat_frac * self.limit
        bat = 0 if ch > max(dis, thr) else (1 if dis > max(ch, thr) else 2)
        dc = setpoint_action(sdc.dc_env.raw_curr_stpt, float(plan['setpoint'][k]))
        return {'agent_ls': ls, 'agent_dc': dc, 'agent_bat': bat}


def make_lp_policy(rest):
    """Build an LPReplayPolicy from 'key=value,...' (the part after 'lp_replay:' in an eval_compare spec)."""
    params = dict(kv.split('=', 1) for kv in rest.split(',') if kv) if rest else {}
    valid = {'defer_frac', 'process_tasks', 'bat_frac', 'verbose'}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f'Unknown lp_replay parameter(s) {sorted(unknown)}; valid: {sorted(valid)}')
    kwargs = {k: (v.lower() in ('1', 'true', 'yes')) if k == 'verbose' else float(v) for k, v in params.items()}
    return LPReplayPolicy(**kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# CLI: the bound on the protocol episodes
# ---------------------------------------------------------------------------------------------------------------------
def main():
    import yaml
    from scipy import stats
    from sdc_eval.constants import AGENTS, ENV_CFG, FINAL_EVAL_SEED
    from sdc_eval.protocol import episode_months, parse_month
    from sdc_eval.rollout import reset_reward_history
    from sustaindc_env import SustainDC

    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument('--location', default=None)
    p.add_argument('--month', type=parse_month, default=None, help='0..11, all or cycle (as eval_compare.py).')
    p.add_argument('--workload_file', default=None, help='Workload trace (default: the env yaml), as eval_compare.py.')
    p.add_argument('--days_per_episode', type=int, default=None)
    p.add_argument('--pv_capacity_kWp', type=float, default=0.0)
    p.add_argument('--ci_year', type=int, default=None,
                   help='Year of the carbon intensity data (default: the env yaml, 2022; 2023: held-out, ca and tx).')
    p.add_argument('--n_episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=FINAL_EVAL_SEED, help='Episode i uses seed + i (as eval_compare.py).')
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()

    with open(ENV_CFG) as f:
        scenario = yaml.safe_load(f)
    for key in ['location', 'month', 'days_per_episode', 'ci_year', 'workload_file']:
        if getattr(args, key) is not None:
            scenario[key] = getattr(args, key)
    scenario['pv_config'] = {**scenario.get('pv_config', {}), 'capacity_kWp': args.pv_capacity_kWp}
    seeds = [args.seed + i for i in range(args.n_episodes)]
    months = episode_months(scenario['month'], len(seeds))
    print(f"LP bound: location={scenario['location']} month={scenario['month']} days={scenario['days_per_episode']} | "
          f"{len(seeds)} episodes, seeds {seeds[0]}..{seeds[-1]}", flush=True)
    envs = {}  # one simulator per episode month (as eval_compare.py)

    rows = []
    energy = None
    for seed, month in zip(seeds, months):
        if month not in envs:
            envs[month] = SustainDC({**scenario, 'month': month, 'agents': list(AGENTS), 'include_dc_setpoint': False,
                                     'include_pv_obs': False})
        sdc = envs[month]
        random.seed(seed)
        np.random.seed(seed)
        reset_reward_history()
        sdc.reset()  # the same episode eval_compare's run_episode draws for this seed
        energy = energy or EnergyModel(sdc)
        sol = build_and_solve(sdc, energy=energy)
        noop = sol['noop']['co2_work_adjusted']
        rows.append({'seed': seed, 'month': month, 'noop_co2_work_adjusted': noop, 'noop_co2': sol['noop']['co2'],
                     'bound_co2_work_adjusted': sol['bound_kg'], 'bound_co2': sol['co2_kg'],
                     'bound_unexecuted_util_steps': sol['unexecuted_util_steps'],
                     'pct_vs_noop': 100 * (sol['bound_kg'] / noop - 1),
                     'solve_s': sol['build_s'] + sol['solve_s'], 'status': sol['status']})
        print(f"  seed {seed}: noop {noop:.1f} kg, bound {sol['bound_kg']:.1f} kg ({rows[-1]['pct_vs_noop']:+.2f}%), "
              f"{rows[-1]['solve_s']:.1f}s", flush=True)

    d = np.array([r['pct_vs_noop'] for r in rows])
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    print(f"\nLower bound on co2_work_adjusted vs noop over {len(d)} episodes: {d.mean():+.2f}% ± {half:.2f} (95% CI)")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'bound.json'), 'w') as f:
        json.dump({'scenario': scenario, 'seeds': seeds, 'setpoints': energy.setpoints if energy else [],
                   'paired_vs_noop': {'co2_work_adjusted': {'mean_pct': float(d.mean()), 'ci95_pct': float(half),
                                                            'per_episode_pct': d.tolist()}},
                   'episodes': rows}, f, indent=2, default=float)
    import csv
    with open(os.path.join(args.out_dir, 'bound.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out_dir}/bound.json and bound.csv")


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    main()
