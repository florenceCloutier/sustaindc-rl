"""Checks for sdc_lp_bound.py (perfect-foresight LP bound on co2_work_adjusted) and mpc_policy's end_cost=metric:
the LP's relaxation contains every simulated trajectory, step by step (utilization, energy, queue, battery), its
optimum is below the co2_work_adjusted of every policy tried, the helpers leave the live episode untouched, and the
replay policy runs in eval_compare. Run from the repo root: python tests/verify_lp_bound.py"""
import os, random, sys
sys.path.insert(0, os.getcwd())
import math

import numpy as np
import yaml

from sdc_eval.policies import make_policy
from sdc_eval.rollout import reset_reward_history, run_episode
from sdc_lp_bound import EnergyModel, build_and_solve, exo_traces, lower_hull_lines, reachable_setpoints
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = yaml.safe_load(f)
DAYS = 2
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_env(location):
    sc = {**BASE, 'location': location, 'month': 6, 'days_per_episode': DAYS,
          'pv_config': {**BASE.get('pv_config', {}), 'capacity_kWp': 0}}
    return SustainDC({**sc, 'agents': list(AGENTS), 'include_dc_setpoint': False, 'include_pv_obs': False})


def seeded_reset(sdc, seed):
    random.seed(seed); np.random.seed(seed)
    reset_reward_history()
    sdc.reset()


def relaxation_violations(sdc, policy, seed, energy):
    """Run one episode like run_episode and check every step against the LP's constraints. Returns
    (worst violation per constraint family, the episode's co2_work_adjusted, the LP bound)."""
    from sdc_eval.metrics import EpisodeAccumulator
    seeded_reset(sdc, seed)
    sol = build_and_solve(sdc, energy=energy)
    exo = exo_traces(sdc)
    ls, bat = sdc.ls_env, sdc.bat_env
    policy.reset(seed)
    acc = EpisodeAccumulator()
    worst = {'u_range': 0.0, 'energy_hull': 0.0, 'queue_balance': 0.0, 'grid_balance': 0.0, 'exo_match': 0.0}
    t = 0
    while True:
        w = float(ls.workload)
        ns = math.ceil(w * ls.non_shiftable_tasks_percentage * 100)
        sh = math.floor(w * ls.shiftable_tasks_percentage * 100)
        cap = max(90 - ns - sh, 0)
        q_before = len(ls.tasks_queue)
        worst['exo_match'] = max(worst['exo_match'], abs(w - exo['w'][t]), abs(sdc.dc_env.ambient_temp - exo['amb'][t]))
        actions = policy.act(sdc)
        _, _, _, truncs, info = sdc.step(actions)
        i = info['agent_ls']
        acc.step(i, actions)
        k = i['ls_shifted_workload'] * 100
        # utilization is a whole number of tasks within [ns, ns + sh + cap]
        worst['u_range'] = max(worst['u_range'], abs(k - round(k)), ns - k, k - min(ns + sh + cap, 100))
        # queue: Q' = Q + d - p with 0 <= d <= sh, 0 <= p <= cap and u = ns + sh - d + p
        dq = i['ls_tasks_in_queue'] - q_before
        net = (ns + sh) - k  # = d - p
        worst['queue_balance'] = max(worst['queue_balance'], abs(dq - net), -net - cap, net - sh)
        # energy above the hull of the step (built exactly as in the LP)
        e, _ = energy.table(exo['amb'][t], exo['wb'][t])
        lines = lower_hull_lines(energy.u[ns:min(ns + sh + cap, 100) + 1], e[ns:min(ns + sh + cap, 100) + 1])
        z = i['dc_total_power_kW'] * 0.25
        hull = max(a + b * k / 100 for a, b in lines)
        worst['energy_hull'] = max(worst['energy_hull'], hull - z)
        # grid = z + charge - discharge, and CO2 = CI x grid
        grid = z + i['bat_energy_in_KWh'] - i['bat_energy_out_KWh']
        worst['grid_balance'] = max(worst['grid_balance'], abs(grid - i['bat_total_energy_with_battery_KWh']),
                                    abs(i['bat_CO2_footprint'] - grid * exo['ci'][t]) / max(exo['ci'][t], 1.0))
        t += 1
        if truncs['__all__']:
            break
    return worst, acc.finalize(sdc)['co2_work_adjusted'], sol['bound_kg']


def main():
    print("1. Helpers")
    s = reachable_setpoints(18, 15.0, 21.6)
    expected = sorted({float(x) for x in range(15, 22)} | {round(21.6 - k, 6) for k in range(0, 7)})
    check("reachable setpoints from 18 C in [15, 21.6]: whole degrees and 21.6 - k", s == expected, f"{s}")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 101)
    y = rng.random(101) + 3 * x ** 2
    lines = lower_hull_lines(x, y)
    below = max(max(a + b * xi for a, b in lines) - yi for xi, yi in zip(x, y))
    check("lower hull lies below every point", below <= 1e-9, f"max excess {below:.2e}")
    check("lower hull touches the lowest point", abs(max(a + b * x[np.argmin(y)] for a, b in lines) - y.min()) < 1e-9)

    sdc = make_env('tx')
    seeded_reset(sdc, 1000)
    state = (random.getstate(), np.random.get_state()[1].copy(), len(sdc.ls_env.tasks_queue), sdc.t_m.current_timestep)
    exo = exo_traces(sdc)
    after = (random.getstate(), np.random.get_state()[1].copy(), len(sdc.ls_env.tasks_queue), sdc.t_m.current_timestep)
    check("exo_traces leaves the live env and the RNG untouched",
          state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:])
    m_noop, _ = run_episode(sdc, make_policy('noop')[1], 1000)
    check("noop metrics from exo_traces match run_episode exactly",
          all(abs(m_noop[k] - exo['noop'][k]) < 1e-9 * max(1.0, abs(m_noop[k]))
              for k in ('co2', 'co2_work_adjusted', 'total_energy')),
          f"co2_work_adjusted {m_noop['co2_work_adjusted']:.3f} vs {exo['noop']['co2_work_adjusted']:.3f}")

    print("2. Every simulated trajectory satisfies the LP's constraints, and the bound is below every policy")
    specs = ['noop', 'random', 'rule:ls,dc,bat', 'oracle:ls,dc,bat', 'setpoint:15',
             'mpc_joint:horizon=16,block=4,samples=8,iters=1', 'lp_replay']
    for location in ('tx', 'ca'):
        sdc = make_env(location)
        seeded_reset(sdc, 1000)
        energy = EnergyModel(sdc)
        for spec in specs:
            for seed in (1000, 1001):
                label, pol = make_policy(spec)
                worst, metric, bound = relaxation_violations(sdc, pol, seed, energy)
                tol = {'u_range': 1e-9, 'energy_hull': 1e-6, 'queue_balance': 1e-9, 'grid_balance': 1e-6,
                       'exo_match': 1e-12}
                bad = {k: v for k, v in worst.items() if v > tol[k]}
                check(f"{location} {label} seed {seed}: constraints hold, metric {metric:.1f} >= bound {bound:.1f} kg",
                      not bad and metric >= bound - 1e-6, f"violations {bad}" if bad else
                      f"gap {100 * (metric / bound - 1):.2f}%")

    print("3. mpc_joint end_cost=metric")
    from mpc_policy import SimCloner, episode_end_task_g
    from sdc_eval.constants import TASK_UTILIZATION
    sdc = make_env('tx')
    seeded_reset(sdc, 1000)
    exo = exo_traces(sdc)
    sim = SimCloner.clone(sdc)
    while True:
        _, _, _, truncs, _ = sim.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
        if truncs['__all__']:
            break
    g = episode_end_task_g(sim)
    ref = exo['kappa_kg'] * TASK_UTILIZATION * 1e3
    check("planner's end-of-episode task charge matches the metric's (within 3%)", abs(g / ref - 1) < 0.03,
          f"{g:.2f} vs {ref:.2f} g per task")
    label, pol = make_policy('mpc_joint:horizon=16,block=4,samples=8,iters=1,end_cost=metric,drop_w=0,overdue_w=0=m')
    m, _ = run_episode(sdc, pol, 1000)
    check("end_cost=metric MPC runs and beats doing nothing", m['co2_work_adjusted'] < exo['noop']['co2_work_adjusted'],
          f"{m['co2_work_adjusted']:.1f} vs noop {exo['noop']['co2_work_adjusted']:.1f} kg")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
