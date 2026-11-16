"""Checks for mppi_policy.py: the ordinal encoding of the sub-actions, the MPPI weights, spec parsing, plans that use
past data only with the persistence forecast, parallel = serial rollouts, PV actions, and MPPI beats doing nothing.
Run from the repo root: python tests/verify_mppi.py"""
import copy, os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import yaml

from eval_compare import NoOpPolicy, run_episode
from mpc_policy import EXOGENOUS_SERIES, NOOP_JOINT, joint_space
from mppi_policy import (MPPIPolicy, effective_sample_size, make_mppi_policy, mppi_weights, ordinal_to_sub,
                         sub_to_ordinal)
from sdc_eval.policies import make_policy, split_label
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS, decode_joint_action

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    SCENARIO = {**yaml.safe_load(f), 'days_per_episode': 1, 'agents': list(AGENTS)}
results = []
SMALL = dict(horizon=32, block=4, samples=16, iters=2)


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def started_env(seed=3, steps=5, scenario=SCENARIO):
    random.seed(seed); np.random.seed(seed)
    sdc = SustainDC(dict(scenario)); sdc.reset()
    for _ in range(steps):
        sdc.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
    return sdc


def main():
    print("1. Ordinal encoding and MPPI weights")
    subs = np.array(np.meshgrid(range(3), range(3), range(3), indexing='ij')).reshape(3, -1).T
    check("sub-action -> ordinal -> sub-action round trip", np.array_equal(ordinal_to_sub(sub_to_ordinal(subs)), subs))
    check("battery order: charge +1, discharge -1, idle 0", sub_to_ordinal([[1, 1, 0]])[0, 2] == 1
          and sub_to_ordinal([[1, 1, 1]])[0, 2] == -1 and sub_to_ordinal([[1, 1, 2]])[0, 2] == 0)
    check("zero vector is the do-nothing action",
          np.ravel_multi_index(tuple(ordinal_to_sub(np.zeros(3))), (3, 3, 3)) == NOOP_JOINT)
    check("rounding and clipping", np.array_equal(ordinal_to_sub(np.array([0.49, -0.51, 7.0])), [1, 0, 0]))
    w = mppi_weights([3.0, 1.0, 2.0], 0.5)
    check("weights sum to 1 and rank by cost", np.isclose(w.sum(), 1) and w[1] > w[2] > w[0])
    check("equal costs give uniform weights", np.allclose(mppi_weights([5.0] * 4, 0.5), 0.25))
    check("low temperature ~ best sample only", mppi_weights([3.0, 1.0, 2.0], 1e-3)[1] > 0.999)
    check("weights invariant to a cost offset and scale",
          np.allclose(mppi_weights([3.0, 1.0, 2.0], 0.5), mppi_weights([3e4 + 7, 1e4 + 7, 2e4 + 7], 0.5)))
    costs = np.random.default_rng(0).gamma(2.0, 1e4, 60)
    for target in (5, 10):
        got = effective_sample_size(mppi_weights(costs, ess=target))
        check(f"ess={target}: lambda reaches the target", abs(got - target) < 0.05, f"ESS {got:.3f}")
    check("ess target invariant to cost offset and scale",
          np.allclose(mppi_weights(costs, ess=5), mppi_weights(3 * costs + 1e5, ess=5)))
    check("ess >= n gives uniform weights", np.allclose(mppi_weights(costs[:4], ess=10), 0.25))
    try:
        mppi_weights(costs, temperature=0.5, ess=5); check("temperature and ess together rejected", False)
    except ValueError:
        check("temperature and ess together rejected", True)
    rng = np.random.default_rng(0)
    vals = np.tile([0.3, -0.7, 1.0], (20000, 1))
    mean_val = sub_to_ordinal(ordinal_to_sub(vals, rng)).mean(axis=0)
    check("stochastic rounding: expected action = continuous value", np.allclose(mean_val, [0.3, -0.7, 1.0], atol=0.02),
          f"means {np.round(mean_val, 3).tolist()}")
    check("stochastic rounding: integer values unchanged",
          np.array_equal(ordinal_to_sub(sub_to_ordinal(subs), rng), subs))

    print("2. Parameters and spec parsing")
    p = make_mppi_policy('horizon=50,block=8,samples=8,iters=2,temperature=0.2,sampling=categorical')
    check("parsed", (p.horizon, p.n_blocks, p.samples, p.temperature, p.ess, p.sampling, p.smoothing)
          == (56, 7, 8, 0.2, None, 'categorical', 0.7))
    d = MPPIPolicy()
    check("defaults: ess=5, execute=best, stochastic rounding, seed plans",
          (d.ess, d.temperature, d.execute, d.rounding, d.seed_plans) == (5.0, None, 'best', 'stochastic', True))
    q = make_mppi_policy('ess=10,rounding=nearest,seed_plans=0,execute=mean')
    check("new parameters parsed", (q.ess, q.rounding, q.seed_plans, q.execute) == (10.0, 'nearest', False, 'mean'))
    for bad in ('temp=1', 'sampling=uniform', 'execute=worst', 'temperature=0', 'ess=1', 'ess=5,temperature=0.5',
                'rounding=up'):
        try:
            make_mppi_policy(bad); check(f"bad parameter rejected: {bad}", False)
        except ValueError:
            check(f"bad parameter rejected: {bad}", True)
    check("eval_compare labels", split_label('mppi:horizon=96,temperature=0.3=lbl')
          == ('mppi:horizon=96,temperature=0.3', 'lbl') and split_label('mppi=lbl') == ('mppi', 'lbl'))
    label, pol = make_policy('mppi:samples=8,iters=1=my_mppi')
    check("make_policy builds an MPPIPolicy", isinstance(pol, MPPIPolicy) and label == 'my_mppi' and pol.samples == 8)

    print("3. Persistence plans from past data only; oracle uses the future")
    sdc = started_env()
    twin = copy.deepcopy(sdc)
    for m, attrs in EXOGENOUS_SERIES.items():
        mgr = getattr(twin, m)
        for a in attrs:
            arr = getattr(mgr, a).copy()
            arr[mgr.time_step + 1:] = arr[mgr.time_step + 1:][::-1]
            setattr(mgr, a, arr)
    for sampling in ('gaussian', 'categorical'):
        noms = {}
        for fc in ('persistence', 'oracle'):
            for name, env in (('real', sdc), ('twin', twin)):
                pol = MPPIPolicy(**SMALL, forecast=fc, sampling=sampling)
                pol.reset(7)
                pol.act_joint(env)
                noms[fc, name] = pol.nominal.copy()
        check(f"{sampling}: persistence nominal independent of the true future",
              np.array_equal(noms['persistence', 'real'], noms['persistence', 'twin']))
        check(f"{sampling}: oracle nominal depends on the true future",
              not np.array_equal(noms['oracle', 'real'], noms['oracle', 'twin']))

    print("4. Receding horizon: warm start and reset")
    pol = MPPIPolicy(**SMALL, forecast='persistence')
    pol.reset(1)
    sdc = started_env()
    pol.act_joint(sdc)
    first = pol.nominal.copy()
    for _ in range(3):
        sdc.step(decode_joint_action(pol.act_joint(sdc), 'discrete'))
    check("pending actions executed before replanning", pol.stats['replans'] == 1)
    pol.act_joint(sdc)
    check("second replan starts from the shifted nominal", pol.stats['replans'] == 2 and pol.nominal.shape == first.shape)
    pol.reset(1)
    check("reset clears the nominal", pol.nominal is None and pol.plan is None)
    for sampling in ('gaussian', 'categorical'):
        for seeded in (True, False):
            pol = MPPIPolicy(**SMALL, sampling=sampling, seed_plans=seeded)
            pol.reset(1)
            pol._init_nominal()
            plans, eps, _ = pol._candidates(pol.samples, pol.noise, first=True)
            consts = {tuple(np.full(pol.n_blocks, a)) for a in range(27)}
            have = consts <= {tuple(p) for p in plans}
            check(f"{sampling}, seed_plans={seeded}: constant plans {'in' if seeded else 'not in'} first iteration",
                  have == seeded and len(plans) == pol.samples + 27 * seeded)
            if sampling == 'gaussian' and seeded:
                nearest = ordinal_to_sub(pol.nominal[None] + eps[pol.samples:])
                check("gaussian seeds: eps reaches the constant plan", np.array_equal(
                    np.ravel_multi_index(tuple(np.moveaxis(nearest, -1, 0)), (3, 3, 3)), plans[pol.samples:]))
            later, _, _ = pol._candidates(pol.samples, pol.noise, first=False)
            check(f"{sampling}, seed_plans={seeded}: later iterations sample only", len(later) == pol.samples)

    print("5. One-day episodes vs do-nothing (same seed)")
    env = SustainDC(dict(SCENARIO))
    noop, fp_noop = run_episode(env, NoOpPolicy(), 1000)
    for kw in (dict(sampling='gaussian'), dict(sampling='categorical'),
               dict(sampling='gaussian', execute='mean', rounding='nearest', seed_plans=False, temperature=0.5),
               dict(sampling='gaussian', forecast='persistence')):
        pol = MPPIPolicy(**SMALL, **kw)
        m, fp = run_episode(env, pol, 1000)
        name = ','.join(f'{k}={v}' for k, v in kw.items())
        check(f"{name}: same exogenous inputs as noop", fp == fp_noop)
        check(f"{name}: work-adjusted CO2 lower than noop", m['co2_work_adjusted'] < noop['co2_work_adjusted'],
              f"{m['co2_work_adjusted']:.1f} vs {noop['co2_work_adjusted']:.1f} kg, {pol.stats['rollouts']} rollouts, "
              f"{pol.stats['plan_seconds']:.0f}s")
    again = MPPIPolicy(**SMALL, sampling='gaussian')
    m2, _ = run_episode(env, again, 1000)
    first_run = MPPIPolicy(**SMALL, sampling='gaussian')
    m1, _ = run_episode(env, first_run, 1000)
    check("deterministic given the seed", m1 == m2)

    print("6. Parallel rollouts match serial ones")
    serial = MPPIPolicy(**SMALL, forecast='persistence')
    m_serial, _ = run_episode(env, serial, 1001)
    parallel = MPPIPolicy(**SMALL, forecast='persistence', workers=4)
    try:
        m_par, _ = run_episode(env, parallel, 1001)
    finally:
        parallel.close()
    check("identical episode metrics with workers=4", m_par == m_serial)

    print("7. Agent-dispatched PV: 81 joint actions")
    pv_act = {**SCENARIO, 'location': 'az', 'pv_config': {'capacity_kWp': 1000}, 'include_pv': True}
    env = SustainDC(dict(pv_act))
    noop, _ = run_episode(env, NoOpPolicy(), 1000)
    bat_seen = []

    class Traced(MPPIPolicy):
        def act(self, sdc):
            a = super().act(sdc)
            bat_seen.append(a['agent_bat'])
            return a
    pol = Traced(**SMALL, forecast='persistence')
    m, _ = run_episode(env, pol, 1000)
    pv_ids = np.array(bat_seen) // 3
    check("plans over 81 actions, PV distribution per block",
          pol.n_joint == 81 and pol.pv_probs is not None and pol.pv_probs.shape == (pol.n_blocks, 3))
    check("do-nothing nominal uses PV ('use')", joint_space(True)[1] == pol._joint(pol._noop_sub()))
    check("PV actions beyond 'use' are chosen", (pv_ids > 0).any(),
          f"use/battery/resell = {np.bincount(pv_ids, minlength=3).tolist()}")
    check("work-adjusted CO2 lower than noop with PV actions", m['co2_work_adjusted'] < noop['co2_work_adjusted'],
          f"{m['co2_work_adjusted']:.1f} vs {noop['co2_work_adjusted']:.1f} kg")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == '__main__':  # required: rollout workers are started with 'spawn', which re-imports this module
    sys.exit(0 if main() else 1)
