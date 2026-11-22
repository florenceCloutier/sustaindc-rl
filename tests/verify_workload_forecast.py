"""Checks for workload_forecast.py and its use in the MPC: the training series is the simulator's workload, the
pinball LP recovers known quantiles, forecasts use past data only and never cross, the adaptive conformal margin
restores coverage of a biased model, the planning model gets the forecast, and the MPC / MPPI run with it.
Run from the repo root: python tests/verify_workload_forecast.py"""
import copy, os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import yaml

from workload_forecast import (DAY, HORIZONS, QUANTILES, WEEK, WorkloadForecaster, WorkloadQuantileModel,
                               critical_quantile, day_mask, evaluate, features, load_trace, pinball,
                               quantile_regression)

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    SCENARIO = yaml.safe_load(f)
TRACES = ('Alibaba_CPU_Data_Hourly_1.csv', 'Alibaba_CPU_Data_Hourly_2.csv', 'GoogleClusteData_CPU_Data_Hourly_1.csv')
GOOGLE = TRACES[2]
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def small_model(y, exclude=None):
    """Quick fit (few origins) for the tests."""
    return WorkloadQuantileModel.fit([y], stride=24, exclude=[exclude], verbose=False)


def main():
    print("1. Training series = the simulator's workload")
    from utils.managers import Workload_Manager
    for fn in TRACES:
        for shift in (0, 3):
            wm = Workload_Manager(workload_filename=fn, timezone_shift=shift)
            wm.reset(init_day=100)
            d = np.abs(load_trace(fn, shift) - wm.cpu_smooth).max()
            check(f"{fn} (timezone shift {shift})", d < 1e-12, f"max diff {d:.2g}")

    print("2. Pinball loss and quantile regression")
    check("critical ratio 20:1 -> 0.952", abs(critical_quantile(20, 1) - 20 / 21) < 1e-12)
    check("pinball penalizes under-forecasts q/(1-q) times more",
          abs(pinball([1.0], [0.0], 0.9) / pinball([0.0], [1.0], 0.9) - 9) < 1e-9)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 4000)
    yy = 1 + 2 * x + (1 + x) * rng.standard_normal(4000)     # q-quantile: 1 + z_q + (2 + z_q) x
    X = np.stack([np.ones_like(x), x], axis=1)
    for q, z in ((0.5, 0.0), (0.9, 1.2816)):
        b = quantile_regression(X, yy, q)
        check(f"LP recovers the {q} quantile line", np.allclose(b, [1 + z, 2 + z], atol=0.2), f"coef {np.round(b, 3)}")
        check(f"  in-sample coverage {q}", abs(np.mean(yy <= X @ b) - q) < 0.01)

    print("3. Forecasts use past data only, never cross, stay in [0, 1]")
    y = load_trace(GOOGLE)
    test_days = (174, 219)
    excl = day_mask(len(y), [test_days])
    model = small_model(y, excl)
    t = 200 * DAY + 37
    scrambled = y.copy()
    scrambled[t + 1:] = scrambled[t + 1:][::-1]
    same = all(np.array_equal(features(y, [t], h), features(scrambled, [t], h)) for h in HORIZONS)
    check("features at t identical whatever comes after t", same)
    check("features from the truncated series identical",
          all(np.array_equal(features(y, [t], h), features(y[:t + 1], [t], h)) for h in HORIZONS))
    try:
        features(y, [t], DAY + 1); check("horizons beyond a day rejected", False)
    except ValueError:
        check("horizons beyond a day rejected", True)
    preds = model.predict(y, np.arange(WEEK, len(y) - DAY, 7))
    check("quantiles never cross", np.all(np.diff(preds, axis=-1) >= 0))
    check("forecasts in [0, 1]", preds.min() >= 0 and preds.max() <= 1)
    fa, fb = WorkloadForecaster(model, q=0.9), WorkloadForecaster(model, q=0.9)
    va, vb = fa.forecast(y, t, 96), fb.forecast(scrambled, t, 96)
    check("online forecaster (margins included) identical whatever comes after t", np.array_equal(va, vb))
    check("forecast covers 24 h, interpolated between anchors", len(va) == 96 and len(fa.forecast(y, t, 500)) == 96)
    fc = WorkloadForecaster(model, q=0.9)
    fc.forecast(y, t, 4)
    fc.forecast(y, t - 10, 4)          # time going backward = new episode
    check("forecaster restarts when time goes backward", fc.start == max(t - 10 - fc.warmup, 0))

    print("4. Save / load")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tmp_wl_model.npz')
    try:
        model.meta['files'] = [GOOGLE]
        model.save(path)
        back = WorkloadQuantileModel.load(path)
        check("round trip", np.array_equal(back.coef, model.coef) and back.horizons == model.horizons
              and back.quantiles == model.quantiles and back.meta['files'] == [GOOGLE])
    finally:
        if os.path.exists(path):
            os.remove(path)
    try:
        model.quantile_index(0.77); check("unfitted quantile rejected", False)
    except ValueError:
        check("unfitted quantile rejected", True)

    print("5. Conservatism and calibration on held-out days (Google trace)")
    res = evaluate(model, y, test_days, q=0.9)
    cov = res['q0.9+aci']['coverage']
    check("conformal bound covers ~90% at every anchor horizon", np.all(np.abs(cov - 0.9) < 0.03), f"{np.round(cov, 3)}")
    hu_p = np.nanmean(res['persistence']['high_under_rate'])
    hu_b = np.nanmean(res['q0.9+aci']['high_under_rate'])
    sf_p = np.nanmean(res['persistence']['high_shortfall'])
    sf_b = np.nanmean(res['q0.9+aci']['high_shortfall'])
    check("at high load, under-forecasts much rarer than persistence", hu_b < 0.5 * hu_p, f"{hu_b:.2f} vs {hu_p:.2f}")
    check("at high load, mean shortfall much smaller than persistence", sf_b < 0.5 * sf_p, f"{sf_b:.3f} vs {sf_p:.3f}")
    check("median model under-forecasts about half the time at short horizons",
          abs(res['median']['coverage'][0] - 0.5) < 0.1, f"{res['median']['coverage'][0]:.2f}")
    biased = WorkloadQuantileModel(model.coef.copy(), model.horizons, model.quantiles, model.scale)
    biased.coef[:, :, 0] -= 0.1                                  # every quantile 0.1 too low
    raw = evaluate(biased, y, test_days, q=0.9, aci=False)['q0.9+aci']['coverage']
    adj = evaluate(biased, y, test_days, q=0.9, aci=True)['q0.9+aci']['coverage']
    check("biased model: raw quantile badly under-covers", raw.max() < 0.75, f"{np.round(raw, 2)}")
    check("biased model: conformal margin restores ~90% coverage", np.all(adj > 0.85), f"{np.round(adj, 3)}")

    print("6. Planning model and MPC")
    from eval_compare import run_episode
    from mpc_policy import MPCPolicy, make_mpc_policy, planning_model
    from mppi_policy import make_mppi_policy
    from sustaindc_env import SustainDC
    from sustaindc_single_env import AGENTS
    scn = {**SCENARIO, 'days_per_episode': 1, 'agents': list(AGENTS), 'workload_file': GOOGLE}
    random.seed(3); np.random.seed(3)
    sdc = SustainDC(dict(scn)); sdc.reset()
    for _ in range(5):
        sdc.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
    wm = sdc.workload_m
    t, live = wm.time_step, wm.cpu_smooth.copy()
    wl = np.full(40, 0.77)
    m = planning_model(sdc, 'oracle', 1, workload=wl)
    s = m.workload_m.cpu_smooth
    check("live workload untouched", np.array_equal(wm.cpu_smooth, live) and wm.cpu_smooth is not s)
    check("planning workload: truth up to now, forecast next", np.array_equal(s[:t + 1], live[:t + 1])
          and np.array_equal(s[t + 1:t + 41], wl))
    check("planning workload after the forecast: persistence, even with forecast=oracle",
          np.array_equal(s[t + 41:t + 1 + DAY], live[t + 41 - DAY:t + 1 + DAY - DAY]))
    check("other series still oracle", m.ci_m.carbon_smooth is sdc.ci_m.carbon_smooth)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tmp_wl_mpc.npz')
    model.save(model_path)
    try:
        twin = copy.deepcopy(sdc)
        arr = twin.workload_m.cpu_smooth.copy()
        arr[t + 1:] = arr[t + 1:][::-1]
        twin.workload_m.cpu_smooth = arr
        plans = {}
        for name, env in (('real', sdc), ('twin', twin)):
            p = MPCPolicy(horizon=32, block=4, samples=16, iters=2, forecast='oracle', wl_model=model_path)
            p.reset(7)
            p.act_joint(env)
            plans[name] = p.plan
        # the CI etc. are oracle but unchanged in the twin, so only the true future workload differs
        check("with wl_model: same plan whatever the true future workload", np.array_equal(plans['real'], plans['twin']))

        p = make_mpc_policy(f'horizon=16,samples=4,iters=1,wl_model={model_path},wl_q=0.95,wl_plan=upper,'
                            f'wl_gamma=0.1,wl_aci=0')
        check("mpc_joint spec parses the wl parameters",
              p.wl is not None and p.wl.q == 0.95 and p.wl.gamma == 0.1 and not p.wl.aci)
        p = make_mppi_policy(f'horizon=16,samples=4,iters=1,wl_model={model_path},wl_plan=median')
        check("mppi spec parses the wl parameters", p.wl is not None and p.wl.q == 0.5 and not p.wl.aci)
        try:
            make_mpc_policy(f'wl_model={model_path},wl_plan=optimistic'); check("bad wl_plan rejected", False)
        except ValueError:
            check("bad wl_plan rejected", True)

        env = SustainDC(dict(scn))
        serial = MPCPolicy(horizon=16, block=4, samples=8, iters=1, forecast='persistence', wl_model=model_path)
        m_serial, _ = run_episode(env, serial, 1001)
        check("MPC episode runs with the forecaster", serial.wl.counts.sum() > 0 and serial.stats['replans'] > 0,
              f"bound coverage per horizon {np.round(serial.wl.coverage(), 2)}")
        parallel = MPCPolicy(horizon=16, block=4, samples=8, iters=1, forecast='persistence', wl_model=model_path,
                             workers=2)
        try:
            m_par, _ = run_episode(env, parallel, 1001)
        finally:
            parallel.close()
        check("parallel rollouts identical to serial with the forecaster", m_par == m_serial)
    finally:
        os.remove(model_path)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == '__main__':  # required: rollout workers are started with 'spawn', which re-imports this module
    sys.exit(0 if main() else 1)
