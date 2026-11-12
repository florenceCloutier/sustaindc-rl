"""Perfect-foresight lower bound on CO2 for GeoJobsDC (geo_jobs.py): a fluid LP relaxation of the job schedule.

Jobs are grouped into classes with the same (arrival step, origin, rigid, latest start step), i.e. the same set of
allowed (site, start) pairs. x[c, i, s] = fraction of class c started at site i at step s. Within a class every job
keeps its own duration and power: a class's load profile after its start is the exact sum of its jobs' profiles
(fractional last step included), so the only relaxation is that a class can be split across sites and start times
(indivisibility is dropped). Hence a valid lower bound on the CO2 any non-preemptive dispatcher can reach, given the
same jobs, weather and CI (cooling setpoint and battery fixed, as in the simulated runs).

    min  sum_{i,t} CI[i,t] * z[i,t]                       (g; z = site energy in kWh during step t)
    s.t. sum_{i,s} x[c,i,s] = 1                           every class starts
         u[i,t] = sum x[c,i,s] * util_profile_c,i[t - s]   site utilization (partial last step)
         sum x[c,i,s] * occ_profile_c,i[t - s] <= cap      committed load (full steps, as the simulator checks)
         z[i,t] >= E_i,t(u_k) + E'_i,t(u_k) (u[i,t] - u_k)  tangents of the convex energy curve (underestimator)
Starts may also fall up to `late_steps` after a class's latest start, with a large penalty, so bursts that no
schedule can absorb on time keep the LP feasible (the simulator lets such jobs start late too). The reported bound is
the CO2 term only.

Variants (allowed starts):
    geo_time   any site, any step in [arrival, latest start]          the bound
    geo        any site, start on arrival                             value of moving work in space only
    time       origin only, any step in [arrival, latest start]       value of moving work in time only
    local      origin only, on arrival                                no flexibility (sanity check vs the simulator)
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from baseline_policies import dc_power
from geo_sustaindc_env import STEP_HOURS

VARIANTS = ('local', 'geo', 'time', 'geo_time')
TANGENT_LOADS = np.linspace(0.0, 1.0, 11)


def exo_traces(geo, n_steps):
    """Per site, the next n_steps of CI (g/kWh), ambient and wet-bulb temperature, and the cooling setpoint."""
    out = []
    for s in geo.sites:
        t0c, t0w = s.ci_m.time_step, s.weather_m.time_step
        ci = np.asarray(s.ci_m.carbon_smooth[t0c:t0c + n_steps], dtype=float)
        amb = np.asarray(s.weather_m.temperature_data[t0w:t0w + n_steps], dtype=float)
        wb = np.asarray(s.weather_m.wet_bulb_data[t0w:t0w + n_steps], dtype=float)
        assert len(ci) == len(amb) == len(wb) == n_steps, 'episode runs past the end of the data year'
        out.append({'ci': ci, 'amb': amb, 'wb': wb, 'stpt': float(s.dc_env.raw_curr_stpt)})
    return out


def energy_curves(geo, traces):
    """Per site and step, IT + cooling energy (kWh) on TANGENT_LOADS and its slope: (N, T, K) values and slopes."""
    N, T, K = geo.n_sites, len(traces[0]['ci']), len(TANGENT_LOADS)
    val, slope = np.zeros((N, T, K)), np.zeros((N, T, K))
    h = 1e-3
    for i, (s, tr) in enumerate(zip(geo.sites, traces)):
        cache = {}
        for t in range(T):
            key = (round(tr['amb'][t], 2), round(tr['wb'][t], 2))
            if key not in cache:
                v, g = [], []
                for u in TANGENT_LOADS:
                    lo, hi = max(u - h, 0.0), min(u + h, 1.0)
                    p = sum(dc_power(s, u, tr['stpt'], *key))
                    p_lo, p_hi = sum(dc_power(s, lo, tr['stpt'], *key)), sum(dc_power(s, hi, tr['stpt'], *key))
                    v.append(p / 1e3 * STEP_HOURS)
                    g.append((p_hi - p_lo) / (hi - lo) / 1e3 * STEP_HOURS)
                cache[key] = (np.array(v), np.array(g))
            val[i, t], slope[i, t] = cache[key]
    return val, slope


def model_co2_kg(geo, traces, util):
    """Exact energy model (dc_power) at a given utilization schedule util (T, N): CO2 in kg. Used to calibrate
    the LP's model against the simulator."""
    total = 0.0
    for i, (s, tr) in enumerate(zip(geo.sites, traces)):
        for t in range(util.shape[0]):
            p = sum(dc_power(s, float(util[t, i]), tr['stpt'], tr['amb'][t], tr['wb'][t]))
            total += tr['ci'][t] * p / 1e3 * STEP_HOURS
    return total / 1e3


def job_classes(geo, variant, late_steps):
    """Group the episode's jobs; returns class arrays and per-class power profiles (W) after the start."""
    T = geo.horizon
    last = np.floor(geo.latest_start + 1e-9).astype(np.int64)
    if variant in ('geo', 'local'):
        last = geo.arrival.copy()
    last = np.minimum(last, T - 1)
    # When work can move, a flexible job's origin changes nothing (free migration): merge origins (origin -1 = any).
    origin = np.where(geo.rigid, geo.origin, -1) if variant in ('geo', 'geo_time') else geo.origin
    key = np.stack([geo.arrival, origin, geo.rigid.astype(np.int64), last], axis=1)
    keys, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    span = int(np.ceil(geo.dur.max()))
    n_cls = len(keys)
    util_prof = np.zeros((n_cls, span))
    occ_prof = np.zeros((n_cls, span))
    offs = np.arange(span)
    for o in offs:
        np.add.at(util_prof[:, o], inv, geo.power_w * np.clip(geo.dur - o, 0.0, 1.0))
        np.add.at(occ_prof[:, o], inv, geo.power_w * (o < np.ceil(geo.dur - 1e-9)))
    return {'arrival': keys[:, 0], 'origin': keys[:, 1], 'rigid': keys[:, 2].astype(bool), 'last': keys[:, 3],
            'util_prof': util_prof, 'occ_prof': occ_prof, 'late_steps': late_steps}


def solve_bound(geo, variant='geo_time', late_steps=4, traces=None, curves=None, verbose=True, method='highs-ipm'):
    """Solve the LP for one variant on geo's current episode (call right after geo.reset()). Returns a dict with
    'co2_kg' (the bound), 'late_job_mwh' (work started after its latest start), 'status' and timing."""
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    t_start = time.time()
    N, T = geo.n_sites, geo.horizon
    traces = traces or exo_traces(geo, T)
    if curves is None:
        curves = energy_curves(geo, traces)
    val, slope = curves
    ci = np.stack([tr['ci'] for tr in traces])  # (N, T)
    cls = job_classes(geo, variant, late_steps)
    span = cls['util_prof'].shape[1]
    cap_w = geo.cap_w
    move = variant in ('geo', 'geo_time')

    # ---- variables: x (class, site, start), u (N*T), z (N*T)
    xs_c, xs_i, xs_s, xs_late = [], [], [], []
    for c in range(len(cls['arrival'])):
        a, last = cls['arrival'][c], cls['last'][c]
        sites = range(N) if cls['origin'][c] < 0 else [cls['origin'][c]]
        hi = last if cls['rigid'][c] else min(last + late_steps, T - 1)
        if cls['rigid'][c]:
            hi = min(a + late_steps, T - 1)  # rigid: on arrival, late only if the origin is full
        starts = np.arange(a, max(hi, a) + 1)
        for i in sites:
            xs_c.append(np.full(len(starts), c))
            xs_i.append(np.full(len(starts), i))
            xs_s.append(starts)
            xs_late.append(np.maximum(starts - (a if cls['rigid'][c] else last), 0))
    xc, xi, xs, xlate = (np.concatenate(v) for v in (xs_c, xs_i, xs_s, xs_late))
    nx = len(xc)
    iu = nx + np.arange(N * T)            # u[i, t] -> iu[i*T + t]
    iz = nx + N * T + np.arange(N * T)
    nvar = nx + 2 * N * T

    # objective: CO2 (g) + lateness penalty (g-equivalent)
    penalty = 10.0 * float((ci[:, :, None] * slope).max())  # per unit-load-step started late
    c_obj = np.zeros(nvar)
    c_obj[iz] = ci.reshape(-1)
    late_w = cls['occ_prof'][xc, 0] / cap_w[xi]  # size of the class at its site
    c_obj[:nx] = penalty * xlate * late_w

    rows_eq, cols_eq, vals_eq, b_eq = [], [], [], []
    # every class starts once
    rows_eq.append(xc)
    cols_eq.append(np.arange(nx))
    vals_eq.append(np.ones(nx))
    n_cls = len(cls['arrival'])
    b_eq.append(np.ones(n_cls))
    # u definition: u[i,t] - sum x * util_prof[t - s] / cap_i = 0
    r_u, c_u, v_u = [iu - nx], [iu], [np.ones(N * T)]  # row index (i*T + t), offset by n_cls below
    r_o, c_o, v_o = [], [], []
    for o in range(span):
        t = xs + o
        ok = t < T
        uv = cls['util_prof'][xc[ok], o] / cap_w[xi[ok]]
        ov = cls['occ_prof'][xc[ok], o] / cap_w[xi[ok]]
        rr = xi[ok] * T + t[ok]
        nz = uv > 0
        r_u.append(rr[nz]); c_u.append(np.flatnonzero(ok)[nz]); v_u.append(-uv[nz])
        nz = ov > 0
        r_o.append(rr[nz]); c_o.append(np.flatnonzero(ok)[nz]); v_o.append(ov[nz])
    rows_eq.append(n_cls + np.concatenate(r_u))
    cols_eq.append(np.concatenate(c_u))
    vals_eq.append(np.concatenate(v_u))
    b_eq.append(np.zeros(N * T))
    A_eq = sp.csr_matrix((np.concatenate(vals_eq), (np.concatenate(rows_eq), np.concatenate(cols_eq))),
                         shape=(n_cls + N * T, nvar))
    b_eq = np.concatenate(b_eq)

    # inequalities: capacity, then tangents  -z + slope*u <= slope*u_k - val
    K = len(TANGENT_LOADS)
    A_cap = sp.csr_matrix((np.concatenate(v_o), (np.concatenate(r_o), np.concatenate(c_o))), shape=(N * T, nvar))
    b_cap = np.full(N * T, geo.cap)
    rk = np.arange(N * T * K)
    it = np.repeat(np.arange(N * T), K)
    sl = slope.reshape(-1)
    A_tan = sp.csr_matrix((np.concatenate([sl, -np.ones(N * T * K)]),
                           (np.concatenate([rk, rk]), np.concatenate([iu[it], iz[it]]))), shape=(N * T * K, nvar))
    b_tan = (slope * TANGENT_LOADS[None, None, :] - val).reshape(-1)
    A_ub = sp.vstack([A_cap, A_tan]).tocsr()
    b_ub = np.concatenate([b_cap, b_tan])

    bounds = np.zeros((nvar, 2))
    bounds[:, 1] = np.inf
    bounds[iu, 1] = 1.0
    t_build = time.time() - t_start
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    out = {'variant': variant, 'status': res.status, 'message': res.message, 'n_vars': nvar, 'n_classes': n_cls,
           'build_s': t_build, 'solve_s': time.time() - t_start - t_build}
    if res.status != 0:
        if verbose:
            print(f'[lp {variant}] failed: {res.message}')
        out['co2_kg'] = float('nan')
        return out
    x = res.x
    out['co2_kg'] = float(ci.reshape(-1) @ x[iz]) / 1e3
    job_mwh = cls['util_prof'].sum(1)[xc] * STEP_HOURS / 1e6  # MWh of the class
    out['late_job_mwh'] = float(np.sum(x[:nx] * job_mwh * (xlate > 0)))
    out['util'] = x[iu].reshape(N, T).T  # (T, N)
    if verbose:
        print(f"[lp {variant}] {out['co2_kg']:.1f} kg, {n_cls} classes, {nvar} vars, build {t_build:.0f}s, "
              f"solve {out['solve_s']:.0f}s, late {out['late_job_mwh']:.2f} MWh")
    return out
