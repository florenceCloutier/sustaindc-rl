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

Overflow: a burst that cannot be placed even `late_steps` late (e.g. origin-only `time`, or start-on-arrival `geo`,
in a busy hour) would make the LP infeasible. Each class therefore has one overflow variable v[c] (the share of the
class not placed in the window) with an even larger penalty, so it is only used when nothing else fits. Overflow work
uses no capacity and is charged a lower bound on its CO2: its load-steps x the lowest CI x marginal energy at zero
load over all of its allowed sites and steps (the energy curve is convex, so the marginal energy at any load is at
least the one at zero load). The result stays a valid lower bound; `overflow_job_mwh` reports how much work used it.

Site flexibility (`site_flex`): the simulated hardness runs hold cooling and battery fixed, and so does the default
bound ('none': the current CRAC setpoint for the whole episode, no battery). With 'battery' and/or 'cooling'
(e.g. 'battery+cooling'), the bound also lets every site
    battery  charge and discharge continuously within its power limit and usable SoC window, with its efficiencies:
             g = z + ch - dis (grid energy), soc' = soc + eff_c ch - dis / eff_d, dis <= z (no export). The simulator's
             battery (three discrete actions, charging taper above 90% SoC) is a subset of this.
    cooling  pick the best CRAC setpoint every step: the energy curve is replaced by lines below min over
             SETPOINT_GRID (+ the current setpoint) of dc_power (envelope_lines; the +-1 C per step limit is ignored).
             About 9x the cost of energy_curves to build.
Both only enlarge the feasible set, so the result is still a valid lower bound for policies that use them.

Solver: HiGHS interior point by default; if it stops with an error (not infeasibility), the solve is retried with the
dual simplex.

Variants (allowed starts):
    geo_time   any site, any step in [arrival, latest start]          the bound
    geo        any site, start on arrival                             value of moving work in space only
    time       origin only, any step in [arrival, latest start]       value of moving work in time only
    local      origin only, on arrival                                no flexibility (sanity check vs the simulator;
                                                                      bursts overflow here, so it reads optimistic)
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from baseline_policies import SETPOINT_GRID, dc_power
from geo_sustaindc_env import STEP_HOURS

VARIANTS = ('local', 'geo', 'time', 'geo_time')
TANGENT_LOADS = np.linspace(0.0, 1.0, 11)
SITE_FLEX = ('none', 'battery', 'cooling', 'battery+cooling')


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


FINE_LOADS = np.linspace(0.0, 1.0, 101)


def sampled_lines(f, x=FINE_LOADS, anchors=TANGENT_LOADS):
    """Lines z >= a + b u below the samples f(x) (x increasing): at each anchor load, the finite-difference slope of
    the samples, with the intercept lowered so the line is below every sample. dc_power has small kinks (it is not
    exactly convex in load), so tangents alone can rise above the curve near a kink; this lowering removes that.
    Returns (slopes, intercepts), each len(anchors)."""
    k = np.searchsorted(x, anchors).clip(1, len(x) - 2)
    b = (f[k + 1] - f[k - 1]) / (x[k + 1] - x[k - 1])
    a = np.min(f[None, :] - b[:, None] * x[None, :], axis=1)
    # Safety margin for the curvature between samples (measured excess <= 1e-4 of the energy on a 4x finer grid).
    return b, a - 5e-4 * float(np.max(f))


def _weather_key(amb, wb, step):
    if step is None:
        return round(float(amb), 2), round(float(wb), 2)
    return round(float(amb) / step) * step, round(float(wb) / step) * step


def energy_curves(geo, traces, loads=FINE_LOADS, weather_step=None):
    """Per site and step, lines below the IT + cooling energy curve (kWh per step) at the site's fixed setpoint, as
    tangent-style (values at TANGENT_LOADS, slopes): (N, T, K) each. Built from FINE_LOADS samples (sampled_lines).
    `loads` and `weather_step` (round the weather to that many C to share curves) trade accuracy for speed; the
    defaults are what keeps the bound valid."""
    N, T, K = geo.n_sites, len(traces[0]['ci']), len(TANGENT_LOADS)
    val, slope = np.zeros((N, T, K)), np.zeros((N, T, K))
    for i, (s, tr) in enumerate(zip(geo.sites, traces)):
        cache = {}
        for t in range(T):
            key = _weather_key(tr['amb'][t], tr['wb'][t], weather_step)
            if key not in cache:
                f = np.array([sum(dc_power(s, u, tr['stpt'], *key)) for u in loads]) / 1e3 * STEP_HOURS
                b, a = sampled_lines(f, loads)
                cache[key] = (a + b * TANGENT_LOADS, b)
            val[i, t], slope[i, t] = cache[key]
    return val, slope


def tangent_lines(curves):
    """Tangents of energy_curves() as lines z >= a + b u: (slopes, intercepts), each (N, T, K)."""
    val, slope = curves
    return slope, val - slope * TANGENT_LOADS[None, None, :]


def envelope_lines(geo, traces, setpoints=SETPOINT_GRID, loads=FINE_LOADS, weather_step=None):
    """Best-setpoint energy curve per site and step: lines below min over setpoints (+ the site's current setpoint) of
    IT + cooling energy (kWh per step), sampled on FINE_LOADS (sampled_lines). Including the current setpoint keeps it
    at or below energy_curves(). Cost: len(setpoints) x len(FINE_LOADS) dc_power calls per distinct weather.
    Returns (slopes, intercepts) each (N, T, K) and the best setpoint (N, T, K) at each of TANGENT_LOADS."""
    N, T, K = geo.n_sites, len(traces[0]['ci']), len(TANGENT_LOADS)
    sl, ic = np.zeros((N, T, K)), np.zeros((N, T, K))
    best = np.zeros((N, T, K))
    kx = np.abs(loads[:, None] - TANGENT_LOADS[None, :]).argmin(axis=0)
    for i, (s, tr) in enumerate(zip(geo.sites, traces)):
        grid = np.unique(np.r_[np.asarray(setpoints, dtype=float), tr['stpt']])
        cache = {}
        for t in range(T):
            key = _weather_key(tr['amb'][t], tr['wb'][t], weather_step)
            if key not in cache:
                e = np.array([[sum(dc_power(s, u, sp, *key)) for u in loads] for sp in grid]) / 1e3 * STEP_HOURS
                cache[key] = (*sampled_lines(e.min(axis=0), loads), grid[np.argmin(e[:, kx], axis=0)])
            sl[i, t], ic[i, t], best[i, t] = cache[key]
    return sl, ic, best


def battery_params(geo):
    """Per site (kWh and kWh per step): capacity, power, efficiencies, usable window and current stored energy."""
    out = []
    for s in geo.sites:
        b = s.bat_env.battery
        cap = b.capacity * 1e3
        out.append({'cap_kwh': cap, 'p_kwh': b.power_mw * 1e3 * STEP_HOURS, 'eta_c': b.eff_charge,
                    'eta_d': b.eff_discharge, 'lo_kwh': b.soc_min * cap, 'hi_kwh': b.soc_max * cap,
                    'soc0_kwh': b.current_load * 1e3})
    return out


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


def solve_bound(geo, variant='geo_time', late_steps=4, traces=None, curves=None, verbose=True, method='highs-ipm',
                site_flex='none', lines=None):
    """Solve the LP for one variant on geo's current episode (call right after geo.reset()). Returns a dict with
    'co2_kg' (the bound), 'late_job_mwh' (work started after its latest start), 'status' and timing."""
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    if site_flex not in SITE_FLEX:
        raise ValueError(f'site_flex must be one of {SITE_FLEX}')
    use_battery, use_cooling = 'battery' in site_flex, 'cooling' in site_flex
    t_start = time.time()
    N, T = geo.n_sites, geo.horizon
    traces = traces or exo_traces(geo, T)
    if lines is None:
        if use_cooling:
            lines = envelope_lines(geo, traces)[:2]
        else:
            lines = tangent_lines(curves if curves is not None else energy_curves(geo, traces))
    l_slope, l_icpt = lines  # energy z >= l_icpt + l_slope * u, each (N, T, L)
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
    n_cls = len(cls['arrival'])
    iu = nx + np.arange(N * T)            # u[i, t] -> iu[i*T + t]
    iz = nx + N * T + np.arange(N * T)
    iv = nx + 2 * N * T + np.arange(n_cls)  # overflow share of each class
    nvar = nx + 2 * N * T + n_cls
    if use_battery:  # grid charge, delivered discharge, stored energy at the end of each step (kWh), per (i, t)
        ich, idis, isoc = (nvar + k * N * T + np.arange(N * T) for k in range(3))
        nvar += 3 * N * T
        bat = battery_params(geo)

    # objective: CO2 (g) + lateness penalty (g-equivalent)
    penalty = 10.0 * float((ci[:, :, None] * l_slope).max())  # per unit-load-step started late
    c_obj = np.zeros(nvar)
    c_obj[iz] = ci.reshape(-1)
    late_w = cls['occ_prof'][xc, 0] / cap_w[xi]  # size of the class at its site
    c_obj[:nx] = penalty * xlate * late_w
    # Overflow: CO2 lower bound of the class (load-steps x min over allowed sites and steps of CI x zero-load slope),
    # plus a penalty above that of the latest allowed start.
    floor_cost = ci * l_slope.min(axis=2)                 # (N, T): g per unit load for one step, lowest slope
    floor_any = floor_cost.min(axis=1)                    # (N,)
    load_steps = cls['util_prof'].sum(1)[:, None] / cap_w[None, :]   # (classes, N)
    allowed = np.where(cls['origin'][:, None] < 0, True, np.arange(N)[None, :] == cls['origin'][:, None])
    co2_floor = np.where(allowed, load_steps * floor_any[None, :], np.inf).min(axis=1)
    # Above the latest allowed start's penalty and above the CO2 of running the class at any site and time (`penalty`
    # is 10x the largest CI x slope), so overflow never pays when a placement exists.
    size = (cls['occ_prof'][:, 0][:, None] / cap_w[None, :]).max(axis=1)
    over_pen = 10 * penalty * ((late_steps + 1) * size + load_steps.max(axis=1))
    c_obj[iv] = co2_floor + over_pen
    if use_battery:
        c_obj[ich] = ci.reshape(-1)
        c_obj[idis] = -ci.reshape(-1)

    rows_eq, cols_eq, vals_eq, b_eq = [], [], [], []
    # every class starts once
    rows_eq.append(np.concatenate([xc, np.arange(n_cls)]))
    cols_eq.append(np.concatenate([np.arange(nx), iv]))
    vals_eq.append(np.ones(nx + n_cls))
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
    n_eq = n_cls + N * T
    if use_battery:
        # soc[i,t] - soc[i,t-1] - eta_c ch[i,t] + dis[i,t] / eta_d = 0   (soc[i,-1] = current stored energy)
        eta_c = np.repeat([b['eta_c'] for b in bat], T)
        eta_d = np.repeat([b['eta_d'] for b in bat], T)
        r = n_eq + np.arange(N * T)
        not_first = np.tile(np.arange(T) > 0, N)
        rows_eq += [r, r, r, r[not_first]]
        cols_eq += [isoc, ich, idis, isoc[np.flatnonzero(not_first) - 1]]
        vals_eq += [np.ones(N * T), -eta_c, 1.0 / eta_d, -np.ones(int(not_first.sum()))]
        soc0 = np.zeros(N * T)
        soc0[np.arange(N) * T] = [b['soc0_kwh'] for b in bat]
        b_eq.append(soc0)
        n_eq += N * T
    A_eq = sp.csr_matrix((np.concatenate(vals_eq), (np.concatenate(rows_eq), np.concatenate(cols_eq))),
                         shape=(n_eq, nvar))
    b_eq = np.concatenate(b_eq)

    # inequalities: capacity, then energy lines  slope*u - z <= -intercept
    K = l_slope.shape[2]
    A_cap = sp.csr_matrix((np.concatenate(v_o), (np.concatenate(r_o), np.concatenate(c_o))), shape=(N * T, nvar))
    b_cap = np.full(N * T, geo.cap)
    rk = np.arange(N * T * K)
    it = np.repeat(np.arange(N * T), K)
    sl = l_slope.reshape(-1)
    A_tan = sp.csr_matrix((np.concatenate([sl, -np.ones(N * T * K)]),
                           (np.concatenate([rk, rk]), np.concatenate([iu[it], iz[it]]))), shape=(N * T * K, nvar))
    b_tan = -l_icpt.reshape(-1)
    blocks, rhs = [A_cap, A_tan], [b_cap, b_tan]
    if use_battery:  # no discharging beyond the DC's own energy (no export): dis - z <= 0
        r = np.arange(N * T)
        blocks.append(sp.csr_matrix((np.r_[np.ones(N * T), -np.ones(N * T)], (np.r_[r, r], np.r_[idis, iz])),
                                    shape=(N * T, nvar)))
        rhs.append(np.zeros(N * T))
    A_ub = sp.vstack(blocks).tocsr()
    b_ub = np.concatenate(rhs)

    bounds = np.zeros((nvar, 2))
    bounds[:, 1] = np.inf
    bounds[iu, 1] = 1.0
    if use_battery:
        bounds[ich, 1] = np.repeat([b['p_kwh'] for b in bat], T)
        bounds[idis, 1] = np.repeat([b['p_kwh'] for b in bat], T)
        bounds[isoc, 0] = np.repeat([b['lo_kwh'] for b in bat], T)
        bounds[isoc, 1] = np.repeat([b['hi_kwh'] for b in bat], T)
    t_build = time.time() - t_start
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    used = method
    if res.status not in (0, 2) and method != 'highs-ds':  # solver error (not infeasibility): retry with dual simplex
        if verbose:
            print(f'[lp {variant}] {method} failed ({res.message}); retrying with highs-ds')
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')
        used = 'highs-ds'
    out = {'variant': variant, 'site_flex': site_flex, 'status': res.status, 'message': res.message, 'n_vars': nvar,
           'n_classes': n_cls, 'method': used, 'build_s': t_build, 'solve_s': time.time() - t_start - t_build}
    if res.status != 0:
        if verbose:
            print(f'[lp {variant}] failed: {res.message}')
        out['co2_kg'] = float('nan')
        return out
    x = res.x
    grid = x[iz] + (x[ich] - x[idis] if use_battery else 0.0)
    out['co2_kg'] = float(ci.reshape(-1) @ grid + co2_floor @ x[iv]) / 1e3
    if use_battery:
        out['battery_discharge_mwh'] = float(x[idis].sum()) / 1e3
    out['overflow_co2_kg'] = float(co2_floor @ x[iv]) / 1e3
    job_mwh = cls['util_prof'].sum(1)[xc] * STEP_HOURS / 1e6  # MWh of the class
    out['late_job_mwh'] = float(np.sum(x[:nx] * job_mwh * (xlate > 0)))
    out['overflow_job_mwh'] = float(np.sum(x[iv] * cls['util_prof'].sum(1) * STEP_HOURS / 1e6))
    out['util'] = x[iu].reshape(N, T).T  # (T, N)
    if verbose:
        print(f"[lp {variant}] {out['co2_kg']:.1f} kg, {n_cls} classes, {nvar} vars, build {t_build:.0f}s, "
              f"solve {out['solve_s']:.0f}s ({used}), late {out['late_job_mwh']:.2f} MWh, "
              f"overflow {out['overflow_job_mwh']:.2f} MWh")
    return out
