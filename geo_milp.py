"""B2: exact job-level optimization for tiny GeoJobsDC instances (time-indexed MILP), and its replay in the simulator.

Unlike the fluid LP bound (geo_lp_oracle) and the index / MPC LPs, nothing is relaxed:
    y[j, i, s] in {0, 1}     job j starts at site i at step s, s in [arrival, latest start + late_steps]
                             (rigid jobs: origin only); sum y + overflow_j = 1
    load      u[i, tau] = sum y x size x (fraction of step tau the job runs);  occupancy (whole steps) <= cap
    cooling   'exact': one-hot setpoint sigma[i, tau, p] over SETPOINT_GRID, |setpoint change| <= 1 C per step from
              the current setpoint, energy z >= lines of that setpoint's curve (convex-hull formulation: load and
              energy split per setpoint, u_p <= sigma_p, z_p >= a sigma_p + b u_p); 'envelope': best setpoint
              each step, no ramp limit (an LP relaxation of 'exact'); 'fixed': the current setpoint
    battery   continuous, as in the LP bound (charge, discharge <= z, usable SoC window, efficiencies)
    objective sum CI x (z + charge - discharge) + lateness / overflow penalties (never cheaper than placing a job)

Scenarios (`scenarios` > 0): a two-stage stochastic program. Routing and start times (y) are decided once, at the start
of the episode, from the nominal CI forecast; each scenario (CI drawn in the forecast box, sufficiency/forecast.py)
gets its own battery and cooling decisions; the objective is the scenario mean. With scenarios = 0, the true CI
(perfect information): the exact job-level hindsight optimum, which the fluid LP bound lower-bounds.

Tiny instances only: binaries ~ jobs x sites x start slots + sites x steps x setpoints. Use small sites
(`tiny_geo`: e.g. 20 kW of IT, so a few hundred jobs per day) and few sites; solve time grows fast with both, which
is the point of reporting it.

    geo = tiny_geo(['US-CAL-CISO', 'DE-LU'], cap_mw=0.02, days=1)
    sol = solve_job_milp(geo, seed=1, cooling='exact', time_limit=600)
    m = run_index_episode(geo, MILPReplay(sol), seed=1)      # simulated CO2 of the schedule
"""
import random
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from baseline_policies import SETPOINT_GRID, dc_power
from geo_lp_oracle import _weather_key, battery_params, exo_traces, sampled_lines
from geo_sustaindc_env import STEP_HOURS

LOADS = np.linspace(0.0, 1.0, 21)


def tiny_geo(regions, cap_mw=0.02, days=1, util=0.5, month=6, year=2023, sla_multiplier=1.5):
    from geo_jobs import GeoJobsDC
    sites = [{'location': f'dcc:{r}:{year}', 'datacenter_capacity_mw': cap_mw, 'max_bat_cap_Mw': 1.2 * cap_mw}
             for r in regions]
    return GeoJobsDC({'sites': sites, 'month': month, 'days_per_episode': days, 'target_util': util,
                      'sla_multiplier': sla_multiplier, 'routing': 'site_queue'})


def setpoint_lines(geo, traces, grid, weather_step=0.5):
    """Per site, step and setpoint: lines z >= a + b u below the IT + cooling energy (kWh per step). (N, T, P, L) x 2."""
    N, T, P = geo.n_sites, len(traces[0]['ci']), len(grid)
    sl = ic = None
    for i, (s, tr) in enumerate(zip(geo.sites, traces)):
        cache = {}
        for t in range(T):
            key = _weather_key(tr['amb'][t], tr['wb'][t], weather_step)
            if key not in cache:
                lines = [sampled_lines(np.array([sum(dc_power(s, u, p, *key)) for u in LOADS]) / 1e3 * STEP_HOURS, LOADS)
                         for p in grid]
                cache[key] = (np.array([b for b, _ in lines]), np.array([a for _, a in lines]))
            b, a = cache[key]
            if sl is None:
                sl, ic = np.zeros((N, T, P, b.shape[1])), np.zeros((N, T, P, b.shape[1]))
            sl[i, t], ic[i, t] = b, a
    return sl, ic


class _Builder:
    def __init__(self):
        self.n, self.lb, self.ub, self.integ, self.c = 0, [], [], [], []
        self.rows, self.cols, self.vals, self.rlo, self.rhi = [], [], [], [], []
        self.nrow = 0

    def var(self, k, lb=0.0, ub=np.inf, integer=False, cost=0.0):
        idx = self.n + np.arange(k)
        self.n += k
        self.lb.append(np.broadcast_to(lb, (k,)).astype(float))
        self.ub.append(np.broadcast_to(ub, (k,)).astype(float))
        self.integ.append(np.full(k, 1 if integer else 0))
        self.c.append(np.broadcast_to(cost, (k,)).astype(float))
        return idx

    def row(self, cols, vals, lo, hi):
        self.rows.append(np.full(len(cols), self.nrow))
        self.cols.append(np.asarray(cols))
        self.vals.append(np.asarray(vals, dtype=float))
        self.rlo.append(lo)
        self.rhi.append(hi)
        self.nrow += 1

    def rows_block(self, r, c, v, lo, hi):
        """Several rows at once: r = local row ids (0..m-1), lo / hi per local row."""
        m = len(lo)
        self.rows.append(np.asarray(r) + self.nrow)
        self.cols.append(np.asarray(c))
        self.vals.append(np.asarray(v, dtype=float))
        self.rlo += list(lo)
        self.rhi += list(hi)
        self.nrow += m

    def set_cost(self, idx, cost):
        cvec = np.concatenate(self.c)
        cvec[idx] += cost
        self.c = [cvec]

    def solve(self, time_limit, gap):
        A = sp.csr_matrix((np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
                          shape=(self.nrow, self.n))
        opts = {'disp': False, 'mip_rel_gap': gap}
        if time_limit:
            opts['time_limit'] = float(time_limit)
        return milp(np.concatenate(self.c), integrality=np.concatenate(self.integ),
                    bounds=Bounds(np.concatenate(self.lb), np.concatenate(self.ub)),
                    constraints=LinearConstraint(A, np.array(self.rlo), np.array(self.rhi)), options=opts)


def solve_job_milp(geo, seed, cooling='exact', scenarios=0, sigma24=0.1, late_steps=4, time_limit=600, gap=1e-3,
                   relax=False, fc_seed=0):
    """Solve the job-level problem on geo's episode for `seed`. relax=True drops integrality (its LP relaxation).
    Returns a dict: status, objective (g, the solver's) and co2_model_kg (true CI, model energy), mip_gap, solve_s,
    sizes, and the schedule: site / start per job, per site battery and setpoint plans (scenario 0)."""
    from sufficiency.forecast import ForecastModel
    random.seed(seed)
    np.random.seed(seed)
    geo.reset()
    N, T, J = geo.n_sites, geo.horizon, geo.n_jobs
    traces = exo_traces(geo, T)
    ci_true = np.stack([tr['ci'] for tr in traces])
    stpt0 = np.array([float(s.dc_env.raw_curr_stpt) for s in geo.sites])
    if cooling == 'exact':
        grid = np.array(SETPOINT_GRID)
    elif cooling in ('envelope', 'fixed'):
        grid = np.array(SETPOINT_GRID) if cooling == 'envelope' else None
    else:
        raise ValueError(cooling)
    if cooling == 'fixed':
        sl, ic = setpoint_lines(geo, traces, [stpt0[0]])
    else:
        sl, ic = setpoint_lines(geo, traces, grid)
    P = sl.shape[2]
    if cooling == 'envelope':      # best setpoint each step: lines of the pointwise min, no binaries
        e = np.stack([ic[..., p, :][..., None, :] + sl[..., p, :][..., None, :] * LOADS[None, None, :, None]
                      for p in range(P)]).max(axis=-1).min(axis=0)       # (N, T, loads)
        lines = [[sampled_lines(e[i, t], LOADS) for t in range(T)] for i in range(N)]
        sl = np.array([[l[0] for l in row] for row in lines])[:, :, None, :]
        ic = np.array([[l[1] for l in row] for row in lines])[:, :, None, :]
        P = 1
    if scenarios:
        fc = ForecastModel(ci_true, [c['location'] for c in geo.site_configs], sigma24=sigma24, seed=fc_seed)
        rng = np.random.default_rng([fc_seed, 7])
        cis = [np.stack([fc.window(i, 0, T, 'nominal') for i in range(N)])]
        for _ in range(scenarios - 1):
            draw = fc.sample(rng, range(N), 0, T)
            cis.append(np.stack([draw[i] for i in range(N)]))
    else:
        cis = [ci_true]
    K = len(cis)
    bat = battery_params(geo)

    t0 = time.time()
    m = _Builder()
    # job start variables
    span = int(np.ceil(geo.dur.max()))
    triples = []
    for j in range(J):
        a = int(geo.arrival[j])
        last = int(np.floor(geo.latest_start[j] + 1e-9))
        sites = [int(geo.origin[j])] if geo.rigid[j] else range(N)
        for i in sites:
            for s in range(a, min(last + late_steps, T - 1) + 1):
                triples.append((j, i, s, max(s - last, 0)))
    tr = np.array(triples, dtype=np.int64).reshape(-1, 4)
    yj, yi, ys, ylate = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
    size = geo.size[yj, yi]
    penalty = 10.0 * float(max(c.max() for c in cis)) * float(sl.max()) * (span + 1)
    iy = m.var(len(tr), 0.0, 1.0, integer=not relax, cost=penalty * ylate * size)
    io = m.var(J, 0.0, 1.0, cost=10 * penalty * ((late_steps + 1) + geo.dur) * np.maximum(geo.power_w / geo.cap_w.min(), 1e-3))
    # each job: sum y + overflow = 1
    m.rows_block(np.r_[yj, np.arange(J)], np.r_[iy, io], np.ones(len(tr) + J), np.ones(J), np.ones(J))
    # load and occupancy per site and step
    iu = m.var(N * T, 0.0, 1.0).reshape(N, T)
    ru, cu, vu, ro, co, vo = [], [], [], [], [], []
    for o in range(span + 1):
        tt = ys + o
        ok = tt < T
        util = size * np.clip(geo.dur[yj] - o, 0.0, 1.0)
        occ = size * (o < np.ceil(geo.dur[yj] - 1e-9))
        keep = ok & (util > 0)
        ru.append(yi[keep] * T + tt[keep]); cu.append(iy[keep]); vu.append(-util[keep])
        keep = ok & (occ > 0)
        ro.append(yi[keep] * T + tt[keep]); co.append(iy[keep]); vo.append(occ[keep])
    m.rows_block(np.r_[np.arange(N * T), np.concatenate(ru)], np.r_[iu.reshape(-1), np.concatenate(cu)],
                 np.r_[np.ones(N * T), np.concatenate(vu)], np.zeros(N * T), np.zeros(N * T))
    m.rows_block(np.concatenate(ro), np.concatenate(co), np.concatenate(vo), np.full(N * T, -np.inf),
                 np.full(N * T, geo.cap))
    # per scenario: energy, cooling choice, battery
    scen = []
    for k, ci in enumerate(cis):
        w = 1.0 / K
        iz = m.var(N * T, 0.0, np.inf, cost=w * ci.reshape(-1)).reshape(N, T)
        ich = m.var(N * T, 0.0, np.repeat([b['p_kwh'] for b in bat], T), cost=w * ci.reshape(-1)).reshape(N, T)
        idis = m.var(N * T, 0.0, np.repeat([b['p_kwh'] for b in bat], T), cost=-w * ci.reshape(-1)).reshape(N, T)
        isoc = m.var(N * T, np.repeat([b['lo_kwh'] for b in bat], T), np.repeat([b['hi_kwh'] for b in bat], T)).reshape(N, T)
        isg = None
        if cooling == 'exact':     # convex hull of the setpoint disjunction: u = sum_p u_p, u_p <= sigma_p, z = sum_p z_p
            ub = np.ones((N, T, P))
            ub[:, 0, :] = (np.abs(grid[None, :] - stpt0[:, None]) <= 1 + 1e-9)
            isg = m.var(N * T * P, 0.0, ub.reshape(-1), integer=not relax).reshape(N, T, P)
            iup = m.var(N * T * P, 0.0, 1.0).reshape(N, T, P)
            izp = m.var(N * T * P, 0.0, np.inf).reshape(N, T, P)
        L = sl.shape[3]
        for i in range(N):
            for t in range(T):
                for p in range(P):
                    for l in range(L):
                        a, b = ic[i, t, p, l], sl[i, t, p, l]
                        if isg is None:      # b u - z <= -a
                            m.row([iu[i, t], iz[i, t]], [b, -1.0], -np.inf, -a)
                        else:                # b u_p - z_p + a sigma_p <= 0
                            m.row([iup[i, t, p], izp[i, t, p], isg[i, t, p]], [b, -1.0, a], -np.inf, 0.0)
                    if isg is not None:
                        m.row([iup[i, t, p], isg[i, t, p]], [1.0, -1.0], -np.inf, 0.0)
                if isg is not None:
                    m.row(isg[i, t], np.ones(P), 1.0, 1.0)
                    m.row(np.r_[iu[i, t], iup[i, t]], np.r_[1.0, -np.ones(P)], 0.0, 0.0)
                    m.row(np.r_[iz[i, t], izp[i, t]], np.r_[1.0, -np.ones(P)], 0.0, 0.0)
                    if t > 0:
                        for q in range(P):
                            adj = np.flatnonzero(np.abs(grid - grid[q]) <= 1 + 1e-9)
                            m.row(np.r_[isg[i, t, q], isg[i, t - 1, adj]], np.r_[1.0, -np.ones(len(adj))], -np.inf, 0.0)
                b_ = bat[i]
                cols = [isoc[i, t], ich[i, t], idis[i, t]] + ([isoc[i, t - 1]] if t else [])
                vals = [1.0, -b_['eta_c'], 1.0 / b_['eta_d']] + ([-1.0] if t else [])
                rhs = 0.0 if t else b_['soc0_kwh']
                m.row(cols, vals, rhs, rhs)
                m.row([idis[i, t], iz[i, t]], [1.0, -1.0], -np.inf, 0.0)
        scen.append((iz, ich, idis, isg))
    build_s = time.time() - t0

    res = m.solve(time_limit, gap)
    out = {'status': int(res.status), 'message': res.message, 'build_s': build_s, 'solve_s': time.time() - t0 - build_s,
           'n_vars': m.n, 'n_binaries': int(np.concatenate(m.integ).sum()), 'n_rows': m.nrow, 'n_jobs': J,
           'cooling': cooling, 'scenarios': K if scenarios else 0, 'relaxed': relax,
           'mip_gap': float(getattr(res, 'mip_gap', np.nan) or 0.0) if res.x is not None else np.nan}
    if res.x is None:
        return out
    x = res.x
    out['objective_g'] = float(res.fun)
    pick = x[iy] > 0.5 if not relax else x[iy] > 1e-6
    site, start = np.full(J, -1), np.full(J, -1)
    best = {}
    for k in np.flatnonzero(pick):
        if x[iy[k]] > best.get(yj[k], 0):
            best[yj[k]] = x[iy[k]]
            site[yj[k]], start[yj[k]] = yi[k], ys[k]
    iz, ich, idis, isg = scen[0]
    g = x[iz] + x[ich] - x[idis]
    out.update({'site': site, 'start': start, 'u': x[iu], 'ch': x[ich], 'dis': x[idis],
                'p_kwh': np.array([b['p_kwh'] for b in bat]), 'overflow': float(x[io].sum()),
                'co2_model_kg': float(np.sum(ci_true * g)) / 1e3})
    if isg is not None:
        out['setpoint'] = grid[np.argmax(x[isg], axis=2)]
    return out


class MILPReplay:
    """Replays a solve_job_milp schedule in the simulator: routes each job to its planned site, starts it at its
    planned step (if it fits; else the env starts it when due), and follows the planned setpoint and battery. The
    simulator's battery has three actions (full-power charge, discharge up to the load, idle), so the planned flows
    are rounded in time: the accumulated planned net charge is tracked against what the battery executed; charge
    when it is half a step's power ahead, discharge when it is behind by 10% of a step's power."""

    def __init__(self, sol, battery=True, setpoint=True):
        self.sol, self.battery, self.setpoint = sol, battery, setpoint

    def reset(self, geo):
        self.stats = {'lp_solves': 0, 'lp_seconds': 0.0, 'replans': 0, 'curve_violations': 0, 'decision_seconds': []}
        self.err = np.zeros(geo.n_sites)   # planned - executed net charge (kWh), for sigma-delta battery rounding

    def act(self, geo):
        from baseline_policies import setpoint_action
        t0 = time.time()
        sol, t = self.sol, geo.t
        arr = geo.arrivals()
        geo.route(np.where(sol['site'][arr] >= 0, sol['site'][arr], -1))
        starts = []
        for i in range(geo.n_sites):
            q = geo.queue(i)
            starts.append(q[(sol['start'][q] >= 0) & (sol['start'][q] <= t)].tolist())
        acts = []
        for i, s in enumerate(geo.sites):
            a = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
            if t < sol['ch'].shape[1]:
                if self.battery:     # track the accumulated planned net charge with what the battery really did
                    p = sol['p_kwh'][i]
                    b = s.bat_env
                    if t > 0:          # what the last step's action executed (kWh): charging draw, delivered energy
                        done = (b.charge_draw * 1e3 if b.bat_mode in ('charge', 'pv_charge') else 0.0) \
                            - (b.discharge_energy * 1e3 if b.bat_mode == 'discharge' else 0.0)
                        self.err[i] -= done
                    self.err[i] += sol['ch'][i, t] - sol['dis'][i, t]
                    if self.err[i] >= 0.5 * p:          # charging runs at full power: wait for half a step's worth
                        a['agent_bat'] = 0
                    elif self.err[i] <= -0.1 * p:       # discharging is capped by the load: start early
                        a['agent_bat'] = 1
                if self.setpoint and 'setpoint' in sol:
                    a['agent_dc'] = setpoint_action(float(s.dc_env.raw_curr_stpt), float(sol['setpoint'][i, t]))
            acts.append(a)
        self.stats['decision_seconds'].append(time.time() - t0)
        return {'route': None, 'starts': starts, 'sites': acts}
