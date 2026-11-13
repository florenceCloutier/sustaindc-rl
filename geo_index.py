"""Lagrangian (Whittle-type) index policy for GeoJobsDC in site_queue mode (geo_jobs.py).

Each site is an arm with its own controlled state: queued jobs, running jobs (committed load), battery state of
charge and CRAC setpoint. The only coupling between sites is routing: every arriving job goes to exactly one site's
queue. Dualizing that constraint with a price per unit of work splits the problem into one small problem per site:

  site LP  SP_i(phi)  (one site, rolling window of H steps; same model as geo_lp_oracle's bound, restricted to it)
      min  sum_tau CI[tau] * (z[tau] + ch[tau] - dis[tau])  + lateness / overflow penalties
      s.t. queued jobs (grouped by latest start) start within their slack (fluid, starts on a 15-min grid for the
           first hour, hourly after); running jobs fixed; load u <= cap; energy z >= lines(u) (IT + cooling, at the
           site's best setpoint with cooling='envelope'); battery: soc' = soc + eff_c ch - dis / eff_d,
           usable window, power limit, dis <= z
           + a fraction phi of a representative bundle of arriving jobs (also within their slack)

  index    nu_i,b(phi) = d SP_i / d (work)  for bundle b: the marginal CO2 (g per W x step of job power) of taking
           more work of that kind at site i, given everything the site can do with it (wait for a cleaner hour,
           cover it from the battery, run cooler). SP_i is convex in phi (an LP's value in its right-hand side), so
           the index is non-decreasing in the amount accepted: the multi-action indexability condition.

Every hour (t % 4 == 0), per site: solve the plan (phi = 0) and, for each slack bundle b, the cost curve at G levels
of phi; its slopes are the index. Bundles: the arrivals of the past hour, split by slack (they stand in for the
coming hour's arrivals; no look-ahead at the arrival process).

Routing (every step, as jobs arrive): each arriving job of bundle b goes to the site with the lowest index at the
amount of bundle-b work it already accepted this hour (streaming water-filling: the clearing price lambda*_b is the
index of the last site filled). Rigid jobs go to their origin. Before the first curves exist, or for an empty bundle,
jobs are routed by the current marginal CO2 (`merit`).

Local execution at each site: the plan (phi = 0) is re-solved every `plan_every` steps (default every step: one small
LP per site, so jobs routed since the last plan are included; slacks are short, often under an hour). Start the queued
jobs the plan starts in this step (per job class, earliest deadline first, up to the planned amount); charge /
discharge / idle the battery as planned; move the setpoint toward the plan's best setpoint (+-1 C per step) with
cooling='envelope'. Jobs not in the current plan start when due (the env forces due jobs at their own site).

Fluid relaxation: the plan treats each job class as divisible; execution rounds it to whole jobs.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from baseline_policies import setpoint_action
from geo_job_baselines import marginal_table
from geo_lp_oracle import TANGENT_LOADS, battery_params, energy_curves, envelope_lines, exo_traces, tangent_lines

STEP_GRID_FIRST_HOUR = 4     # 15-min start slots in the first hour, hourly after
LOADS_FAST = np.linspace(0.0, 1.0, 21)


def _start_slots(last, H):
    """Allowed start offsets (relative to now) up to `last` (inclusive), within the first H steps."""
    slots = list(range(min(STEP_GRID_FIRST_HOUR, H)))
    slots += list(range(STEP_GRID_FIRST_HOUR, H, 4))
    return np.array([s for s in slots if s <= last] or [0])


def _profiles(size, dur, span):
    """Per-offset utilization (fraction of the step run) and occupancy (full steps) of jobs summed: (span,) each."""
    o = np.arange(span)[None, :]
    util = (size[:, None] * np.clip(dur[:, None] - o, 0.0, 1.0)).sum(0)
    occ = (size[:, None] * (o < np.ceil(dur[:, None] - 1e-9))).sum(0)
    return util, occ


class SiteLP:
    """One site's planning LP at the current step (see the module docstring). Build once per (site, step) from the
    site's state; solve() for any amount phi of the extra bundle(s) of jobs and, optionally, another CI forecast.
    `bundle`: None, one dict {'jobs', 'slack'} or a list of them (each gets its own phi: bundle id = list position)."""

    def __init__(self, geo, i, t, H, ci, lines, cap, late_steps=4, bundle=None):
        self.i, self.t = i, t
        span = int(np.ceil(geo.trace['max_dur_steps'])) + 1
        W = int(min(H + span, len(ci) - t))
        H = int(min(H, W))
        self.W, self.H = W, H
        size_i = geo.size[:, i]

        # committed load of running jobs over the window
        run = geo.running(i)
        tau = t + np.arange(W)
        self.R_util = (size_i[run, None] * np.clip(geo.end_step[run, None] - tau[None, :], 0.0, 1.0)).sum(0)
        self.R_occ = (size_i[run, None] * (tau[None, :] < np.ceil(geo.end_step[run, None] - 1e-9))).sum(0)

        # queued jobs, grouped by latest start (relative), EDF inside a class
        q = geo.queue(i)
        last = np.floor(geo.latest_start[q] - t + 1e-9).astype(int)
        self.classes = []  # dicts: jobs, last, util, occ, work (load-steps), kind ('queue' or 'bundle')
        for L in np.unique(last):
            jobs = q[last == L]
            u, o = _profiles(size_i[jobs], geo.dur[jobs], span)
            self.classes.append({'jobs': jobs, 'last': int(L), 'util': u, 'occ': o, 'kind': 'queue'})
        # representative bundle(s) (a fraction phi of each must start)
        bundles = [] if bundle is None else [bundle] if isinstance(bundle, dict) else list(bundle)
        for bid, bun in enumerate(bundles):
            if not len(bun['jobs']):
                continue
            bj = bun['jobs']
            slack = np.floor(bun['slack']).astype(int)
            for L in np.unique(slack):
                jobs = bj[slack == L]
                u, o = _profiles(size_i[jobs], geo.dur[jobs], span)
                self.classes.append({'jobs': jobs, 'last': int(L), 'util': u, 'occ': o, 'kind': 'bundle', 'bid': bid})

        self.ci = np.asarray(ci[t:t + W], dtype=float)
        l_slope, l_icpt = lines[0][t:t + W], lines[1][t:t + W]
        self.l_slope = l_slope
        bat = battery_params(geo)[i]
        self.bat = bat
        self._build(span, cap, late_steps, l_slope, l_icpt, bat)

    def _build(self, span, cap, late_steps, l_slope, l_icpt, bat):
        W, H, C = self.W, self.H, len(self.classes)
        xc, xs, xlate = [], [], []
        for c, cl in enumerate(self.classes):
            slots = _start_slots(max(cl['last'], 0) + late_steps, H)
            xc.append(np.full(len(slots), c))
            xs.append(slots)
            xlate.append(np.maximum(slots - max(cl['last'], 0), 0) * (slots > cl['last']))
        xc = np.concatenate(xc) if C else np.zeros(0, int)
        xs = np.concatenate(xs) if C else np.zeros(0, int)
        xlate = np.concatenate(xlate) if C else np.zeros(0)
        nx = len(xc)
        iu, iz = nx + np.arange(W), nx + W + np.arange(W)
        ich, idis, isoc = (nx + (2 + k) * W + np.arange(W) for k in range(3))
        iv = nx + 5 * W + np.arange(C)
        nvar = nx + 5 * W + C
        self.xc, self.xs, self.iu, self.iz, self.ich, self.idis, self.iv = xc, xs, iu, iz, ich, idis, iv

        # Lateness must never pay: per step late, more than the CO2 of running the longest job at the dirtiest hour.
        penalty = 10.0 * float((self.ci[:, None] * l_slope).max()) * (span + 1)
        size0 = np.array([cl['occ'][0] for cl in self.classes]) if C else np.zeros(0)
        work = np.array([cl['util'].sum() for cl in self.classes]) if C else np.zeros(0)
        c_obj = np.zeros(nvar)
        c_obj[iz], c_obj[ich], c_obj[idis] = self.ci, self.ci, -self.ci
        if C:
            c_obj[:nx] = penalty * xlate * size0[xc]
            c_obj[iv] = 10 * penalty * ((late_steps + 1) * size0 + work)  # overflow: never cheaper than placing
        self.c_obj = c_obj

        rows, cols, vals = [], [], []
        # every class starts (share 1 for queued jobs, phi for the bundle): rows 0..C-1, rhs set in solve()
        rows += [xc, np.arange(C)]
        cols += [np.arange(nx), iv]
        vals += [np.ones(nx), np.ones(C)]
        # u[tau] = R_util[tau] + sum x * util[tau - s]   (rows C..C+W-1);  occupancy <= cap (inequality block)
        ru, cu, vu, ro, co, vo = [], [], [], [], [], []
        for o in range(span):
            tt = xs + o
            ok = tt < W
            if not ok.any():
                continue
            uv = np.array([self.classes[c]['util'][o] for c in xc[ok]])
            ov = np.array([self.classes[c]['occ'][o] for c in xc[ok]])
            ru.append(tt[ok]); cu.append(np.flatnonzero(ok)); vu.append(-uv)
            ro.append(tt[ok]); co.append(np.flatnonzero(ok)); vo.append(ov)
        rows += [C + np.arange(W)] + [C + r for r in ru]
        cols += [iu] + cu
        vals += [np.ones(W)] + vu
        # battery: soc[t] - soc[t-1] - eta_c ch[t] + dis[t] / eta_d = 0   (rows C+W..C+2W-1)
        r = C + W + np.arange(W)
        rows += [r, r, r, r[1:]]
        cols += [isoc, ich, idis, isoc[:-1]]
        vals += [np.ones(W), np.full(W, -bat['eta_c']), np.full(W, 1.0 / bat['eta_d']), -np.ones(W - 1)]
        self.n_eq = C + 2 * W
        self.A_eq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(self.n_eq, nvar))
        b_eq = np.zeros(self.n_eq)
        b_eq[C:C + W] = self.R_util
        b_eq[C + W] = bat['soc0_kwh']
        self.b_eq = b_eq

        # inequalities: occupancy <= cap - R_occ;  energy lines: slope*u - z <= -icpt;  dis - z <= 0
        L = l_slope.shape[1]
        blocks = [sp.csr_matrix((np.concatenate(vo) if vo else np.zeros(0),
                                 (np.concatenate(ro) if ro else np.zeros(0, int),
                                  np.concatenate(co) if co else np.zeros(0, int))), shape=(W, nvar))]
        rhs = [np.maximum(cap - self.R_occ, 0.0)]
        rk = np.arange(W * L)
        it = np.repeat(np.arange(W), L)
        blocks.append(sp.csr_matrix((np.r_[l_slope.reshape(-1), -np.ones(W * L)], (np.r_[rk, rk], np.r_[iu[it], iz[it]])),
                                    shape=(W * L, nvar)))
        rhs.append(-l_icpt.reshape(-1))
        rw = np.arange(W)
        blocks.append(sp.csr_matrix((np.r_[np.ones(W), -np.ones(W)], (np.r_[rw, rw], np.r_[idis, iz])), shape=(W, nvar)))
        rhs.append(np.zeros(W))
        self.A_ub = sp.vstack(blocks).tocsr()
        self.b_ub = np.concatenate(rhs)

        bounds = np.zeros((nvar, 2))
        bounds[:, 1] = np.inf
        bounds[iu, 1] = 1.0
        bounds[ich, 1] = bounds[idis, 1] = bat['p_kwh']
        bounds[isoc, 0], bounds[isoc, 1] = bat['lo_kwh'], bat['hi_kwh']
        self.bounds = bounds
        self.is_bundle = np.array([cl['kind'] == 'bundle' for cl in self.classes], dtype=bool)
        self.bid = np.array([cl.get('bid', -1) for cl in self.classes], dtype=np.int64)

    def objective(self, ci=None):
        """Objective vector for a CI forecast over the window (W,) (None: the one the LP was built with). The lateness
        and overflow penalties stay those of the build-time CI."""
        if ci is None:
            return self.c_obj
        ci = np.asarray(ci, dtype=float)[:self.W]
        c = self.c_obj.copy()
        c[self.iz], c[self.ich], c[self.idis] = ci, ci, -ci
        return c

    def rhs(self, phi):
        """Equality right-hand side for bundle share(s) phi: a number (every bundle) or {bundle id: share}."""
        b_eq = self.b_eq.copy()
        C = len(self.classes)
        if isinstance(phi, dict):
            share = np.array([phi.get(int(b), 0.0) for b in self.bid])
        else:
            share = np.full(C, float(phi))
        b_eq[:C] = np.where(self.is_bundle, share, 1.0)
        return b_eq

    def solve(self, phi=0.0, ci=None):
        """Optimal value (g, with penalties) and solution when a share phi of the bundle(s) must be placed."""
        args = dict(A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.rhs(phi), bounds=self.bounds)
        c = self.objective(ci)
        for method in ('highs', 'highs-ds', 'highs-ipm'):   # always feasible (overflow); failures are numerical
            res = linprog(c, method=method, **args)
            if res.status == 0:
                break
        if res.status != 0:
            return np.inf, None
        return float(res.fun), res.x

    def grid_energy(self, x):
        """Grid energy (kWh) per step of the window in solution x: IT + cooling + charge - discharge."""
        return x[self.iz] + x[self.ich] - x[self.idis]


class IndexPolicy:
    """Index policy for GeoJobsDC(routing='site_queue'). act(geo) -> (route, starts, site actions); see the module
    docstring. Options:
        window_h       planning window of the site LPs (hours)
        slack_edges    bundle boundaries on the job slack, in steps (default: < 1 h, 1-2 h, > 2 h)
        levels         number of phi levels per cost curve (G)
        cooling        'envelope' (best setpoint, executed by moving the setpoint) or 'fixed' (current setpoint)
        execute_local  True: battery and setpoint follow the site plan; False: they are left to the caller
        plan_every     re-solve each site's plan every this many steps (cost curves are always hourly)
        cap_margin     plan to cap - cap_margin (fluid plans fill capacity exactly; whole jobs then overflow it)
        reserve_steps  start a planned job early only if the queued jobs due within this many steps still fit at
                       their latest start (0: off, as before)
        admission      'plan': a job may go to a site where it fits at some start before its latest start, given the
                       site's plan; 'latest': only where it fits at its latest start with every job already queued
                       there also placed at its latest start (a feasible schedule always exists; use with
                       reserve_steps so that early starts keep it feasible)
        bat_threshold  charge / discharge (at full power, the simulator's battery actions) when this step's planned
                       charge / discharge is at least this share of the battery's power. Plans spread discharges
                       thinly over many steps; with a high threshold they are dropped and the battery only charges.
    """

    def __init__(self, window_h=24, slack_edges=(4, 8), levels=4, cooling='envelope', execute_local=True,
                 plan_every=1, cap_margin=0.0, reserve_steps=0, execute_battery=True, bat_threshold=0.1,
                 admission='plan'):
        self.plan_every = int(plan_every)
        self.cap_margin = float(cap_margin)
        self.reserve_steps = int(reserve_steps)
        self.execute_battery = bool(execute_battery)
        self.bat_threshold = float(bat_threshold)
        if admission not in ('plan', 'latest'):
            raise ValueError("admission must be 'plan' or 'latest'")
        self.admission = admission
        self.H = int(window_h * 4)
        self.edges = np.asarray(slack_edges, dtype=float)
        self.G = int(levels)
        self.cooling = cooling
        self.execute_local = execute_local

    # -----------------------------------------------------------------------------------------------------------------
    def reset(self, geo):
        assert geo.routing == 'site_queue', 'IndexPolicy needs GeoJobsDC(routing="site_queue")'
        T = geo.horizon
        traces = exo_traces(geo, T)
        self.ci = np.stack([tr['ci'] for tr in traces])  # (N, T)
        if self.cooling == 'envelope':
            sl, ic, self.best_stpt = envelope_lines(geo, traces, loads=LOADS_FAST, weather_step=0.5)
        else:
            sl, ic = tangent_lines(energy_curves(geo, traces, loads=LOADS_FAST, weather_step=0.5))
            self.best_stpt = None
        self.lines = (sl, ic)
        _, _, self.A = marginal_table(geo, T + 2)        # fallback routing score (merit)
        self.plans = [None] * geo.n_sites
        self.curves = {}                                   # (site, bundle) -> (phis, nus, work_total)
        self.acc = np.zeros((geo.n_sites, len(self.edges) + 1))
        self.started = {}                                  # class plan bookkeeping
        self.stats = {'lp_solves': 0, 'lp_seconds': 0.0, 'replans': 0, 'decision_seconds': [], 'curve_violations': 0}

    def _bundle_of(self, slack):
        return np.searchsorted(self.edges, slack, side='right')

    def _site_lp(self, geo, i, bundle=None):
        return SiteLP(geo, i, geo.t, self.H, self.ci[i], (self.lines[0][i], self.lines[1][i]), geo.cap - self.cap_margin,
                      bundle=bundle)

    def _ci_for(self, geo, i, lp):
        """CI over lp's window that site i plans with (None: the true CI; ScreenedIndexPolicy uses forecasts)."""
        return None

    def _solve(self, lp, phi, ci=None):
        t0 = time.time()
        out = lp.solve(phi, ci)
        self.stats['lp_solves'] += 1
        self.stats['lp_seconds'] += time.time() - t0
        return out

    def _bundles(self, geo):
        """The past hour's non-rigid arrivals, split by slack: {bundle: {'jobs', 'slack', 'work' (W x steps)}}."""
        t = geo.t
        lo, hi = np.searchsorted(geo.arrival, [t - 4, t])         # arrivals are sorted
        recent = np.arange(lo, hi)
        recent = recent[~geo.rigid[recent]]
        slack_all = geo.latest_start - geo.arrival
        groups = {}
        for b in range(len(self.edges) + 1):
            jobs = recent[self._bundle_of(slack_all[recent]) == b]
            if len(jobs):
                groups[b] = {'jobs': jobs, 'slack': slack_all[jobs], 'work': float(np.sum(geo.power_w[jobs] * geo.dur[jobs]))}
        return groups

    @staticmethod
    def _phi_max(geo, lp, g):
        """Largest share of bundle g site lp could take: free capacity over the steps the bundle can use (its longest
        slack + its longest job) vs the bundle's work here; queued jobs due in that span use part of it."""
        reach = min(int(np.ceil(g['slack'].max() + geo.dur[g['jobs']].max())) + 1, lp.W)
        free = float(np.sum(np.maximum(geo.cap - lp.R_occ[:reach], 0.0))) \
            - sum(cl['util'][:reach].sum() for cl in lp.classes if cl['kind'] == 'queue' and cl['last'] < reach)
        work_here = sum(cl['util'].sum() for cl in lp.classes if cl['kind'] == 'bundle')
        return min(1.0, max(free, 0.0) / max(work_here, 1e-12))

    def _curve(self, lp, phis, v0, work, ci=None):
        """Index (g per W x step of job power) on each segment of phis, from the LP values (v0 = value at phi = 0).
        Returns (nus, values, solutions)."""
        vals, xs = [v0], []
        for ph in phis[1:]:
            v, x = self._solve(lp, ph, ci)
            vals.append(v)
            xs.append(x)
        vals = np.array(vals, dtype=float)
        with np.errstate(invalid='ignore'):
            nus = np.diff(vals) / (np.diff(phis) * work)
        nus[~(np.isfinite(vals[1:]) & np.isfinite(vals[:-1]))] = np.inf     # a failed solve: unknown, never cheap
        fin = nus[np.isfinite(nus)]
        if len(fin) > 1 and np.any(np.diff(fin) < -1e-6 * max(1.0, np.abs(fin).max())):
            self.stats['curve_violations'] += 1
            nus = np.maximum.accumulate(nus)
        return nus, vals, xs

    def _replan(self, geo):
        """Hourly: each site's plan (phi = 0) and its cost curve per bundle."""
        t = geo.t
        self.stats['replans'] += 1
        groups = self._bundles(geo)
        self.curves = {}
        self.acc[:] = 0.0
        for i in range(geo.n_sites):
            base = self._site_lp(geo, i)
            v0, x0 = self._solve(base, 0.0, self._ci_for(geo, i, base))
            self.plans[i] = self._make_plan(base, x0, t) if x0 is not None else None
            if not np.isfinite(v0):
                continue
            for b, g in groups.items():
                lp = self._site_lp(geo, i, bundle=g)
                phi_max = self._phi_max(geo, lp, g)
                if phi_max <= 1e-6:
                    continue
                phis = np.linspace(0.0, phi_max, self.G + 1)
                nus = self._curve(lp, phis, v0, g['work'], self._ci_for(geo, i, lp))[0]
                self.curves[(i, b)] = (phis, nus, g['work'])

    def _make_plan(self, lp, x, t):
        """Per queued class: planned start share per step (relative), and battery / load per step."""
        starts = {}
        for k, (c, s) in enumerate(zip(lp.xc, lp.xs)):
            if lp.classes[c]['kind'] == 'queue' and x[k] > 1e-9:
                starts.setdefault(c, {})[int(s)] = starts.get(c, {}).get(int(s), 0.0) + float(x[k])
        # planned occupancy (full steps, as the env checks capacity): committed + planned starts' occupancy
        occ = lp.R_occ.copy()
        for k, (c, s) in enumerate(zip(lp.xc, lp.xs)):
            if x[k] > 1e-9 and lp.classes[c]['kind'] == 'queue':   # bundles are forecasts, not reservations
                o = lp.classes[c]['occ'][:lp.W - s]
                occ[s:s + len(o)] += x[k] * o
        return {'t0': t, 'classes': [cl['jobs'] for cl in lp.classes if cl['kind'] == 'queue'], 'starts': starts,
                'ch': x[lp.ich], 'dis': x[lp.idis], 'u': x[lp.iu], 'occ': occ, 'p_kwh': lp.bat['p_kwh']}

    # -----------------------------------------------------------------------------------------------------------------
    def _index(self, i, b, extra_work):
        """Index of site i for bundle b after accepting `extra_work` (W x steps) of it this hour; inf if full."""
        cur = self.curves.get((i, b))
        if cur is None:
            return None
        phis, nus, work = cur
        phi = (self.acc[i, b] + extra_work) / work
        k = int(np.searchsorted(phis[1:], phi, side='right'))
        return nus[k] if k < len(nus) else np.inf

    def _planned_load(self, geo, i, n):
        """Site i's planned occupancy for the next n steps (from its latest plan; committed load if none)."""
        plan = self.plans[i]
        run = geo.running(i)
        tau = geo.t + np.arange(n)
        committed = (geo.size[run, i][:, None] * (tau[None, :] < np.ceil(geo.end_step[run, None] - 1e-9))).sum(0)
        if plan is None:
            return committed
        k = geo.t - plan['t0']
        occ = plan['occ'][k:k + n]
        occ = np.pad(occ, (0, n - len(occ)), constant_values=0.0)
        return np.maximum(occ, committed)

    def route(self, geo):
        """Each arrival to the site with the lowest index, among sites where it fits in time: some start no later
        than its latest start at which the site's planned load (its latest plan) plus the jobs routed earlier in this
        step plus the job stay within capacity over the job's whole run. The job reserves that slot. (The hourly
        curves cannot see this step's congestion; without the check, urgent jobs pile onto a full site.)"""
        arr = geo.arrivals()
        out = np.full(len(arr), -1, dtype=np.int64)
        if len(arr) == 0:
            return out
        slack = geo.latest_start[arr] - geo.arrival[arr]
        n = int(np.ceil(slack.max() + geo.dur[arr].max())) + 2
        load = np.stack([self._planned_load(geo, i, n) for i in range(geo.n_sites)])  # (N, n), updated as we route
        latest = self.admission == 'latest'
        if latest:
            lat = np.stack([self._latest_profile(geo, i, n) for i in range(geo.n_sites)])
        for k in np.argsort(geo.latest_start[arr], kind='stable'):
            j = arr[k]
            last = int(max(np.floor(geo.latest_start[j] - geo.t + 1e-9), 0))
            span = int(np.ceil(geo.dur[j] - 1e-9))
            sites = [int(geo.origin[j])] if geo.rigid[j] else range(geo.n_sites)
            slot = {}
            for i in sites:
                if latest:
                    if lat[i, last:last + span].max(initial=0.0) + geo.size[j, i] <= geo.cap + 1e-12:
                        slot[i] = last
                    continue
                win = np.lib.stride_tricks.sliding_window_view(load[i], span)[:last + 1].max(axis=1)
                ok = np.flatnonzero(win + geo.size[j, i] <= geo.cap + 1e-12)
                if len(ok):
                    slot[i] = int(ok[0])
            b = int(self._bundle_of(slack[k]))
            w = float(geo.power_w[j] * geo.dur[j])
            if geo.rigid[j]:
                i = int(geo.origin[j])
            else:
                idx = [self._index(i, b, w / 2) for i in range(geo.n_sites)]  # index at the job's midpoint
                score = self.A[:, geo.t].copy() if all(v is None for v in idx) else \
                    np.array([np.inf if v is None else v for v in idx])          # no curve yet: merit now
                score = np.where([i in slot for i in range(geo.n_sites)], score, np.inf)
                if np.isfinite(score).any():
                    i = int(np.argmin(score))
                elif latest:  # admitted nowhere: the site whose latest-start schedule overflows least
                    self.stats['admission_fallbacks'] = self.stats.get('admission_fallbacks', 0) + 1
                    i = int(np.argmin(lat[:, last:last + span].max(axis=1, initial=0.0)
                                      + geo.size[j, np.arange(geo.n_sites)]))
                else:  # fits nowhere in time: the site with the most room soonest
                    i = int(np.argmin(load[:, :last + 1].min(axis=1)))
                if self.curves.get((i, b)) is not None:
                    self.acc[i, b] += w
            s0 = slot.get(i, 0)
            load[i, s0:s0 + span] += geo.size[j, i]
            if latest:
                lat[i, last:last + span] += geo.size[j, i]
            out[k] = i
        return out

    @staticmethod
    def _latest_profile(geo, i, n):
        """Site i's occupancy over the next n steps if every job queued there starts at its latest start (and the
        running jobs run on): the admission test of admission='latest'."""
        tau = geo.t + np.arange(n)
        run = geo.running(i)
        prof = (geo.size[run, i][:, None] * (tau[None, :] < np.ceil(geo.end_step[run, None] - 1e-9))).sum(0)
        q = geo.queue(i)
        if len(q):
            s0 = np.maximum(np.floor(geo.latest_start[q] + 1e-9).astype(int) - geo.t, 0)
            d = np.ceil(geo.dur[q] - 1e-9).astype(int)
            o = np.arange(n)[None, :]
            prof += (geo.size[q, i][:, None] * ((o >= s0[:, None]) & (o < (s0 + d)[:, None]))).sum(0)
        return prof

    def starts(self, geo):
        """Per site: queued jobs the plan starts in this step (per class, earliest deadline first)."""
        out = []
        for i in range(geo.n_sites):
            plan = self.plans[i]
            if plan is None:
                out.append(None)
                continue
            k = geo.t - plan['t0']
            todo = []
            for c, by_step in plan['starts'].items():
                jobs = plan['classes'][c]
                waiting = jobs[geo.status[jobs] == 1]
                if not len(waiting):
                    continue
                target = sum(v for s, v in by_step.items() if s <= k)   # cumulative planned share of the class
                work = geo.power_w[jobs] * geo.dur[jobs]
                done = float(np.sum(work[geo.status[jobs] >= 2])) / float(work.sum())
                order = waiting[np.argsort(geo.latest_start[waiting], kind='stable')]
                acc = done
                for j in order:
                    share = float(geo.power_w[j] * geo.dur[j]) / float(work.sum())
                    if acc + share / 2 > target + 1e-9:
                        break
                    todo.append(int(j))
                    acc += share
            # Jobs due now always come first, then earliest deadline first across classes: otherwise a less urgent
            # class's planned starts can take the capacity a due job needs (it would start one step late).
            q = geo.queue(i)
            due = q[geo.latest_start[q] < geo.t + 1]
            todo = np.unique(np.r_[np.asarray(todo, dtype=np.int64), due])
            todo = todo[np.argsort(geo.latest_start[todo], kind='stable')]
            if self.reserve_steps:
                todo = self._reserve(geo, i, q, todo)
            out.append(todo.tolist())
        return out

    def _reserve(self, geo, i, q, todo):
        """Drop planned (not yet due) starts that would leave no room for the queued jobs due within reserve_steps,
        each placed at its latest start. Fluid plans round to whole jobs; without this, a site near capacity starts
        a non-urgent job and a job due the next step cannot start (an SLA violation)."""
        t, R = geo.t, self.reserve_steps
        L = R + int(np.ceil(geo.trace['max_dur_steps'])) + 1
        tau = t + np.arange(L)
        run = geo.running(i)
        prof = (geo.size[run, i][:, None] * (tau[None, :] < np.ceil(geo.end_step[run, None] - 1e-9))).sum(0)
        last = np.floor(geo.latest_start[q] + 1e-9).astype(int) - t          # relative latest start slot
        soon = q[(last >= 1) & (last <= R)]
        slot = {}
        for j in soon:                                                     # reservations at the latest start
            s0, d = int(np.floor(geo.latest_start[j] + 1e-9)) - t, int(np.ceil(geo.dur[j] - 1e-9))
            prof[s0:s0 + d] += geo.size[j, i]
            slot[int(j)] = (s0, d)
        keep = []
        for j in todo:
            j = int(j)
            d = int(np.ceil(geo.dur[j] - 1e-9))
            if geo.latest_start[j] < t + 1:                                # due now: always tried
                keep.append(j)
                continue
            trial = prof.copy()
            if j in slot:
                s0, dj = slot[j]
                trial[s0:s0 + dj] -= geo.size[j, i]
            trial[:d] += geo.size[j, i]
            over = lambda p: float(np.maximum(p - geo.cap, 0.0).sum())
            if over(trial) <= over(prof) + 1e-12:                         # no new overflow (moving a crowded job up helps)
                prof = trial
                keep.append(j)
        return np.array(keep, dtype=np.int64)

    def site_actions(self, geo):
        acts = []
        for i, s in enumerate(geo.sites):
            a = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
            plan = self.plans[i]
            if self.execute_local and plan is not None:
                k = geo.t - plan['t0']
                if k < len(plan['ch']) and self.execute_battery:
                    if plan['ch'][k] >= self.bat_threshold * plan['p_kwh'] and plan['ch'][k] >= plan['dis'][k]:
                        a['agent_bat'] = 0
                    elif plan['dis'][k] >= self.bat_threshold * plan['p_kwh']:
                        a['agent_bat'] = 1
                    if self.best_stpt is not None:
                        u = plan['u'][k]
                        target = self.best_stpt[i, geo.t, int(np.abs(TANGENT_LOADS - u).argmin())]
                        a['agent_dc'] = setpoint_action(float(s.dc_env.raw_curr_stpt), float(target))
            acts.append(a)
        return acts

    def _plan_only(self, geo):
        for i in range(geo.n_sites):
            lp = self._site_lp(geo, i)
            _, x = self._solve(lp, 0.0, self._ci_for(geo, i, lp))
            self.plans[i] = self._make_plan(lp, x, geo.t) if x is not None else None

    def act(self, geo):
        t0 = time.time()
        if geo.t % 4 == 0:
            self._replan(geo)            # cost curves (and plans) from the state before this step's arrivals
        route = self.route(geo)
        geo.route(route)
        if geo.t % self.plan_every == 0:
            self._plan_only(geo)         # plans that include the jobs just routed
        starts = self.starts(geo)
        acts = self.site_actions(geo)
        self.stats['decision_seconds'].append(time.time() - t0)
        return {'route': None, 'starts': starts, 'sites': acts}


def run_index_episode(geo, policy, seed):
    """One seeded episode with the index policy; returns geo.episode_metrics() + policy timing stats."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    geo.reset()
    policy.reset(geo)
    steps = 0
    while True:
        _, _, _, truncated, _ = geo.step(policy.act(geo))
        steps += 1
        if truncated:
            break
    m = geo.episode_metrics()
    d = np.array(policy.stats['decision_seconds'])
    m.update({'steps': steps, 'lp_solves': policy.stats['lp_solves'], 'lp_seconds': policy.stats['lp_seconds'],
              'replans': policy.stats['replans'], 'curve_violations': policy.stats['curve_violations'],
              'decision_s_mean': float(d.mean()), 'decision_s_max': float(d.max()),
              'replan_s_mean': float(d[::4].mean())})
    return m
