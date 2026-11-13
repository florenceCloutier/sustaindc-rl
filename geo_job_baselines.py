"""Heuristic job dispatchers for GeoJobsDC (geo_jobs.py), and run_job_episode().

Every dispatcher sees pending jobs in earliest-latest-start order and, for each, picks a site to start at now or
waits (-1). It tracks the load it commits during the step, so it never sends a job to a full site. Myopic rules use only
the current state; `forecast_greedy` uses the true future (perfect forecast).

    local            start at the origin (waits only when the origin is full)
    greedy_ci        start now at the site with the lowest current carbon intensity that has room
    merit            start now at the site with the lowest marginal CO2 now: CI x marginal IT + cooling energy at its
                     current weather (a hot site with a high cooling overhead ranks lower)
    merit_defer      merit, but wait (while slack remains) if the best site's CI is above the `q` quantile of its own
                     past-24 h CI: a reactive, forecast-free temporal shift (the job version of baseline_policies.LSRule)
    forecast_greedy  perfect foresight: for each job, the (site, start time within its slack) with the lowest CO2 over
                     its whole run (true CI and weather), given the load already committed or planned by earlier jobs
                     this step; start now if the best start is now, else wait (re-planned every step)

Marginal CO2 uses a precomputed table, per site and step, of gCO2 per unit of load for one step:
    A[i, t] = CI_i(t) x dP_i/du (t) x 0.25 h,
dP/du = slope of IT + cooling power with load (baseline_policies.dc_power at the site's setpoint and that step's
weather). The power curve is nearly linear in load, so one slope per step is enough.
"""
import numpy as np

from baseline_policies import dc_power
from geo_sustaindc_env import STEP_HOURS

JOB_DISPATCHERS = ('local', 'greedy_ci', 'merit', 'merit_defer', 'forecast_greedy')
SLOPE_LOADS = (0.3, 0.7)


def marginal_table(geo, n_steps):
    """(ci, kwh_per_unit, A) over the next n_steps of the episode, each (N, n_steps):
    ci gCO2/kWh; kwh_per_unit = kWh for one unit of load (the whole site) during one step; A = ci x kwh_per_unit (g)."""
    lo, hi = SLOPE_LOADS
    ci = np.zeros((geo.n_sites, n_steps))
    kwh = np.zeros((geo.n_sites, n_steps))
    for i, s in enumerate(geo.sites):
        t0c, t0w = s.ci_m.time_step, s.weather_m.time_step
        c = np.asarray(s.ci_m.carbon_smooth[t0c:t0c + n_steps], dtype=float)
        amb = np.asarray(s.weather_m.temperature_data[t0w:t0w + n_steps], dtype=float)
        wb = np.asarray(s.weather_m.wet_bulb_data[t0w:t0w + n_steps], dtype=float)
        n = min(len(c), len(amb), len(wb))
        c, amb, wb = np.pad(c[:n], (0, n_steps - n), mode='edge'), np.pad(amb[:n], (0, n_steps - n), mode='edge'), \
            np.pad(wb[:n], (0, n_steps - n), mode='edge')
        stpt = float(s.dc_env.raw_curr_stpt)
        cache = {}
        for k in range(n_steps):
            key = (round(amb[k], 1), round(wb[k], 1))
            if key not in cache:
                p_lo = sum(dc_power(s, lo, stpt, key[0], key[1]))
                p_hi = sum(dc_power(s, hi, stpt, key[0], key[1]))
                cache[key] = (p_hi - p_lo) / (hi - lo) / 1e3 * STEP_HOURS  # kWh per unit load per step
            kwh[i, k] = cache[key]
        ci[i] = c
    return ci, kwh, ci * kwh


def run_cost(A_cum, i, tau, dur):
    """CO2 (g per unit load) of running at site i from step(s) tau for dur steps: sum of A with a partial last step.
    A_cum: (N, T+1) prefix sums of A. tau: int array; dur: float."""
    full = int(np.floor(dur))
    part = dur - full
    T = A_cum.shape[1] - 1
    a = np.minimum(tau, T)
    b = np.minimum(tau + full, T)
    cost = A_cum[i, b] - A_cum[i, a]
    if part > 0:
        nxt = np.minimum(tau + full, T - 1)
        cost = cost + part * (A_cum[i, nxt + 1] - A_cum[i, nxt])
    return cost


class JobDispatcher:
    def __init__(self, kind, q=0.5):
        if kind not in JOB_DISPATCHERS:
            raise ValueError(f'Unknown job dispatcher {kind!r}; choose from {JOB_DISPATCHERS}')
        self.kind = kind
        self.q = float(q)

    def reset(self, geo):
        """Called after geo.reset(). Precomputes the marginal-CO2 table over the episode (+ the longest job)."""
        self.n = geo.horizon + int(np.ceil(geo.trace['max_dur_steps'])) + 2
        self.ci, self.kwh, self.A = marginal_table(geo, self.n)
        self.A_cum = np.concatenate([np.zeros((geo.n_sites, 1)), np.cumsum(self.A, axis=1)], axis=1)
        if self.kind == 'merit_defer':
            past = []
            for i, s in enumerate(geo.sites):
                t0 = s.ci_m.time_step
                p = np.asarray(s.ci_m.carbon_smooth[max(0, t0 - 96):t0], dtype=float)
                # Episodes starting less than a day into the data year (and sites in other time zones start at other
                # local steps) have a shorter past: pad on the left with the earliest known value, so every site
                # has the same 96-step window.
                fill = p[0] if len(p) else self.ci[i, 0]
                past.append(np.concatenate([np.full(96 - len(p), fill), p]))
            self.ci_hist = np.concatenate([np.stack(past), self.ci], axis=1)
            self.hist0 = self.ci_hist.shape[1] - self.ci.shape[1]

    def assign(self, geo, view):
        jobs, t = view['jobs'], view['t']
        out = np.full(len(jobs), -1, dtype=np.int64)
        if len(jobs) == 0:
            return out
        load = view['load'].copy()
        cap = view['cap']
        size, origin, rigid = view['size'], view['origin'], view['rigid']
        if self.kind == 'local':
            for k in range(len(jobs)):
                i = origin[k]
                if load[i] + size[k, i] <= cap:
                    out[k] = i
                    load[i] += size[k, i]
            return out
        if self.kind == 'forecast_greedy':
            return self._forecast_greedy(geo, view)
        score = self.ci[:, t] if self.kind == 'greedy_ci' else self.A[:, t]
        order = np.argsort(score)
        if self.kind == 'merit_defer':
            h = self.hist0 + t
            ref = np.array([np.quantile(self.ci_hist[i, h - 96:h], self.q) for i in range(geo.n_sites)])
        for k in range(len(jobs)):
            if rigid[k]:
                i = origin[k]
                if load[i] + size[k, i] <= cap:
                    out[k] = i
                    load[i] += size[k, i]
                continue
            for i in order:
                if load[i] + size[k, i] <= cap:
                    if self.kind == 'merit_defer' and self.ci[i, t] > ref[i] and view['latest_start'][k] >= t + 1:
                        break  # wait: the best open site is expensive relative to its own recent CI
                    out[k] = i
                    load[i] += size[k, i]
                    break
        return out

    def _forecast_greedy(self, geo, view):
        """List scheduling with perfect foresight: a per-(site, step) load profile of running jobs, plus reservations
        of the jobs planned later this step; each job takes its cheapest feasible (site, start)."""
        jobs, t = view['jobs'], view['t']
        N, H = geo.n_sites, self.n
        prof = np.zeros((N, H))  # committed load per site and step (relative to the episode start)
        run = np.flatnonzero(geo.status == 2)
        for j in run:
            i = geo.site[j]
            prof[i, t:int(np.ceil(geo.end_step[j]))] += geo.size[j, i]
        out = np.full(len(jobs), -1, dtype=np.int64)
        cap = view['cap']
        for k, j in enumerate(jobs):
            d = view['dur'][k]
            last = int(np.floor(view['latest_start'][k] + 1e-9))
            taus = np.arange(t, max(last, t) + 1)
            span = int(np.ceil(d))
            sites = [view['origin'][k]] if view['rigid'][k] else range(N)
            best = (np.inf, -1, -1)
            for i in sites:
                sz = view['size'][k, i]
                cost = run_cost(self.A_cum, i, taus, d) * sz
                # feasibility: the site must have room over the whole run
                win = np.lib.stride_tricks.sliding_window_view(prof[i, t:t + len(taus) + span - 1], span)[:len(taus)]
                ok = (win.max(axis=1) + sz <= cap + 1e-12)
                if view['rigid'][k]:
                    ok[1:] = False
                if ok.any():
                    c = np.where(ok, cost, np.inf)
                    m = int(np.argmin(c))
                    if c[m] < best[0]:
                        best = (c[m], i, taus[m])
            c, i, tau = best
            if i < 0:
                continue  # no feasible slot: wait (the env force-starts it at its latest start if it can)
            prof[i, tau:tau + span] += view['size'][k, i]
            if tau == t:
                out[k] = i
        return out


def run_job_episode(geo, dispatcher, seed, local_spec='noop'):
    """One seeded episode; returns geo.episode_metrics() (+ 'steps')."""
    import random
    from eval_compare import make_policy
    random.seed(seed)
    np.random.seed(seed)
    geo.reset()
    dispatcher.reset(geo)
    local = [make_policy(local_spec)[1] for _ in range(geo.n_sites)]
    for k, p in enumerate(local):
        p.reset(seed + 1000 * k)
    steps = 0
    while True:
        view = geo.pending_view()
        assign = dispatcher.assign(geo, view)
        _, _, _, truncated, _ = geo.step({'assign': assign, 'sites': [p.act(s) for p, s in zip(local, geo.sites)]})
        steps += 1
        if truncated:
            break
    m = geo.episode_metrics()
    m['steps'] = steps
    return m


# ---------------------------------------------------------------------------------------------------------------------
# site_queue mode (GeoJobsDC(routing='site_queue')): route each arrival to a site's queue; sites start jobs themselves
# ---------------------------------------------------------------------------------------------------------------------
JOB_ROUTERS = ('origin', 'greedy_ci', 'merit')


class JobRouter:
    """Routes this step's arrivals to site queues, using only the current state:
        origin     every job to its origin (with earliest-deadline-first starts, the site_queue version of `local`)
        greedy_ci  the lowest-current-CI site whose committed + queued load leaves room for the job
        merit      the same with the marginal CO2 now (CI x marginal IT + cooling energy, as `merit`)
    If no site has room, the job goes to the one with the least committed + queued load (it waits there)."""

    def __init__(self, kind):
        if kind not in JOB_ROUTERS:
            raise ValueError(f'Unknown router {kind!r}; choose from {JOB_ROUTERS}')
        self.kind = kind

    def reset(self, geo):
        if self.kind != 'origin':
            self.n = geo.horizon + 2
            self.ci, _, self.A = marginal_table(geo, self.n)

    def route(self, geo):
        arr = geo.arrivals()
        out = np.full(len(arr), -1, dtype=np.int64)
        if self.kind == 'origin' or len(arr) == 0:
            return out
        proj = geo.committed_load()
        for i in range(geo.n_sites):
            proj[i] += geo.size[geo.queue(i), i].sum()
        score = (self.ci if self.kind == 'greedy_ci' else self.A)[:, geo.t]
        order = np.argsort(score)
        for k in np.argsort(geo.latest_start[arr], kind='stable'):
            j = arr[k]
            if geo.rigid[j]:
                i = geo.origin[j]
            else:
                fits = [i for i in order if proj[i] + geo.size[j, i] <= geo.cap + 1e-12]
                i = fits[0] if fits else int(np.argmin(proj))
            out[k] = i
            proj[i] += geo.size[j, i]
        return out


def run_queue_episode(geo, router, seed, local_spec='noop', scheduler=None):
    """One seeded episode in site_queue mode: `router.route(geo)` routes arrivals, `scheduler` (None = earliest
    deadline first) starts queued jobs: scheduler.starts(geo) -> per-site lists. Returns geo.episode_metrics()."""
    import random
    from eval_compare import make_policy
    assert geo.routing == 'site_queue', 'run_queue_episode needs GeoJobsDC(routing="site_queue")'
    random.seed(seed)
    np.random.seed(seed)
    geo.reset()
    router.reset(geo)
    if scheduler is not None:
        scheduler.reset(geo)
    local = [make_policy(local_spec)[1] for _ in range(geo.n_sites)]
    for k, p in enumerate(local):
        p.reset(seed + 1000 * k)
    steps = 0
    while True:
        route = router.route(geo)
        geo.route(route)
        starts = scheduler.starts(geo) if scheduler is not None else None
        _, _, _, truncated, _ = geo.step({'route': None, 'starts': starts,
                                          'sites': [p.act(s) for p, s in zip(local, geo.sites)]})
        steps += 1
        if truncated:
            break
    m = geo.episode_metrics()
    m['steps'] = steps
    return m
