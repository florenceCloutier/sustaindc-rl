"""Conservative (quantile) workload forecaster for the MPC planners, with online conformal calibration.

Why: the planners (mpc_policy.py, mppi_policy.py) size load shifting to the workload they are told about. The queue
can only be drained with the capacity left by the load (envs/carbon_ls.py: 90% minus the current tasks), so a
forecast that is too low exactly when load is high makes the planner defer work it then cannot run: the queue fills,
deferred tasks are dropped and overdue tasks are forced through at whatever CI. A forecast trained for the mean or
median is too low about half the time, and its misses cluster at surges. The `persistence` forecast of the MPC (same
time of day, averaged over past days) is the extreme case: it under-predicts at nearly every top-10%-load step.

What this module provides:
  1. WorkloadQuantileModel: direct multi-horizon linear quantile regression. One model per (anchor horizon,
     quantile), fitted with the pinball loss
        L_q(y, f) = q (y - f)  if y >= f      (under-forecast)
                    (1 - q)(f - y) otherwise   (over-forecast)
     solved exactly as a linear program (scipy HiGHS). With q = 0.9 an under-forecast costs 9x an over-forecast of the
     same size, so the model learns the conditional 90th percentile. Several quantiles are fitted at once and are
     sorted at prediction time so they never cross. Features use the series up to the forecast origin t only:
     current load, 1 h / 4 h slopes, last-24 h mean, same time yesterday, same time last week, how far the load is
     from last week's (|y_t - y_{t-week}|) and how fast it moves (|1 h slope|), the last two letting the spread grow
     at ramps and when the week does not repeat, plus the time of day of the target.
  2. WorkloadForecaster: the online forecaster the planners use. It adds an adaptive conformal margin to the
     quantile model's output so that the bound covers the realized load a fraction q of the time even when the
     model is off (the margin form of Adaptive Conformal Inference, Gibbs & Candes 2021, as the "quantile tracker" of
     Angelopoulos, Candes & Tibshirani 2023). For every anchor horizon h, once the load at t is known:
        miss = 1[y_t > qhat_h(t - h) + m_h]
        m_h <- m_h + gamma * scale_h * (miss - (1 - q))
     where the bound is the one issued at t - h (delayed feedback) and scale_h is the larger of the median model's
     training MAE at h and a running (1-day) mean of the margin-free model's |y - qhat|, so the margin still moves when a model that was
     near-certain in training meets a trace it does not fit. The margin grows after misses and shrinks slowly
     otherwise, which gives long-run coverage q whatever the drift. At the start of an episode the margin is warmed up on the
     `warmup_days` days before it (past data). The forecast for horizons between the anchors is interpolated; past
     the last anchor (96 steps = 24 h) the planner keeps the persistence forecast.
  3. critical_quantile(c_under, c_over): the newsvendor choice of q from the cost of one unit of under-forecast
     versus one unit of over-forecast, q* = c_under / (c_under + c_over). E.g. a dropped task 20x worse than a unit
     of idle capacity gives q* = 0.95.

How the planners use it (mpc_joint / mppi parameters wl_model, wl_q, wl_plan, wl_gamma, wl_aci): the planning
model's workload series after the current step becomes the forecaster's output for the next 24 h (then persistence).
  wl_plan=upper  (default) plan on the calibrated q-bound: the planner assumes every future step is at its upper
                 quantile, a pessimistic path that keeps capacity headroom for surges.
  wl_plan=median plan on the median model (q = 0.5, no margin): the learned point forecast, for ablations.
The workload forecast is independent of `forecast` (oracle / persistence for CI, weather and PV), so
`forecast=oracle,wl_model=...` isolates the effect of the workload forecast.

Data caveat: both Alibaba traces in data/Workload are a single week repeated 52 times, so "same time last week" is
almost exactly the future and the fitted model is near-certain on them (the conformal margin then stays ~0). The
Google trace does not repeat; that is where the quantile and the margin matter. Train on days or traces you do not
evaluate on (fit --exclude_days); the model file records what it was trained on.

Usage (from the repo root):
    # fit on the Google trace, holding out the evaluation protocol's July window (days 174-219) and two weeks after
    python workload_forecast.py fit --files GoogleClusteData_CPU_Data_Hourly_1.csv --exclude_days 174:233 \\
        --out trained_models/workload_forecast/google_nojuly.npz
    # forecast quality on the held-out days: coverage, pinball loss, shortfall at high load, vs persistence
    python workload_forecast.py eval --model trained_models/workload_forecast/google_nojuly.npz \\
        --files GoogleClusteData_CPU_Data_Hourly_1.csv --days 174:219 --q 0.9
    # in the planners
    python eval_compare.py --policy mpc_joint:forecast=persistence=mpc_persist \\
        --policy mpc_joint:forecast=persistence,wl_model=trained_models/workload_forecast/google_nojuly.npz,wl_q=0.9=mpc_wlq90
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

DAY = 96          # 15-minute steps per day
WEEK = 7 * DAY
HORIZONS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)   # anchor horizons (steps); others are interpolated
QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)
FEATURES = ('bias', 'y_t', 'slope_1h', 'slope_4h', 'abs_slope_1h', 'mean_24h', 'y_target_minus_day', 'y_t_minus_day',
            'y_target_minus_week', 'y_t_minus_week', 'abs_dev_week', 'sin_tod', 'cos_tod', 'sin_2tod', 'cos_2tod')
PLANS = ('upper', 'median')
REPO = os.path.dirname(os.path.abspath(__file__))


def critical_quantile(c_under, c_over):
    """Newsvendor-optimal quantile: q* = c_under / (c_under + c_over)."""
    if c_under < 0 or c_over < 0 or c_under + c_over <= 0:
        raise ValueError(f'costs must be >= 0 and not both 0, got c_under={c_under}, c_over={c_over}')
    return c_under / (c_under + c_over)


def pinball(y, f, q):
    """Mean pinball loss of forecast f for quantile q."""
    d = np.asarray(y) - np.asarray(f)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# ---------------------------------------------------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------------------------------------------------
def load_trace(filename, timezone_shift=0):
    """The workload series SustainDC runs on (utils/managers.py Workload_Manager after reset): the hourly trace
    interpolated to 15 minutes, rolled by the time zone, scaled so its 5th / 95th percentiles map to 0.2 / 0.8,
    clipped to [0, 1] and smoothed with a 16-step moving average. Reimplemented here so fitting does not need the
    simulator's dependencies; tests/verify_workload_forecast.py checks it matches the manager."""
    path = filename if os.path.isabs(filename) else os.path.join(REPO, 'data', 'Workload', filename)
    raw = pd.read_csv(path)['cpu_load'].values[:8760].astype(float)
    y = np.interp(np.linspace(0, len(raw), len(raw) * 4), range(len(raw)), raw)
    y = np.roll(y, -4 * int(timezone_shift))
    p5, p95 = np.percentile(y, 5), np.percentile(y, 95)
    y = np.clip(0.2 + (y - p5) * (0.8 - 0.2) / (p95 - p5), 0, 1)
    return np.convolve(y, np.ones(16) / 16, mode='same')


def features(y, origins, h):
    """Feature matrix (len(origins), len(FEATURES)) to forecast y[t + h] from y[:t + 1], for each origin t.
    Only indices <= t are read (h <= DAY), clipped at 0 before the start of the data. Needs 1 <= h <= DAY."""
    if not 1 <= h <= DAY:
        raise ValueError(f'horizon must be in [1, {DAY}], got {h}')
    y = np.asarray(y, dtype=float)
    t = np.asarray(origins, dtype=int)
    if np.any(t >= len(y)) or np.any(t < 0):
        raise IndexError('origin outside the series')
    at = lambda idx: y[np.clip(idx, 0, t)]            # noqa: E731 (never later than the origin)
    csum = np.concatenate([[0.0], np.cumsum(y)])
    lo = np.maximum(t - DAY + 1, 0)
    mean_24h = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    y_t = y[t]
    tod = 2 * np.pi * ((t + h) % DAY) / DAY
    cols = [np.ones_like(y_t), y_t, y_t - at(t - 4), y_t - at(t - 16), np.abs(y_t - at(t - 4)), mean_24h,
            at(t + h - DAY), at(t - DAY), at(t + h - WEEK), at(t - WEEK), np.abs(y_t - at(t - WEEK)),
            np.sin(tod), np.cos(tod), np.sin(2 * tod), np.cos(2 * tod)]
    return np.stack(cols, axis=1)


def quantile_regression(X, y, q):
    """Linear quantile regression: argmin_b sum pinball_q(y - X b), as an LP solved by HiGHS.
    min q 1'u + (1-q) 1'v  s.t.  X b + u - v = y,  u, v >= 0."""
    from scipy.optimize import linprog
    import scipy.sparse as sp
    n, p = X.shape
    c = np.concatenate([np.zeros(p), np.full(n, q), np.full(n, 1 - q)])
    eye = sp.identity(n, format='csr')
    A = sp.hstack([sp.csr_matrix(X), eye, -eye], format='csr')
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f'quantile regression LP failed (q={q}): {res.message}')
    return res.x[:p]


# ---------------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------------
class WorkloadQuantileModel:
    """Linear quantile models, one per (anchor horizon, quantile). coef: (len(horizons), len(quantiles), n_features).
    scale: training MAE of the median model per horizon (step size unit of the conformal margin)."""

    def __init__(self, coef, horizons=HORIZONS, quantiles=QUANTILES, scale=None, meta=None):
        self.coef = np.asarray(coef, dtype=float)
        self.horizons = tuple(int(h) for h in horizons)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.scale = np.ones(len(self.horizons)) if scale is None else np.asarray(scale, dtype=float)
        self.meta = dict(meta or {})
        if self.coef.shape != (len(self.horizons), len(self.quantiles), len(FEATURES)):
            raise ValueError(f'coef shape {self.coef.shape} does not match horizons x quantiles x features')

    @classmethod
    def fit(cls, series_list, horizons=HORIZONS, quantiles=QUANTILES, stride=4, exclude=None, verbose=True):
        """Fit on the given series. Origins every `stride` steps from one week in (so every feature is real data).
        exclude: per series, a boolean mask over its steps; samples whose origin or target is excluded are dropped."""
        coef = np.zeros((len(horizons), len(quantiles), len(FEATURES)))
        scale = np.zeros(len(horizons))
        n_samples = []
        for i, h in enumerate(horizons):
            Xs, ys = [], []
            for k, y in enumerate(series_list):
                t = np.arange(WEEK, len(y) - h, stride)
                if exclude is not None and exclude[k] is not None:
                    t = t[~exclude[k][t] & ~exclude[k][t + h]]
                Xs.append(features(y, t, h))
                ys.append(y[t + h])
            X, Y = np.concatenate(Xs), np.concatenate(ys)
            n_samples.append(len(Y))
            t0 = time.time()
            for j, q in enumerate(quantiles):
                coef[i, j] = quantile_regression(X, Y, q)
            j50 = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
            scale[i] = max(float(np.mean(np.abs(Y - X @ coef[i, j50]))), 1e-4)
            if verbose:
                print(f'  h={h:3d}: {len(Y)} samples, median MAE {scale[i]:.4f}, {time.time() - t0:.1f}s', flush=True)
        meta = {'n_samples': n_samples, 'stride': stride}
        return cls(coef, horizons, quantiles, scale, meta)

    def predict(self, y, origins):
        """Quantile forecasts (len(origins), len(horizons), len(quantiles)) of y[t + h] from y[:t + 1]: non-crossing
        (sorted over the quantiles) and clipped to [0, 1]."""
        origins = np.atleast_1d(origins)
        out = np.stack([features(y, origins, h) @ self.coef[i].T for i, h in enumerate(self.horizons)], axis=1)
        return np.clip(np.sort(out, axis=-1), 0.0, 1.0)

    def quantile_index(self, q):
        i = int(np.argmin(np.abs(np.asarray(self.quantiles) - q)))
        if abs(self.quantiles[i] - q) > 1e-9:
            raise ValueError(f'quantile {q} not fitted; available: {self.quantiles}')
        return i

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, coef=self.coef, horizons=np.array(self.horizons), quantiles=np.array(self.quantiles),
                 scale=self.scale, features=np.array(FEATURES), meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if tuple(str(f) for f in d['features']) != FEATURES:
                raise ValueError(f'{path} was fitted with other features: {list(d["features"])}')
            return cls(d['coef'], d['horizons'], d['quantiles'], d['scale'], json.loads(str(d['meta'])))


# ---------------------------------------------------------------------------------------------------------------------
# Online forecaster with adaptive conformal calibration
# ---------------------------------------------------------------------------------------------------------------------
class WorkloadForecaster:
    """Stateful forecaster for one episode at a time: forecast(series, t, n) returns the next n values after t, built
    from series[:t + 1] only. With plan='upper', the q-quantile plus an adaptive conformal margin per anchor horizon;
    with plan='median', the median model. Call reset() at the start of each episode (it also resets itself if time
    goes backward)."""

    def __init__(self, model, q=0.9, plan='upper', gamma=0.05, aci=True, warmup_days=7):
        if plan not in PLANS:
            raise ValueError(f'wl_plan must be one of {PLANS}, got {plan!r}')
        if not 0 < q < 1:
            raise ValueError(f'q must be in (0, 1), got {q}')
        self.model = WorkloadQuantileModel.load(model) if isinstance(model, str) else model
        self.plan = plan
        self.q = 0.5 if plan == 'median' else float(q)
        self.qi = self.model.quantile_index(self.q)
        self.aci = bool(aci) and plan == 'upper'
        self.gamma = float(gamma)
        self.warmup = int(warmup_days) * DAY
        self.horizons = np.array(self.model.horizons)
        self.reset()

    def reset(self):
        self.margin = np.zeros(len(self.horizons))
        self.err = self.model.scale.copy()  # running mean |realized - issued bound| per horizon (1-day EWMA)
        self.t_done = None          # last origin whose forecast has been issued (and loads up to it scored)
        self.start = None           # first origin issued this episode (warm-up included)
        self._issued = {}           # origin -> bound issued at that origin per anchor horizon (q-quantile + margin)
        self._raw = {}              # origin -> margin-free q-quantile forecast per anchor horizon
        self.hits = np.zeros(len(self.horizons))
        self.counts = np.zeros(len(self.horizons))

    def update(self, y, t):
        """Issue the forecasts of every origin up to t and update the margins with the realized loads up to t.
        Each realized load y[tau] is scored against the bound issued h steps before (delayed feedback), which is
        what makes the long-run miss rate of the issued bounds 1 - q. Only y[:t + 1] is read."""
        y = np.asarray(y)[:t + 1]
        if self.t_done is not None and t < self.t_done:
            self.reset()
        if self.t_done is None:
            self.start = max(t - self.warmup, 0)
            self.t_done = self.start - 1
        if t <= self.t_done:
            return
        taus = np.arange(self.t_done + 1, t + 1)
        preds = self.model.predict(y, taus)[:, :, self.qi]      # margin-free, so computed in one batch
        for k, tau in enumerate(taus):
            if self.aci:
                origins = tau - self.horizons
                idx = np.flatnonzero(origins >= self.start)
                if len(idx):
                    bound = np.array([self._issued[s][i] for s, i in zip(origins[idx], idx)])
                    raw = np.array([self._raw[s][i] for s, i in zip(origins[idx], idx)])
                    miss = (y[tau] > bound).astype(float)
                    # error scale of the margin-free model (not of the bound: that would feed the margin back)
                    self.err[idx] += (np.abs(y[tau] - raw) - self.err[idx]) / DAY
                    step = self.gamma * np.maximum(self.model.scale[idx], self.err[idx])
                    self.margin[idx] += step * (miss - (1 - self.q))
                    self.hits[idx] += 1 - miss
                    self.counts[idx] += 1
            self._issued[int(tau)] = np.clip(preds[k] + (self.margin if self.aci else 0.0), 0.0, 1.0)
            self._raw[int(tau)] = preds[k]
        self.t_done = t
        oldest = t - int(self.horizons.max())                  # no anchor horizon scores origins before this
        if len(self._issued) > 2 * self.horizons.max():
            self._issued = {s: v for s, v in self._issued.items() if s >= oldest}
            self._raw = {s: v for s, v in self._raw.items() if s >= oldest}

    def anchors(self, y, t):
        """Forecast at the anchor horizons from origin t (margins updated with the loads up to t)."""
        self.update(y, t)
        return self._issued[t]

    def forecast(self, y, t, n):
        """Forecast of y[t + 1 .. t + n] (at most the last anchor horizon) from y[:t + 1], interpolated between the
        anchor horizons (horizon 0 = the current load)."""
        n = min(int(n), int(self.horizons.max()))
        vals = self.anchors(y, t)
        hs = np.arange(1, n + 1)
        return np.interp(hs, np.concatenate([[0], self.horizons]), np.concatenate([[float(y[t])], vals]))

    def coverage(self):
        """Empirical coverage of the bound per anchor horizon over the updates so far (warm-up included)."""
        return np.where(self.counts > 0, self.hits / np.maximum(self.counts, 1), np.nan)


# ---------------------------------------------------------------------------------------------------------------------
# Command line: fit / eval
# ---------------------------------------------------------------------------------------------------------------------
def parse_days(spec):
    """'a:b' -> (a, b) inclusive day range (0-indexed days of the year)."""
    a, b = (int(x) for x in spec.split(':'))
    if not 0 <= a <= b <= 364:
        raise ValueError(f'day range must be a:b with 0 <= a <= b <= 364, got {spec!r}')
    return a, b


def day_mask(n, ranges):
    mask = np.zeros(n, dtype=bool)
    for a, b in ranges:
        mask[a * DAY:(b + 1) * DAY] = True
    return mask


def evaluate(model, y, days, q=0.9, gamma=0.05, aci=True, warmup_days=7, high_pct=90):
    """Online evaluation over the origins of the given day range (every step, as the planner would use it). Returns
    a dict of per-horizon arrays: coverage and pinball of the raw q-quantile, of the conformal bound, and of the
    persistence / weekly / median forecasts; and at high load (target above the trace's `high_pct` percentile) the
    rate and mean size of under-forecasts."""
    a, b = days
    origins = np.arange(max(a * DAY, WEEK), (b + 1) * DAY - int(max(model.horizons)))
    fc = WorkloadForecaster(model, q=q, plan='upper', gamma=gamma, aci=aci, warmup_days=warmup_days)
    hs = np.array(model.horizons)
    bound = np.stack([fc.anchors(y, t) for t in origins])           # (n, H), online margins
    preds = model.predict(y, origins)                               # (n, H, Q)
    raw_q = preds[:, :, model.quantile_index(q)]
    median = preds[:, :, model.quantile_index(0.5)]
    target = y[origins[:, None] + hs[None, :]]
    # persistence as in mpc_policy.persistence_forecast (days=1): same time of day yesterday
    persist = y[origins[:, None] + hs[None, :] - DAY]
    weekly = y[origins[:, None] + hs[None, :] - WEEK]
    high = target > np.percentile(y, high_pct)
    out = {'horizons': hs, 'n_origins': len(origins), 'high_threshold': float(np.percentile(y, high_pct))}
    for name, f in (('persistence', persist), ('weekly', weekly), ('median', median), (f'q{q:g}', raw_q),
                    (f'q{q:g}+aci', bound)):
        under = target - f
        out[name] = {
            'coverage': np.mean(under <= 0, axis=0),
            'pinball': np.array([pinball(target[:, i], f[:, i], q) for i in range(len(hs))]),
            'mae': np.mean(np.abs(under), axis=0),
            'over': np.mean(np.maximum(-under, 0), axis=0),       # mean headroom (cost of conservatism)
            'high_under_rate': np.array([np.mean(under[high[:, i], i] > 0) if high[:, i].any() else np.nan
                                         for i in range(len(hs))]),
            'high_shortfall': np.array([np.mean(np.maximum(under[high[:, i], i], 0)) if high[:, i].any() else np.nan
                                        for i in range(len(hs))]),
        }
    return out


def print_eval(res, show=(1, 4, 16, 48, 96)):
    hs = list(res['horizons'])
    cols = [hs.index(h) for h in show if h in hs]
    print(f"{res['n_origins']} origins; high load = target > {res['high_threshold']:.3f}")
    print(f"{'forecast':>14} {'h':>4} {'coverage':>9} {'pinball':>8} {'MAE':>6} {'headroom':>9} "
          f"{'hi-load under%':>15} {'hi-load shortfall':>18}")
    for name, m in res.items():
        if not isinstance(m, dict):
            continue
        for i in cols:
            print(f"{name:>14} {hs[i]:>4} {m['coverage'][i]:>9.3f} {m['pinball'][i]:>8.4f} {m['mae'][i]:>6.3f} "
                  f"{m['over'][i]:>9.3f} {100 * m['high_under_rate'][i]:>14.1f}% {m['high_shortfall'][i]:>18.4f}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit', help='fit the quantile models and save them')
    f.add_argument('--files', nargs='+', default=['Alibaba_CPU_Data_Hourly_1.csv'], help='traces in data/Workload')
    f.add_argument('--exclude_days', nargs='*', default=[], help='a:b day ranges left out of training (every file)')
    f.add_argument('--stride', type=int, default=4, help='steps between training origins')
    f.add_argument('--timezone_shift', type=int, default=0)
    f.add_argument('--out', required=True)
    e = sub.add_parser('eval', help='online forecast quality on a day range')
    e.add_argument('--model', required=True)
    e.add_argument('--files', nargs='+', default=['Alibaba_CPU_Data_Hourly_1.csv'])
    e.add_argument('--days', default='174:219', help='a:b origins (default: the July protocol window)')
    e.add_argument('--q', type=float, default=0.9)
    e.add_argument('--gamma', type=float, default=0.05)
    e.add_argument('--no_aci', action='store_true')
    e.add_argument('--timezone_shift', type=int, default=0)
    args = p.parse_args()

    if args.cmd == 'fit':
        ranges = [parse_days(s) for s in args.exclude_days]
        series = [load_trace(fn, args.timezone_shift) for fn in args.files]
        exclude = []
        for y in series:
            m = day_mask(len(y), ranges)
            # origins in the two weeks after an excluded range read it through their lag features: drop them too
            m |= day_mask(len(y), [(a, min(b + 14, 364)) for a, b in ranges])
            exclude.append(m)
        print(f'fitting {len(HORIZONS)} horizons x {len(QUANTILES)} quantiles on {args.files}'
              + (f', excluding days {args.exclude_days} (+14 days after each)' if ranges else ''))
        model = WorkloadQuantileModel.fit(series, stride=args.stride, exclude=exclude)
        model.meta.update({'files': args.files, 'exclude_days': args.exclude_days,
                           'timezone_shift': args.timezone_shift})
        model.save(args.out)
        print(f'saved {args.out}')
    else:
        model = WorkloadQuantileModel.load(args.model)
        print(f"model trained on {model.meta.get('files')} excluding {model.meta.get('exclude_days')}")
        for fn in args.files:
            print(f'\n== {fn}, origins on days {args.days}')
            res = evaluate(model, load_trace(fn, args.timezone_shift), parse_days(args.days), q=args.q,
                           gamma=args.gamma, aci=not args.no_aci)
            print_eval(res)


if __name__ == '__main__':
    main()
