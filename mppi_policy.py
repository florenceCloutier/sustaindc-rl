"""Model predictive path integral control (MPPI) for SustainDC from the single-agent viewpoint, next to the CEM MPC of
mpc_policy.py (policy kind `mpc_joint`). Everything except the optimizer is shared with it: the planning model (an
exact clone of the simulator, with the `oracle` or `persistence` forecast of the exogenous inputs), the rollout cost
and terminal value (rollout_cost), move blocking (`block`), receding horizon (`replan`), `end_cost`, and the parallel
rollout workers. Differences between `mpc_joint` and `mppi` at the same settings and rollout budget are therefore
differences between the two optimizers.

MPPI update. Each replan, for `iters` iterations: sample `samples` perturbed plans around the nominal plan, roll each
out on the planning model to get its cost S_i (gCO2), weight it by
    w_i = exp(-(S_i - min S) / lambda) / sum_j exp(-(S_j - min S) / lambda)
and move the nominal plan to the weighted average of the samples. lambda (the temperature) is chosen every iteration
so that the effective sample size ESS = 1 / sum_i w_i^2 equals `ess` (default 5): about that many samples drive the
update, whatever the scale and spread of the costs (which change by orders of magnitude with location, season and
horizon). ESS -> 1 keeps only the best sample, ESS -> number of samples averages all of them. Alternatively
`temperature` fixes lambda = temperature * std(S) (the first version of this module). (No control-cost term: the
actions have no cost of their own here, i.e. gamma = 0 in the information-theoretic MPPI.)

The actions are discrete (3 choices per sub-action), so there are two variants (`sampling`):
  * gaussian (default): the classical MPPI on a continuous relaxation. The three sub-actions are ordinal, so each
    becomes a value in [-1, 1]:
        ls:  -1 defer,     0 do nothing,  +1 process the queue
        dc:  -1 setpoint down, 0 hold,    +1 setpoint up
        bat: -1 discharge, 0 idle,        +1 charge
    The nominal plan U (one 3-vector per block) is perturbed with Gaussian noise (std `noise`, shrunk by `noise_decay`
    every iteration), and U <- U + sum_i w_i eps_i (clipped to [-1, 1]). Each sample is turned into actions by
    stochastic rounding (`rounding`): 0.3 becomes +1 with probability 0.3 and 0 otherwise, so the expected action is
    the continuous value and the update sees the effect of small perturbations (nearest rounding maps most of them to
    the same plan). The continuous nominal is kept between replans (shifted by the executed blocks). With
    agent-dispatched PV (include_pv) the PV action (use / battery / resell) is not ordinal: it gets a categorical
    distribution per block, updated with the same weights (step 0.7, and mixed 20% with uniform at every warm start,
    so it cannot collapse for good).
  * categorical: one categorical distribution per block over the 27 (81 with include_pv) joint actions, as in the
    CEM MPC, but updated with the MPPI weights of all samples instead of the frequencies of the elite set:
        p_b <- (1 - smoothing) p_b + smoothing * sum_i w_i onehot(a_i,b)
    This isolates the update rule (soft weights vs hard elites) from the action representation.

Candidates: the first sample of every iteration is the nominal plan itself (rounded to the nearest action, or the
most likely action of each block). With `seed_plans` (default) the first iteration also rolls out the 27 (81) constant
plans, as the CEM MPC does; they enter the weighted update like the other samples (for gaussian sampling, as the
perturbation that reaches them from the nominal).

Executed plan (`execute`): `best` (default) executes the lowest-cost plan rolled out during the replan, the nominal
plan after the last update included (it is rolled out once more); `mean` (pure MPPI) executes that nominal plan even
if it was never rolled out.

Usage (from the repo root), policy kind `mppi`:
    python eval_compare.py --policy noop --policy mpc_joint=cem --policy mppi=mppi --days_per_episode 3
    python eval_compare.py --policy mppi:horizon=96,block=8,replan=4,samples=128,iters=5,workers=16=mppi_H96S ...
    python eval_compare.py --policy mppi:sampling=categorical,ess=10=mppi_cat ...
Parameters (key=value, comma separated): horizon, block, replan, samples, iters, ess, temperature, noise,
noise_decay, sampling, rounding, seed_plans, smoothing, execute, drop_w, overdue_w, forecast, forecast_days, end_cost,
workers, seed, verbose, and the conservative workload forecast wl_model, wl_q, wl_plan, wl_gamma, wl_aci
(workload_forecast.py). See MPPIPolicy.__init__.
"""
import time

import numpy as np

from mpc_policy import WL_BOOLS, WL_FLOATS, WL_STRINGS, MPCPolicy
from sustaindc_single_env import sub_actions

SAMPLINGS = ('gaussian', 'categorical')
EXECUTES = ('mean', 'best')
ROUNDINGS = ('stochastic', 'nearest')
N_ORDINAL = 3  # ls, dc, bat
PV_STEP = 0.7  # step size of the PV distribution's update (gaussian sampling with include_pv), as categorical's default
# Ordinal value of each sub-action index, and back (value + 1 -> index).
ORD_VALUE = np.array([[-1, 0, 1],    # ls:  0 defer, 1 do nothing, 2 process
                      [-1, 0, 1],    # dc:  0 setpoint -1, 1 hold, 2 setpoint +1
                      [1, -1, 0]])   # bat: 0 charge, 1 discharge, 2 idle
ORD_INDEX = np.argsort(ORD_VALUE, axis=1)  # ORD_INDEX[k, v + 1] = sub-action index of value v


def effective_sample_size(w):
    """1 / sum w_i^2 of normalized weights: 1 if one sample has all the weight, n if the weights are uniform."""
    return 1.0 / float(np.sum(np.square(w)))


def _boltzmann(costs, lam):
    w = np.exp(-(costs - costs.min()) / lam)
    return w / w.sum()


def mppi_weights(costs, temperature=None, ess=None):
    """Normalized MPPI weights exp(-(S - min S) / lambda).

    ess:         lambda such that the effective sample size is `ess` (bisection on log lambda; ESS grows with lambda).
                 ess >= number of samples gives uniform weights, ess <= 1 the best sample only.
    temperature: lambda = temperature * std(S) instead.
    Exactly one of the two must be given. Equal costs give uniform weights."""
    if (temperature is None) == (ess is None):
        raise ValueError('give exactly one of temperature and ess')
    costs = np.asarray(costs, dtype=float)
    n = len(costs)
    spread = costs.std()
    if spread <= 1e-12 * max(1.0, abs(costs.min())):
        return np.full(n, 1.0 / n)
    if temperature is not None:
        if not np.isfinite(temperature):
            return np.full(n, 1.0 / n)
        return _boltzmann(costs, temperature * spread)
    if ess >= n:
        return np.full(n, 1.0 / n)
    if ess <= 1:
        w = (costs == costs.min()).astype(float)
        return w / w.sum()
    lo, hi = np.log(spread * 1e-6), np.log(spread * 1e6)  # ESS(lo) ~ ties at the min, ESS(hi) ~ n
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if effective_sample_size(_boltzmann(costs, np.exp(mid))) < ess:
            lo = mid
        else:
            hi = mid
    return _boltzmann(costs, np.exp(0.5 * (lo + hi)))


def ordinal_to_sub(values, rng=None):
    """Continuous ordinal values (..., 3) -> sub-action indices (..., 3). Nearest rounding, or stochastic rounding if
    `rng` is given (v in [k, k+1] becomes k+1 with probability v - k, so the expected value is v)."""
    v = np.clip(values, -1.0, 1.0)
    if rng is None:
        q = np.rint(v)
    else:
        lo = np.floor(v)
        q = lo + (rng.random(v.shape) < (v - lo))
    q = np.clip(q, -1, 1).astype(int) + 1
    return np.stack([ORD_INDEX[k][q[..., k]] for k in range(N_ORDINAL)], axis=-1)


def sub_to_ordinal(sub):
    """Sub-action indices (..., 3) -> ordinal values (..., 3)."""
    sub = np.asarray(sub, dtype=int)
    return np.stack([ORD_VALUE[k][sub[..., k]] for k in range(N_ORDINAL)], axis=-1).astype(float)


class MPPIPolicy(MPCPolicy):
    """Receding-horizon MPPI over the joint (ls, dc, bat[, pv]) action, on the same planning model and cost as the
    CEM MPC (MPCPolicy). eval_compare policy interface: reset(seed), act(sdc) -> {agent: action}; act_joint(sdc)
    returns the Discrete(27) / Discrete(81) index."""

    def __init__(self, horizon=48, block=4, replan=None, samples=32, iters=3, ess=None, temperature=None, noise=0.6,
                 noise_decay=1.0, sampling='gaussian', rounding='stochastic', seed_plans=True, smoothing=None,
                 execute='best', drop_w=10.0, overdue_w=1.0, forecast='oracle', forecast_days=1, end_cost='proxy',
                 workers=0, seed=0, verbose=False, wl_model=None, wl_q=0.9, wl_plan='upper', wl_gamma=0.05,
                 wl_aci=True):
        """
        horizon, block, replan, drop_w, overdue_w, forecast, forecast_days, end_cost, workers, seed, verbose,
        wl_model, wl_q, wl_plan, wl_gamma, wl_aci:
                       as for MPCPolicy (mpc_policy.py).
        samples:       sampled plans rolled out per MPPI iteration (the first is the nominal plan itself).
        iters:         MPPI iterations per replan.
        ess:           target effective sample size of the weights (> 1); lambda is set to reach it every iteration.
                       Default 5 (unless `temperature` is given).
        temperature:   fixed lambda relative to the standard deviation of the sampled costs (> 0), instead of `ess`.
        noise:         std of the Gaussian perturbation of the ordinal actions in [-1, 1] (gaussian sampling).
        noise_decay:   factor applied to `noise` after every iteration of a replan (1 = constant).
        sampling:      'gaussian' (continuous relaxation of the ordinal sub-actions) or 'categorical' (categorical
                       distribution over the joint actions per block). See the module doc.
        rounding:      'stochastic' (default) or 'nearest': how gaussian samples are turned into actions.
        seed_plans:    also roll out the constant plans in the first iteration of each replan (as the CEM MPC).
        smoothing:     step size of the update (1 = full MPPI step). Default 1 for gaussian, 0.7 for categorical
                       (as the CEM MPC's smoothing; a full step would put all mass on the sampled actions).
        execute:       'best' (lowest-cost plan rolled out, default) or 'mean' (nominal plan after the update).
        """
        super().__init__(horizon=horizon, block=block, replan=replan, samples=samples, iters=iters,
                         drop_w=drop_w, overdue_w=overdue_w, forecast=forecast, forecast_days=forecast_days,
                         end_cost=end_cost, workers=workers, seed=seed, verbose=verbose, wl_model=wl_model,
                         wl_q=wl_q, wl_plan=wl_plan, wl_gamma=wl_gamma, wl_aci=wl_aci)
        if sampling not in SAMPLINGS:
            raise ValueError(f'sampling must be one of {SAMPLINGS}, got {sampling!r}')
        if execute not in EXECUTES:
            raise ValueError(f'execute must be one of {EXECUTES}, got {execute!r}')
        if rounding not in ROUNDINGS:
            raise ValueError(f'rounding must be one of {ROUNDINGS}, got {rounding!r}')
        if ess is not None and temperature is not None:
            raise ValueError('give ess or temperature, not both')
        if temperature is not None and not float(temperature) > 0:
            raise ValueError(f'temperature must be > 0, got {temperature}')
        if temperature is None:
            ess = 5.0 if ess is None else float(ess)
            if not ess > 1:
                raise ValueError(f'ess must be > 1, got {ess}')
        if not float(noise) > 0 or not 0 < float(noise_decay) <= 1:
            raise ValueError(f'need noise > 0 and 0 < noise_decay <= 1, got {noise}, {noise_decay}')
        if self.samples < 2:
            raise ValueError(f'samples must be >= 2, got {self.samples}')
        self.sampling, self.execute, self.rounding = sampling, execute, rounding
        self.seed_plans = bool(seed_plans)
        self.ess = ess
        self.temperature = float(temperature) if temperature is not None else None
        self.noise, self.noise_decay = float(noise), float(noise_decay)
        self.smoothing = float(smoothing) if smoothing is not None else (1.0 if sampling == 'gaussian' else 0.7)
        if not 0 < self.smoothing <= 1:
            raise ValueError(f'smoothing must be in (0, 1], got {self.smoothing}')

    def reset(self, seed):
        super().reset(seed)
        self.nominal = None  # gaussian: (n_blocks, 3) ordinal values; categorical: (n_blocks, n_joint) probabilities
        self.pv_probs = None  # gaussian with include_pv: (n_blocks, 3) PV action probabilities

    # -- plans --------------------------------------------------------------------------------------------------------
    def _joint(self, sub):
        """Sub-action indices (..., n_sub) -> joint action indices (...)."""
        return np.ravel_multi_index(tuple(np.moveaxis(sub, -1, 0)), sub_actions(self.include_pv))

    def _sub(self, joint):
        """Joint action indices (...) -> sub-action indices (..., n_sub)."""
        return np.stack(np.unravel_index(joint, sub_actions(self.include_pv)), axis=-1)

    def _noop_sub(self):
        return np.array(np.unravel_index(self.noop_joint, sub_actions(self.include_pv)))

    def _shift(self, x, pad=None):
        """Shift a per-block array by the executed blocks (as MPCPolicy._warm_start), padding with its last row."""
        shift = min(-(-self.replan // self.block), self.n_blocks)
        tail = np.repeat((x[-1:] if pad is None else pad[None]), shift, axis=0)
        return np.concatenate([x[shift:], tail])

    def _init_nominal(self):
        """Nominal plan at the start of a replan: the previous one shifted, or do-nothing after a reset (or a change of
        the action space, where MPCPolicy.act_joint clears self.plan)."""
        fresh = self.plan is None or self.nominal is None
        if self.sampling == 'gaussian':
            if fresh:
                noop = self._noop_sub()
                self.nominal = np.tile(sub_to_ordinal(noop[:N_ORDINAL]), (self.n_blocks, 1))
                self.pv_probs = None
                if self.include_pv:
                    self.pv_probs = np.full((self.n_blocks, 3), 0.5 / 2)
                    self.pv_probs[:, noop[N_ORDINAL]] = 0.5
            else:
                self.nominal = self._shift(self.nominal)
                if self.include_pv:  # keep exploration mass (PV choices are cost-neutral at night)
                    self.pv_probs = 0.8 * self._shift(self.pv_probs) + 0.2 / 3
        else:
            uniform = np.full(self.n_joint, 1.0 / self.n_joint)
            if fresh:
                self.nominal = np.tile(uniform, (self.n_blocks, 1))
                self.nominal[:, self.noop_joint] += 1.0
                self.nominal /= self.nominal.sum(axis=1, keepdims=True)
            else:
                # Keep some mass everywhere so a distribution that collapsed can still move.
                self.nominal = 0.8 * self._shift(self.nominal) + 0.2 * uniform

    def _nominal_plan(self):
        """Joint-action plan represented by the current nominal (nearest action / most likely action per block)."""
        if self.sampling == 'gaussian':
            sub = ordinal_to_sub(self.nominal)
            if self.include_pv:
                sub = np.concatenate([sub, self.pv_probs.argmax(axis=1)[:, None]], axis=1)
            return self._joint(sub).astype(int)
        return self.nominal.argmax(axis=1).astype(int)

    def _sample_categorical(self, probs, n):
        """n samples per block from per-block probabilities (n_blocks, k) -> (n, n_blocks)."""
        u = self.rng.random((n, probs.shape[0], 1))
        return (probs.cumsum(axis=1)[None] < u).sum(axis=2).clip(0, probs.shape[1] - 1)

    def _candidates(self, n, sigma, first):
        """Plans of one iteration (n samples, plus the constant plans in the first iteration with seed_plans).
        Returns (plans (m, n_blocks), eps (m, n_blocks, 3) or None, pv (m, n_blocks) or None)."""
        constants = (np.repeat(np.arange(self.n_joint)[:, None], self.n_blocks, axis=1)
                     if first and self.seed_plans else np.zeros((0, self.n_blocks), dtype=int))
        if self.sampling == 'categorical':
            plans = self._sample_categorical(self.nominal, n).astype(int)
            plans[0] = self.nominal.argmax(axis=1)
            return np.concatenate([plans, constants]), None, None
        eps = self.rng.normal(0.0, sigma, (n, self.n_blocks, N_ORDINAL))
        eps[0] = 0.0
        rng = self.rng if self.rounding == 'stochastic' else None
        sub = ordinal_to_sub(self.nominal[None] + eps, rng)
        sub[0] = ordinal_to_sub(self.nominal)  # the nominal plan itself, nearest actions
        pv = None
        if self.include_pv:
            pv = self._sample_categorical(self.pv_probs, n)
            pv[0] = self.pv_probs.argmax(axis=1)
            sub = np.concatenate([sub, pv[..., None]], axis=2)
        plans = self._joint(sub).astype(int)
        if len(constants):
            csub = self._sub(constants)
            eps = np.concatenate([eps, sub_to_ordinal(csub[..., :N_ORDINAL]) - self.nominal[None]])
            if self.include_pv:
                pv = np.concatenate([pv, csub[..., N_ORDINAL]])
            plans = np.concatenate([plans, constants])
        return plans, eps, pv

    # -- planning -----------------------------------------------------------------------------------------------------
    def _replan(self, sdc):
        t0 = time.time()
        evaluate = self._evaluator(sdc)
        self._init_nominal()
        seen = {}

        def costs_of(plans):
            new = list({p.tobytes(): p for p in plans if p.tobytes() not in seen}.values())
            if new:
                for p, c in zip(new, evaluate(new)):
                    seen[p.tobytes()] = float(c)
            return np.array([seen[p.tobytes()] for p in plans])

        best_plan, best_cost = None, np.inf
        sigma = self.noise
        for it in range(self.iters):
            plans, eps, pv = self._candidates(self.samples, sigma, first=(it == 0))
            plans = list(plans)
            costs = costs_of(plans)
            i_best = int(np.argmin(costs))
            if costs[i_best] < best_cost:
                best_cost, best_plan = float(costs[i_best]), plans[i_best].copy()

            w = mppi_weights(costs, temperature=self.temperature, ess=None if self.temperature else self.ess)
            a = self.smoothing
            if self.sampling == 'gaussian':
                self.nominal = np.clip(self.nominal + a * np.tensordot(w, eps, axes=1), -1.0, 1.0)
                if self.include_pv:
                    target = np.stack([np.bincount(pv[:, b], weights=w, minlength=3) for b in range(self.n_blocks)])
                    self.pv_probs = (1 - PV_STEP) * self.pv_probs + PV_STEP * target
                sigma *= self.noise_decay
            else:
                stacked = np.stack(plans)
                target = np.stack([np.bincount(stacked[:, b], weights=w, minlength=self.n_joint)
                                   for b in range(self.n_blocks)])
                self.nominal = (1 - a) * self.nominal + a * target

        plan = self._nominal_plan()
        if self.execute == 'best':
            c = costs_of([plan])[0]
            if best_cost < c:
                plan = best_plan
        self.plan = plan
        self.pending = list(np.repeat(plan, self.block)[:self.replan].astype(int))
        self.stats['replans'] += 1
        self.stats['rollouts'] += len(seen)
        self.stats['plan_seconds'] += time.time() - t0
        if self.verbose:
            sub = [tuple(int(x) for x in np.unravel_index(a, sub_actions(self.include_pv))) for a in plan]
            names = '(ls,dc,bat,pv)' if self.include_pv else '(ls,dc,bat)'
            print(f'  [mppi] t={sdc.t_m.current_timestep} best sampled={best_cost / 1e3:.1f} kg '
                  f'plan{names}={sub} ({len(seen)} rollouts, {time.time() - t0:.2f}s)', flush=True)


def make_mppi_policy(rest):
    """Build an MPPIPolicy from 'key=value,...' (the part after 'mppi:' in an eval_compare spec)."""
    params = dict(kv.split('=', 1) for kv in rest.split(',') if kv) if rest else {}
    ints = {'horizon', 'block', 'replan', 'samples', 'iters', 'seed', 'forecast_days', 'workers'}
    floats = {'ess', 'temperature', 'noise', 'noise_decay', 'smoothing', 'drop_w', 'overdue_w'} | WL_FLOATS
    strings = {'forecast', 'end_cost', 'sampling', 'execute', 'rounding'} | WL_STRINGS
    bools = {'verbose', 'seed_plans'} | WL_BOOLS
    valid = ints | floats | strings | bools
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f'Unknown mppi parameter(s) {sorted(unknown)}; valid: {sorted(valid)}')
    kwargs = {}
    for k, v in params.items():
        if k in ints:
            kwargs[k] = int(v)
        elif k in floats:
            kwargs[k] = float(v)
        elif k in bools:
            kwargs[k] = v.lower() in ('1', 'true', 'yes')
        else:
            kwargs[k] = v
    return MPPIPolicy(**kwargs)
