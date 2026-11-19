"""Next-hour job-arrival forecaster, trained on the real (non-repeated) 7 weeks of the Alibaba 2020 GPU trace.

Why only 7 weeks. DCcluster-Opt's full-year trace is one 7-week (49-day) block of the real trace repeated ~7.5 times
(sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py): every hour equals the hour 49 days
earlier. Training across copies and testing on another copy measures memory, not forecasting. So everything here
uses ONE copy, block days 0-48 (the trace starts on 2020-01-01 at a block start).

What it predicts. Job work (power x duration) arriving in the next hour, per urgency class = the slack bundles of
geo_index.IndexPolicy (slack = 0.5 x duration < 1 h, 1-2 h, > 2 h), as log(1 + work / scale).

Split by block day (the evaluation episodes, dccluster5 seeds 1-3 and n20 seeds 1-2 in July, fall on block days
29.3-41.6):
    days 0-6     history only (inputs of the first targets)
    days 7-23    train
    days 24-27   validation (model choice, early stopping)
    days 28-41   TEST (hold-out; contains every evaluation episode)
    days 42-48   unused (their 7-day inputs would include test hours)
No train / validation sample has an input or a target in days 28-41 (asserted). Nothing reaches across the block
boundary.

Inputs (past only): hour of day and day of week (sin, cos; 2020-01-01 is a Wednesday), the last 6 hours per class,
the same hour 1, 2 and 7 days earlier per class, the 24-hour mean per class.
Models: ridge (closed form, strength chosen on validation) and an MLP (1 hidden layer of 32, weight decay, absolute
error loss, early stopping); the one with the lower validation error is kept. Baselines on the test hours: the last
hour (what dual uses now) and the seasonal average (same hour on the previous 7 days).

    python geo_arrivals.py --out <dir>      # <dir>/arrival_model.pt, report.json, test_forecast.png

ArrivalForecaster(path).bundles(geo, t) turns a prediction into forecast jobs for dual (next step: geo_dual_fc.py).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())
from geo_jobs import JOB_DEFAULTS, load_trace  # noqa: E402

EDGES = (4, 8)                    # slack class edges in steps, as IndexPolicy's slack_edges
BLOCK_DAYS = 49
LAGS_H, DAY_LAGS = 6, (1, 2, 7)
HIST = 24 * max(DAY_LAGS)         # hours of history a sample needs
TRAIN, VAL, TEST = (7, 24), (24, 28), (28, 42)
FIRST_WEEKDAY = 2                 # 2020-01-01 was a Wednesday (Monday = 0)
CLASSES = ('urgent (<1 h slack)', 'medium (1-2 h)', 'relaxed (>2 h)')


def hourly_work(trace, n_hours=None, sla=JOB_DEFAULTS['sla_multiplier']):
    """(3, n_hours) work per slack class and trace hour, and the class of every trace job."""
    slack = (sla - 1.0) * trace['dur_steps']
    cls = np.searchsorted(np.asarray(EDGES, float), slack, side='right')
    n_h = n_hours or trace['n_steps'] // 4
    hour = trace['arrival'] // 4
    W = np.zeros((3, n_h))
    for c in range(3):
        m = (cls == c) & (hour < n_h)
        W[c] = np.bincount(hour[m], weights=(trace['power_w'] * trace['dur_steps'])[m], minlength=n_h)
    return W, cls


def block_series(trace):
    """W over one copy of the block (block days 0-48), and a check that the trace really repeats with that period."""
    W, _ = hourly_work(trace)
    n = 24 * BLOCK_DAYS
    assert np.allclose(W[:, n:2 * n], W[:, :n]), 'the trace does not repeat every 49 days'
    return W[:, :n]


def input_hours(h):
    """The hours a sample at hour h reads (for the leakage assertion)."""
    return sorted({h - k for k in range(1, 25)} | {h - 24 * d for d in DAY_LAGS})


def features(W, h, scale):
    L = np.log1p(W / scale)
    lags = L[:, h - LAGS_H:h][:, ::-1].reshape(-1)
    days = L[:, [h - 24 * d for d in DAY_LAGS]].reshape(-1)
    mean24 = np.log1p(W[:, h - 24:h].mean(1) / scale)
    hod, dow = (h % 24) / 24.0, ((h // 24 + FIRST_WEEKDAY) % 7) / 7.0
    return np.r_[np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod), np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
                 lags, days, mean24].astype(np.float32)


def split(W, scale):
    out = {}
    for name, (d0, d1) in (('train', TRAIN), ('val', VAL), ('test', TEST)):
        hs = [h for h in range(24 * d0, 24 * d1) if h >= HIST]
        if name != 'test':                                    # no input or target in the test days
            for h in hs:
                assert all(not (24 * TEST[0] <= g < 24 * TEST[1]) and g >= 0 for g in input_hours(h) + [h])
        X = np.stack([features(W, h, scale) for h in hs])
        Y = np.log1p(W[:, hs].T / scale).astype(np.float32)
        out[name] = (hs, X, Y)
    return out


def to_work(Y, scale):
    return np.expm1(np.asarray(Y)).clip(0) * scale


def errors(P, T):
    """(error of total next-hour work, error per class), each as mean absolute error / mean actual."""
    return (float(np.abs(P.sum(1) - T.sum(1)).mean() / T.sum(1).mean()),
            float(np.abs(P - T).mean() / T.mean()))


# -- models -----------------------------------------------------------------------------------------------------
class Ridge:
    def __init__(self, mu, sd, beta):
        self.mu, self.sd, self.beta = mu, sd, beta

    def __call__(self, X):
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        return Z @ self.beta


def fit_ridge(Xtr, Ytr, Xva, Yva, scale, W, va_hours):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    best = None
    for lam in 10.0 ** np.arange(-3, 4):
        I = np.eye(Z.shape[1])
        I[-1, -1] = 0.0
        beta = np.linalg.solve(Z.T @ Z + lam * I, Z.T @ Ytr)
        m = Ridge(mu, sd, beta)
        e = errors(to_work(m(Xva), scale), W[:, va_hours].T)[0]
        if best is None or e < best[0]:
            best = (e, lam, m)
    return best[2], {'lambda': best[1], 'val_error_total': best[0]}


class MLP(nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


class MLPModel:
    def __init__(self, net, mu, sd):
        self.net, self.mu, self.sd = net, mu, sd

    def __call__(self, X):
        with torch.no_grad():
            return self.net(torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))).numpy()


def fit_mlp(Xtr, Ytr, Xva, Yva, scale, W, va_hours, seed=0, epochs=2000):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(Ytr)
    net = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        loss = (net(xt) - yt).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        net.eval()
        e = errors(to_work(MLPModel(net, mu, sd)(Xva), scale), W[:, va_hours].T)[0]
        if e < best - 1e-6:
            best, state, bad = e, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 100:
                break
    net.load_state_dict(state)
    net.eval()
    return MLPModel(net, mu, sd), {'val_error_total': best, 'epochs': ep + 1}


# -- training and report ------------------------------------------------------------------------------------------
def train(out_dir):
    tr = load_trace()
    W = block_series(tr)
    d = split(W, scale=1.0)                                   # scale from the training targets only
    scale = float(W[:, d['train'][0]].mean())
    d = split(W, scale)
    (htr, Xtr, Ytr), (hva, Xva, Yva), (hte, Xte, Yte) = d['train'], d['val'], d['test']
    ridge, rinfo = fit_ridge(Xtr, Ytr, Xva, Yva, scale, W, hva)
    mlp, minfo = fit_mlp(Xtr, Ytr, Xva, Yva, scale, W, hva)
    chosen = 'ridge' if rinfo['val_error_total'] <= minfo['val_error_total'] else 'mlp'
    T = W[:, hte].T
    last = np.stack([W[:, h - 1] for h in hte])                               # baselines straight from W
    season = np.stack([W[:, [h - 24 * k for k in range(1, 8)]].mean(1) for h in hte])
    preds = {'last_hour': last, 'seasonal_7d': season,
             'ridge': to_work(ridge(Xte), scale), 'mlp': to_work(mlp(Xte), scale)}
    rep = {'hours': {'train': len(htr), 'val': len(hva), 'test': len(hte)},
           'test_error_total_work': {k: errors(P, T)[0] for k, P in preds.items()},
           'test_error_per_class': {k: errors(P, T)[1] for k, P in preds.items()},
           'chosen': chosen, 'ridge': rinfo, 'mlp': minfo}
    # leak check: the chosen model with shuffled test inputs should do no better than the seasonal average
    model = ridge if chosen == 'ridge' else mlp
    perm = np.random.default_rng(0).permutation(len(hte))
    rep['test_error_total_work_shuffled_inputs'] = errors(to_work(model(Xte[perm]), scale), T)[0]
    os.makedirs(out_dir, exist_ok=True)
    blob = {'chosen': chosen, 'scale': scale, 'd_in': Xtr.shape[1]}
    if chosen == 'ridge':
        blob.update({'mu': ridge.mu, 'sd': ridge.sd, 'beta': ridge.beta})
    else:
        blob.update({'mu': mlp.mu, 'sd': mlp.sd, 'state': mlp.net.state_dict()})
    torch.save(blob, os.path.join(out_dir, 'arrival_model.pt'))
    json.dump(rep, open(os.path.join(out_dir, 'report.json'), 'w'), indent=1)
    plot(hte, T, preds, chosen, os.path.join(out_dir, 'test_forecast.png'))
    return rep


def plot(hours, T, preds, chosen, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = (np.asarray(hours) / 24.0)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(x, T.sum(1) / 1e6, color='k', lw=1.2, label='actual')
    ax.plot(x, preds['last_hour'].sum(1) / 1e6, lw=0.8, alpha=.7, label='last hour (dual now)')
    ax.plot(x, preds['seasonal_7d'].sum(1) / 1e6, lw=0.8, alpha=.7, label='seasonal average')
    ax.plot(x, preds[chosen].sum(1) / 1e6, lw=1.2, label=f'{chosen} (trained)')
    ax.set_xlabel('block day (test weeks)')
    ax.set_ylabel('next-hour job work (trace units, x1e6)')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# -- use in the policy ----------------------------------------------------------------------------------------------
class ArrivalForecaster:
    """predict(trace_hour) and bundles(geo, t). Inputs come from the episode's actual past (the looped trace before
    hour h, which is what an operator has seen)."""

    def __init__(self, path):
        b = torch.load(path, weights_only=False)
        self.scale = b['scale']
        if b['chosen'] == 'ridge':
            self.model = Ridge(b['mu'], b['sd'], b['beta'])
        else:
            net = MLP(b['d_in'])
            net.load_state_dict(b['state'])
            net.eval()
            self.model = MLPModel(net, b['mu'], b['sd'])
        self.trace = load_trace()
        self.W, self.cls = hourly_work(self.trace)

    def predict(self, h):
        """(3,) predicted trace work per class for trace hour h (reads hours < h only)."""
        return to_work(self.model(features(self.W, h, self.scale)[None]), self.scale)[0]

    def bundles(self, geo, t, days=7):
        """Forecast bundles for episode steps [t, t + 4): {class: {power_w, dur, slack, release, work}}. The predicted
        work per class (x the environment's replication factor, minus the rigid share) rescales the job mix of the
        same hour on the previous `days` days (durations, slacks and minutes of arrival kept)."""
        tr = self.trace
        h = int(((geo.utc0 + t) % tr['n_steps']) // 4)
        if h < HIST:
            return {}
        factor = geo.job_cfg['target_util'] * float(geo.cap_w.sum()) / tr['mean_running_w']
        pred = self.predict(h) * factor * (1.0 - geo.job_cfg['rigid_fraction'])
        sla = geo.job_cfg['sla_multiplier']
        hour = tr['arrival'] // 4
        groups = {}
        for c in range(3):
            idx = np.flatnonzero(np.isin(hour, [h - 24 * k for k in range(1, days + 1)]) & (self.cls == c))
            if not len(idx) or pred[c] <= 0:
                continue
            assert np.all(hour[idx] < h), 'forecast jobs must come from the past'
            p, d = tr['power_w'][idx], tr['dur_steps'][idx]
            alpha = pred[c] / float(np.sum(p * d))
            groups[c] = {'power_w': p * alpha, 'dur': d, 'slack': (sla - 1.0) * d,
                         'release': (tr['arrival'][idx] % 4).astype(float), 'work': float(pred[c])}
        return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    print(json.dumps(train(args.out), indent=1))


if __name__ == '__main__':
    main()
