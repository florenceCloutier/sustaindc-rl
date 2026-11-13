"""B3: centralized deep RL router for GeoJobsDC(routing='site_queue'), trained with PPO (train_geo_job_ppo.py).

Decision: every 15-min step, each arriving job picks a site (factored: one categorical over the N sites per job,
all jobs of the step sampled independently given the joint observation). Sites then start their queues earliest
deadline first, with battery and cooling idle (as the JobRouter baselines), so the router is compared on routing.

Observation (centralized, grows with N x H):
    per site    CI now, nominal CI forecast (hourly means over the next `obs_hours`, sufficiency/forecast.py, so
                --sigma24 noise applies), ambient temperature, committed load, queued load, battery SoC, capacity,
                local hour (sin, cos)
    per job     log power, duration, slack, rigid, its size at every site (N), origin one-hot (N)
Policy: logits[j, i] = MLP(context(global obs), job features)[i]; rigid jobs are masked to their origin.
Value: MLP(global obs).
Reward per step: -(CO2 kg + late_kg x late work), late work = sum over waiting jobs past their latest start of their
size (fraction of their site's capacity) for the step, late_kg = 2 x the CO2 of a whole site for one step at
1000 g/kWh: lateness never pays.

PPO: per-job clipped ratios sharing their step's advantage (GAE over steps), as in parameter-shared multi-agent PPO.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from sufficiency.forecast import ForecastModel

SITE_FEATS = 8
JOB_FEATS = 4


class JobRoutingEnv:
    """Steps GeoJobsDC with routes given per arrival. reset(seed) -> obs; step(actions) -> obs, reward, done, info."""

    def __init__(self, geo, obs_hours=12, sigma24=0.0, rho=0.3, fc_seed=0, late_kg=None):
        self.geo, self.H, self.N = geo, int(obs_hours), geo.n_sites
        self.sigma24, self.rho, self.fc_seed = sigma24, rho, fc_seed
        self.late_kg = late_kg if late_kg is not None else 2 * 1.0 * 1e3 * 0.25 * float(geo.capacity_mw.max())
        self.obs_dim = self.N * (SITE_FEATS + self.H)
        self.job_dim = JOB_FEATS + 2 * self.N

    def reset(self, seed):
        import random
        from geo_lp_oracle import exo_traces
        random.seed(seed)
        np.random.seed(seed)
        self.geo.reset()
        tr = exo_traces(self.geo, self.geo.horizon + 4 * self.H + 4)
        self.ci = np.stack([t['ci'] for t in tr])
        self.amb = np.stack([t['amb'] for t in tr])
        self.fc = ForecastModel(self.ci, [c['location'] for c in self.geo.site_configs], self.sigma24, rho=self.rho,
                                seed=self.fc_seed + seed)
        return self.observe()

    def observe(self):
        geo, N, t = self.geo, self.N, self.geo.t
        g = geo.site_groups()
        committed = geo.committed_load()
        queued = np.array([geo.size[g['queue'][i], i].sum() for i in range(N)])
        hour = ((geo.utc0 + t) // 4 + geo.utc_offsets) % 24
        site = np.stack([self.ci[:, t] / 1000, self.amb[:, t] / 40, committed, queued,
                         [s.bat_env.get_battery_soc() for s in geo.sites], geo.capacity_mw / geo.capacity_mw.max(),
                         np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)], axis=1)
        fcst = np.stack([self.fc.window(i, t, 4 * self.H, 'nominal').reshape(self.H, 4).mean(1) / 1000 for i in range(N)])
        glob = np.concatenate([site, fcst], axis=1).reshape(-1).astype(np.float32)
        arr = geo.arrivals()
        slack = geo.latest_start[arr] - t
        jobs = np.concatenate([np.stack([np.log(geo.power_w[arr] / 1e3 + 1e-3), geo.dur[arr] / 26, slack / 13,
                                         geo.rigid[arr].astype(float)], axis=1),
                               geo.size[arr] * 20, np.eye(N)[geo.origin[arr]]], axis=1).astype(np.float32)
        mask = np.ones((len(arr), N), dtype=bool)
        r = np.flatnonzero(geo.rigid[arr])
        mask[r] = False
        mask[r, geo.origin[arr[r]]] = True
        return {'glob': glob, 'jobs': jobs, 'mask': mask, 'arr': arr}

    def step(self, actions):
        geo = self.geo
        geo.route(np.asarray(actions, dtype=np.int64) if len(actions) else None)
        _, _, _, trunc, info = geo.step({'route': None, 'starts': None, 'sites': None})
        co2 = sum(c['bat_CO2_footprint'] for c in info['sites']) / 1e3
        a = geo.active
        wait = a[(geo.status[a] == 1) & (geo.latest_start[a] < geo.t)]
        late = float(geo.size[wait, geo.site[wait]].sum()) if len(wait) else 0.0
        reward = -(co2 + self.late_kg * late)
        return self.observe(), reward, bool(trunc), {'co2_kg': co2, 'late': late}


class RouterNet(nn.Module):
    def __init__(self, obs_dim, job_dim, n_sites, hidden=256):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.job = nn.Sequential(nn.Linear(job_dim, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_sites))
        self.value = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def logits(self, glob, jobs, mask, step_idx=None):
        """glob (S, D) per step; jobs (J, F); step_idx (J,) step of each job (None: one step)."""
        c = self.ctx(glob)
        c = c[step_idx] if step_idx is not None else c.expand(len(jobs), -1)
        lg = self.head(torch.cat([c, self.job(jobs)], dim=1))
        return lg.masked_fill(~mask, -1e9)


class PPORouter:
    """Router interface (reset / route) around a trained RouterNet; deterministic (argmax) by default."""

    def __init__(self, net, cfg, greedy=True):
        self.net, self.cfg, self.greedy = net, cfg, greedy

    @classmethod
    def load(cls, run_dir, name='best', **overrides):
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        net = RouterNet(cfg['obs_dim'], cfg['job_dim'], cfg['n_sites'], cfg['hidden'])
        net.load_state_dict(torch.load(os.path.join(run_dir, f'{name}.pt')))
        net.eval()
        return cls(net, cfg)

    def reset(self, geo):
        if geo.n_sites != self.cfg['n_sites']:
            raise ValueError(f"router trained for {self.cfg['n_sites']} sites, env has {geo.n_sites}")
        self.env = JobRoutingEnv(geo, self.cfg['obs_hours'], self.cfg.get('sigma24', 0.0), self.cfg.get('rho', 0.3),
                                 self.cfg.get('fc_seed', 0))
        from geo_lp_oracle import exo_traces
        tr = exo_traces(geo, geo.horizon + 4 * self.env.H + 4)
        self.env.ci = np.stack([t['ci'] for t in tr])
        self.env.amb = np.stack([t['amb'] for t in tr])
        self.env.fc = ForecastModel(self.env.ci, [c['location'] for c in geo.site_configs], self.env.sigma24,
                                    rho=self.env.rho, seed=self.env.fc_seed)
        self.decision_seconds = []

    def route(self, geo):
        t0 = time.time()
        o = self.env.observe()
        if not len(o['arr']):
            self.decision_seconds.append(time.time() - t0)
            return np.zeros(0, dtype=np.int64)
        with torch.no_grad():
            lg = self.net.logits(torch.from_numpy(o['glob'])[None], torch.from_numpy(o['jobs']),
                                 torch.from_numpy(o['mask']))
        a = lg.argmax(1) if self.greedy else torch.distributions.Categorical(logits=lg).sample()
        self.decision_seconds.append(time.time() - t0)
        return a.numpy()


def collect(env, net, seed):
    """One episode with the stochastic policy. Returns per-step arrays and per-job arrays (with their step index)."""
    o = env.reset(seed)
    G, R, V, D = [], [], [], []
    J, M, A, LP, SI = [], [], [], [], []
    co2 = late = 0.0
    done, s = False, 0
    while not done:
        g = torch.from_numpy(o['glob'])[None]
        with torch.no_grad():
            V.append(float(net.value(g)))
            if len(o['arr']):
                jobs, mask = torch.from_numpy(o['jobs']), torch.from_numpy(o['mask'])
                dist = torch.distributions.Categorical(logits=net.logits(g, jobs, mask))
                a = dist.sample()
                J.append(jobs); M.append(mask); A.append(a); LP.append(dist.log_prob(a))
                SI.append(torch.full((len(a),), s, dtype=torch.long))
                acts = a.numpy()
            else:
                acts = []
        G.append(o['glob'])
        o, r, done, info = env.step(acts)
        R.append(r)
        co2 += info['co2_kg']
        late += info['late']
        s += 1
    cat = (lambda x, dt=None: torch.cat(x) if x else torch.zeros(0, dtype=dt))
    return {'glob': torch.from_numpy(np.stack(G)), 'rew': np.array(R), 'val': np.array(V),
            'jobs': cat(J), 'mask': cat(M), 'act': cat(A, torch.long), 'logp': cat(LP), 'step': cat(SI, torch.long),
            'co2_kg': co2, 'late': late}


def ppo_update(net, opt, batch, clip=0.2, epochs=4, minibatch=4096, vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95,
               rew_scale=1e-2):
    """batch: list of collect() outputs. Per-job clipped surrogate with the step's GAE advantage."""
    globs, rets, advs, jobs, masks, acts, logps, steps = [], [], [], [], [], [], [], []
    off = 0
    for ep in batch:
        r, v = ep['rew'] * rew_scale, ep['val']
        adv = np.zeros_like(r)
        last = 0.0
        for k in reversed(range(len(r))):
            nv = v[k + 1] if k + 1 < len(r) else 0.0
            delta = r[k] + gamma * nv - v[k]
            last = delta + gamma * lam * last
            adv[k] = last
        globs.append(ep['glob']); rets.append(torch.from_numpy(adv + v).float()); advs.append(torch.from_numpy(adv).float())
        jobs.append(ep['jobs']); masks.append(ep['mask']); acts.append(ep['act']); logps.append(ep['logp'])
        steps.append(ep['step'] + off)
        off += len(r)
    glob, ret, adv = torch.cat(globs), torch.cat(rets), torch.cat(advs)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    jobs, mask, act, logp_old, step = torch.cat(jobs), torch.cat(masks), torch.cat(acts), torch.cat(logps), torch.cat(steps)
    n = len(act)
    info = {}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, max(n, 1), minibatch):
            idx = perm[k:k + minibatch]
            dist = torch.distributions.Categorical(logits=net.logits(glob, jobs[idx], mask[idx], step[idx]))
            lp = dist.log_prob(act[idx])
            ratio = torch.exp(lp - logp_old[idx])
            a = adv[step[idx]]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            vl = ((net.value(glob).squeeze(1) - ret) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            info = {'pg': float(pg), 'vf': float(vl), 'entropy': float(ent)}
    return info
