"""Carbon-intensity forecasts with errors, and the uncertainty box the screening works with.

Errors are multiplicative and grow with lead time; they are drawn per *region* (a site's location), so sites on the
same grid share the same forecast error. At issue time t0 (the start of the hour) and lead k = tau - t0 steps:

    nominal   c_hat[tau] = c[tau] * exp(sigma_k * z_k)          sigma_k = sigma24 * sqrt(k / 96)
    box       [c_hat * exp(-q sigma_k), c_hat * exp(+q sigma_k)]
    refined   c_ref[tau] = c[tau] * exp(rho * sigma_k * z_k)    the same error, shrunk (a better forecast product);
                                                                its box uses rho * sigma_k

z_k is a standard normal at every lead, correlated across leads (a normalized random walk: an error at 3 h persists
at 4 h). The current step (k = 0) is known exactly. q = 1.645 makes a 90% box per step. sigma24 = 0 gives perfect
forecasts (the box is then a point and nothing is uncertain).

Errors depend only on (seed, region, issue hour), so every method sees the same forecasts in the same episode.
"""
import zlib

import numpy as np

STEPS_PER_DAY = 96


class ForecastModel:
    def __init__(self, ci, regions, sigma24=0.10, q=1.645, rho=0.3, seed=0):
        """ci: (N, T) true CI per site; regions: (N,) region label per site (sites with one label share errors)."""
        self.ci = np.asarray(ci, dtype=float)
        self.regions = list(regions)
        self.sigma24, self.q, self.rho, self.seed = float(sigma24), float(q), float(rho), int(seed)
        self._z = {}

    def _errors(self, region, t0, n):
        key = (region, t0)
        z = self._z.get(key)
        if z is None or len(z) < n:
            rng = np.random.default_rng([self.seed, zlib.crc32(region.encode()), int(t0)])
            w = np.cumsum(rng.standard_normal(n))
            z = np.r_[0.0, w[:n - 1] / np.sqrt(np.arange(1, n))]  # k = 0 exact, then N(0, 1), correlated
            self._z[key] = z
        return z[:n]

    def sigma(self, n):
        return self.sigma24 * np.sqrt(np.arange(n) / STEPS_PER_DAY)

    def window(self, i, t, n, kind='nominal'):
        """CI over steps t .. t+n-1 for site i, issued at the start of t's hour. kind: 'true', 'nominal', 'refined',
        'low', 'high' (box corners around the nominal forecast), 'ref_low', 'ref_high' (around the refined one).
        Steps past the trace repeat its last value."""
        t0 = t - t % 4
        T = self.ci.shape[1]
        idx = np.minimum(np.arange(t, t + n), T - 1)
        c = self.ci[i, idx]
        if kind == 'true' or self.sigma24 == 0:
            return c
        lead = np.arange(t, t + n) - t0
        z = self._errors(self.regions[i], t0, int(lead[-1]) + 1)[lead]
        s = self.sigma(int(lead[-1]) + 1)[lead]
        if kind.startswith('ref'):
            s = self.rho * s
        c_hat = c * np.exp(s * z)
        if kind in ('nominal', 'refined'):
            return c_hat
        if kind in ('low', 'ref_low'):
            return c_hat * np.exp(-self.q * s)
        if kind in ('high', 'ref_high'):
            return c_hat * np.exp(self.q * s)
        raise ValueError(f'Unknown forecast kind {kind!r}')

    def sample(self, rng, i_list, t, n, refined=False):
        """One draw of the true CI given the forecasts, uniformly in the box of each region (shared within a region):
        {site: (n,) CI}. Used to test the screening rule."""
        out, per_region = {}, {}
        for i in i_list:
            r = self.regions[i]
            if r not in per_region:
                per_region[r] = rng.uniform(-1.0, 1.0, n)
            lo = self.window(i, t, n, 'ref_low' if refined else 'low')
            hi = self.window(i, t, n, 'ref_high' if refined else 'high')
            u = per_region[r]
            out[i] = np.exp(np.log(lo) + (u + 1) / 2 * (np.log(hi) - np.log(lo)))
        return out
