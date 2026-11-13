"""Screening rule for capacity-weighted top-M selection (fluid market clearing), and the Phase-1 toy check.

Setting. Items a = 1..n (in the dispatcher: one segment of one site's index curve) have a width w_a (how much work
they can take) and an index nu_a(c) that depends on an uncertain cost vector c in a box C. Demand D is filled
cheapest-first (the fluid clearing): item a gets min(w_a, D - work of the items cheaper than a)^+.

Classification from index intervals nu_a(c) in [lo_a, hi_a] for c in C, and an optional "comparable" relation:
    possibly_cheaper(b, a)   b can be at least as cheap as a for some c in C
                             = lo_b <= hi_a by default (independent uncertainties);
                             for pairs with shared uncertainty, pass `cheaper[b, a]` from joint scenarios
    surely in    sum of w_b over b != a possibly cheaper than a, + w_a  <=  D     (filled for every c)
    surely out   sum of w_b over b != a surely cheaper than a           >=  D     (empty for every c)
                 (surely cheaper: not possibly_cheaper(a, b), e.g. hi_b < lo_a)
    boundary     everything else
Proposition (checked here and in tests/verify_screening.py): for every c in C the clearing fills surely-in items and
leaves surely-out items empty, so only boundary items depend on c. With linear indices nu_a = pi_a . c, the
measurements (pi_a - pi_b) . c for boundary pairs a, b decide the rest: at most |B| - 1 numbers instead of all of c.

Run `python -m sufficiency.top_m` for the toy: 5 unit items, M = 2, boundary {B, C}, one measurement c_B - c_C.
"""
import numpy as np


def clear(nu, w, D):
    """Fluid clearing: amount given to each item (cheapest first, ties by index)."""
    out = np.zeros(len(nu))
    left = float(D)
    for a in np.argsort(nu, kind='stable'):
        if left <= 0:
            break
        out[a] = min(w[a], left)
        left -= out[a]
    return out


def classify(lo, hi, w, D, cheaper=None, tol=0.0):
    """Labels per item: +1 surely in, -1 surely out, 0 boundary.
    cheaper[b, a] (bool, optional): b can be at least as cheap as a for some c (from joint scenarios, for pairs with
    shared uncertainty); it replaces the interval test lo_b <= hi_a. Ties count as possibly cheaper (the clearing may
    break them either way). tol: margin added to every interval test."""
    lo, hi, w = (np.asarray(x, dtype=float) for x in (lo, hi, w))
    poss = lo[:, None] <= hi[None, :] + tol              # poss[b, a]: b possibly <= a
    if cheaper is not None:
        poss = np.asarray(cheaper, dtype=bool).copy()
    np.fill_diagonal(poss, False)
    strict = ~poss.T                                     # strict[b, a]: a never <= b, so b < a for every c
    np.fill_diagonal(strict, False)
    pin = (poss * w[:, None]).sum(0) + w <= D + 1e-12
    pout = (strict * w[:, None]).sum(0) >= D - 1e-12
    lab = np.zeros(len(lo), dtype=int)
    lab[pin] = 1
    lab[pout & ~pin] = -1
    return lab


def check_by_sampling(nu_of_c, sampler, lab, w, D, n_samples=2000, rng=None):
    """Proposition check: for sampled c, surely-in items are filled and surely-out items are empty. Returns the
    number of violating samples and the set of items whose amount changed across samples."""
    rng = rng or np.random.default_rng(0)
    bad, changed, first = 0, set(), None
    for _ in range(n_samples):
        x = clear(nu_of_c(sampler(rng)), w, D)
        if np.any((lab == 1) & (x < w - 1e-9)) or np.any((lab == -1) & (x > 1e-9)):
            bad += 1
        if first is None:
            first = x
        changed |= set(np.flatnonzero(np.abs(x - first) > 1e-9).tolist())
    return bad, changed


def toy():
    """Spec Phase 1: arms A..E, nu_a = c_a, c_a uniform in a box; unit widths, M = 2."""
    names = 'ABCDE'
    lo = np.array([1.0, 3.0, 4.0, 8.0, 10.0])
    hi = np.array([2.0, 5.0, 6.0, 9.0, 11.0])
    w, D = np.ones(5), 2.0
    lab = classify(lo, hi, w, D)
    boundary = [names[a] for a in np.flatnonzero(lab == 0)]
    bad, changed = check_by_sampling(lambda c: c, lambda r: r.uniform(lo, hi), lab, w, D)
    # sufficiency of the single measurement c_B - c_C: given it, the selection is determined
    rng = np.random.default_rng(1)
    determined = True
    for _ in range(500):
        c1, c2 = rng.uniform(lo, hi), rng.uniform(lo, hi)
        c2[1] = c2[2] + (c1[1] - c1[2])          # same measurement c_B - c_C, other coordinates free
        if not (lo[1] <= c2[1] <= hi[1]):
            continue
        if np.any(clear(c1, w, D) != clear(c2, w, D)):
            determined = False
    return {'labels': dict(zip(names, lab.tolist())), 'boundary': boundary, 'violations': bad,
            'changed': sorted(names[a] for a in changed), 'measurement_c_B_minus_c_C_sufficient': determined}


if __name__ == '__main__':
    out = toy()
    print(out)
    assert out['boundary'] == ['B', 'C'] and out['violations'] == 0 and out['changed'] == ['B', 'C']
    assert out['measurement_c_B_minus_c_C_sufficient']
    print('toy: PASS')
