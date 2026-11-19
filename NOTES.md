# Sufficiency-screened index policy: statements, proofs and scope

Code: `geo_index.py` (site LP, index, routing), `sufficiency/` (forecast box, screening, measurements, exact
analysis). Checks: `tests/verify_screening.py`, `tests/verify_measurements.py`.

## 1. The routing decision

Every hour, the past hour's non-rigid arrivals of one slack class form a **bundle** b (`IndexPolicy._bundles`).
Each site i solves its planning LP with a share φ of the bundle added (`SiteLP`):

    V_i(φ, c) = min { c · g(x) + p(x) : x ∈ X_i(φ) }

- x: start shares of queued jobs and of the bundle, load, energy, battery flows over the window;
- g(x) ∈ R^W: grid energy per step (IT + cooling + battery charge − discharge);
- p(x): lateness and overflow penalties (no carbon intensity in them);
- X_i(φ): a polyhedron (capacity, deadlines, energy lines, battery dynamics), with φ on the right-hand side;
- c ∈ R^W: the site's carbon-intensity forecast over the window.

The curve is cut at φ_0 = 0 < φ_1 < … < φ_G = φ_max. **Item** a = (i, k) is segment k of site i, with width
w_a = φ_{k+1} − φ_k and index

    ν_a(c) = [V_i(φ_{k+1}, c) − V_i(φ_k, c)] / (w_a · work_b).

The **fluid clearing** fills demand D = 1 (one bundle) cheapest index first:
x*(c) = argmin { Σ_a ν_a(c) x_a : Σ_a x_a = D, 0 ≤ x_a ≤ w_a } (`top_m.clear`; ties may be broken either way).

**Uncertainty.** The forecast of region r lies in a box C_r = [ℓ_r, h_r] (`forecast.py`, a 90% interval per
step). Sites of one region share one c_r. C = Π_r C_r.

## 2. Screening is safe (Proposition)

Suppose that for every item a an interval [lo_a, hi_a] ∋ ν_a(c) for all c ∈ C is known, and a relation
`cheaper[b, a]` that is true whenever ν_b(c) ≤ ν_a(c) for some c ∈ C (it may be true more often). Define
(`top_m.classify`)

- **surely in**: Σ_{b ≠ a, cheaper[b,a]} w_b + w_a ≤ D;
- **surely out**: Σ_{b : ¬cheaper[a,b]} w_b ≥ D (not surely in);
- **boundary** B: the rest.

**Proposition.** For every c ∈ C and every tie-breaking of the clearing, x*_a(c) = w_a for surely-in items and
x*_a(c) = 0 for surely-out items.

*Proof.* Fix c. The clearing gives a its full width unless the items placed before it already use more than
D − w_a. Every item placed before a has ν_b(c) ≤ ν_a(c) (ties included), so cheaper[b, a] holds for it. For a
surely-in item their total width is at most D − w_a, so a gets w_a. For a surely-out item: ¬cheaper[a, b] means
ν_a(c) > ν_b(c) for every c ∈ C, so b is placed strictly before a whatever the tie-breaking. These b have total
width ≥ D, so nothing is left for a. ∎

The Proposition needs **no linearity**: it holds whenever the intervals and the relation are valid. Everything
approximate in the method is in how they are computed (Section 3). Only the boundary items' shares depend on c.

What it does not cover: the policy routes arriving jobs one at a time against the curves (streaming clearing,
`IndexPolicy.route`) with an admission test, not the fluid clearing of the forecast bundle; the Proposition is
about the latter.

## 3. When the intervals are exact

**Lemma (linear regime).** For a site and a segment k, let x_k and x_{k+1} be optimal solutions at φ_k and
φ_{k+1} for the nominal forecast. If x_k stays optimal at φ_k, and x_{k+1} at φ_{k+1}, for every c ∈ C, then on C

    ν_a(c) = π_a · c + κ_a,   π_a = [g(x_{k+1}) − g(x_k)] / (w_a work_b),   κ_a = [p(x_{k+1}) − p(x_k)] / (w_a work_b).

*Proof.* V(φ, c) = c · g(x*) + p(x*) with x* fixed on C, at both ends of the segment; subtract. ∎

π_a is the grid energy per step of the marginal work (when, and how much, the site draws for it, including what its
battery and cooling do in response); κ_a is its change of penalties. `ScreenedIndexPolicy.site_curves` computes
both from the solutions it already has.

**Corollary (exact intervals).** In the linear regime the range of ν_a over the box is

    lo_a = κ_a + Σ_t min(π_{a,t} ℓ_t, π_{a,t} h_t),   hi_a = κ_a + Σ_t max(π_{a,t} ℓ_t, π_{a,t} h_t),

and for two items on one grid (shared c), ν_b(c) ≤ ν_a(c) for some c ∈ C iff
κ_b − κ_a + Σ_t min((π_b − π_a)_t ℓ_t, (π_b − π_a)_t h_t) ≤ 0 (linear functions over a box reach their extremes
at a vertex chosen coordinate by coordinate). Items on different grids have independent boxes, so the interval test
lo_b ≤ hi_a is exact for them. This is `interval='linear'` in `screen.py`. The older `interval='scenario'`
(min / max over the nominal forecast and the two box corners) equals it when π_a ≥ 0 in every step, and can be
too narrow otherwise, e.g. when the battery's response gives π negative entries.

**Outside the linear regime** V(φ, ·) is concave and piecewise linear in c, and ν_a, a difference of two such
functions, is neither concave nor monotone, so no finite set of corners bounds it in general. The method then
falls back to the scenario rule, which is an approximation.

**The guard.** `_linear_ok` accepts an item when its nominal linear form reproduces the LP's index at every
scenario it solved (the box corners and `n_samples` random draws) within `lin_tol` (10⁻³). This is a necessary
condition for the lemma's premise, not a certificate: the solution can change at a point no scenario visits.
`tests/verify_measurements.py` measures how often that happens (Section 6).

## 4. The boundary needs |B| − 1 numbers (sufficiency)

**Theorem.** Let the boundary items be in the linear regime. Fix r ∈ B. The measurements

    m_a = (π_a · c_{r(a)} + κ_a) − (π_r · c_{r(r)} + κ_r),   a ∈ B \ {r},

determine x*(c) for every c ∈ C (up to ties among items of equal index): for c, c' ∈ C with equal measurements,
x*(c) and x*(c') give every item the same share, ties aside.

*Proof.* By the Proposition, surely-in and surely-out items have fixed shares, so the boundary items share the
remaining D' = D − Σ_in w_a. Their shares are the clearing of D' over B, which depends on c only through the order
of (ν_a(c))_{a ∈ B}. The order is decided by the signs of ν_a − ν_r and of their pairwise differences
m_a − m_b, all functions of the measurements. ∎

The measurements are linear in c (the κ's are constants), so the dataset is the set of directions
{π_a − π_r : a ∈ B \ {r}} (each placed in its region's block of c): |B| − 1 directions, of rank ≤ |B| − 1. Two
items on one grid share c, so their direction is (π_a − π_r) applied to one forecast; directions of items on one
grid are often nearly collinear and the rank drops well below |B| − 1.

**Relation to Bennouna et al.** For the clearing LP with cost P c + κ, the minimal sufficient dataset spans the
relevant directions P^T Δ(X, C), where Δ(X, C) are the edge directions e_j − e_i at optimal vertices whose tie face
meets C (Bennouna Thm 2; Ye Thm 3: span Δ = dir X*(C)). An edge moves mass between items i and j, so both shares
change at some c ∈ C; by the Proposition both are boundary items. Hence P^T(e_j − e_i) = π_j − π_i ∈
span{π_a − π_r : a ∈ B}, and

    d*_c := rank P^T Δ(X, C)  ≤  rank{π_a − π_r}  ≤  |B| − 1,

with the boundary differences a sufficient (generally not minimal) dataset. `sufficiency/exact.py` computes
X*(C), Δ(X, C), d* and Ye's Algorithm 2 exactly on the linearized decision (LPs over the box replace dstar.py's
ball formulas) and checks the inclusion and Ye's certificate with our dataset.

**In the policy** (`refine='measure'`): for each bundle, boundary items in the linear regime are keyed by the
measured differences (one number each, plus one absolute value of the reference when some boundary item failed
the guard and is compared through a refitted curve); items that failed the guard get their site's whole refined
forecast, as before (`refine='forecast'`). The information requested per hour is reported as
`scalars_per_hour` and `forecast_values_per_hour`.

## 5. Scope and limits

- **Uncertainty only in the objective**: carbon intensity. Weather, the energy model and the jobs are treated as
  known; the bundle (next hour's arrivals) is the past hour's arrivals.
- **Fluid relaxation**: the guarantees are for the fluid clearing of the bundle. Jobs are indivisible, routed one
  at a time and rounded by execution.
- **Linear regime**: needed for exact intervals and for the measurement theorem; checked by a finite guard
  (Section 3), measured in Section 6. Outside it, intervals come from scenarios (approximate).
- **The box**: per-step 90% intervals. Over a ~120-step window, a forecast often leaves the box at some step, and
  then the Proposition says nothing about it. Section 6 reports how often the refined forecast falls outside.
- **Modelling restrictions**: per-site queues (a routed job never moves); cooling as the best-setpoint envelope
  (ramp limit ignored in planning); battery executed as full-power actions when the plan moves ≥ 10% of its power.
- **Deviations from the plan** (`~/.claude/plans/in-this-new-branch-velvety-hearth.md`): streaming clearing instead
  of bisection on λ*_k; latest-start admission; failed LP solves give +∞ indices and unknown intervals; the
  boundary measurements are differences to a reference item (plus an anchor when needed); the exact Bennouna check
  uses the linearized decision with box uncertainty rather than dstar.py's MDP/ball code.

## 6. What has been checked

| Claim | Check | Result |
|---|---|---|
| Proposition (scenario intervals, real LPs) | `verify_screening.py`, n20, σ₂₄ 0.15 | 0 violations in 3712 (draw, item) pairs |
| Boundary ⊇ items of relevant edges; relevant directions ⊆ span of ours; Ye's certificate with ours | `verify_measurements.py` (`exact.py`), dccluster5 and n20 | 18/18 decisions (residual ≤ 6e-16, certificate at 40/40 sampled costs each) |
| d*_vert = d*_Δ in decision space (Ye Thm 3) | same | 18/18 |
| Linear form accuracy for guard-accepted items (guard: corners + 4 draws) | same, 4 random draws in the box per decision | 99.1–99.6% within 10⁻³, 100% within 10⁻², worst 2.3 × 10⁻³ (1744 item-draws) |
| Measurement keys route like LP re-solves at the refined forecast | same, per-site shares | 18/18 decisions (3 with the refined forecast inside the box, 15 outside it) |

## Runs

**`tests/verify_measurements.py`** (Slurm job 11756854; σ₂₄ = 0.1, 3 hours × 3 bundles per scenario, linearity guard
with 4 extra draws): 14/14 checks passed. Per decision (items, boundary |B|, d*_c, rank of our differences):

| Scenario | Decisions with a boundary | \|B\| range | d*_c vs \|B\| − 1 | Ye Alg. 2 (40 samples) |
|---|---|---|---|---|
| dccluster5 (20 items per bundle) | 7 of 9 | 1–8 | equal in 4, below in 3 (e.g. 1 vs 5, 4 vs 7: same-grid items share directions) | 0–3 directions |
| n20 (72–80 items per bundle) | 5 of 9 | 1–10 | equal in all (e.g. 9 vs 9 at t = 12) | 0–6 directions |

Observations:
- The boundary always contained every item of a relevant edge, and only a few more (|B| − 1 ≥ d*).
- Ye's Algorithm 2 with 40 sampled costs often returned fewer directions than d* (e.g. 6 vs 9): rare tie faces are
  seldom sampled. The screening's dataset needs no sampling and covers all of them.
- At 20 sites the refined forecast left the nominal box at some step in every decision, so the in-box routing check
  had no cases there; routing still matched LP re-solves in all 9.
- With the guard on corners only (no extra draws), 80% of accepted item-draws were within 10⁻³ and the worst error
  was 8.7 × 10⁻³ (dccluster5, 1 hour): the corner check alone misses some changes of solution inside the box.
- About half of all items pass the guard at σ₂₄ = 0.1 (the rest use the scenario rule and, on the boundary, a
  whole refined forecast).

**CO₂ and information** (`geo_scaling.py`, methods screen / screen_lin / screen_meas): submitted as `meas_n5_v1`
(dccluster5, 3 seeds, with index, index_full and the bound) and `meas_scale_v1` (n20, n50, n200; σ₂₄ 0.1 and 0.2;
2 seeds); results to be added.
