# Stage 1: action labels as a weak prior, and the dead-reckoning reference

Status: **3 of 4 acceptance criteria pass; the dev-vs-test mean-agreement criterion fails marginally** (details below, thresholds not loosened). Prior fitted on dev seeds 7 to 10 (12,490 consecutive-frame pairs), checked on test seeds 21 to 25 (16,347 pairs).

## Built

| file | what |
|---|---|
| `source/mapping/se2.py` | wrap, compose, inverse, relative, analytic Jacobians, numerical Jacobian, integrate |
| `source/mapping/motion.py` | `load_pairs` (true relative motion per pair with labels), `MotionPrior` (fit, save, load, `get(label, prev)`, `mean_increments`) |
| `source/mapping/stage1.py` | fits on dev, evaluates on test, dead-reckons, writes figures |
| `cache/mapping/motion_prior.json` | per-label and per-(previous, current) mixtures: weights, means, covariances, moment mean/cov, 95 % density threshold, sample counts, dev seeds |
| `cache/mapping/stage1_results.json` | every number below |
| `tests/mapping/test_se2.py`, `test_motion.py` | 9 + 8 tests; `uv run pytest`: 41 passed |

Reproduce: `uv run source/mapping/stage1.py`.

## What the data looks like

The true relative motion `(dx, dy, dtheta)` of a pair, in the first pose's frame, given the label of the pair's record (dev seeds):

| label | n | mean dx [m] | std dx | mean dtheta [rad] | std dtheta |
|---|---|---|---|---|---|
| FORWARD | 4,012 | 0.112 | 0.046 | +0.002 | 0.076 |
| LEFT | 3,368 | 0.022 | 0.039 | +0.219 | 0.081 |
| RIGHT | 3,122 | 0.022 | 0.038 | -0.219 | 0.081 |
| IDLE | 1,988 | 0.002 | 0.011 | +0.012 | 0.206 |

`dy` is essentially zero (std 0.0007 to 0.003 m): the robot is non-holonomic. The distributions are **strongly non-Gaussian**. The samples sit on a lattice (`stage1_label_scatter.png`): `dtheta` in multiples of 0.06 rad and `dx` in multiples of 0.03 m, i.e. the number of turning and forward ticks in the 5-tick window, plus a thin tail of collision events (rotations up to 0.9 rad under a FORWARD or IDLE label, found in stage 0). A label of LEFT is mostly a window of 1 to 5 left ticks, sometimes several forward ticks first. The mean of a label is therefore not a motion that ever happens; this is what makes label-only dead reckoning so poor (below).

## Prior model

- **One mixture per group, at most 3 Gaussian components**, chosen by BIC on `dx, dy, dtheta` scaled by `(0.004, 0.001, 0.01)` with a variance floor of one scale unit (so the near-constant `dy` and the quantised angles never give a singular covariance). The plan's alternatives were a mixture or an inflated covariance; a single Gaussian cannot describe a lattice with tails.
- **95 % region = the 95 % highest-density region of the mixture** (threshold at the 5th percentile of dev log-densities). This is what the calibration criterion tests.
- **Moments for the pose graph.** Each group also stores the mixture's mean and covariance, inflated by `max(1, p95(Mahalanobis^2) / 7.81)` on dev so that its 95 % ellipse covers 95 % of dev samples. The inflation came out 1.0 for every group except FORWARD>IDLE (1.79): the mixture moments are already wide, so the ellipse is conservative (it covers 96.8 to 97.9 % of test pairs). Stage 6 should use the mixture density where it can, and these moments where it needs a single Gaussian.

## Does the previous label add information? Yes, a lot (kept)

Per current label, the covariance determinant of the (previous, current) groups against the label alone, weighted geometric mean over groups with at least 30 samples:

| label | determinant ratio | shrink |
|---|---|---|
| FORWARD | 0.0101 | 99.0 % |
| LEFT | 0.0270 | 97.3 % |
| RIGHT | 0.0287 | 97.1 % |
| IDLE | 5e-6 | 99.9995 % |

All pass the 30 % bar by a wide margin, so the prior is conditioned on the previous label (`MotionPrior.get(label, prev)`; a pair with fewer than 100 dev samples, including the very first record after the spawn, falls back to the label-only mixture). The determinant overstates the practical gain, because it is dominated by `dy` and `dx`. Pooled within-group standard deviations show where the information is:

| label | std dx none -> prev | std dtheta none -> prev | with the *next* label too |
|---|---|---|---|
| FORWARD | 0.046 -> 0.019 | 0.076 -> 0.066 | 0.019 / 0.066 |
| LEFT | 0.039 -> 0.028 | 0.081 -> 0.075 | 0.027 / 0.071 |
| RIGHT | 0.038 -> 0.027 | 0.081 -> 0.075 | 0.027 / 0.072 |
| IDLE | 0.011 -> 0.009 | 0.206 -> 0.115 | 0.004 / 0.108 |

The previous label mostly explains a label's translation (a FORWARD after a FORWARD is a full 0.149 m, after a turn only 0.059 m) and splits IDLE's heading into "still turning left", "still turning right" and "stationary" (an IDLE after a LEFT turns by +0.205 rad). The heading within a group stays at 0.07 to 0.12 rad (4 to 7 degrees). Conditioning on the next label as well (available offline) gains little (only IDLE's `dx` improves noticeably) and was **not adopted**: the plan specifies the previous label only.

## Calibration on the test seeds

Fraction of true test motions inside the prior's 95 % region, the prior fitted on dev only (target 0.90 to 0.99):

| label | n | (prev, current) prior | label-only prior | moment ellipse (chi-square 95 %) | worst (prev, current) group |
|---|---|---|---|---|---|
| FORWARD | 5,092 | **0.934** | 0.943 | 0.977 | 0.929 (after RIGHT) |
| LEFT | 4,205 | **0.935** | 0.937 | 0.968 | 0.923 (after IDLE) |
| RIGHT | 4,337 | **0.948** | 0.950 | 0.973 | 0.942 |
| IDLE | 2,713 | **0.939** | 0.943 | 0.979 | 0.902 (after FORWARD, n = 328) |

Dev (in-sample) is 0.949 to 0.950 by construction. Every label and every group is inside 0.90 to 0.99, so the prior is neither overconfident nor uselessly wide; the drop from 0.95 to 0.93 to 0.95 on unseen seeds is what a slight overfit looks like. IDLE after FORWARD has the smallest margin.

## Dev vs test means (not seed-specific?)

z = (test mean - dev mean) / combined standard error, per component `(dx, dy, dtheta)`. The naive error treats pairs as independent; the clustered one uses the spread of per-drive means (pairs in a drive are correlated, so it is the more honest of the two).

| label | z naive | z clustered | dev -> test mean difference (dx [m], dtheta [rad]) |
|---|---|---|---|
| FORWARD | -1.0, -0.2, **-2.1** | -0.8, -0.2, -1.9 | -0.0009, -0.0038 |
| LEFT | 0.6, -0.4, -1.2 | 0.6, -0.5, -1.4 | +0.0005, -0.0022 |
| RIGHT | 0.0, -0.7, 0.3 | -0.1, -0.8, 0.3 | 0.0000, +0.0005 |
| IDLE | 2.0, 1.5, -2.0 | **2.2**, 1.6, -1.7 | +0.0007, -0.0121 |

**Criterion not met as written:** FORWARD's `dtheta` is 2.1 naive standard errors off, and IDLE's `dx` 2.2 clustered (IDLE's `dx` and `dtheta` sit at 2.0 naive). I did not loosen it. In physical terms the differences are at most 0.9 mm in translation and 0.012 rad (0.7 degrees) in heading, an order of magnitude below the within-group spread, and with 12 comparisons about one exceeding 2 by chance is expected; the clustered values exceed it only once. The conclusion I draw is that the prior is not seed-specific in any way that matters, but the statistical statement of the criterion is not satisfied.

## Dead-reckoning reference (no threshold)

Prior means integrated from the true spawn (0.2, 2.6, 0) over each whole drive; position error is RMS against truth with no alignment (the start is known). Cells are ATE / final error [m] / heading RMS [degrees], per drive 0, 1, 2, for the (previous, current) prior means. Dev rows use a prior fitted on the same seeds (in-sample).

| seed | drive 0 | drive 1 | drive 2 | label-only means: ATE m / heading deg per drive |
|---|---|---|---|---|
| 7 (dev) | 1.93 / 2.31 / 60 | 3.27 / 4.29 / 96 | 3.88 / 3.93 / 117 | 3.30/78, 4.42/89, 3.93/111 |
| 8 (dev) | 3.94 / 2.89 / 103 | 3.03 / 1.49 / 94 | 4.45 / 6.26 / 84 | 4.10/105, 6.59/85, 4.58/106 |
| 9 (dev) | 4.27 / 8.84 / 106 | 2.01 / 5.29 / 78 | 10.71 / 14.00 / 88 | 2.80/53, 4.84/106, 8.89/109 |
| 10 (dev) | 1.19 / 2.24 / 37 | 5.72 / 9.60 / 103 | 3.26 / 4.97 / 76 | 2.14/66, 7.13/92, 6.16/131 |
| 21 | 4.06 / 9.32 / 86 | 4.00 / 8.36 / 115 | 5.05 / 6.41 / 66 | 3.06/100, 3.57/101, 9.19/77 |
| 22 | 3.57 / 3.09 / 104 | 2.86 / 1.61 / 82 | 4.04 / 6.84 / 69 | 3.52/122, 2.02/66, 9.24/106 |
| 23 | 1.81 / 0.65 / 36 | 3.90 / 3.51 / 112 | 2.18 / 4.48 / 55 | 6.51/75, 4.13/114, 8.03/89 |
| 24 | 2.32 / 5.33 / 57 | 4.05 / 9.52 / 89 | 7.53 / 6.76 / 90 | 6.75/105, 6.03/89, 8.88/118 |
| 25 | 4.41 / 8.00 / 94 | 1.86 / 1.96 / 39 | 4.51 / 11.01 / 78 | 5.06/93, 7.89/121, 5.35/100 |

On the test seeds the median drive has an ATE of **4.0 m** (range 1.8 to 7.5) and a heading RMS of **82 degrees** with the (previous, current) means, against 6.0 m and 100 degrees with label-only means. The maze is 6.2 m across: **dead reckoning from the labels is useless as odometry**; the heading error is tens of degrees RMS over a drive. This is the number stage 4's visual odometry must beat, and it confirms the revision-2 decision to use labels only as a weak prior. The previous-label conditioning improves the median ATE by a third but does not change that conclusion. It is a reference, not a method.

## Acceptance

| criterion | result | pass |
|---|---|---|
| SE(2) unit tests: `inv(a) (+) a = identity`, associativity, wrapping at +-pi, numerical vs analytic Jacobians to 1e-6 | all pass (9 tests, 30 random poses per Jacobian test, also near the +-pi cut) | pass |
| Test-seed calibration, every label 0.90 to 0.99 | 0.934, 0.935, 0.948, 0.939 (every (previous, current) group also inside) | pass |
| Per-label dev vs test means within 2 standard errors | max abs z: 2.1 naive (FORWARD dtheta), 2.2 clustered (IDLE dx); differences at most 0.9 mm and 0.7 degrees | **fail (marginal)** |
| Dead-reckoning ATE and heading RMS reported per seed | table above | reported |

## Figures

`stage1_label_scatter.png` (per label `dx` vs `dtheta`, dev and test, points outside the 95 % region in red, mixture component means as crosses, moment mean as a plus), `stage1_pair_scatter.png` (the same per (previous, current) group on test seeds), `stage1_dead_reckoning_<seed>.png` for all nine seeds (dead-reckoned vs true trajectory, position error and heading error against frame index).

## Open issues and notes for later stages

- The mean-agreement criterion is statistically marginal, physically negligible. Decision for the reviewer: accept as is, or re-state the criterion with a tolerance in metres and degrees.
- The mixtures reproduce the quantised structure only approximately (three smooth components for a lattice). If stage 6 wants sharper constraints from a label, a direct model of the window composition (number of turning and forward ticks given the labels around it) would be tighter than the heading spread of 4 to 7 degrees seen here. Not needed unless stage 4 turns out to lean on the prior.
- The tail from wall collisions (about 0.3 % of pairs rotate over 30 degrees under a FORWARD or IDLE label) is inside the mixtures' 95 % region only partly; the rest is outside it by construction. Stage 4 must not rely on this prior to catch those rotations (see the stage 0 addendum).
- The prior was fitted on local recordings made without `motion_bias`. The challenge server's bias, if any, is unknown, and would shift these means.
