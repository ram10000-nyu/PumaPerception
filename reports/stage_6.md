# Stage 6: pose-graph optimisation

Plan revision in force: **revision 9**. Implemented by the user's Claude session on the Linux workspace (system Python; no simulator needed), from the stage 4b odometry and the stage 5b closures on `gt4`.

Status: **every criterion passes except scale, which misses by a hair.**
- Joint trajectories of all three drives: rigid-aligned ATE **0.013 to 0.027 m** on test (target ≤ 0.08 m), down from 0.060 to 0.120 m.
- Heading RMS ≤ 0.72°.
- Every false closure is inactive, including 5 injected random closures and all 17 real stage 5 aliases.
- **Scale:** the median drive is 0.55 % small (target 0.5 %), and 14 of 15 drives are within 1 % (worst 1.07 %).
- **One test-seed fix disclosed:** a reweighting order bug, found on test seed 22. Details below.

## Built

| file | what |
|---|---|
| `source/mapping/posegraph.py` | **pipeline**: SE(2) graph; vectorised whitened residuals and analytic sparse Jacobian; `least_squares` (trf, lsmr) |
| `source/mapping/stage6.py` | builds the graph (odometry, spawn prior, absolute heading priors, stage 5b union closures thinned), robust IRLS on loop edges, two-pass χ² gating, evaluation (reads ground truth only to score), robustness injections |
| `tests/mapping/test_posegraph.py` | toy square loops (plan criteria) and a numerical Jacobian check; 3 pass |
| `cache/mapping/<seed>/poses_optimized.json` | optimised poses, all 9 seeds, with the frozen parameters |
| `cache/mapping/stage6/results_gt4_{dev,test,test_random,test_alias}.json` | every number below |

Reproduce: `python3 source/mapping/stage6.py run dev|test [--inject random|alias]`.

## The graph

- **Nodes:** every frame of the three drives (2,856 to 3,600 per seed), and a fixed **spawn** anchor (0.2, 2.6, 0).
- **Odometry edges:** the relative pose between consecutive stage 4b variant C poses. These are heading-corrected, with stage 4b's increment covariance **inflated 16×**. Its stated per-frame uncertainty (≈ 1.5 mm, 0.3°) is far smaller than the real drift of about 10 cm per drive. At 1× the closures could not move the graph (ATE unchanged); at 64× one dev false closure took over (heading error 16°). 16 was best on dev.
- **Spawn prior:** first frame of each drive relative to the spawn.
- **Absolute heading priors**, every 10th frame: the filter's heading, which the stage 3 compass makes drift-free, σ = max(filter σ, 1°). Without them the map was internally consistent but rotated about the spawn (up to 1.07° on dev seed 10, 10 cm at 6 m). With them the residual rotation is at most 0.21° on dev and 0.12° on test.
- **Loop closures:** the stage 5b union rule (gate without contradiction, or boundary-verified), method (c) pose and covariance, thinned to one per (drive, 5-frame window) pair. That gives 310 to 453 per seed, about 20 % of them boundary closures.

**Solver.** The robust part is iteratively reweighted least squares (Cauchy, c = 3 on whitened residuals, 4 rounds) on **loop edges only**, so odometry stays Gaussian. Two passes:
1. Pass 1 optimises.
2. Every closure is re-gated against the pass-1 solution: χ² against its covariance plus a floor of 2 cm / 2 cm / 1°, at the 3-dof 99 % bound (11.34).
3. Pass 2 optimises with the survivors.

Runtime is 4.5 to 10 s per seed.

**Parameters, chosen on dev and frozen before test:** inflation 16, heading prior every 10 frames with a 1° floor, c = 3, 4 rounds, gate floor 2 cm / 1°, thinning window 5.

## Results

| seed | rigid ATE before → after [m] | raw ATE after (spawn frame) [m] | residual rotation | heading RMS | closures (boundary) | dropped (false / true) |
|---|---|---|---|---|---|---|
| 7 (dev) | 0.086 → **0.020** | 0.025 | −0.01° | 0.70° | 310 (49) | 0 / 0 |
| 8 (dev) | 0.100 → **0.040** | 0.045 | 0.05° | 0.29° | 361 (82) | 1 / 0 |
| 9 (dev) | 0.095 → **0.038** | 0.055 | 0.13° | 0.54° | 348 (52) | 0 / 0 |
| 10 (dev) | 0.072 → **0.018** | 0.038 | 0.21° | 0.31° | 445 (59) | 0 / 0 |
| **21** | 0.120 → **0.027** | 0.037 | 0.12° | 0.37° | 363 (66) | 1 / 0 |
| **22** | 0.067 → **0.017** | 0.043 | 0.10° | 0.29° | 453 (80) | 3 / 0 |
| **23** | 0.060 → **0.013** | 0.036 | −0.04° | 0.31° | 320 (60) | 0 / 0 |
| **24** | 0.094 → **0.022** | 0.032 | 0.06° | 0.42° | 441 (80) | 4 / 0 |
| **25** | 0.086 → **0.025** | 0.031 | 0.04° | 0.72° | 310 (55) | 0 / 0 |

Every false closure in the union set is dropped by the second pass on every seed, including seed 22's 3.6 m boundary false closure (χ² 23,190), and no true closure is dropped. "Before" is the stage 4b odometry: a 3- to 6-fold reduction.

**Robustness** (test):

| injection | false closures | active after two passes | raw ATE change, worst seed |
|---|---|---|---|
| 5 random closures per seed (pose within ±0.4 m, ±0.4 rad, tight covariance) | 5 + union false | **0** | +10 % (seed 25: 0.031 → 0.034 m) |
| all 17 real stage 5 aliases (the 0.6 m clusters) | 5 (seed 22), 12 (seed 24) | **0** | +12 % (seed 24: 0.032 → 0.036 m) |

**Scale on loop-closed trajectories** (revision 8 criterion):
- Similarity-alignment scale of each drive, test: median **+0.55 %**, meaning the estimate is 0.55 % small. **14 of 15** drives within 1 %, worst +1.07 % (seed 21 drive 1).
- Dev: median +0.55 %, 7 of 12 within 1 %, worst +1.3 %.
- Straight-segment ratio, test: median **−0.49 %** (n = 9; stage 4b −0.64 %). Dev: −0.26 % (n = 11).
- The map is therefore about half a percent small: **1.5 cm at 3 m**, under one 5 cm map cell. **Stage 7 can remove it exactly**: the wall lattice is 0.2 m by construction (a simulator constant, rule 10 allows it), so the detected lattice spacing gives the global scale.

## Toy tests (plan criteria)

- (a) Exact square from a perturbed start converges within 1e-6: **pass**.
- (b) 0.1 rad drift on one edge with an exact closure: the plan's "closure residual below 1e-3" was a wrong expectation. With equal weights, least squares spreads the drift over all four edges, the closure included (each about 0.025 rad). The test checks the spread (no edge carries more than half) and the closure's share (under 1/3 of the drift): **pass**.
- The analytic Jacobian matches the numerical one to 1e-4.

## Disclosure: a fix found on a test seed

The first test run failed on seed 22 (rigid ATE 0.067 → 0.172 m): **two closures of a stage 5 alias cluster stayed active.** The cause was an ordering bug in the reweighting:
- Round 1 solved with *full* weight on every closure before any weight was computed.
- So a mutually consistent pair of false closures, χ² ≈ 430 against the odometry, pulled the graph to itself.
- Every later round then saw them as consistent (χ² 3 to 5).

**Fix:** compute the first weights at the starting estimate (the odometry), before solving. This is the standard IRLS order, not a tuned parameter.
- Dev results were **identical** before and after the fix (no aliases on dev).
- The test numbers above are the rerun.
- Seed 22's other false closure, the 3.6 m boundary one, was dropped by the second pass in both runs.

No other change was made after looking at test.

## Acceptance

| criterion | result | |
|---|---|---|
| Toy-loop unit tests | (a) pass; (b) restated (spread, not a 1e-3 closure residual) and passes | pass |
| Optimised ATE ≤ 0.08 m on every test seed (after one rigid alignment) | 0.013 to 0.027 m | pass |
| Scale: median drive within 0.5 %, every drive within 1 % | +0.55 %; 14 / 15 (worst 1.07 %) | **fail, marginally**; stage 7 lattice scale proposed |
| Straight-segment ratio reported with CI | −0.49 % (n = 9) | reported |
| Optimised heading RMS ≤ 2° | ≤ 0.72° | pass |
| Robustness: 5 injected false closures, ATE +25 % at most | +10 % at most; none active | pass |
| Real stage 5 aliases injected; none active after two-pass gating | 17 of 17 dropped; +12 % at most | pass |
| Runtime < 30 s per maze | 4.5 to 10 s | pass |
| Two-pass gating, thinning (revision 9 tasks) | done | done |

## Not done

- `official` (`gt`) frames were not run: boundaries were only extracted on `gt4`.
- NIS of the optimised graph was not computed.
- The scale correction is proposed for stage 7, not applied here.

## For stage 7

- Use `cache/mapping/<seed>/poses_optimized.json`.
- Expect about 0.5 % shrink; calibrate the global scale from the detected 0.2 m lattice before snapping walls.
- Residual position error is about 2 to 4 cm (raw, spawn frame), smaller than one 5 cm cell, so the occupancy map should approach the stage 2 oracle maps (IoU about 0.80 to 0.87 against the true layout).
