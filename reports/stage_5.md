# Stage 5: loop closures

Plan revision in force: **revision 8** (read from disk; it names stage 5 as next).

Status: **two of four criteria pass.** The closures are precise and accurate.
- Precision is **0.997** on `gt4` test (17 false of 6,594), counted by the plan's own revisit definition.
- The median relative-pose error is **2.1 mm and 0.22°**.

Two criteria fail:
- **Pair recall** is 0.086 against ≥ 0.25. Only about 20 % of ground-truth revisit pairs ever become candidates. Per-frame recall is 0.34.
- **Drive coverage:** on test seed 23, drives 0 and 1 have no direct closure. They barely overlap (119 ground-truth revisit pairs, all near the spawn, at most 8 inliers); both join drive 2.

The main finding: **appearance alone cannot make closures here.** The 200 wall textures repeat across a maze. A third of dev pairs with 30 or more homography inliers are 1 to 5 m apart, and they have as many inliers as true revisits, so no inlier threshold separates them. Closures are therefore gated against the odometry, which shares one frame across drives because every drive starts at the spawn.

## What was built

| file | what |
|---|---|
| `source/mapping/loops.py` (pipeline) | wall-mask SIFT features; VLAD candidates (the baseline's `vlad.VLADExtractor` encoding, fixed vocabulary, PCA whitening); RANSAC-homography verification; relative pose by (a) homography decomposition, (b) symmetric scan ICP, (c) wall-feature points with two-scan depth; revisit envelope; odometry gate; `measure`, `find_closures` |
| `source/mapping/stage5_extract.py` | per-frame features cache (`cache/mapping/stage5/`, keyed by a code hash) |
| `source/mapping/stage5.py` | evaluation: vocabulary fit, candidates and verification, pose methods, ground truth, PR sweep, figures |
| `cache/mapping/stage5_vocab.npz` | vocabulary (64 words) and PCA whitening (256 dims), fitted on dev `gt4` (every 3rd frame, 4,168 frames); a property of the texture pack, not of a maze |
| `tests/mapping/test_loops.py` (+ `loops.py` in the rule 10 test) | 11 tests; `uv run pytest`: 153 pass; ruff clean |

Reproduce:
```
uv run python source/mapping/stage5_extract.py gt4 official
uv run python source/mapping/stage5.py vocab
uv run python source/mapping/stage5.py verify gt4 && uv run python source/mapping/stage5.py evaluate gt4 && uv run python source/mapping/stage5.py plots gt4
(same for official)
```
Numbers are in `cache/mapping/stage5_results_<variant>.json`; figures exist for both frame types (`stage5_*_gt4.png`, `stage5_*_official.png`). The odometry used by the gate is the stage 4b variant C output (calibrated priors, `stage4b_results_<variant>.pkl`), which is pipeline output.

## Method, and what changed from the plan

1. **Candidates (task 1).** Each frame's 20 most similar frames of any drive by VLAD, excluding its own drive within 10 frames. **Deviation:** the plan says 5. Twenty was chosen on dev: closures +32 %, pair recall 0.078 → 0.097, precision unchanged. This gives about 38,000 candidate pairs per seed.
2. **Verification (task 2).** RootSIFT ratio test (0.8), RANSAC homography (3 px). The threshold sweep is below. **Chosen: 20 inliers on dev.** Above 15, dev precision is 1.000 at every threshold. Twenty keeps 0.35 of revisit frames with a margin over 15.
3. **Relative pose (task 3)**, all three methods on dev and test. Accepted test closures (`gt4`, error against the true relative pose):

   | method | n | median | p95 |
   |---|---|---|---|
   | (a) homography decomposition, scale from the scan | 4,088 (fails on 38 %) | 5.3 mm, 0.61° | 49 mm, 5.8° |
   | (b) symmetric scan ICP, initialised from (c) | 6,587 | 2.4 mm, 0.14° | 50 mm, 0.72° |
   | **(c) wall features as metric points, depth from each frame's own scan** (the stage 4b fix) | 6,594 | **2.1 mm, 0.22°** | **16 mm, 1.57°** |

   **(c) is kept** as the measurement. (b) must agree with it within 0.1 m and 5° whenever (b) exists, the plan's extra check.
   - Method (c) is an addition to the plan's (a) and (b). It uses the same homography inliers but measures depth instead of inferring it.
   - Method (a) needed one fix found by a synthetic unit test: of the decomposition's two physically valid solutions, the one whose plane normal matches the scan's wall normal is the right one.
4. **Covariance (task 4):** method (c)'s point-to-point Hessian, scaled by the residual. On dev, 99.5 % of accepted closures have NIS under the 3-dof 95 % bound, so the covariance is conservative. Inflation is kept at 1.0.
5. **Two additions not in the plan, both needed and both chosen on dev:**
   - **Revisit envelope.** A pair is claimed as a closure only if its *measured* relative pose is within 0.45 m and 27°, a margin inside the plan's revisit definition (0.5 m, 30°). Without it, about 27 % of accepted pairs are correct wide-baseline matches that the plan's precision metric counts as false.
   - **Odometry gate.** A closure must agree with the odometry's relative pose within 0.5 m and 10°. Odometry drift stays under 0.25 m (stage 4b); on dev, 99 % of correct closures are within 0.34 m of the odometry and 99.9 % within 0.55 m. This is what removes the aliases. Without it, precision is 0.4 to 0.6 at any threshold (`stage5_pr_gt4.png`, faint curves).

## Results (`gt4` primary; threshold 20)

Precision-recall over the threshold: `stage5_pr_gt4.png`. Pose-error histograms: `stage5_pose_errors_gt4.png`. Closures on the true trajectories: `stage5_closures_gt4.png`.

| | dev `gt4` | **test `gt4`** | test `gt` | threshold |
|---|---|---|---|---|
| accepted closures | 5,510 | **6,594** | 5,405 | |
| precision (closure is a ground-truth revisit, plan definition) | 0.9998 | **0.9974** (17 false) | 0.9981 (10 false) | ≥ 0.98 |
| precision (pose within 0.3 m and 10° of the truth) | 1.0000 | **0.9974** | 0.9980 | |
| recall, ground-truth revisit pairs | 0.097 | **0.086** | 0.071 | ≥ 0.25 |
| recall, frames with a revisit that get a closure | 0.348 | **0.344** | 0.304 | |
| relative-pose error, median | 2.4 mm, 0.24° | **2.1 mm, 0.22°** | 2.7 mm, 0.30° | < 0.08 m, < 2° |
| drive pairs joined, per seed | all 3 on all 4 | **all 3 on 4 of 5** (seed 23: 0–2, 1–2) | all 3 on 4 of 5 (seed 23 as `gt4`) | every pair |

Per test seed (`gt4`):

| seed | closures | false | recall pairs / frames | drive pairs |
|---|---|---|---|---|
| 21 | 1,290 | 0 | 0.092 / 0.326 | 0–1, 0–2, 1–2 |
| 22 | 1,601 | 5 | 0.085 / 0.378 | 0–1, 0–2, 1–2 |
| 23 | 936 | 0 | 0.063 / 0.280 | 0–2, 1–2 |
| 24 | 1,613 | 12 | 0.101 / 0.417 | 0–1, 0–2, 1–2 |
| 25 | 1,154 | 0 | 0.090 / 0.314 | 0–1, 0–2, 1–2 |

## The failures

**Pair recall (0.086 < 0.25).** A frame has on average about 9 ground-truth revisit partners (within 0.5 m and 30°, across all drives).
- **Candidates cap it.** Only about 21 % of revisit pairs ever become candidates even at 20 per frame (seed 7: 2,526 of 11,729).
- **Verification caps it further.** About half of those reach 20 inliers (1,275), because many revisit pairs look at different walls or at open floor.
- **Recall is flat in the threshold.** At 10 inliers, ungated, pair recall is still only 0.16.

Reaching 0.25 would need several times more candidates per frame, for closures that repeat the same constraint. Per-frame recall (a third of revisiting frames get a closure) is the measure that matters for a pose graph, and it is reported next to the plan's.

**Drive coverage (seed 23, drives 0 and 1).** The two drives share only 119 revisit pairs, all in their first 30 to 120 frames near the spawn; the best of them has 8 inliers. Both drives are joined to drive 2, so the graph is connected. All three are also tied to the spawn by stage 6's spawn constraint. The criterion fails as written.

## The false closures (disclosed: test seeds)

All 17 false test closures (seeds 22 and 24) have one pattern. The same arrangement of textures recurs about 0.4 to 0.6 m away, and the rotation is right. The measured pose says "same spot" while the odometry predicts a 0.40 to 0.45 m offset, inside the 0.5 m gate.

A 0.35 m gate would remove all of them. I found this by looking at test, so the dev-chosen gate (0.5 m) stays. These are the closures stage 6's robust kernel has to absorb. Its 5-false-closure robustness test is the right check, and real aliases of this kind could be used as the injected set.

I also looked at test seed 23's drive pair for the coverage diagnosis; nothing was changed because of it.

A bug found on the `official` run: method (a) could return a non-finite pose, which crashed ICP initialisation when (c) had failed. Both are now guarded. It does not change any `gt4` number: that run never hit the case.

## Open issues

- **The gate depends on odometry drift staying under about 0.4 m.** On these drives it does: ATE max 0.24 m. On longer drives, or on the challenge data's per-tick frames (stage 9), a fixed gate may reject true closures or admit aliases. A gate scaled by the odometry covariance, or a two-pass scheme (gate, optimise, re-gate), would be safer.
- **Alias closures at 0.4 to 0.6 m** pass the gate (17 in 6,594 on test). Stage 6 needs a robust kernel or switchable constraints.
- **Pair recall** as a criterion is bounded by the candidate design. The plan may want frame recall instead.
- Closures are many and redundant (about 1,300 per seed). Stage 6 may thin them, for example one per frame pair cluster, to keep the graph small.
