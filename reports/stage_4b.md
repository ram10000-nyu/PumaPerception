# Stage 4b: holdout seeds, forward scale bias, prior self-calibration

Plan revision in force: **revision 7** (read from disk; it names stage 4b as next).

Status: **mostly done; two scale criteria and one calibration criterion fail.**
- Holdout seeds are recorded and guarded.
- The forward bias has a mechanism and a fix.
- Every stage 4 criterion passes on `gt4` test, and the trajectories improve: 10-frame p95 translation error goes from 6.87 % to 4.64 %, and median ATE from 0.12 m to 0.094 m.
- The self-calibration recovers injected biases from 0.44× to 1.29×.
- **Fail:** the straight-segment ratio is −0.64 % (limit ±0.5 %).
- **Fail:** the per-drive similarity scale is within 1 % on only 10 of 15 drives.
- **Fail as written:** the turn scale fitted on unbiased local seeds is up to 1.6 % from 1.

Stage 4b also found two things the plan does not expect:
1. `Simulator.record` draws a **separate motion bias for each route**, so the three drives of one dataset can differ (0.44× to 1.29× forward speed in one recording).
2. **The challenge download `ab7efaeb1abf` holds one frame per tick, not one per five ticks.** The stage 1 prior and the "label = last tick of a 5-tick window" fact do not apply to it. This affects stage 9 and should go into the next plan revision.

## What was built

| file | what |
|---|---|
| `source/mapping/record_gt.py` | holdout guard: `gt_dir` raises `HoldoutError` for a holdout seed unless `allow_final()` was called; `record_biased` / `--motion-bias` writes `data/bias-<seed>` |
| `source/mapping/seeds.json` | `holdout: [31, 32, 33, 34, 35]` |
| `source/mapping/vo.py` | `icp_symmetric`; `icp(init_sigma=...)`; wall features with frame k's depth from its own scan (`scan_cur`), vetoed by the row ratio; coincident keypoints counted once; wall fits with a rotation s.d. above 10° rejected |
| `source/mapping/odometry.py` | uses symmetric ICP and two-scan wall features (`SYMMETRIC_ICP`, `WALL_TWO_SCANS`), with a fallback to the row-ratio fit when too few features have two scan depths |
| `source/mapping/motion.py` | `GroupPrior.scaled`, `MotionPrior.scaled` |
| `source/mapping/calibration.py` (new, pipeline) | per-drive speed and turn scale from the drive's own vision; drives pooled when they agree |
| `source/mapping/stage4b_bias.py` | evaluation: estimator bias on straight pairs, rows against ray-cast truth, synthetic scans, ablations |
| `source/mapping/stage4b.py` | evaluation: scale metrics plus every stage 4 criterion, with calibrated priors |
| `source/mapping/stage4b_calib.py` | evaluation: calibration on local seeds, on the biased recordings, on the challenge data; recovery figure |
| `tests/mapping/test_calibration.py`, additions to `test_vo.py`, `test_record_gt.py`, `test_odometry.py` | 142 tests pass (`uv run pytest`); ruff clean |

The stage 3 cue caches are untouched: `stage3_extract.code_hash()` is `cdbbc5b8` before and after.

Reproduce:
```
uv run source/mapping/record_gt.py --seeds 31 32 33 34 35 --verify
uv run source/mapping/record_gt.py --seeds 31 32 33 34 35 --msaa 4 --verify
uv run source/mapping/record_gt.py --seeds 7 --msaa 4 --motion-bias 0.1 0.05 0.0 --motion-seed 206
uv run source/mapping/record_gt.py --seeds 8 --msaa 4 --motion-bias 0.3 0.15 0.0 --motion-seed 224
uv run python source/mapping/stage4b_bias.py {estimators dev|estimators test|rows|synthetic|arms}
uv run python source/mapping/stage4b_calib.py local gt4 && uv run python source/mapping/stage4b_calib.py local official
uv run python source/mapping/stage4b.py gt4 official
uv run python source/mapping/stage4b_calib.py biased && uv run python source/mapping/stage4b_calib.py plot
uv run python -I source/mapping/stage4b_calib.py challenge
```
Numbers are in `cache/mapping/stage4b_*.json`.

## Task 0: holdout seeds

Seeds **31 to 35** are recorded as `gt-<seed>` and `gt4-<seed>` (3,323 / 3,105 / 3,196 / 3,396 / 2,979 frames).

- The `gt` recordings are byte-identical to `cli.record_exploration_data` ("student-equal", no differing file).
- `poses.json`, `data_info.json`, frame counts, layout, spawn and target are identical between `gt` and `gt4` on all five seeds.
- That check compares bytes only. The pose-alignment check was skipped for the holdout seeds because it decodes frames.
- Nothing has opened, plotted or run on them since.

The guard sits in `record_gt.gt_dir`, which every loader goes through. A test checks that both prefixes are refused without `--final` and allowed after `allow_final()`. **Not done yet:** a `--final` command-line flag in an evaluation script. Only the `allow_final()` hook exists; the final evaluation (end of stage 8) will wire it.

## Task 1: scale metrics

`stage4b.py` reports both measures from the plan:
- **Similarity-alignment scale per drive:** Umeyama with scale on the positions.
- **Straight-segment ratio:** maximal runs of at least 10 consecutive FORWARD pairs with a true heading change under 2° and more than 0.3 m of travel. The ratio is estimated over true endpoint displacement. The test seeds have only 15 such segments (dev 16).

The old pooled path ratio is kept for comparison.

**The similarity scale is not a scale measure for these drives.** On `gt4` test the drives outside ±1 % are all *positive* (+1.1 % to +4.8 %), while their path ratios are −0.7 % to +0.3 % and their segment ratios are negative. Umeyama's scale absorbs shape distortion from accumulated drift: the worst drive, 21-1, has ATE 0.24 m. The criterion is reported as measured, but it tests drift as much as scale. The segment ratio is the cleaner scale measure, but 15 segments give a wide interval. Figure: `stage4b_similarity_scale.png`.

## Task 2: the forward bias

Straight pairs are FORWARD after FORWARD, true rotation under 1°, true step at least 5 cm, 80 per drive. Bias is `sum(est dx) / sum(true dx) − 1` over estimates within 5 cm of the truth; the dropped share is about 10 % for ICP and 7 % for wall features. These are dev seeds on `gt4`; test numbers were checked after the fixes were chosen.

### Hypothesis 1: ICP prior shrinkage — ruled out as the main cause

If the weak prior around ICP's initial guess caused the bias, the bias would follow the guess.

| ICP (stage 4) started from | bias (dev) | s.e. | median pair |
|---|---|---|---|
| the true motion | −0.72 % | 0.21 | −0.84 % |
| the true motion × 0.8 | −0.82 % | 0.20 | −0.86 % |
| the true motion × 1.2 | −0.62 % | 0.22 | −0.83 % |
| the label prior's mean | −0.75 % | 0.22 | −0.83 % |
| zero | −0.97 % | 0.24 | −0.88 % |

- Starting 40 % apart moves the bias by about 0.2 points.
- Starting at the truth still leaves −0.7 %.
- Removing the initial-guess prior altogether: −0.54 % (s.e. 0.22), so the prior contributes at most about 0.2 points of the −0.7 %.

### What it is instead: noise acting through a one-sided fit

Three controlled checks:

1. **The scan is not biased.** The cached floor-edge rows against the ray-cast truth at the true pose (dev, every 4th frame, 500,000 columns) are off by +0.02 px (0.3 to 0.5 m) to +0.06 px (2.5 to 3 m) on average. A row offset moves the ICP bias by about 2.9 % per pixel (`row_offset` sweep), so this explains about 0.1 %.
2. **Noise-free synthetic scans:** scans ray-cast at the true poses give an ICP bias of 0.00 % (s.e. 0.41). With 0.3 px of row noise it is −0.52 % (s.e. 0.38), and symmetric ICP gives −0.13 %. These samples are small (276 pairs), so this check is consistent with the mechanism but does not prove it.
3. **Ablations on the real pairs** (n ≈ 800, s.e. about 0.2): 

   | ICP variant (real dev pairs, started from the truth) | forward run | forward + reverse averaged | rotation median, averaged |
   |---|---|---|---|
   | stage 4 ICP | −0.72 % | **+0.18 %** | 0.18° (forward alone 0.33°) |
   | no Cauchy weights | −0.19 % | −0.06 % | 0.32° |
   | Huber weights | −0.53 % | −0.12 % | 0.28° |
   | no normal-agreement check | −1.02 % | −0.10 % | 0.18° |
   | no initial-guess prior | −0.54 % | +0.37 % | 0.19° |


The bias needs noise and sits in how ICP uses it. One run fits the noisy points of frame k to straight lines fitted on frame k−1, and the robust weights and the normal-agreement test then treat near and far, early and late points unequally. Switching off any single mechanism moves the bias by ±0.3 to 0.9 points in either direction; no single one is "the" cause. **Symmetrising cancels the asymmetry:** forward ICP and the inverted reverse ICP, averaged, give +0.1 to +0.2 % (within one s.e. of zero), and the median rotation error halves (0.33° to 0.18°) with the same translation error.

**Wall features.** The −2.7 % came from frame k's feature depth `Z2 = Z1 (v1 − cy) / (v2 − cy)`.
- The reverse direction (frame k's scan, ratio for frame k−1) is just as biased (−2.8 %), so it is not the convexity of 1/x alone.
- A symmetric average does not help either: −2.7 %.
- These two numbers, and the −0.41 % for a symmetric two-scan fit below, come from exploratory runs on the same dev pairs; `stage4b_bias.py` does not reproduce them.
- **Taking frame k's depth from frame k's own scan** brings it to −0.4 % (median −0.15 %). The rest is a tail of low estimates, not a uniform scale: symmetrising the two-scan fit gives −0.41 %.

| estimator (straight pairs, `gt4`) | dev | test |
|---|---|---|
| ICP stage 4, from the label mean | −0.75 % (s.e. 0.22) | −0.93 % (0.17) |
| **ICP symmetric (4b)**, from the label mean | **+0.14 %** (0.19) | **+0.24 %** (0.16) |
| ICP symmetric, from zero | −0.50 % (0.21) | −0.15 % (0.17) |
| wall features, row ratio (stage 4 depth; current code otherwise) | −2.71 % (0.17), median −1.69 % | −2.77 % (0.14), median −1.98 % |
| **wall features, two scans (4b)** | **−0.43 %** (0.15), median −0.17 % | **−0.51 %** (0.12), median −0.18 % |

Figure: `stage4b_bias_before_after.png`.

### Side effects, found and fixed (disclosed: on test drives)

The first two-scan wall version made three drives lose their heading: 10-1 at 71° RMS, 23-1 at 44° and 25-0 at 60°. Two of these are test drives. There were two causes:
1. The scan column in frame k can belong to another wall near corners and after rotations. Fix: the row ratio now **vetoes** a two-scan depth that disagrees by more than 5 cm + 10 %.
2. A wall fit with most of its inliers at one spot (SIFT returns several orientations per keypoint) has a rotation s.d. of about 70° and still "agrees" with any rotation source at that width. That weakness was in the stage 4 acceptance rule too. Fix: coincident points are counted once, and fits with a rotation s.d. above 10° are rejected (typical is 1°, p99 3.9° on dev; 0.2 % of fits are affected).

The final full run then showed a new acceptance of a 16°-wrong heading (23-2 frame 307, `gt4`). There, two-scan features were too few, neither vision measurement was accepted, and the filter's heading was loose enough to let the bad heading through. Fix: when the two-scan fit has too few features, the row-ratio fit is used (about 2 % of pairs try it, 0.3 % get a measurement). This restores stage 4's behaviour on those pairs.

## Task 3: label-prior self-calibration

**Finding: the bias is per drive.** `Simulator.drawn_bias()` reports one draw, but `record()` draws a new one for every route. Per-drive true forward speed against the prior's mean:
- `bias-7` (σ = 0.1, 0.05; reported draw 0.896, 1.042): 1.086 / 0.978 / 1.049
- `bias-8` (σ = 0.3, 0.15; reported 0.818, 1.150): **1.292 / 0.437 / 0.920**

The calibration is therefore per drive.

**Method** (`calibration.py`, images and labels only):
- Pairs used: those where symmetric ICP (initial-guess prior widened to 0.3 m and 40°, so it does not pull towards the label) and two-scan wall features agree (χ² ≤ 15), with fused s.d. under 1 cm / 2 cm / 1.5° and |rotation| under 20°.
- **Speed scale:** summed measured forward motion over summed prior-mean forward motion on FORWARD pairs.
- **Turn scale:** `Σ v m / Σ m²` on the rotation of LEFT and RIGHT pairs.
- Both are applied to every label's prior by `MotionPrior.scaled`, which transforms means, covariances and density thresholds. The collision component is left alone: it is a wall event, not a motion scale.
- Each drive is fitted alone, then **pooled** when the three drives agree (χ², p > 0.01).

A per-label model was tried first. LEFT and RIGHT forward motion is a few centimetres, so its scale scattered by ±5 %. The simulator's bias model has exactly one speed and one turn factor, which is the model used now.

**Local seeds** (no bias recorded; all 27 drives pooled on both frame types):

| | speed scale | turn scale |
|---|---|---|
| `gt4`, 9 seeds | 0.992 to 1.011; within 1 % on 8 of 9 (seed 24: 1.011) | 0.984 to 1.003; within 1 % on 5 of 9 |
| `gt`, 9 seeds | 0.994 to 1.008; within 1 % on 9 of 9 | 0.979 to 0.999; within 1 % on 5 of 9 |

**The turn criterion fails as written, and the reason matters.** The seeds' *true* turn against the prior's mean is itself about 1 % below 1: 0.989 to 0.991 on seeds 9, 21, 22 and 25, measured from ground truth. The fit is a further 0.5 % low, because symmetric ICP under-measures rotation by about 0.7 % (wall features −0.2 %). Pair selection contributes nothing: the ratio on the selected pairs equals the ratio on all pairs within 0.2 %. So the fit tracks each drive's real turn to about 0.5 %, but "within 1 % of 1" is not reachable when the truth is not 1. I looked at test seeds 21, 22 and 25 for this diagnosis; nothing was tuned on it.

**Biased recordings** (dev seeds 7 and 8, `msaa=4`, same maze and routes as `gt4-<seed>`; figure `stage4b_calibration_recovery.png`):

| drive | true speed | fitted | error | true turn | fitted | error |
|---|---|---|---|---|---|---|
| bias-7 drive 0 | 1.086 | 1.079 ± 0.013 | -0.6 % | 0.993 | 0.986 ± 0.015 | -0.7 % |
| bias-7 drive 1 | 0.978 | 0.963 ± 0.012 | -1.5 % | 1.038 | 1.034 ± 0.017 | -0.5 % |
| bias-7 drive 2 | 1.049 | 1.050 ± 0.009 | +0.1 % | 1.063 | 1.059 ± 0.012 | -0.3 % |
| bias-8 drive 0 | 1.292 | 1.287 ± 0.018 | -0.4 % | 0.981 | 0.973 ± 0.016 | -0.8 % |
| bias-8 drive 1 | 0.437 | 0.438 ± 0.004 | +0.3 % | 0.777 | 0.772 ± 0.012 | -0.6 % |
| bias-8 drive 2 | 0.920 | 0.915 ± 0.007 | -0.6 % | 1.116 | 1.116 ± 0.014 | +0.0 % |

- **Recovery:** 11 of 12 scales are within 1 % of the truth; the worst is bias-7 d1 speed at −1.5 % (1.2 s.e.).
- **Variant C ATE**, mean of 3 drives:

  | | unbiased run (calibrated) | biased, uncalibrated | biased, calibrated |
  |---|---|---|---|
  | seed 7 | 0.085 m | 0.106 m | **0.099 m (+16 %)** |
  | seed 8 | 0.108 m | 0.098 m | **0.053 m** |

  The 20 % limit is met on both. Vision dominates the odometry, so even the uncalibrated run copes; the calibration mainly helps the gate.

**Challenge data `ab7efaeb1abf`** (reported only; run with `python -I`). The fitted scales hit the 0.3 floor on all three drives. The raw ratios are 0.20 to 0.21 (speed) and 0.25 (turn) against the 5-tick prior, which led to the finding:

| drive | frames | label same as previous | forward motion per frame | turn per frame (vision) | turn per frame (heading cue) |
|---|---|---|---|---|---|
| 0 | 1,714 | 0.80 | 0.0289 m = 0.96 tick | 3.48° = 1.01 tick | 3.50° |
| 1 | 2,342 | 0.80 | 0.0314 m = 1.05 tick | 3.46° = 1.01 tick | 3.48° |
| 2 | 3,146 | 0.80 | 0.0297 m = 0.99 tick | 3.56° = 1.04 tick | 3.60° |

One tick is 0.03 m or 3.44°; locally (5 ticks per frame) the label repeats 45 % of the time. **This download is one frame per tick.** The per-tick ratios suggest the server's drives also carry small per-drive biases (forward 0.96 to 1.05). None of this is used anywhere; the stage 1 prior does not apply to such data.

## Task 4: re-run with the fixed estimators and the calibrated prior

Variant C uses each drive's calibrated prior; A is the uncalibrated labels-only reference; B is vision only. Test seeds, no loop closures.

| | stage 4, `gt4` | **4b, `gt4`** | stage 4, `gt` | **4b, `gt`** | threshold |
|---|---|---|---|---|---|
| heading RMS, worst of 15 (median) | 0.83° (0.47) | **1.01° (0.33)** | 1.57° (0.83) | **1.32° (0.56)** | < 2° |
| ATE per drive, median / max | 0.12 / 0.20 m | **0.094 / 0.24 m** | 0.13 / 0.31 m | **0.093 / 0.22 m** | |
| 10-frame translation error, median / p95 | 1.99 / 6.87 % | **0.79 / 4.64 %** | 1.90 / 7.98 % | **0.91 / 6.30 %** | < 3 / < 8 % |
| 10-frame rotation error, median | 0.25° | **0.19°** | 0.30° | **0.24°** | < 1° |
| collision events on the right branch | 63 / 63 | **63 / 63** (max error 5 frames on 1.2°) | 63 / 63 | **63 / 63** | all |
| C ATE below A on every test seed | yes | **yes** | yes | **yes** | |
| NIS in the 95 % bound | 97.5 % | **97.8 %** | 97.3 % | **97.8 %** | ≥ 90 % |
| ICP / wall features accepted | 95.5 / 97.7 % | 96.4 / 97.8 % | 93.3 / 94.7 % | 94.6 / 94.9 % | |
| path length, pooled (old metric) | −1.10 % | **+0.01 %** | −0.87 % | **+0.22 %** | |
| similarity scale, median drive | | **+0.14 %** | | **+0.45 %** | within 0.5 % |
| similarity scale, every drive | | **10 / 15 within 1 % (worst +4.8 %)** | | **10 / 15 (worst +3.9 %)** | within 1 % |
| straight-segment ratio, median (n = 15) | | **−0.64 %** (95 % CI −0.90 to −0.11) | | **−0.83 %** (CI −1.58 to −0.23) | within 0.5 % |

On dev (`gt4`): windows 0.74 / 3.87 %, segment ratio −0.92 % (n = 16), worst heading RMS 0.96°, collisions 25 / 25.

**The remaining forward shortfall on straight segments** (about −0.6 %) is where the filter combines symmetric ICP (+0.1 to +0.2 %) with the wall features (−0.4 %, tail-driven) and with the label prior. On straight pairs the prior's mean is about 0.7 % short of the pairs' true motion, because the FORWARD-after-FORWARD mixture also covers mixed windows and collisions. The calibration cannot remove that: it matches the prior to *all* pairs, which are unbiased on local seeds. The fused straight-pair bias inside the filter is about −0.5 % (−0.6 % on the 88 % of pairs where both vision measurements are accepted, −3 to −6 % on the few corridor pairs where only one or neither is).

**Confidently wrong headings (revision 7 wording):**
- `gt4`: 7 test frames. No frame wrong by more than 10° is accepted. One frame wrong by 3.1° (22-2-684) is accepted, filtered error 1.52°. The worst filtered error at any of them is 1.52°, under 3°: **pass**.
- `gt`: 8 frames. None over 10° accepted. 23-2-1072 (wrong by 6.2°) is accepted, filtered error 6.11° (reported).

**B vs C:** B's ATE is 2.0× C's on `gt4` (median 0.21 against 0.094 m) and 5× on `gt` (0.47 m; worst heading RMS 68.7°). This is unchanged in kind from stage 4: the pipeline uses the action labels. The calibration now also makes it robust to a per-drive motion bias, the risk stage 4 named.

**Leakage (rule 10):**
- `vo.py`, `odometry.py`, `pipeline.py` and `calibration.py` never name ground truth (test).
- Images to trajectory, and images to the calibration's measurements, are identical on a copy without `poses.json` / `gt.json` (tests).

## Acceptance

| criterion | result | |
|---|---|---|
| Holdout seeds recorded, poses and labels identical between `gt` and `gt4`, guarded, not opened | yes; guard tested; the `--final` CLI flag is a hook only (above) | pass |
| Mechanism of the forward bias identified with a controlled experiment, or hypotheses ruled out | ICP prior shrinkage ruled out (initial guess); scan bias ruled out (rows against truth); ICP: noise through a one-sided fit (noise-free vs noisy synthetic; ablations); wall features: the row-ratio depth (two-scan ablation) | pass |
| Similarity scale: median drive within 0.5 %, every drive within 1 %; segment ratio median within 0.5 % | median +0.14 %; every drive: **10 / 15**; segments **−0.64 %** | **fail** (two of three) |
| All stage 4 criteria (revision 7 wording) on `gt4` test | all pass (table above) | pass |
| Self-calibration: fitted scale within 1 % of 1 on every local seed | speed 8 / 9 (`gt4`; seed 24 at 1.07 %), 9 / 9 (`gt`); turn 5 / 9 (truth itself about 1 % off; see above) | **fail as written** |
| Recovers an injected bias within 1 % on a biased dev seed; C ATE within 20 % of the unbiased run | 11 / 12 drive scales within 1 % (worst 1.5 %); ATE +16 % (seed 7), −51 % (seed 8) | pass on ATE, 11 / 12 on recovery |
| Fitted scales on the challenge data reported | reported; the 5-tick prior does not apply to that data (one frame per tick) | reported |
| Changes made after looking at a test-seed result disclosed | below | done |

## Disclosure: what was tuned and where I looked

- The two fixes (symmetric ICP, two-scan wall depth) were chosen on **dev** straight pairs; the test straight-pair numbers were computed afterwards as a check.
- The heading-loss fixes (ratio veto, rotation s.d. cap, coincident keypoints) and the row-ratio fallback were found on drives 10-1 (dev), 23-1, 25-0 and 23-2 (**test**). Each targets a mechanism, not a seed, and each was checked on the dev drives. As in stages 3b and 4, the test numbers above are selected in a way the dev numbers are not.
- For the turn-scale diagnosis I looked at test seeds 21, 22 and 25; nothing was changed because of it.
- The holdout seeds have not been opened.

## Open issues and notes for the next plan revision

- **The challenge data is one frame per tick.**
  - The action labels are then exact per-tick actions, not a weak prior.
  - The stage 1 prior, the collision statistics (stage 0 / 3b) and every per-frame threshold assume 5 ticks.
  - Stage 9 needs either a 1-tick prior and retuned gates, or the challenge frames subsampled to every 5th (and labels aggregated).
  - Whether every server download is like this, or only this one, is unknown; checking another download would settle it.
- **Per-route motion bias** in `Simulator.record`. Any local recording with `motion_bias` has three different biases; the calibration handles it per drive.
- **Straight-segment shortfall about −0.6 %** (CI excludes 0). The likely sources are the label prior's straight-pair conditional mean and the wall features' low tail. Stage 6 loop closures will not remove a per-pair scale.
- **Similarity-alignment scale** as a criterion mostly measures drift on these drives. The plan may want to restate it (for example, on loop-closed trajectories in stage 6).
- The turn scale is fitted 0.5 % low (symmetric ICP under-measures rotation by about 0.7 %). A heading-cue rotation measurement would avoid it.
- The `--final` command-line flag still has to be wired into the stage 8 final evaluation.
- Not changed in this stage: the plan file, and `CLAUDE.md` (it still says stages 0 to 4 are done).
