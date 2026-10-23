# Stage 8: goal placement on the map

Plan revision in force: **revision 9**. Implemented by the user's Claude session in the Linux VM on the user's Mac, from the stage 7 maps. Developed on the dev seeds only; the test seeds were scored once, with the user's agreement, after the parameters were frozen.

Status: **every criterion passes, except the final holdout evaluation, which is deferred to after stage 9 by the user's decision.**
- The goal was found on **5 of 5 test seeds**: position error 2 to 3 mm, heading error 0.0 to 0.2° (bounds 0.10 m and 10°).
- Confidence on test is 0.31 to 0.38, against a dev-chosen threshold of 0.10. There are no high-confidence failures anywhere.
- Runtime is 0.55 to 1.0 s per placement given a built map (target < 2 s); loading the map takes another 0.12 s.

**New fact: the goal views come from a lower camera than the robot's,** 0.160 m against 0.210 m. This holds on every local seed and on the challenge download. The method measures it from the images themselves.

## Built

| file | what |
|---|---|
| `source/mapping/goal.py` | **pipeline**: goal scan from the four views (camera height measured per goal), scan compass, global likelihood-field search over the map's free cells, Gauss-Newton refinement, wall-face texture term, confidence. `place(map, target.jpg or info.targets)` |
| `source/mapping/stage8.py` | evaluation (reads the true goal): real goal views, a stress test with 2- and 3-view subsets, a geometry-only ablation, figures. Refuses test seeds without `--allow-test`. |
| `source/mapping/final_eval.py` | the final holdout evaluation: `--final` wired to `record_gt.allow_final()`. Without the flag the holdout seeds are refused. Runs `build_map.py` and scores every stage's key metric (stages 2, 3, 3b, 4, 5, 6, 7, 8). Checked on dev seed 7; **not yet run on the holdout.** |
| `tests/mapping/test_goal.py` | 7 tests: camera height, heading modulo 90°, geometric placement on a synthetic map, the holdout gate, rule 10 leakage. The full mapping suite passes (164; `test_record_gt` needs the real simulator and was not run in the VM). |
| `cache/mapping/stage8/*.json` | every number below |
| `reports/figures/stage8_goals_test_gt4.png`, `stage8_confidence_vs_error.png` | |

Reproduce:
```
python3 source/mapping/stage8.py run dev [--subsets | --notex | --prefix gt]
python3 source/mapping/stage8.py run test --allow-test
python3 source/mapping/stage8.py figures
```

## Two facts found on dev

1. **The view order is counter-clockwise.** The four views face θ, θ + 90°, θ + 180° and θ + 270° (front, left, back, right). On dev, the other order fits worse on every seed: median residual 0.06 against 0.01 to 0.03 m.

2. **The goal views come from a camera at 0.160 m, not 0.210 m.**
   - At the true goal pose, the goal scans fit the walls only after a range scale of about 0.75.
   - The direct measurement compares, for each wall column, the floor edge and wall top offsets from the horizon. Their ratio is h / (H − h). It gives:
     - h = 0.1600 m on every dev seed's goal views;
     - 0.1599 m on the challenge download's `target.jpg`;
     - 0.210 m on exploration frames, both local and challenge.
   - The intrinsics are unchanged.
   - `goal.py` measures h from each goal image (a median over about 1,200 columns), so a different goal height in the exam would be picked up.

## Method

1. **Goal scan.**
   - Segment each view and range it with the measured camera.
   - Rotate view k by k × 90° and merge into one scan: about 1,200 points, about 360°.
   - Adjacent views overlap by about 30°; their disagreement there is 1.3 to 7.8 mm (median of nearest points). This is the plan's consistency check.
2. **Heading modulo 90°.** Take the circular mean of 4 × the angles of the scan's straight-piece normals (stage 3's scan compass). This gives 4 heading branches, each searched at ±1.5°.
3. **Global search.**
   - The likelihood field is a Gaussian (σ = 3 cm) of the distance to the nearest rendered wall of the stage 7 map, at 1 cm.
   - Candidates are every known-free 5 cm cell more than 8 cm from the collision segments (about 9,700), × 12 headings, scored with 240 points.
   - The best 12 distinct places are kept: more than 0.25 m apart, or a different heading branch.
4. **Refinement.** Per place: a fine grid (±5 cm at 1 cm, ±2° at 1°) with all points, then point-to-wall Gauss-Newton with a Cauchy weight.
5. **Texture term.**
   - At each refined pose, the goal views' wall pixels are assigned to the map's wall faces, the same way stage 7 built them.
   - Each face's hue-saturation histogram is compared with the stored signature (Hellinger distance, pixel-weighted, faces with at least 100 pixels).
   - Score = geometric + 0.5 × (1 − mean distance).
6. **Output.** The best pose; confidence = its score minus the best *different* place's; the top 3.

The parameters were set before any scoring and kept: σ, 12 candidates, weight 0.5. The confidence threshold of 0.10 was chosen on the dev stress test (below).

## Results

**Dev** (`gt4`, 4 views): errors 2 to 3 mm and 0.1 to 0.5°, confidence 0.24 to 0.31. The `gt` (msaa 1) goal views give the same: 1 to 3 mm, confidence 0.25 to 0.31.

**Geometry alone fails** (dev ablation, `--notex`):
- Seed 10 is placed 4.0 m away at the wrong heading. Another place fits the walls better (0.978 vs 0.964) and the margins are tiny (0.01 to 0.05).
- With the texture term, the true place scores 0.75 to 0.88 on texture, and every rival 0.26 to 0.41.
- The term is what makes the placement unique. This is the stage 5b finding (pairs of faces are near unique), used across the 20 to 30 faces a goal sees.

**Stress test on dev** (each 3-view and 2-view subset of the goal views, 40 queries):
- 39 are correct, with confidence 0.18 to 0.31.
- The one failure (seed 10, front and left views only) has confidence 0.02.
- A threshold of 0.10 separates them; it is in `goal.CONF_MIN`, and `GoalEstimate.reliable` tests it.

**Test** (frozen parameters, scored once):

| seed | error (gt4) | heading | confidence | runner-up (geo / texture) | error (gt, msaa 1) | runtime |
|---|---|---|---|---|---|---|
| 21 | 2 mm | 0.2° | 0.343 | 0.989 / 0.174, 0.63 m away | 3 mm | 0.88 s |
| 22 | 2 mm | 0.0° | 0.320 | 0.902 / 0.341, 3.4 m | 1 mm | 0.96 s |
| 23 | 3 mm | 0.0° | 0.381 | 0.850 / 0.299, 4.6 m | 2 mm | 0.97 s |
| 24 | 2 mm | 0.0° | 0.319 | 0.948 / 0.272, 3.6 m | 2 mm | 0.89 s |
| 25 | 2 mm | 0.1° | 0.306 | 0.978 / 0.267, 0.85 m | 3 mm | 0.55 s |

The best's geometric score is 0.945 to 0.991. On seeds 21 and 25 a runner-up fits the walls almost as well: 0.989 vs 0.991, and 0.978 vs 0.987. Texture decides.

The runtimes above are with 4 placements in parallel; on dev, one at a time, they are 0.56 to 1.02 s.

## Acceptance

| criterion | result | |
|---|---|---|
| Position error < 0.10 m and heading < 10° on ≥ 80 % of test seeds (≥ 5 seeds) | 5 of 5 (2 to 3 mm, ≤ 0.2°) | pass |
| No high-confidence failures (threshold chosen on dev) | none on test; the one dev failure (stress test) has confidence 0.02 < 0.10 | pass |
| Runtime < 2 s per placement given a built map | 0.55 to 1.0 s (+0.12 s to load the map) | pass |
| Leakage test (rule 10) | `goal.py` static scan; the goal views and the map only | pass |
| `--final` wired, holdout refused without it (test) | `final_eval.py`, `test_final_evaluation_refuses_holdout_seeds_without_the_flag` | pass |
| Final evaluation on the holdout seeds | **deferred to after stage 9** (user's decision: spend the holdout once, on the pipeline including the challenge adapter) | open |

## Disclosures

- Before the user asked for test seeds to be kept for testing only, one convention check (the order of the four views and what `gt.json`'s target refers to) looked at the goal images and true goals of **test seeds 21 and 22** alongside dev seed 7. It measured residuals and tuned nothing. The view order and the camera height were then established on dev seeds only.
- The test seeds were scored once, after the user agreed. Nothing was changed after that.
- The confidence threshold rests on 4 dev goals plus 40 subset queries made from them, which is a small sample. The test margins (≥ 0.31) are far above it.

## For stage 9

- Run `build_map.py` on the challenge download through the stage 9 adapter (one frame per tick → every 5th). Then `goal.place(load_map(map.npz), target.jpg)`, and compare with `place(..., info.targets)` at runtime: the four views as a list.
- A placement with `reliable == False` should not be trusted. The fallback is to explore more, or the baseline's behaviour.
- The final holdout run: `final_eval.py --final build <seed> --steps ...`, then `--final score 31,32,33,34,35`, once, after stage 9.
