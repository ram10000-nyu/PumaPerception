# Implementation plan: top-down map from exploration data

**Repo:** fork of `ai4ce/vis_nav_player` (NYU ROB-GY 6203 Visual Navigation Challenge)
**End goal:** from the provided exploration trajectories, build a metric top-down (zenithal) occupancy map of the maze with wall-texture signatures, and place the goal pose on it.
**Out of scope for this plan:** live localization during a run, path planning, autonomous control. Those come in a follow-up plan that consumes this map.

## Revision 10 (2026-10-18): stages 7 and 8 done; stage 9 rewritten; seed discipline

Stages 5b to 8 were implemented by the user's Claude session: `reports/stage_5b.md` to `reports/stage_8.md`. They are **accepted; do not rerun them.** Next stage to run: **stage 9**, as rewritten below. **Stage 10 (new, at the user's request):** press P in `my_agent.py` to locate the robot on the map with an in-place 360° spin and show a green circle there. It runs after stage 9 and is the only live-localization piece in this plan. Read the stage 7 and stage 8 status notes and the facts table first.

**Seed discipline (the user's instruction; it overrides anything older in this plan):**
- Tune and debug on the **dev seeds (7 to 10) only**.
- **Ask the user before any test seed (21 to 25) is scored**, and score each stage on test only once, after freezing. `stage8.py` refuses test seeds without `--allow-test`; any new evaluation script must do the same.
- The **holdout seeds (31 to 35)** are opened once, by `final_eval.py --final`, **after stage 9 passes and after the user confirms**. The user chose to move this final run from the end of stage 8 to the end of stage 9.
- Do not open a fresh local seed for tuning either. A new maze is a test.
- Never run on the server (it spends an attempt), and never download another challenge's exploration data, without asking the user.

**Built ahead of stage 9 (by the user's Claude session).** Keep and extend these; do not rewrite them:

| file | what |
|---|---|
| `source/mapping/build_map.py` | raw data → `map.npz`, resumable steps, `timing.json` (173 s per local maze on a 4-core VM) |
| `source/mapping/slam.py` | stages 5b and 6 as pipeline functions of raw inputs |
| `source/mapping/grid.py` | stage 7: occupancy, lattice and scale, snapping, collision geometry, configuration space, signatures |
| `source/mapping/goal.py` | stage 8: `place(load_map(...), target.jpg or info.targets)` |
| `source/mapping/final_eval.py` | the holdout run: `--final build <seed>`, then `--final score 31,32,33,34,35` |
| `source/my_agent.py` | display-only agent (stage 9 task 5 below). Builds or loads the map in `__init__`, places the goal in `setup`, shows the map and goal in `panel()`, keyboard driving. One open bug: see task 5. |

**New facts:**
- the goal views come from a 0.160 m camera (stage 8);
- the collision radius is about 0.12 m to the unextended wall centre lines (stage 7);
- a post can be missing between two walls (stage 7).

All three are in the facts table.

## Revision 9 (2026-10-15): stage 5 accepted; texture adjacency (stage 5b)

Stage 5 (`reports/stage_5.md`) is **accepted; do not rerun it.** Closures: precision 0.997 on `gt4` test (17 false of 6,594), relative-pose error median 2.1 mm / 0.22 deg. Two criteria were mis-specified and are restated (both pass): **frame recall** (revisiting frames that get at least one closure) >= 0.25 replaces pair recall (measured 0.34); **graph connectivity** (closures plus the spawn constraint connect all drives) replaces "every drive pair directly joined" (seed 23 drives 0 and 1 connect through drive 2).

Key finding: the ~200 wall textures repeat across a maze, so appearance alone cannot make closures; stage 5 gates them against the odometry (0.5 m, 10 deg). Closures are therefore a local refinement that depends on drift staying under about 0.4 m. The 17 false closures are aliases 0.4 to 0.6 m away (the same texture arrangement one or two cells along a corridor).

| Where | Change |
|---|---|
| Stage 5 | Accepted; criteria restated as above. |
| **Stage 5b (new)** | **Texture adjacency**: an ordered pair of textures meeting at a wall is far more distinctive than one texture. Go/no-go measurement first, then boundary-verified closures. **Being implemented by the user's Claude session, not the coding agent: do not start it.** |
| Stage 6 | Two-pass Mahalanobis gating, closure thinning, and the real stage 5 aliases as the robustness set. |
| Stage 8 | Textures repeat: the texture term cannot identify a wall alone; geometry leads, and stage 5b boundary signatures (if they pass) identify the goal's surroundings. |
| Stage 9 | The closure gate assumes drift under about 0.4 m; per-tick or biased challenge drives must be checked against that. |

Stages 5b, 6, 7 and 8 are **done** (`reports/stage_5b.md` to `reports/stage_8.md`). Next stage to run: **stage 9** (challenge data). **The final holdout evaluation is deferred until after stage 9** (user's decision, 2026-10-18: spend the holdout once, on the pipeline including the challenge adapter); its script is ready (`final_eval.py --final`). **Seed discipline (user's instruction):** develop on dev seeds only, and ask the user before any test seed is scored. Earlier note: stage 8 was to run next (goal placement; read the stage 7 status notes first). Earlier note: stage 7 was to run next (task 0, then lattice scale calibration). Earlier note: stage 6 was to run, using the union closures (`stage5b.py` rules: gate without contradiction, or boundary-verified) and the stage 5b notes on gate-free boundary closures.

---

## Revision 8 (2026-10-15): stage 4b accepted; per-tick challenge data

Stage 4b (`reports/stage_4b.md`) is **accepted; do not rerun it.** Holdout seeds 31 to 35 are recorded and guarded. The forward bias was traced (ICP: noise through a one-directional fit, fixed by symmetric ICP; wall features: the row-ratio depth, fixed by two-scan depth). Odometry on `gt4` test improved to ATE median 0.094 m, 10-frame drift 0.79 / 4.64 %. Per-drive prior self-calibration recovers injected biases (11 of 12 scales within 1 %).

Its three failed criteria are settled as follows:
- **Straight-segment ratio -0.64 %** (CI excludes 0): a real residual, about 2 cm over 3 m, inside the stage 6 budget. **Accepted and recorded**; re-measured in stage 6.
- **Similarity scale per drive (10 of 15 within 1 %)**: the metric absorbs drift-induced shape distortion as scale. **Moved to stage 6**, measured on loop-closed trajectories.
- **Turn scale within 1 % of 1**: the true turn scale of those drives is itself about 0.99. **Restated:** fitted scale within 1 % of each drive's *true* scale (which it meets to about 0.5 %).

Two findings change the plan:
1. **The challenge download `ab7efaeb1abf` is one frame per tick**, not one per five. Measured from vision: 0.03 m and 3.5 deg per frame; the label repeats 80 % of the time (45 % locally). The stage 1 prior, the collision statistics and every per-frame threshold assume 5-tick spacing. Stage 9 gets a data adapter (task 0b).
2. `Simulator.record` draws a **separate motion bias per route**; the server's drives probably do too (vision suggests forward 0.96 to 1.05 per drive). Calibration stays per drive.

| Where | Change |
|---|---|
| Facts | Per-tick challenge data, per-route bias, holdout seeds 31 to 35, stage 4b fixes. |
| Stage 4b | Status: accepted; criteria settled as above. |
| Stage 6 | Gains the similarity-scale and straight-segment checks, on loop-closed trajectories. |
| Stage 8 | The `--final` command-line flag must be wired before the final holdout run. |
| Stage 9 | New task 0b: frame-rate detection and a per-tick to 5-tick adapter with exact window composition; check a second download. |

Next stage to run: **stage 5**.

---

## Revision 7 (2026-10-15): stage 4 results; holdout seeds; stage 4b

Stage 4 (`reports/stage_4.md`) produced per-drive trajectories from images with ATE median 0.12 m and worst heading RMS 0.83 deg on `gt4` test (labels alone: 3.5 to 4.6 m), all 63 collision events on the right branch. It did **not** pass cleanly: path length 1.1 % short (`gt4`), and one heading wrong by 3.1 deg (`gt4`) / 6.2 deg (`gt`) got through the gate. The report also states that the **test seeds were used to find and fix failures** in stage 4 (and two disclosed adjustments in stage 3b), so test-seed numbers are now optimistic.

What changed in revision 7:

| Where | Change |
|---|---|
| Rule 4 | **Holdout seeds** added: recorded once, never opened until the end of stage 8. Test seeds keep their working role. |
| Stage 4 | Status: accepted pending stage 4b. Heading criterion restated to the actual danger. Scale criterion moved to stage 4b with a better metric. Deviations recorded (no floor homography, no parallel hypotheses). |
| Stage 4b (new) | Record holdout seeds; diagnose and fix the forward scale bias; label-prior self-calibration on each dataset (against an unknown server `motion_bias`). |

Next stage to run: **stage 4b**.

---

## Revision 6 (2026-10-13): stage 3 accepted; server-like recordings; heading gating

Stage 3 (`reports/stage_3.md`) is **complete and accepted; do not rerun it.** Fused heading on test seeds: 96.0 % of frames valid, median 0.21 deg, p95 0.75 deg, 99.95 % of valid frames under 3 deg; wall-top ranging lifts scan coverage from 0.80 to 0.95.

What changed in revision 6:

| Where | Change |
|---|---|
| Rule 9 | Exception: stage 3b task 0 records anti-aliased (`msaa=4`) datasets. |
| Facts | Anti-aliasing mismatch (local `msaa=1`, server `msaa=4`); scan heading cue and wall-top ranging delivered; 15 confidently wrong headings. |
| Stage 3b | New task 0: record `msaa=4` datasets for all seeds; from 3b on, tune on `msaa=4` dev and pass on both frame types. |
| Stage 4 | New criterion: none of the confidently wrong stage 3 headings is accepted by the filter. |
| Stage 7 | New task 0: regenerate the stage 2 oracle maps with the current scan code before scoring. |

Next stage to run: **stage 3b** (start with its task 0).

---

## Revision 5 (2026-10-13): stage 2 accepted; small local edits

Stage 2 (`reports/stage_2.md`) is **complete and accepted; do not rerun it.** Local changes only: facts table (wall rule, half-thickness, JPEG-aware floor palette); stage 3 adds a scan-based heading cue and wall-top ranging; stages 4 and 5 weight scan returns by range instead of cutting at `R_MAX`; stage 7 scores against an observable layout and reports missing walls. Next stage to run: **stage 3**.

---

## Revision 4 (2026-10-13): stage 1 accepted; collision handling moved to stage 3b

**Reload this file from disk before working.** An earlier session overwrote revision 3 with a stale revision 2 copy, so stage 1 ran without the revision 3 collision tasks. Never save an in-memory copy of this plan over the file; edit the file on disk.

Stage 1 (`reports/stage_1.md`) is **accepted and complete; do not rerun it.** The prior (per (previous, current) label Gaussian mixtures) is calibrated on test seeds (0.934 to 0.948 inside the 95 % region). The one failed criterion, dev-vs-test means within 2 standard errors (max z 2.1 naive, 2.2 clustered), is restated below as a practical tolerance; the measured differences (at most 0.9 mm and 0.7 degrees) pass it. Reason: with thousands of samples a standard-error test flags differences far too small to matter, and with 12 comparisons about one |z| > 2 is expected by chance.

What changed in revision 4:

| Where | Change |
|---|---|
| Stage 1 | Marked complete. Mean-agreement criterion restated as an equivalence tolerance. The two revision 3 collision tasks (collision component, collision detector) moved out. |
| Stage 3b (new) | Collision component of the prior and the collision detector, built after stage 3 because the detector's best signals (wall-feature inliers, floor-boundary orientation, vanishing-point disagreement) come from stages 2 and 3. Must pass before stage 4 starts. |
| Stage 4 | References point to stage 3b. |

Order from here: **stage 2 -> stage 3 -> stage 3b -> stage 4.** Next stage to run: **stage 2** (done; see revision 5).

---

## Revision 3 (2026-10-13): stage 0 complete; findings folded in

Stage 0 revision 2 (`reports/stage_0.md`, "Addendum (revision 2)") is **complete; do not rerun it.** Seven of eight criteria passed. The rotation gate failed: 27 consecutive-frame pairs (0.09 %) rotate more than 45 degrees (max 54.8), all **wall collisions** under FORWARD/IDLE labels, on 8 of 9 seeds. That failure is a design input, not something to fix in stage 0; this revision handles it in stages 1 and 4.

What changed in revision 3:

| Where | Change |
|---|---|
| Rules | New rule 10: ground truth is evaluation-only, with a leakage test. |
| Facts | Measured camera height, zero offset, thin-plane wall geometry, collision radius, far-plane rule, floor palette, cue statistics. The revision-2 statement "walls are whole cells, faces at +/- 0.1 m" was **wrong** and is corrected. |
| Stage 1 | Prior becomes a mixture with a collision component; collision detector added. |
| Stage 2 | Reference geometry is `camera_fit.Scene(...).segments`; investigate the 6.6 % of boundary columns the scene does not explain first; start `r_max` at 2.5 m. |
| Stage 3 | Must build a real validity test (the stage 0 detector agreed with true heading on only 39 % of frames). |
| Stage 4 | Collision handling: branch hypotheses on flagged pairs, rotation-global scan matching, mixture-based gating; new criterion on collision events. |
| Stage 7 | Walls are segments on cell-centre lines; adds a configuration-space map inflated by the 0.08 m collision radius. |
| Stage 9 | Leakage test before touching challenge data. |

Next stage to run: **stage 1**.

---

## Revision 2 (2026-10-13): vision-first odometry

Stage 0 (`reports/stage_0.md`) found that a record's action label is only the **last tick** of its 5-tick window (62 % of windows mix actions; FORWARD records advance 0.110 +/- 0.046 m, not 0.15) and that the server's `motion_bias` is unknown. Action labels are therefore demoted from the motion model to a **weak prior**; odometry is driven by vision.

What changed:

| Stage | Change | Rerun? |
|---|---|---|
| 0 | Everything that passed stays. Adds **0.6** camera height and offset (now required, not deferred), **0.7** trackability audit, **0.8** far-plane behaviour. Report goes in an addendum. | **Incremental**: do only 0.6 to 0.8. Do not re-record. |
| 1 | Rewritten: characterize action labels as an empirical prior instead of fitting per-action constants. | Not started yet: run as written now. |
| 2 | Camera height comes from stage 0.6; colours and range limit updated. | As written. |
| 4 | Rewritten: visual odometry, three variants compared, absolute targets. | As written. |
| 6 | Spawn constraint added (all drives start at the same known pose). | As written. |
| 7 | True layout is available from the simulator, so the layout IoU criterion is mandatory. | As written. |

---

## How to work through this plan (read first)

1. **One stage at a time.** Do not start stage N+1 until stage N's acceptance criteria pass. If a criterion fails after reasonable effort, **stop and report**: what failed, the numbers, what you tried, your best guess at the cause. Never loosen a threshold silently.
2. **Do not modify the baseline.** `source/baseline_agent.py`, `source/vlad.py`, `source/keyboard_agent.py` and `source/cli.py` stay untouched. New code goes in `source/mapping/`. Reuse baseline functions by importing them.
3. **Ground truth exists only in the local simulator** (`vis_nav_sim`, installed by `uv sync`). Every metric is measured on local maze seeds. The challenge data is only processed after the pipeline passes on local seeds.
4. **Fixed seed split, decided in stage 0 and never changed:**
   - dev seeds (tune anything you like): at least 3
   - test seeds (report only; never tune on them): at least 3
   Every acceptance number below is measured on the **test** seeds unless stated otherwise.
   - **holdout seeds (revision 7):** 5 new seeds recorded in stage 4b as `gt-<seed>` and `gt4-<seed>`, listed in `seeds.json` under `holdout`. Nothing may read their frames, cues or results before the **final evaluation, moved to the end of stage 9 by the user in revision 10**, which runs the frozen pipeline on them exactly once and reports every stage's key metric (stages 2, 3, 3b, 4, 6, 7, 8) next to the test-seed numbers. No change to the pipeline is allowed after that run; if it reveals a failure, report it rather than fix and re-run. Rationale: the test seeds were used to find and fix failures in stages 3b and 4, so their numbers are no longer an unbiased estimate.
5. **Every stage writes a report** `reports/stage_N.md`: what was built, a results table (metric, value, threshold, pass/fail), the plots listed under that stage, and open issues. Plots go in `reports/figures/`.
6. **Units:** metres, radians in code, degrees in reports. Angles wrapped to (-pi, pi]. Poses are SE(2) `(x, y, theta)`.
7. **Reproducibility:** every script takes `--seed` / `--data` arguments, fixes RNG seeds, and caches intermediate results under `cache/mapping/` with the stage name and a config hash in the filename (the baseline's caches are keyed only by frame count; do not repeat that mistake).
8. **Tests:** each stage adds `pytest` tests for its pure functions (geometry, SE(2) composition, log-odds update, etc.) under `tests/mapping/`. `uv run pytest` must pass at every stage.
9. **Reruns after a plan revision:** keep every deliverable that still satisfies its criteria. Do the new or changed tasks only, and record them in an `## Addendum (revision N)` section at the end of that stage's report rather than rewriting the report. Never re-record datasets unless a task says so. (Revision 6: stage 3b task 0 says so, for the `msaa=4` datasets only; the existing `data/gt-<seed>/` sets stay untouched.)
10. **Ground truth is evaluation-only.** `gt.json` and `poses.json` exist only because we run the simulator ourselves (`record_gt.py` reads the simulator's internal state: `world.grid`, `world.spawn`, `world.target`, `record(student_files=False)` for true poses). The server never sends any of it, and challenge data contains neither file. Therefore:
    - Pipeline code (everything `build_map.py` and the agent import) must never read `gt.json`, `poses.json`, `Simulator` internals, or `camera_fit.Scene`. Only evaluation, calibration and report scripts may.
    - Simulator **constants** measured once (K, `h`, motion constants, wall rendering, far plane, palette) are allowed in the pipeline as configuration, because they are properties of the simulator, not of a particular maze. Record each one in `cache/mapping/camera.json` or `motion_prior.json` with its provenance.
    - Never try to identify the challenge maze by matching it against locally generated seeds.
    - Leakage test (stage 9, also run at the end of stages 4, 7 and 8): copy a local dataset, delete `gt.json` and every `poses.json`, and run the pipeline on the copy; outputs must be identical to the run on the original.

### Known facts from the starter code (verified)

| Fact | Where |
|---|---|
| Exploration data = 3 drives per maze, one frame every 5 ticks, plus `target.jpg` (local) | `cli.record_exploration_data` |
| Recording uses motion noise: `motion_noise=sim.DEFAULT_NOISE` | same |
| `data_info.json` records carry `image` and `action` (a list; in all data seen so far it holds exactly one action, see stage 0) | `vlad.load_frames` (which drops multi-action records) |
| Intrinsic matrix and image size are given to the agent in `info.camera` | `my_agent.py` docstring |
| Frames are (240, 320, 3) uint8 BGR | `my_agent.py` docstring |
| World units are metres; goal = within 0.2 m and 30 degrees | `rl/env.py` (`GOAL_DISTANCE_M`, `GOAL_HEADING_DEG`) |
| Local simulator: `sim.Simulator(textures, seed, motion_noise=..., motion_seed=...)`, `.step(action, ticks)`, `.pose() -> (x, y, yaw)`, `.distance_heading()`, `.targets()`, `.record(...)` | `rl/env.py`, `cli.py` |
| Pose, map and goal position are never sent by the server | `my_agent.py` docstring |

### Established by stage 0 (see `reports/stage_0.md`)

| Fact | Value |
|---|---|
| Seeds | dev 7, 8, 9, 10; test 21, 22, 23, 24, 25 (`source/mapping/seeds.json`, frozen) |
| Ground-truth datasets | `data/gt-<seed>/` with `traj_<i>/poses.json` (true `[x, y, yaw]` per saved frame) and `gt.json` |
| Intrinsics | fx = fy = 92, cx = 160, cy = 120, 320 x 240; far plane 5.0 m (`cache/mapping/camera.json`); horizontal field of view about 120 deg |
| Motion constants | tick 0.01 s; FORWARD 0.03 m/tick; LEFT/RIGHT 0.06 rad/tick; default noise sigma (0.3, 0.1) per tick |
| Motion between saved frames | normal: translation p95 <= 0.155 m (max 0.174), rotation p95 <= 17.3 deg. **Collisions:** 88 pairs (0.31 %) rotate > 30 deg and 27 (0.09 %) > 45 deg, max 54.8 deg, under FORWARD/IDLE labels with only 0.04 to 0.07 m translation, on 8 of 9 seeds |
| Action labels | one action per record, but it is the last tick of the 5-tick window only (62 % of windows mix actions) |
| Spawn | every drive starts at (0.2, 2.6, 0); first saved frame is 5 ticks later |
| Maze grid (simulator) | 31 x 31 cells of 0.2 m in `gt.json["grid"]` (`#` = wall cell), cell (r, c) centred at (c * 0.2, r * 0.2) |
| **Rendered wall geometry** | walls are **thin vertical planes through wall-cell centres**, not 0.2 m blocks: a wall cell contributes a 0.2 m segment along x if a left/right neighbour is a wall, along y if an up/down neighbour is; an isolated wall cell contributes an x segment. *Revision 5 (stage 2 correction):* a segment joins the centres of every pair of 4-adjacent wall cells; the wall extends 0.1 m past a cell centre only at a free end; corners and junctions end at the centre; half-thickness 2.2 mm (`camera_fit.wall_segments`, explains 99.2 % of boundary columns). Exact set: `camera_fit.Scene(...).segments` (evaluation only). A one-cell corridor is 0.4 m wide between wall planes |
| Wall height | 0.30 m |
| Camera height `h` | **0.2099 m** (+/- 0.001), = `height_offset` 0.15 + robot origin 0.06 |
| Camera offset from rotation centre | axial 0.3 +/- 0.8 mm, lateral -0.8 +/- 0.9 mm: **treat as zero** (turning in place is pure rotation) |
| Goal-view camera | **The four goal views are rendered from h = 0.160 m** (exploration and live frames: 0.210 m), same intrinsics; counter-clockwise order (front, left, back, right = theta + k 90 deg). Measured on every local seed and on the challenge `target.jpg` (stage 8); `goal.py` measures h per goal image. |
| Robot collision radius | **about 0.12 m to the collision segments** (centres of 4-adjacent wall cells, *without* the rendered 0.1 m free-end extension; an isolated cell is a point). True poses never come closer than 0.117 m to them on any of the 9 seeds (stage 7). Stage 0's 0.08 m was the closest approach to the *rendered* planes, reached at wall ends; along faces the closest approach is 0.117 m. |
| Far plane | 5.0 m depth; nothing drawn beyond; background white (255, 255, 255), same as sky. Floor reaching the far plane ends at row 123.86 |
| Floor palette (BGR) | blue tile (218, 181, 157) drifting to about (222, 195, 178) with distance (fog); grey tile (232, 232, 232). Stage 0's figures were on clean renders. *Revision 5:* on the JPEG frames the pipeline sees, floor = pixels within 5 levels of the line from (218, 181, 157) to (232, 232, 232); boundary found with a loose 30-level palette and a per-column step fit because of JPEG ringing next to dark walls (`segment.py`). Stage 0's trackability valid-column statistics used the old palette and are pessimistic |
| Depth resolution | one boundary pixel = z^2 / (fy h): 0.05 m at 1 m, 0.21 m at 2 m, 0.47 m at 3 m, 0.83 m at 4 m |
| Cue statistics (stage 0 audit, provisional detectors) | SIFT inliers vs previous frame: median about 30, 16.9 % of frames under 15 (mostly corridor views of dark walls). Valid boundary columns: median 0.54 to 0.58, 0.11 % of frames under 0.20. Vanishing point "found" on 88 % of frames but agreeing with true heading within 3 deg on only 39 % of those. Blind frames: 1 in 28,837 |
| Challenge data | has `target.jpg` (4 views, 240 x 1280); no `gt.json` or `poses.json` |
| **Challenge frame rate (stage 4b)** | `ab7efaeb1abf` is **one frame per tick** (vision: 0.029 to 0.031 m and 3.5 deg per frame; label repeats 80 % vs 45 % in 5-tick local data). Its labels are therefore exact per-tick actions. Unknown whether every server download is like this |
| Motion bias per route (stage 4b) | `Simulator.record` draws a new bias per route (e.g. 0.44x to 1.29x forward speed within one biased recording); calibration is per drive (`calibration.py`). Server drives suggest per-drive forward scales 0.96 to 1.05 (vision, no ground truth) |
| Holdout seeds | 31, 32, 33, 34, 35 (`seeds.json` `holdout`), recorded as `gt` and `gt4`, guarded by `record_gt.gt_dir` / `allow_final()`; unopened |
| Odometry estimators (stage 4b) | symmetric ICP (forward and inverted reverse, averaged) and wall features with frame k's depth from its own scan (row ratio as veto and fallback); straight-pair bias +0.14 % (ICP) and -0.43 % (wall), fused straight-segment ratio -0.64 % |
| Camera level | pitch < 0.3 deg; fitted cy = 120.000; stage 3: roll < 0.1 deg, no pitch trend |
| **Anti-aliasing mismatch (stage 3)** | local recordings use `msaa=1` (simulator default, not overridden by `record_exploration_data`); the server renders with `msaa=4`. Challenge frames: 70 % of boundary columns have sub-pixel blends vs 13 % locally. Results on `msaa=4` re-renders are equal or better, so `msaa=1` numbers are conservative |
| Heading (stage 3) | fused VP + scan-direction cue, valid only when the cues agree within 3 deg (`heading.py`). Scan cue is the stronger one (median 0.17 deg vs 0.39 deg). 4 % of frames are conflicts (no heading). **15 confidently wrong frames in about 32,000** (both cues agree, error up to 37 to 44 deg), all with estimated yaw mod 90 >= 31 deg, mostly in-place rotation in open areas: a valid heading is a measurement to gate, not ground truth |
| Wall-top ranging (stage 3) | `Z = f_y (H - h) / (c_y - v_top)`, status `TOP_RETURN` in `scan.py`; recovers 14.9 % of columns (error 0.45 mm median at 0.085 to 0.16 m); coverage 0.80 -> 0.95; stage 2 criteria still hold. 1,599 columns under 0.15 m near corners show 0.37 m error on both edges (unexplained; use a robust loss) |
| Coverage | the three drives visit 100 % of free cells and pass the goal cell on every seed |

### Unknowns stage 0 had to resolve (all answered; kept for the record)

- Full schema of `data_info.json` (any extra fields? how combined actions are encoded?).
- Whether downloaded **challenge** data includes `target.jpg`.
- Whether the 3 drives start at the same pose.
- Whether `Simulator.record()` can log true poses; if not, how to obtain a true pose for each recorded frame.
- How to get the intrinsic matrix offline (from the simulator or a `SessionInfo` instance) so the offline pipeline does not need a server session.
- Whether the camera is level (zero pitch and roll) and at the robot's rotation centre.

---

## Stage 0: Data audit and ground-truth harness

**Status (revision 3): complete. Do not rerun.** Tasks 1 to 8 done; one failed criterion (rotation gate) is handled in stages 1 and 4. The text below is kept as the record of what was asked.

**Goal:** know exactly what the data contains, and have local datasets with a true pose for every frame.

**Tasks**
1. Inspect `vis_nav_sim` (`help(sim.Simulator)`, source if available) and document the API relevant to recording, stepping, pose and camera intrinsics.
2. Inspect one local exploration dataset (`uv run source/keyboard_agent.py --local 7` records `data/local-7/`) and, if available, one challenge dataset. Document the schema.
3. Write `source/mapping/record_gt.py`: for a seed, produce a dataset in the **same format** as `record_exploration_data` plus a `poses.json` with the true `(x, y, yaw)` of every saved frame, aligned by frame index. If `record()` cannot log poses, reproduce its driving logic with `step()` and log `pose()` at each capture.
4. Record ground-truth datasets for all dev and test seeds. Choose seeds and write them to `source/mapping/seeds.json`.
5. Save the intrinsic matrix K and image size to `cache/mapping/camera.json`.

**Revision 2 tasks (rerun only these; tasks 1 to 5 are done):**

6. **Camera height `h` and camera offset from the rotation centre.** Vision-only odometry takes its metric scale entirely from `h`, so this can no longer be deferred. The earlier estimate was confounded by not knowing where wall faces sit; the true grid now fixes them (a wall cell centred at (c * 0.2, r * 0.2) has faces at +/- 0.1 m). *(Revision 3 note: this assumption was wrong; stage 0 found walls render as thin planes through cell centres. See the facts table.)*
    - Height: with `Simulator.render(x, y, yaw)`, place the camera facing a known wall face squarely at 8 or more known distances (0.3 m to 3 m). In each render, find the floor-wall boundary row in the central columns (sub-pixel, e.g. from the vertical intensity profile). Fit `v_b - c_y = f_y * h / Z` jointly for `h` and an axial offset `a` (Z = true distance from the rotation centre to the face minus `a`). Also fit the wall-top row against the known wall height if the simulator exposes it, as a cross-check.
    - Offset: render one position at yaws spanning 0 to 360 degrees in steps of at most 15 degrees. If the camera sits on the rotation axis, view pairs are related exactly by H = K R K^-1. Measure the residual parallax (e.g. RANSAC homography fit residual on SIFT matches, or direct comparison against the warped image); fit the axial and lateral offset that explains it. Cross-check against `a` from the height fit.
    - Reconcile with the simulator's stated `height_offset` = 0.15 and the earlier 0.21 m estimate: explain any difference (for example, eye height measured from the floor vs from the robot origin).
    - Save `h`, axial offset, lateral offset and their uncertainties to `cache/mapping/camera.json` (add fields; do not change the existing ones).
7. **Trackability audit** (decides whether vision-only odometry is feasible). On every seed and drive:
    - From `poses.json`, the true motion between consecutive saved frames: distributions of the translation and rotation, their maxima, and the fraction of pairs with rotation above 45 degrees (must be 0 for vanishing-point unwrapping).
    - For every consecutive frame pair: number of wall-feature RANSAC homography inliers (SIFT on the wall mask, ratio test 0.8), the fraction of image columns with a valid floor-wall boundary in each frame, and whether a vanishing point is found (a quick version of stage 3 is fine; note it is provisional).
    - Classify a frame as **blind** when all three cues fail (inliers below 15, valid columns below 20 %, no vanishing point). Report the blind fraction and the longest run of consecutive blind frames per drive, and show 10 example blind frames.
8. **Far plane.** Render a long straight corridor view beyond 5 m. Document what appears past the far plane (sky colour, background, nothing) and how a column whose floor reaches the far plane should be classified (no return vs wall). This sets `r_max` in stage 2.

**Deliverables**
- `reports/stage_0.md`: API notes, `data_info.json` schema, answers to every unknown listed above, chosen seeds.
- `source/mapping/record_gt.py`, `source/mapping/seeds.json`, `cache/mapping/camera.json`.
- `reports/figures/stage0_true_trajectories_<seed>.png`: the 3 true drives per seed, coloured per drive, start and end marked.

**Acceptance criteria**
- [ ] Every unknown in the list above is answered with evidence (code reference or measurement), or explicitly marked unresolvable with a stated fallback.
- [ ] Pose alignment check: for 20 random frames per seed, re-render (or re-step) the simulator at the logged pose and confirm the image matches the saved frame (mean absolute pixel difference < 2 on 0-255 scale, or exact match if rendering is deterministic). If re-rendering is impossible, document an equivalent check.
- [ ] Our recorder produces frames identical to `record_exploration_data` for the same seed (same count, same images, same actions).
- [ ] At least 3 dev and 3 test seeds recorded; the split is frozen in `seeds.json`.

**Revision 2 acceptance (tasks 6 to 8):**

- [ ] `h` determined to +/- 2 mm and the axial offset to +/- 5 mm (95 % interval from the fit), with RMS boundary-row residual below 0.5 px. Reason: a 1 % error in `h` is a 1 % scale error over the whole 6.2 m maze (about 6 cm), most of the stage 6 pose budget.
- [ ] Offset model explains the rotate-in-place renders: residual reprojection error below 0.5 px after compensation.
- [ ] Trackability: rotation between consecutive saved frames never exceeds 45 degrees on any seed.
- [ ] Trackability: the longest blind run is at most 3 frames on every seed, **or**, if not, the report lists where blind runs occur and the drives are marked for segmentation (stage 4 will then need relocalization to rejoin segments). This criterion is a feasibility gate: report the numbers either way.
- [ ] Far-plane behaviour documented, with a rule for classifying such columns.
- [ ] `reports/stage_0.md` gains an `## Addendum (revision 2)` with the above, plus figures `stage0_height_fit.png`, `stage0_rotation_parallax.png`, `stage0_trackability_<seed>.png` (inliers, valid columns and vanishing-point validity per frame), `stage0_blind_examples.png`.
- [ ] `uv run pytest` passes, including new tests for the height-fit function on synthetic data.

---

## Stage 1: Action labels as a weak prior, and the dead-reckoning reference

**Status (revision 4): complete and accepted. Do not rerun.** Collision tasks moved to stage 3b.

**Goal (revision 2):** a calibrated, honest prior on the motion between consecutive saved frames given only the action label, plus the dead-reckoning reference numbers. Vision drives odometry (stage 4); this prior only fills gaps (blind frames) and regularizes.

**Why this changed:** stage 0 showed a label is only the last tick of a 5-tick window, so a label does not determine the motion. Fitting nominal per-action constants (0.15 m, 0.30 rad) would be wrong; the prior must be the empirical distribution of true motion given the label.

**Tasks**
1. Implement `source/mapping/se2.py`: compose, inverse, relative pose, angle wrap, Jacobians.
2. On **dev** seeds, for every consecutive pair of saved frames, compute the true relative motion `(dx, dy, dtheta)` in the robot frame of the first pose (from `poses.json`). Group by the label of the first record (FORWARD, LEFT, RIGHT, IDLE).
3. Per label, fit the prior: mean and 3 x 3 covariance, plus a description of the shape (histograms; note multimodality, e.g. a LEFT label after a mostly-forward window). If a single Gaussian is a poor description, use a Gaussian mixture with at most 3 components, or inflate the covariance so the 95 % ellipse covers 95 % of the samples; document the choice.
4. Check whether the previous record's label adds information (prior conditioned on the label pair). Keep it only if it shrinks the covariance determinant by more than 30 % on dev seeds.
5. Implement `source/mapping/motion.py`: label (and optionally previous label) -> prior mean and covariance.
6. Dead reckoning with prior means from the true spawn (0.2, 2.6, 0); ATE and heading RMS per drive against truth. Reference numbers only.

**Deliverables**
- `cache/mapping/motion_prior.json` (per-label mean, covariance or mixture, sample counts, dev seeds used).
- `source/mapping/se2.py`, `source/mapping/motion.py`, tests.
- `reports/stage_1.md` with: per-label scatter of true `(dx, dtheta)`, fitted ellipses, dead-reckoned vs true trajectories, position and heading error vs frame index.

**Acceptance criteria**
- [ ] SE(2) unit tests pass: `inv(a) (+) a = identity`, composition associativity, angle wrapping at +/-pi, numerical vs analytic Jacobians agree to 1e-6.
- [ ] Prior calibration on **test** seeds: for every label, the fraction of true relative motions inside the prior's 95 % region lies between 0.90 and 0.99. Below 0.90 the prior is overconfident (the pose graph will trust it too much); above 0.99 it is uselessly wide.
- [ ] ~~Per-label prior means agree between dev and test seeds within 2 standard errors.~~ Revision 4: per-label prior means agree between dev and test seeds within a practical tolerance: |delta dx| and |delta dy| < 5 mm, |delta dtheta| < 1 degree (the prior is not seed-specific in any way that matters). Measured: at most 0.9 mm and 0.7 degrees: **pass**.
- [ ] Dead-reckoning ATE and heading RMS are reported per seed. **No threshold**: these are the reference for stage 4.

---

## Stage 2: Floor segmentation and the virtual laser scan

**Status (revision 5): complete and accepted. Do not rerun.** `R_MAX` = 1.5 m for mapping; scan endpoint error median 2.8 mm, p95 23 mm.

**Goal:** convert each frame into a 2-D range scan in the robot frame.

**Tasks**
0. **Reference check first (revision 3).** Stage 0's `camera_fit.Scene(...).segments` explains 93 % of boundary columns within 0.5 px, but 6.6 % deviate by more than 2 px, cause unknown (hypothesis: grazing views, segment ends). Find the cause on dev seeds (plot the deviating columns' back-projected points over the segments; check segment ends, T-junctions, isolated wall cells) and fix the segment rule if it is wrong. The reference must be trustworthy before it scores anything.
1. Floor/sky/wall segmentation. Reuse stage 0's measured palette and segmentation (blue tile (218, 181, 157) drifting to about (222, 195, 178) with distance, grey tile (232, 232, 232), sky and far-plane background (255, 255, 255); recall 0.995, no wall pixels accepted). Promote it from the stage 0 audit code to `segment.py` with tests; keep the border-connectivity prior from the `mask.py` prototype.
2. Per image column, find the floor-wall boundary row `v_b(u)` (lowest wall pixel above the connected floor), sub-pixel from the anti-aliased blend as in stage 0. Far-plane rule (stage 0): a column is a **wall return** only if the first non-floor pixel above the floor is a wall pixel and the pixel two rows above it is not white; if it is white, the floor reaches the far plane: **no return** (free space up to `r_max`, nothing beyond), never a wall at 5 m.
3. Back-project with `h` = 0.2099 m and zero camera offset: `Z = f_y * h / (v_b - c_y)`, `X = Z * (u - c_x) / f_x`, giving range and bearing per column. Clip at `r_max`: start at 2.5 m (depth error per pixel grows as z^2 / (fy h): 0.21 m per pixel at 2 m, 0.47 m at 3 m) and settle it from the error-vs-range plot; 5.0 m is the hard ceiling.
4. Cross-check `h` (do not re-estimate it): with true poses, a systematic range bias proportional to range means `h` is off, so go back to stage 0.
5. Reference geometry (evaluation only, rule 10): the wall **segments** from `camera_fit.Scene(...).segments` (thin planes through wall-cell centres; corridors 0.4 m wide between planes). Score scan endpoints by distance to the nearest segment. Also build an **oracle map** (occupancy grid from scans placed at true poses, simple version of stage 7) for stage 7 to compare against.

**Deliverables**
- `source/mapping/segment.py`, `source/mapping/scan.py`, tests.
- `reports/stage_2.md` with: mask overlays for 50 random frames (contact sheet), range error vs range plot, chosen `r_max` with its justification.

**Acceptance criteria**
- [ ] Mask review: on the 50-frame contact sheet, no frame has wall texture labelled as floor over more than 5 % of the wall area (visual check, list the failures).
- [ ] Task 0 done: the cause of the unexplained boundary columns is identified and the segment reference explains at least 99 % of boundary columns within 0.5 px on dev seeds (or the remaining columns are characterized and excluded from scoring, with the reason).
- [ ] Scan accuracy against the reference segments (scans placed at true poses): for returns with range <= `r_max`, median absolute distance from the scan endpoint to the nearest segment < 0.03 m, 95th percentile < 0.10 m.
- [ ] No range bias: the median signed range error, binned by range, stays within +/- 1 % of range in every bin up to `r_max`.
- [ ] Coverage: at least 60 % of columns per frame return a valid range on average (otherwise the scan is too sparse to match).
- [ ] Unit test: synthetic floor point projected with K, then back-projected, recovers `(X, Z)` to 1e-9.

---

## Stage 3: Vanishing-point compass

**Status (revision 6): complete and accepted (including the revision 5 additions below stage 3b). Do not rerun.**

**Goal:** an absolute yaw measurement (modulo 90 degrees) per frame, relative to the maze axes.

**Tasks**
1. Line segments with LSD (`cv2.createLineSegmentDetector`; fall back to `cv2.ximgproc.createFastLineDetector` if unavailable). Keep only segments on floor/seam regions (mask from stage 2, dilated 2 px; segment counted if > 60 % of its samples fall in the band). Drop segments shorter than 8 px.
2. Verify the level-camera assumption from data: near-vertical segments should have angle 90 +/- 1 degree. If not, stop and report.
3. Intersect each kept segment's line with the horizon row `v = c_y`, histogram the x values (length-weighted), find peaks, and solve for yaw with the Manhattan constraint `(x1 - c_x)(x2 - c_x) = -f^2` when two peaks exist, or from the single near-centre peak otherwise.
4. Output per frame: yaw mod 90 degrees, a confidence, and a validity flag.
    - **Validity test (revision 3).** Stage 0's lenient detector reported a vanishing point on 88 % of frames but agreed with true heading within 3 degrees on only 39 % of them, so "found" must not mean "valid". Require at least: two peaks consistent with the Manhattan constraint and the known `f` (within 5 %), or one strong near-centre peak supported by segments from both sides of the image; reject segments from oblique seams (wall bottoms of walls not aligned with the peak direction); minimum total supporting segment length. Tune on dev seeds.
5. Maze-axis offset: the true grid is axis-aligned with the world frame and yaw is measured CCW from +x (stage 0), so the offset should be 0. Verify on dev seeds and report.

**Deliverables**
- `source/mapping/vp.py`, tests.
- `reports/stage_3.md` with: yaw error histogram, error vs confidence plot, the focal length implied by VP pairs compared with K.

**Acceptance criteria**
- [ ] Validity coverage: a valid yaw on at least 70 % of frames.
- [ ] Accuracy on valid frames (error computed modulo 90 degrees against true yaw): median < 1.5 degrees, 95th percentile < 4 degrees.
- [ ] Focal length from VP pairs agrees with K's `f_x` within 5 % (median over frames with two VPs).
- [ ] Validity is meaningful: among frames flagged valid, at least 95 % have yaw error under 3 degrees (stage 0's detector: 39 %).

---

## Stage 3b: Collision handling inputs (revision 4; moved from stage 1)

**Goal:** before visual odometry starts, (a) a motion prior that does not veto collision rotations and (b) a detector that flags collision pairs, built on the cues from stages 2 and 3.

**Why here:** stage 1 showed the collision tail is only partly inside its mixtures' 95 % region, so stage 4 gating needs an explicit component; and the detector's best signals come from stage 2 (floor boundary, scans, wall mask) and stage 3 (vanishing points).

**Tasks**
0. **Server-like recordings (revision 6).** For every dev and test seed, record a second dataset `data/gt4-<seed>/` with the same seed, routes, `capture_every=5`, motion noise and `motion_seed` as `data/gt-<seed>/`, but `msaa=4`, and with poses (`student_files=False`). Checks:
    - `poses.json` and `data_info.json` identical to the `gt-<seed>` versions (only rendering changes); if not, stop and report.
    - Boundary blend fraction comparable to the challenge frames (about 70 % of boundary columns fractional).
    - Compare the JPEG quantization tables of the challenge frames with the local encoder's; report whether they match (quality, chroma subsampling).
    - From here on: **tune thresholds on `gt4` dev seeds; every acceptance criterion must pass on both `gt4` and `gt` test seeds.** Re-run nothing from stages 0 to 3 except as needed to produce the cue inputs for 3b and 4 on `gt4`.
1. **Collision component.** Stage 0 found wall collisions: under FORWARD/IDLE labels, 0.31 % of pairs rotate more than 30 degrees with only 0.04 to 0.07 m of translation. Extend the stage 1 prior (`MotionPrior`, per (previous, current) label mixtures) with a **collision component** (weight fitted on dev seeds, about 0.3 %; mean near zero translation; rotation broad, e.g. uniform over +/- 60 degrees or a wide Gaussian). Downstream gating (stage 4) must use this mixture so that a label never vetoes a large rotation seen by vision.
2. **Collision detector.** A function, using only pipeline-legal inputs (images, labels), that flags a consecutive frame pair as a likely collision. Candidate signals: low SIFT inliers vs the previous frame, a large change of the dominant floor-boundary orientation, a vanishing-point reading far from the prediction, small apparent translation under a FORWARD label. Tune on dev seeds against true events (true rotation > 30 degrees under FORWARD/IDLE).
3. Reuse stage 2 and 3 code for every signal; do not re-implement segmentation or vanishing points.

**Deliverables**
- Updated `cache/mapping/motion_prior.json` (collision component added, previous fields unchanged) and `source/mapping/motion.py`.
- `source/mapping/collision.py` (detector), tests.
- `reports/stage_3b.md`; `reports/figures/stage3b_collisions_<seed>.png`: the true trajectory with every collision event (> 30 degrees) marked, and for 6 events the frame before and after with the detector's signals.

**Acceptance criteria**
- [ ] Stage 1's calibration criterion (every label and (previous, current) group 0.90 to 0.99 inside the 95 % region on test seeds) still holds **with** the collision component included.
- [ ] Every test-seed collision event (true rotation > 30 degrees under FORWARD/IDLE) lies inside the extended prior's 99 % region.
- [ ] Collision detector on test seeds: recall 1.0 on events with true rotation > 45 degrees (19 events), recall >= 0.9 on events > 30 degrees, and at most 1 % of all pairs flagged. Reason: a missed > 45 degree event can cause a permanent 90 degree heading error; a false flag only costs extra computation in stage 4.
- [ ] Leakage test (rule 10): the detector uses only images and labels.
- [ ] Task 0: `gt4` datasets recorded for all 9 seeds with identical poses and labels; JPEG settings comparison reported; detector criteria pass on both `gt4` and `gt` test seeds.

---

## Stage 3 additions (revision 5)

- **Second heading cue from scans.** In each frame's virtual scan (stage 2), fit the dominant wall direction (e.g. a histogram of local segment orientations mod 90 degrees, length-weighted). Output yaw mod 90 with a confidence, fused with the vanishing-point yaw. Report accuracy and validity for VP alone, scan alone and fused; the stage 3 acceptance criteria apply to the **fused** output.
- **Wall-top ranging.** 15.2 % of columns are INVALID (wall nearer than about 0.16 m, base below the image). Wall height is 0.30 m, so the top edge gives Z = f_y (H - h) / (c_y - v_top) = 92 x 0.09 / (c_y - v_top). Add it to `scan.py` for INVALID columns and as a cross-check elsewhere; report its error vs range on test seeds (expect about 2.3 x the floor-edge error per pixel). Stage 2's criteria must still hold with it on.

---

## Stage 4: Visual odometry with an action prior

**Status (revision 7): accepted pending stage 4b.** Do not rerun stage 4 as a whole; stage 4b changes the estimators and re-checks the stage 4 criteria. Recorded deviations, accepted: (1) the floor homography was not built (scan ICP plus wall-feature motion constrain every direction on these data); (2) parallel filter hypotheses were not built (the single best hypothesis was right on all 88 events; 2 test events had a margin below 0.15; the `ambiguous` code path stays and is a known risk). Note for later stages: variant B (vision only) is about 2 x worse than C on `gt4`, so the action labels matter, mostly along corridors and around collisions.

**Goal (revision 2):** per-drive trajectories driven by vision, with action labels only as a weak prior; and a measured answer to "how much do the actions add?".

**Measurements available per consecutive frame pair (k, k+1):**
- Absolute yaw mod 90 degrees from the vanishing points (stage 3), unwrapped to the branch nearest the predicted heading **except on collision-flagged pairs** (stage 3b detector), where the nearest-branch rule is unsafe: stage 0 found 27 pairs rotating more than 45 degrees during wall collisions, and a wrong branch is a permanent 90 degree heading error for the rest of the drive.
- Relative planar motion from the **floor homography**: floor correspondences (tile edges, seam points; or dense alignment of the inverse-perspective-mapped floor) between k and k+1, solved for `(dx, dy, dtheta)` with known K and `h` = 0.2099 m. Metric, no scale ambiguity. Use near floor only (far floor aliases and depth error grows as z^2).
- Relative planar motion from **scan-to-scan ICP** between the virtual scans of k and k+1 (stage 2; *revision 5:* use returns up to 2.5 to 3 m weighted by a range-dependent standard deviation sigma(z) proportional to z^2 / (f_y h), not the 1.5 m mapping cut; robust loss), with its covariance (e.g. Censi's closed form, or from the Hessian). This covariance exposes the long-corridor degeneracy: motion along a corridor with featureless walls is weakly constrained.
- Relative motion from **wall-feature homographies** (SIFT on the wall mask, RANSAC), decomposed with the wall depth from the scan. This constrains motion along corridors when walls are textured.
- Camera offset from the rotation centre is zero (stage 0), so camera motion equals robot motion.

**Tasks**
1. Implement each relative-motion estimator as a function returning `(dx, dy, dtheta)` and a 3 x 3 covariance, or "no measurement". Unit-test each on rendered pairs with known true motion (dev seeds).
2. Fuse per frame pair: combine the available measurements (information-weighted) into one relative-motion measurement. Gate outliers against the stage 1 **mixture** prior plus the stage 3b collision component, never against the normal component alone, so a FORWARD label cannot veto a large rotation that vision sees.
3. **Collision pairs (revision 3).** For every pair flagged by the stage 3b detector:
    - Estimate rotation globally first: correlative scan matching over the full rotation range (ICP alone converges only from roughly 20 to 30 degrees away), then refine with ICP.
    - Enumerate heading hypotheses `theta_pred + vp_residual + k * 90 deg` for k in {-1, 0, +1}; score each by scan-to-scan alignment (strong at collisions: boundary columns stay at 0.4 to 0.8) plus wall-feature inliers (weak at collisions: about 8).
    - If the best hypothesis does not win clearly (margin threshold tuned on dev seeds), keep the top two as parallel filter hypotheses for up to 5 frames and let later frames decide (a small multi-hypothesis filter).
    - Inflate the covariance of the pair's odometry edge accordingly.
4. Filter (EKF on `(x, y, theta)`): predict with the **action prior** from stage 1; update with the fused relative motion and the vanishing-point yaw. On blind frames (stage 0.7), predict only, so covariance grows.
5. Run and compare three variants on every seed:
    - **A**: action prior only (= stage 1 dead reckoning).
    - **B**: vision only (predict with a constant-velocity model with wide covariance instead of the action prior).
    - **C**: vision + action prior (the default).
6. Blind runs: stage 0 found at most 1 consecutive blind frame, so no drive needs splitting. Keep the splitting code path only if a test seed ever produces a blind run longer than 3.
7. Output per drive (variant C): poses, covariances, and the relative-pose measurements with covariances between consecutive frames (odometry edges for stage 6).

**Deliverables**
- `source/mapping/vo.py` (estimators), `source/mapping/odometry.py` (filter), tests.
- `reports/stage_4.md` with: A/B/C trajectories vs truth per test seed; an error table; relative pose error vs window length; NIS over time; a plot of the along-corridor standard deviation from ICP vs frame index; a per-collision-event table (true rotation, chosen branch, hypotheses kept, heading error 5 frames later).

**Acceptance criteria** (test seeds, per drive, no loop closures)
- [ ] Heading (variant C): RMS < 2 degrees.
- [ ] Relative pose error over 10-frame windows (variant C): translation median < 3 % of the true distance travelled in the window, 95th percentile < 8 %; rotation median < 1 degree.
- [ ] ~~Scale (variant C): estimated total path length within 1 % of the true path length.~~ Revision 7: moved to stage 4b with a better metric (summing step lengths counts noise on turns as path, which hides a forward bias).
- [ ] **Collisions (revision 3):** every test-seed event with true rotation > 45 degrees (19 events) and every event > 30 degrees resolves to the correct 90 degree branch: heading error under 5 degrees five frames after the event. A single wrong branch fails the stage.
- [ ] Variant C ATE lower than variant A on every test seed.
- [ ] Filter consistency (variant C): NIS of the fused relative-motion update within the 95 % chi-square bound for 3 degrees of freedom (7.81) on at least 90 % of updates.
- [ ] **Report, no threshold:** variant B vs C. If B's ATE is within 20 % of C's on every test seed, the pipeline does not depend on the action labels (robust to an unknown server `motion_bias`): state this conclusion explicitly. If B is much worse, say where (collisions, long corridors) with figures.
- [ ] ~~Confidently wrong headings (revision 6): none accepted, filtered error < 2 deg.~~ Revision 7, restated to the actual danger: **no** stage 3 heading wrong by more than **10 deg** is accepted by the gate, and the filtered heading error at every confidently wrong frame stays under **3 deg on `gt4`** (report `gt` too). Stage 4 result: every frame wrong by > 10 deg rejected; the accepted ones were wrong by 3.1 and 6.2 deg; filtered error 1.81 deg on `gt4` (6.10 on `gt`, reported): **pass**.
- [ ] Leakage test (rule 10) passes for the stage 4 pipeline.

---

## Stage 4b: Holdout seeds, scale bias, prior self-calibration (revision 7)

**Status (revision 8): accepted. Do not rerun.** Residual straight-segment ratio -0.64 % accepted and re-measured in stage 6; the similarity-scale criterion moved to stage 6; the turn-scale criterion restated as "within 1 % of each drive's true scale" (met to about 0.5 %). Known gaps: the `--final` CLI flag is a hook only (wire it in stage 8); turn scale fitted about 0.5 % low (symmetric ICP under-measures rotation by about 0.7 %; a heading-cue rotation measurement would avoid it).

**Goal:** an unbiased final exam, odometry without a forward scale bias, and a label prior that adapts to whatever motion bias a dataset has.

**Tasks**
0. **Holdout seeds.** Choose 5 new seeds (not 7 to 10, 21 to 25), record them as `gt-<seed>` and `gt4-<seed>` exactly like the others (with poses), check poses and labels are identical between the two, and add them to `seeds.json` under `holdout`. Do not open, plot or run anything on them after recording (rule 4). Add a test that the pipeline's evaluation scripts refuse holdout seeds unless called with an explicit `--final` flag.
1. **Scale metric.** Replace "total path length" with two measures that do not count noise as path:
    - the scale factor of a **similarity alignment** (Umeyama with scale) of each estimated drive to the true one;
    - **straight-segment displacement**: for runs of at least 10 consecutive FORWARD pairs with true rotation under 2 deg in total, the ratio of estimated to true displacement (endpoint to endpoint).
2. **Find the forward bias.** Stage 4 showed ICP about -1.2 % and wall-feature motion -1.7 % on straight pairs, while raw frontal-wall depth differences are unbiased (0.3 %) and `h` is right. Test at least these two hypotheses on dev seeds, each with a controlled experiment:
    - *ICP prior shrinkage:* the weak Gaussian prior around ICP's initial guess pulls toward the label prior's mean, which for FORWARD (0.112 m, label-only; 0.149 m after FORWARD) can be short. Run ICP from the true motion, from zero and from the label prior; if the bias tracks the initial guess, weaken the prior or make it non-informative along the motion direction.
    - *Errors-in-variables in the wall-feature depth:* `Z2 = Z1 (v1 - cy) / (v2 - cy)` divides by a noisy quantity, and E[1 / (a + noise)] > 1 / a, so Z2 is biased high and forward motion low; the rigid fit with noise in both point sets also shrinks scale (regression dilution). Try depths taken from each frame's own scan, and a total-least-squares or symmetric fit.
    Report the bias of each estimator before and after on straight pairs (dev and test).
3. **Label-prior self-calibration** (rule 10 compliant: images and labels only). On each dataset, fit a per-label scale (and offset) of the stage 1 prior from the pairs where ICP and wall features are both accepted and well constrained, by maximum likelihood or EM; apply it on that dataset. Validate: on local seeds the fitted scale must be close to 1 (no bias recorded locally); then inject a synthetic bias by re-recording **one dev seed** with `motion_bias` set (e.g. 0.9 x forward, 1.05 x turn) and show the calibration recovers it and variant C's ATE stays within 20 % of the unbiased case. Report the fitted scales on the challenge data `ab7efaeb1abf` (reported only; no ground truth there).
4. Re-run variant C (and B, A for reference) on dev and test seeds with the fixed estimators and the calibrated prior; re-check every stage 4 criterion.

**Deliverables**
- `reports/stage_4b.md`, updated `vo.py` / `odometry.py` / `motion.py`, tests; `seeds.json` with `holdout`; holdout datasets on disk (unopened).
- Figures: bias per estimator before / after on straight pairs; similarity-scale per drive; recovered vs injected bias.

**Acceptance criteria** (test seeds; `gt4` primary, `gt` reported)
- [ ] Holdout seeds recorded, poses and labels identical between `gt` and `gt4`, guarded by the `--final` test, and not opened.
- [ ] Mechanism of the forward bias identified with a controlled experiment, or the report states which hypotheses were ruled out and how.
- [ ] Similarity-alignment scale of variant C within **0.5 %** of 1 for the median drive and within **1 %** for every drive; straight-segment displacement ratio median within **0.5 %** of 1.
- [ ] All stage 4 criteria (revision 7 wording) still pass on `gt4` test: heading RMS < 2 deg; 10-frame RPE median < 3 %, p95 < 8 %; collisions 63 / 63; C beats A; NIS >= 90 %; confidently wrong headings as restated; leakage.
- [ ] Self-calibration: fitted scale within 1 % of 1 on every local seed; recovers an injected bias within 1 % on the biased dev seed, with variant C ATE within 20 % of the unbiased run. Fitted scales on the challenge data reported.
- [ ] Any change made after looking at a test-seed result is disclosed, as in stages 3b and 4.

---

## Stage 5: Loop closures

**Goal:** confirmed revisit pairs with a measured relative pose and covariance, within and across drives.

**Tasks**
1. Candidates: VLAD place recognition (reuse `vlad.VLADExtractor`, but compute SIFT on the **wall-only mask** from stage 2; use a separate cache key). For each frame, top-5 candidates excluding frames within 10 indices of it in the same drive.
2. Verification: SIFT ratio-test matches + RANSAC homography; accept if inliers >= a threshold tuned on dev seeds.
3. Relative pose, two methods, compared on dev seeds:
   - (a) decompose the homography (`cv2.decomposeHomographyMat`), keep the solution with positive depth and a horizontal-plane normal consistent with a vertical wall, set scale from the scan range to that wall;
   - (b) 2-D ICP between the two frames' virtual scans (range-weighted returns up to 2.5 to 3 m, as in stage 4), initialized from (a) or from the identity.
   Keep whichever is more accurate; if both work, require agreement within 0.1 m and 5 degrees as an extra check.
4. Covariance per closure from ICP residuals or an empirical value from dev seeds.
5. Ground-truth revisit definition (for scoring only): true positions within 0.5 m and headings within 30 degrees.

**Deliverables**
- `source/mapping/loops.py`, tests.
- `reports/stage_5.md` with: precision-recall curve over the inlier threshold, chosen threshold, relative-pose error histograms, a plot of accepted closures drawn on the true trajectories.

**Acceptance criteria**
- [ ] Precision >= 0.98 at the chosen threshold (at most 1 false closure in 50).
- [ ] ~~At least one accepted closure between every pair of drives on every test seed.~~ Revision 9: the closure graph plus the spawn constraint connects all drives on every test seed. Stage 5 result: **pass**.
- [ ] ~~Recall >= 0.25 of ground-truth revisit pairs.~~ Revision 9: frame recall (revisiting frames with at least one closure) >= 0.25. Stage 5 result: 0.344 (`gt4`), 0.304 (`gt`): **pass**.
- [ ] Relative-pose error of accepted closures: median < 0.08 m and < 2 degrees.

---

## Stage 5b: Texture adjacency (revision 9)

**Owner: the user's Claude session (not the coding agent).**

**Status: done (`reports/stage_5b.md`).** Task 0: go (98.7 to 100 % of boundaries unique in their maze; every triple unique; three quarters of single faces have a twin). Pipeline `texture.py`: boundary-verified closures reach precision 0.998 on test **without the odometry gate** (2 false of 1,174, both 3.6 to 4.2 m off), frame recall 0.086; the union with the stage 5 gate keeps frame recall 0.344 at precision 0.998. The veto removes 4 of the 17 stage 5 aliases (13 have no face edge visible in one frame); those come from 3 places, each a 0.6 m (three-cell) shift. **Stage 6 uses the union closures, marks boundary closures gate-free, and keeps two-pass gating for the alias clusters and for the rare long-range boundary false closures.**

**Idea.** One texture out of about 200 repeats several times per maze. An *ordered pair* of textures meeting along a wall (A|B) has about 40,000 possibilities, so with K boundaries per maze the expected number of repeated pairs is about K^2 / 80,000 (3 for K = 500, 12 for K = 1,000); a triple is effectively unique. A boundary is also a single physical point, so it pins position along a corridor where tiled textures alias.

**Tasks**
0. **Go/no-go measurement (evaluation only, rule 10).** From recorded frames at true poses and the true wall segments, rebuild every wall face's appearance and measure: how textures are laid out (per cell face, per run, stretched or tiled), how many texture boundaries a maze has, and how unique ordered pairs and triples are (within a maze and across seeds). **Go** if, on dev seeds, at least 90 % of boundaries carry a pair that occurs once in their maze.
1. **Texture IDs (pipeline).** Per frame, cut each scan wall into faces, describe each visible face (colour statistics and a small rectified patch), and cluster into IDs across all frames of a dataset.
2. **Boundaries.** Where the ID changes along one continuous wall (the scan rejects occlusion edges: depth jumps), record the ordered pair, the side, and the boundary's position in the robot frame.
3. **Boundary-verified closures.** A stage 5 candidate is accepted without the odometry gate if both frames see the same ordered pair and the measured relative pose aligns the boundary positions within 5 cm.

**Acceptance criteria** (dev chooses, test reports, `gt4` primary)
- [ ] Task 0 measurement reported; go/no-go stated with the numbers.
- [ ] If go: precision >= 0.98 **without** the odometry gate; none of the stage 5 alias closures (0.4 to 0.6 m) accepted; frame recall >= 0.25; leakage test (rule 10).
- [ ] If no-go: the report says why, and stages 6 to 8 proceed with the stage 5 closures.

---

## Stage 6: Pose-graph optimization

**Status (revision 9): done by the user's Claude session (`reports/stage_6.md`).** Rigid-aligned ATE 0.013 to 0.027 m on test (stage 4b: 0.060 to 0.120 m), heading RMS <= 0.72 deg, all false closures (union set, 5 random, 17 real aliases) inactive, 4.5 to 10 s per seed. Additions beyond the plan: odometry covariance inflation (16x, chosen on dev), absolute heading priors every 10 frames (remove a global rotation about the spawn), IRLS on loop edges with first weights at the odometry. Scale misses marginally (median drive 0.55 % small, 14/15 drives within 1 %); **stage 7 should calibrate global scale from the detected 0.2 m lattice before snapping walls.** Optimised poses: `cache/mapping/<seed>/poses_optimized.json`.

**Goal:** one globally consistent trajectory for all 3 drives.

**Tasks**
1. Graph: nodes = every frame of every drive (or drive segment, if stage 4 split any); odometry edges from stage 4 variant C; loop-closure edges from stage 5.
2. Spawn constraint (revision 2): stage 0 showed every drive starts at the known spawn (0.2, 2.6, 0) and the first saved frame is 5 ticks later. Add one fixed spawn node at that pose and connect each drive's first frame to it with an edge whose mean and covariance come from the stage 1 action prior for its label. This fixes the gauge and puts the map in the true world frame. Segments that do not start at the spawn float and are tied in only by loop closures.
3. Solver: Levenberg-Marquardt on SE(2) with angle-wrapped residuals and a robust kernel (Cauchy or Huber; or switchable constraints). Use `scipy.optimize.least_squares` with a sparse Jacobian, or GTSAM if installable; document the choice.
4. Robustness test: inject 5 random false closures (random frame pairs with random relative poses) and re-optimize. *Revision 9:* also inject the real stage 5 alias closures (the 17 false test closures, 0.4 to 0.6 m off), which are much harder than random ones.
5. **Two-pass gating (revision 9).** Optimize once, then re-gate every closure with a Mahalanobis test against the optimized relative pose and its covariance (chi-square, 3 dof, 99 %), and optimize again. After the first pass the uncertainty is far smaller, so 0.4 m aliases fall outside the gate without a hand-tuned distance.
6. **Closure thinning (revision 9).** Stage 5 makes about 1,300 closures per seed from consecutive frames; they are strongly correlated, and treating them as independent makes the graph overconfident. Keep about one closure per revisit cluster (e.g. per pair of 5-frame windows), or down-weight by cluster size; report the effect on ATE and NIS.

**Deliverables**
- `source/mapping/posegraph.py`, tests (toy square loops, see criteria).
- `cache/mapping/<seed>/poses_optimized.json`.
- `reports/stage_6.md` with: before/after trajectories vs truth, ATE table (dead reckoning, EKF, optimized), robustness test results.

**Acceptance criteria**
- [ ] Toy-loop unit tests: (a) a 4-edge square with noise-free odometry and closure, started from a perturbed initial guess, converges to the true square within 1e-6; (b) the same square with 0.1 rad heading drift on one edge and an exact closure ends with the closure residual below 1e-3 and the error spread over all edges (no single edge carries more than half of it).
- [ ] Optimized ATE (all drives jointly, after one rigid alignment) <= 0.08 m RMS on every test seed.
- [ ] **Scale on loop-closed trajectories (revision 8, moved from stage 4b):** similarity-alignment scale within 0.5 % of 1 for the median drive and within 1 % for every drive; straight-segment ratio median reported with its confidence interval (stage 4b residual: -0.64 %). If the straight-segment shortfall still exceeds 0.5 %, report its effect on wall positions in the stage 7 map rather than calibrating it away.
- [ ] Optimized heading RMS <= 2 degrees.
- [ ] Robustness: with 5 injected false closures, ATE increases by no more than 25 %. *Revision 9:* the same with the real stage 5 aliases included, and after two-pass gating none of the 17 aliases remains active.
- [ ] Runtime for the whole graph < 30 s per maze on a laptop CPU.

---

## Stage 7: Occupancy grid, grid regularization, wall textures

**Status (revision 9): done by the user's Claude session (`reports/stage_7.md`).** All criteria pass on every test seed: IoU vs oracle 0.945 to 0.979; snapped wall cells 98.7 to 100 % recovered, 0 to 0.22 % false; lattice scale detected within 0.07 % and offset within 1.1 mm (Umeyama scale after correction 0.9998 to 1.0007, so stage 6's scale shortfall is removed); 100 % of true poses in the configuration space; `build_map.py` raw images to `map.npz` in 173 s. Regenerated oracle maps reach 0.93 to 0.98 IoU against the true layout (stage 2: 0.80 to 0.87); the drives see every wall, so the observable and full layouts coincide. Facts for later stages: (1) collision geometry and radius corrected (facts table); (2) in the true layouts a post can be missing between two walls (1.5 % of collinear, 2.4 % of perpendicular pairs; the gap is visible), and an isolated vertical edge cell renders as a horizontal 0.2 m piece; the snapping handles both; (3) wall signatures are stored per wall face (cell and side), not per segment: same-face distance between drives about 0.15, different faces about 0.8 (AUC 0.98), but 75 % of faces have a twin (stage 5b), so they confirm a pose hypothesis and cannot choose one alone; (4) pipeline modules: `grid.py`, `slam.py` (stages 5b and 6 as functions of raw inputs), `build_map.py` (resumable steps, `timing.json`).

**Goal:** the top-down map itself.

**Tasks**
0. **Regenerate the stage 2 oracle maps (revision 6)** with the current `segment.py` / `scan.py` (stage 3 changed the boundary sub-pixel step and added wall-top returns), for both `gt` and `gt4`, before using them as references.
1. Log-odds occupancy grid at 0.05 m resolution: Bresenham ray casting of every scan at its optimized pose; `l_occ`, `l_free` tuned on dev seeds; clamp at +/-5; down-weight far returns; "no return" columns update free space only up to `r_max`.
2. Lattice detection: histogram occupied cells along the maze axes, estimate cell size L and offset; report them. **Walls lie on lines through cell centres** (stage 0: thin planes through wall-cell centres), so the lattice to detect is the set of cell-centre lines, and a snapped wall is a 0.2 m segment between two adjacent lattice points. Produce the snapped wall map (a set of such segments, plus the equivalent wall-cell grid) as a second output.
3. Wall-texture signatures: for each lattice wall segment observed, store an HSV hue-saturation histogram (16x4 bins, Hellinger-normalized) and the ids of the frames that saw it.
4. Rendering: `map.png` (grid + all three optimized trajectories + start points) and `map_snapped.png`. Draw walls as segments, not as filled 0.2 m squares (the stage 0 trajectory figures used squares, which misrepresents the free space).
5. **Configuration-space map (revision 3; radius corrected in stage 7, see the facts table):** free space for the robot centre = free cells farther than the robot's collision radius from any collision segment. Save it in `map.npz` for the follow-up planning plan, and check that every true pose on dev seeds lies in it (allowing 0.01 m).

**Deliverables**
- `source/mapping/grid.py`, `source/mapping/build_map.py` (end-to-end CLI: `uv run source/mapping/build_map.py --data <dir>`), tests.
- `cache/mapping/<seed>/map.npz` (log-odds grid, resolution, origin, lattice, wall signatures).
- `reports/stage_7.md` with map images for all test seeds next to the oracle maps.

**Acceptance criteria**
- [ ] Map IoU of occupied cells vs the oracle map (stage 2, true poses) >= 0.75 on every test seed, computed after dilating both by one cell (0.05 m tolerance).
- [ ] *Revision 5:* IoU of occupied cells vs the **observable** true walls (reference segments that some true-pose ray reached within `R_MAX`), rasterized at 0.05 m, >= 0.75, same tolerance. Also report IoU vs the full true layout (no threshold; the stage 2 oracle ceiling is 0.80 to 0.87), and list which true walls are missing and why (never in range, grazing only, pose error).
- [ ] Snapped wall map vs the reference: at least 95 % of true wall segments recovered and at most 5 % false segments on every test seed.
- [ ] Detected lattice cell size within 2 % of the true 0.2 m, and the lattice offset within 0.02 m of the true one (cell centres at multiples of 0.2 m).
- [ ] Configuration-space check: at least 99.5 % of true poses on test seeds lie in the configuration-space free region (0.01 m allowance).
- [ ] Leakage test (rule 10) passes for `build_map.py`.
- [ ] End-to-end `build_map.py` runs from raw data to `map.npz` in < 5 minutes per maze on a laptop CPU, with caches cleared.

---

## Stage 8: Goal placement on the map

**Status (revision 9): done by the user's Claude session (`reports/stage_8.md`), except the final holdout run, deferred to after stage 9 by the user.** Test seeds (scored once, with the user's agreement, after freezing): 5 of 5 goals within 3 mm and 0.2 deg, confidence 0.31 to 0.38 (threshold 0.10 chosen on dev), 0.55 to 1.0 s per placement. The texture term is necessary: geometry alone places dev seed 10 4 m away, and on test seeds 21 and 25 a runner-up fits the walls almost as well as the goal. `final_eval.py` (`--final` wired to `record_gt.allow_final()`, tested) runs `build_map.py` and scores stages 2 to 8 on given seeds; checked on dev seed 7.

**Goal:** goal pose `(x_g, y_g, theta_g)` in the map frame, from the 4 goal views.

**Tasks**
1. Virtual scan for each goal view (front, left, back, right); rotate by 0, 90, 180, 270 degrees and merge into one goal scan in the goal's frame.
2. Global search: correlative scan matching over the free cells of the map, coarse-to-fine in `(x, y, theta)`, scored by a blurred likelihood field.
3. Texture term (*revision 9: textures repeat across a maze, so this term only breaks ties between geometric candidates; if stage 5b passed, use its boundary signatures instead, which are near unique*): for the top-N geometric candidates, compare the hue-saturation histograms of the walls seen in the goal views with the map's wall signatures at the predicted walls; combine scores.
4. Refine the winner with ICP. Output pose, a confidence (score margin between best and second-best candidate), and the top-3 candidates.
5. Source of goal views: `target.jpg` from the exploration data (stage 0: present in challenge data too, 240 x 1280, view 0 = the heading to arrive with). Also expose a function that takes the 4 views at runtime (`info.targets`) and runs in < 2 s, as a fallback.
6. The goal views overlap: the horizontal field of view is about 120 degrees, so adjacent views 90 degrees apart share about 30 degrees. Use the overlaps as a consistency check on the merged goal scan.

**Deliverables**
- `source/mapping/goal.py`, tests.
- `reports/stage_8.md` with: goal estimate vs true goal drawn on each test map, error table, confidence vs error plot.

**Acceptance criteria**
- [ ] Goal position error < 0.10 m and heading error < 10 degrees on at least 80 % of test seeds. Increase the number of test seeds to at least 5 for this stage if fewer were recorded.
- [ ] **Final evaluation (revision 7; revision 8: first wire the `--final` command-line flag into the evaluation scripts, calling `record_gt.allow_final()`, with a test that the holdout seeds are still refused without it):** after this stage passes and the pipeline is frozen, run it once on the 5 holdout seeds (`--final`) and report every stage's key metric next to the test-seed numbers. No fixes after this run; failures are reported.
- [ ] No high-confidence failures: every seed whose error exceeds those bounds has confidence below the threshold chosen on dev seeds.
- [ ] Runtime < 2 s per goal placement given a built map.
- [ ] Leakage test (rule 10) passes.

---

## Stage 9: Challenge data run, agent display, final holdout evaluation (rewritten in revision 10)

**Goal:**
1. Run the validated pipeline on real challenge data, which is one frame per tick.
2. Have `my_agent.py` show the map and goal reliably.
3. Run the one-time holdout evaluation.

No live localization, planning or control: those belong to the follow-up plan.

**Why an adapter.** Every frame-counted setting was tuned on frames 5 ticks apart: the motion prior, the collision thresholds, the closure exclusion window (10 frames), the thinning window (5 frames), heading priors every 10 frames, and the 16× odometry inflation. The challenge download `ab7efaeb1abf` is one frame per tick (stage 4b): the label repeats 80 % of the time, and the motion per frame is 0.029 to 0.031 m and 3.5°, i.e. 1 tick. On that data the stage 4b calibration hit its 0.3 floor.

Keeping every 5th frame makes the data **identical in kind to the local recordings**, so nothing tuned needs to change. The ticks in between are not wasted: they give each kept frame its exact action composition.

**Tasks**

0. **Leakage and import check** (rule 10) for every new or changed pipeline module (`adapter.py`, `build_map.py`, `my_agent.py`'s map path):
   - a static scan for `gt.json`, `poses.json`, `camera_fit`, `record_gt`, `vis_nav_sim`;
   - an identical-output run on a copy without the truth files. `tests/mapping/test_grid.py` has the pattern.
1. **Frame-rate detector** in `source/mapping/adapter.py` (pipeline): `ticks_per_frame(drive_dir) -> (k, evidence)`, from the data alone.
   - **Evidence:**
     - (a) the label-repeat rate (local 5-tick: 0.45; per tick: 0.80);
     - (b) the median forward motion per FORWARD-labelled frame from consecutive-frame scan ICP (`vo.icp_symmetric`), divided by the 0.03 m tick;
     - (c) the median absolute heading change per LEFT/RIGHT frame from the stage 3 compass, divided by the 3.44° tick.
   - Take k = the rounded median of (b) and (c), and require (a) to agree. If they disagree, raise an error rather than guess.
   - Record k and the evidence in `build_map.py`'s outputs.
2. **Local per-tick recordings, for validation only.**
   - Add `--capture-every N` to `record_gt.py` (default 5; this changes no existing recording). It writes `data/tick<N>-<seed>/` with `--msaa 4`.
   - Record **dev seeds 7 and 8** at N = 1. These are the same mazes, routes and noise as `gt4-7` and `gt4-8`, so the frames must coincide.
   - Find the offset: which per-tick frame equals which 5-tick frame. Byte-compare the JPEGs, and report the maximum pixel difference if they are not identical.
3. **Subsampling** (`adapter.subsample(drive_dir, k, out_dir)`):
   - keep frame 0 and every k-th frame after it, at the offset found in task 2;
   - write `data_info.json` with `action` = the **last** tick's label (the stage 1 convention: a window's label is its last tick), plus a new field `composition` = counts of each action over the window's k ticks;
   - copy or link the images.

   Check:
   - on `tick1-7` and `tick1-8`, the subsampled `data_info.json` actions equal `gt4-7` / `gt4-8` on 100 % of frames;
   - the images are identical.
4. **`build_map.py --ticks auto|N`** (default `auto`): detect the rate; if it isn't 5, subsample into the work directory and build from that. Write the detected rate to `adapter.json` in the output directory.

   **Transparency check (dev only):** `build_map.py` on `tick1-7` must reproduce the `gt4-7` build:
   - poses within 1 mm;
   - snapped wall cells identical;
   - goal within 1 cm.

   If not, find out why before going on.
5. **Agent (`my_agent.py`).**
   - **5a. Fix the open bug first.** On `uv run source/my_agent.py --local 7` the user saw **no red arrow**, although `goal.place` on `data/local-7/target.jpg` with the same map gives (5.00, 0.40, 139°), confidence 0.29, i.e. reliable, which is the true goal. The difference must be in what the session hands `setup()`. Diagnose with this, then fix:
     - compare `info.targets` from `vis_nav_sdk.local.local_session('local:7')` with `target.jpg`: shapes, dtype, `camera.color_order`, the mean absolute difference as is and with channels reversed;
     - run `goal.place` on both.

     Candidate causes:
     - RGB instead of BGR;
     - a different rendering of the live goal views;
     - an exception swallowed in `setup()`;
     - the arrow drawn orange because confidence < 0.10.

     Also reset `self.status` in `setup()`: a pre-flight failure message can currently linger.
   - **5b.** Replace the `five_tick_data()` name check with the task 1 detector. On challenge data, build through the adapter, so `--map` is no longer needed.
   - **5c.** Draw the HUD goal line in the same colour as the arrow (red when reliable, orange when not).
6. **Challenge run:** `build_map.py --data data/ab7efaeb1abf` (no ground truth; parameters frozen since stage 8), then `goal.place(..., target.jpg)`.
7. **Sanity checks without ground truth.** Report each challenge value next to the dev-seed range and the test range already in reports 4b to 8; flag and explain any value outside them:
   - detected rate and its evidence;
   - per-drive calibrated scales (expected 0.9 to 1.1);
   - closures per drive pair and by kind;
   - closures dropped by the chi-square re-check;
   - final graph cost per edge;
   - lattice scale and strength;
   - snapped wall cells (local mazes: 420 to 480) and whether the outer wall is closed;
   - configuration-space cells;
   - goal camera height (0.160);
   - goal confidence and overlap residual;
   - runtime.
8. **Final holdout evaluation** (moved from stage 8 by the user). Only after tasks 0 to 7 pass, the pipeline is frozen, **and the user confirms**:
   - run `python3 source/mapping/final_eval.py --final build <seed>` for 31 to 35;
   - then `--final score 31,32,33,34,35`;
   - report every stage's key metric next to the test-seed numbers.

   No fixes after this run; failures are reported. Run it on `gt4` (server-like frames). The holdout recordings are 5-tick, so the adapter is exercised only through detection (k must come out 5).

**Deliverables**
- `source/mapping/adapter.py`; `record_gt.py --capture-every`; `build_map.py --ticks`; `my_agent.py` fixes; `tests/mapping/test_adapter.py` (detector on synthetic label streams, subsampling bookkeeping, the leakage tests).
- `data/tick1-7`, `data/tick1-8` (dev only).
- `reports/stage_9.md`, containing:
  - adapter validation;
  - the transparency check;
  - the root cause of the arrow bug and its fix;
  - the challenge map and goal (figure);
  - the sanity-check table;
  - the holdout table.

**Acceptance criteria**
- [ ] Leakage and import checks pass for `adapter.py`, `build_map.py` and the agent's map path.
- [ ] The detector returns 5 on every local 5-tick dataset it is run on (dev seeds; holdout through `final_eval`), 1 on `tick1-7`, `tick1-8` and `ab7efaeb1abf`, with evidence (a) to (c) reported.
- [ ] Subsampled `tick1` data equals the native 5-tick data: labels on 100 % of frames; images identical, or the maximum difference is reported and explained.
- [ ] Transparency: a `tick1-7` build reproduces the `gt4-7` build (poses ≤ 1 mm, same snapped cells, goal ≤ 1 cm).
- [ ] `--local 7` (dev) shows the red arrow at the true goal, with the root cause of the earlier missing arrow stated. The user drives to it manually and checks in; report the score.
- [ ] Challenge: end to end with no manual edits, < 5 min. Every sanity value is in range or flagged and explained.
- [ ] Final holdout evaluation done once, with the user's confirmation, and reported without fixes.
- [ ] Optional, only if everything above passes: a **composition prior** (motion given the window's exact action counts, fitted on `tick1-7`, `tick1-8`) used for subsampled data. Show calibration (95 % region covers 90 to 99 %) on a dev drive held out from the fit, and the ATE change of variant C. Adopt it only if it helps on dev, and mention it before the holdout run.

---

## Stage 10: Ping: locate the robot on the map on demand (revision 10)

**Goal:** while driving with the keyboard in `my_agent.py`, pressing **P** locates the robot on the map from what the camera sees *now*. The panel then redraws with a **green circle** at the estimated position, plus a short tick for the heading. This is a deliberate, on-demand exception to "live localization is out of scope": no continuous tracking, no planning, no control.

**Order and freezing.** Run this stage after stage 9, including its holdout run. It is additive: the frozen pipeline (stages 2 to 8) must behave identically, so the one change to `goal.py` comes with a byte-identical check (task 3). The holdout seeds are never used here.

**Why a ping is a spin, not a single frame.**
- A single 120° view is often ambiguous: corridors look alike, and stage 8 showed that even 2 of the 4 goal views failed once on dev.
- Turning in place through 360° yields the same kind of panoramic scan the goal views give. `goal.place`, which needs every view of it to place well, then works almost unchanged.
- The robot rotates about its camera (stage 0: offset zero), so the position does not move during the spin.
- The cost is steps, not attempts: 360° / 3.44° per tick ≈ **105 ticks**, out of a 60,000-step budget.

**Tasks**

1. **The ping sweep** (`source/mapping/ping.py`, pipeline code):
   - `start()` begins a sweep. While a sweep runs, the agent returns `(Action.LEFT, 1)` each step and passes every `obs.image` to `add(image)`. This is about 105 frames, one per tick.
   - The sweep ends after the stage 3 compass, combined with the tick count (3.44° per LEFT tick), says the total turn is ≥ 360°. Use the tick count to unwrap the compass's modulo-90° heading.
   - Collisions do not apply (an in-place turn), but stop and report if the compass and the tick count disagree by more than 10°.
2. **Sweep → scan:**
   - For each kept frame (every 3rd is plenty, about 35 frames), take its virtual scan (stage 2, **robot camera height 0.21 m**).
   - Rotate it by the frame's heading relative to the first frame. Use the heading change from the compass (drift-free modulo 90°) unwrapped by the ticks, rather than the ticks alone, which carry motion noise.
   - Merge into one scan in the frame of the first view. Keep each frame's image and its relative angle for the texture term.
3. **Generalise `goal.py`** so that a scan's views can carry **any relative angles** (today `texture_score` assumes k × 90°). Add an angle per entry of `GoalScan.views` and use it in `texture_score`. Goal placement must give byte-identical results before and after this change: check it on the dev goals.
4. **Localise:** `ping.locate(map, sweep) -> (pose of the first frame, confidence, top 3)`, through `goal.place` on the merged scan. The robot's pose at the end of the sweep is the first frame's pose plus the final relative heading.
   - Reuse the stage 8 confidence rule.
   - Choose `PING_CONF_MIN` on dev as stage 8 did: the margin that separates right from wrong on the dev evaluation in task 6. Do not assume the goal's 0.10.
5. **Agent** (`my_agent.py`):
   - **Keys.** `act()` reads `self.viewer.keys()` **once** per step and handles both the arrows and **"p"** (the pygame key name). `Viewer.keys()` clears the tap record, so calling it twice per step loses taps. Do not call `KeyboardAgent.act` after reading the keys yourself. Copy its few lines instead, and do not modify `keyboard_agent.py` (rule 2).
   - **During a sweep,** ignore the arrows (escape still quits).
   - **When the sweep ends,** locate and redraw the panel: a green filled circle at the estimate with a heading tick. If not reliable, draw a hollow green circle, write "NOT reliable" in the HUD, and show the runner-ups as small grey circles.
   - **Keep the last 5 pings** as fading green circles, so the user sees the track.
   - **HUD:** the ping state ("spinning 210/360°", then "located x, y, heading, confidence, 0.4 s"), and the ticks the ping spent.
   - The map itself is not changed by a ping: the walls stay as built, and only the overlay updates.
6. **Evaluation** (`source/mapping/stage10.py`, evaluation code, local simulator only):
   - **Recording.** Record ping sweeps with true poses on **dev seeds 7 to 10**: at least 25 random free positions per seed, at least 0.15 m from walls, random initial headings, each a full 360° in-place sweep. Use `vis_nav_sim` directly, as `record_gt.py` does, with `msaa=4`. Save the frames and the true poses.
   - **Placement metrics:** position and heading error of the pose at the end of the sweep, confidence, runtime.
   - **Ablations:** a single frame (no spin), a half spin (180°), and no texture term. These show what each part buys.
   - **Test seeds:** only once and **after asking the user**, same protocol, with a `--allow-test` guard.
7. **The display, for real:** `uv run source/my_agent.py --local 7` (dev). Drive somewhere, press P, and check the green circle against where you are. The user does this run. Report what they saw, and the positions of 3 or more pings compared with the simulator's true pose. The local SDK does not report the pose, so read it from the recorded evaluation harness or a debug print in the local simulator session.

**Deliverables**
- `source/mapping/ping.py`; the `goal.py` generalisation; `my_agent.py` with P.
- `source/mapping/stage10.py`; `tests/mapping/test_ping.py`:
  - sweep bookkeeping on synthetic rotations;
  - angle unwrapping;
  - the 4-view case reproducing `goal.place` exactly;
  - leakage.
- `reports/stage_10.md`:
  - error table per seed;
  - confidence vs error plot;
  - the ablations;
  - a panel screenshot with pings.

**Acceptance criteria**
- [ ] Goal placement is unchanged by the `goal.py` generalisation: dev goals byte-identical.
- [ ] Dev sweeps: ≥ 90 % within 0.10 m and 10°, and no high-confidence failures at the chosen `PING_CONF_MIN`.
- [ ] Test sweeps, scored once with the user's agreement: ≥ 90 % within 0.10 m and 10°, no high-confidence failures.
- [ ] Locating takes < 1 s of computation after the sweep (the sweep itself is about 105 ticks), and the panel updates in the same step.
- [ ] A sweep that collides or whose compass and ticks disagree is reported in the HUD, not silently used.
- [ ] Leakage (rule 10): `ping.py` and the agent's ping path use only images, labels and the map.
- [ ] The user sees the green circle in the right place in `--local 7` on 3 or more pings.

---

## Summary of thresholds

| Stage | Key metric | Threshold |
|---|---|---|
| 0 | done | h = 0.2099 m, offset 0, blind run 1 frame; rotation gate failed (collisions), handled in 1 and 4 |
| 1 | done | prior calibrated (0.934 to 0.948 on test); means within 0.9 mm / 0.7 deg; dead reckoning median ATE 4.0 m, heading RMS 82 deg (reference) |
| 2 | Scan endpoint error vs true walls (<= r_max) | median < 0.03 m, p95 < 0.10 m; range bias within 1 % |
| 2 | done | endpoint error median 2.8 mm, p95 23 mm; coverage 0.80 (0.95 with wall top) |
| 3 | done | fused heading: 96.0 % valid, median 0.21 deg, p95 0.75 deg, 99.95 % of valid within 3 deg |
| 3b | Collision component; collision detector | calibration still 0.90 to 0.99 with the component; detector recall 1.0 on > 45 deg events, >= 0.9 on > 30 deg, <= 1 % flagged |
| 4 | Visual odometry (variant C) | heading RMS < 2 deg; 10-frame RPE median < 3 %; beats actions-only; every collision event on the correct branch; no heading wrong by > 10 deg accepted; ATE median 0.12 m (`gt4`) |
| 4b | done | symmetric ICP + two-scan wall depth; ATE median 0.094 m; 10-frame drift 0.79 / 4.64 %; straight-segment -0.64 % (accepted); per-drive calibration recovers injected bias; holdout seeds 31 to 35 |
| 6 (rev 8) | Scale on loop-closed trajectories | similarity scale within 0.5 % (median) / 1 % (every drive); straight-segment ratio reported |
| 9 (rev 8) | Challenge adapter | frame rate detected; subsample to 5 ticks with exact composition prior; validated locally; closure gate (drift < 0.4 m) checked |
| 5 | done | precision 0.997, pose error 2.1 mm / 0.22 deg; frame recall 0.34; graph connected |
| 5b (rev 9) | Texture adjacency | go/no-go: >= 90 % of boundaries unique in their maze; then precision >= 0.98 without odometry gate, no 0.4 to 0.6 m aliases |
| 8+ | Final holdout evaluation | run once on the frozen pipeline; reported, no fixes |
| 5 | Loop-closure precision | >= 0.98; >= 1 closure per drive pair |
| 6 | Optimized ATE | <= 0.08 m RMS; heading <= 2 deg |
| 7 | Map IoU vs oracle / true walls | >= 0.75 / >= 0.70 (1-cell tolerance); snapped segments >= 95 % recall, <= 5 % false; C-space holds 99.5 % of true poses |
| 9 | Leakage | outputs identical without `gt.json` / `poses.json`; no ground-truth imports |
| 8 | Goal pose error | < 0.10 m and < 10 deg on >= 80 % of seeds |
