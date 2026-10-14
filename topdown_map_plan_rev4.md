# Implementation plan: top-down map from exploration data

**Repo:** fork of `ai4ce/vis_nav_player` (NYU ROB-GY 6203 Visual Navigation Challenge)
**End goal:** from the provided exploration trajectories, build a metric top-down (zenithal) occupancy map of the maze with wall-texture signatures, and place the goal pose on it.
**Out of scope for this plan:** live localization during a run, path planning, autonomous control. Those come in a follow-up plan that consumes this map.

## Revision 4 (2026-10-13): stage 1 accepted; collision handling moved to stage 3b

**Reload this file from disk before working.** An earlier session overwrote revision 3 with a stale revision 2 copy, so stage 1 ran without the revision 3 collision tasks. Never save an in-memory copy of this plan over the file; edit the file on disk.

Stage 1 (`reports/stage_1.md`) is **accepted and complete; do not rerun it.** The prior (per (previous, current) label Gaussian mixtures) is calibrated on test seeds (0.934 to 0.948 inside the 95 % region). The one failed criterion, dev-vs-test means within 2 standard errors (max z 2.1 naive, 2.2 clustered), is restated below as a practical tolerance; the measured differences (at most 0.9 mm and 0.7 degrees) pass it. Reason: with thousands of samples a standard-error test flags differences far too small to matter, and with 12 comparisons about one |z| > 2 is expected by chance.

What changed in revision 4:

| Where | Change |
|---|---|
| Stage 1 | Marked complete. Mean-agreement criterion restated as an equivalence tolerance. The two revision 3 collision tasks (collision component, collision detector) moved out. |
| Stage 3b (new) | Collision component of the prior and the collision detector, built after stage 3 because the detector's best signals (wall-feature inliers, floor-boundary orientation, vanishing-point disagreement) come from stages 2 and 3. Must pass before stage 4 starts. |
| Stage 4 | References point to stage 3b. |

Order from here: **stage 2 -> stage 3 -> stage 3b -> stage 4.** Next stage to run: **stage 2**.

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
5. **Every stage writes a report** `reports/stage_N.md`: what was built, a results table (metric, value, threshold, pass/fail), the plots listed under that stage, and open issues. Plots go in `reports/figures/`.
6. **Units:** metres, radians in code, degrees in reports. Angles wrapped to (-pi, pi]. Poses are SE(2) `(x, y, theta)`.
7. **Reproducibility:** every script takes `--seed` / `--data` arguments, fixes RNG seeds, and caches intermediate results under `cache/mapping/` with the stage name and a config hash in the filename (the baseline's caches are keyed only by frame count; do not repeat that mistake).
8. **Tests:** each stage adds `pytest` tests for its pure functions (geometry, SE(2) composition, log-odds update, etc.) under `tests/mapping/`. `uv run pytest` must pass at every stage.
9. **Reruns after a plan revision:** keep every deliverable that still satisfies its criteria. Do the new or changed tasks only, and record them in an `## Addendum (revision N)` section at the end of that stage's report rather than rewriting the report. Never re-record datasets unless a task says so.
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
| **Rendered wall geometry** | walls are **thin vertical planes through wall-cell centres**, not 0.2 m blocks: a wall cell contributes a 0.2 m segment along x if a left/right neighbour is a wall, along y if an up/down neighbour is; an isolated wall cell contributes an x segment. Exact set: `camera_fit.Scene(...).segments` (evaluation only). A one-cell corridor is 0.4 m wide between wall planes |
| Wall height | 0.30 m |
| Camera height `h` | **0.2099 m** (+/- 0.001), = `height_offset` 0.15 + robot origin 0.06 |
| Camera offset from rotation centre | axial 0.3 +/- 0.8 mm, lateral -0.8 +/- 0.9 mm: **treat as zero** (turning in place is pure rotation) |
| Robot collision radius | about 0.08 m (closest approach of the robot centre to a wall plane) |
| Far plane | 5.0 m depth; nothing drawn beyond; background white (255, 255, 255), same as sky. Floor reaching the far plane ends at row 123.86 |
| Floor palette (BGR) | blue tile (218, 181, 157) drifting to about (222, 195, 178) with distance (fog); grey tile (232, 232, 232). Stage 0's segmentation: recall 0.995, no wall pixel accepted |
| Depth resolution | one boundary pixel = z^2 / (fy h): 0.05 m at 1 m, 0.21 m at 2 m, 0.47 m at 3 m, 0.83 m at 4 m |
| Cue statistics (stage 0 audit, provisional detectors) | SIFT inliers vs previous frame: median about 30, 16.9 % of frames under 15 (mostly corridor views of dark walls). Valid boundary columns: median 0.54 to 0.58, 0.11 % of frames under 0.20. Vanishing point "found" on 88 % of frames but agreeing with true heading within 3 deg on only 39 % of those. Blind frames: 1 in 28,837 |
| Challenge data | has `target.jpg` (4 views, 240 x 1280); no `gt.json` or `poses.json` |
| Camera level | pitch < 0.3 deg; fitted cy = 120.000 |
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

---

## Stage 4: Visual odometry with an action prior

**Goal (revision 2):** per-drive trajectories driven by vision, with action labels only as a weak prior; and a measured answer to "how much do the actions add?".

**Measurements available per consecutive frame pair (k, k+1):**
- Absolute yaw mod 90 degrees from the vanishing points (stage 3), unwrapped to the branch nearest the predicted heading **except on collision-flagged pairs** (stage 3b detector), where the nearest-branch rule is unsafe: stage 0 found 27 pairs rotating more than 45 degrees during wall collisions, and a wrong branch is a permanent 90 degree heading error for the rest of the drive.
- Relative planar motion from the **floor homography**: floor correspondences (tile edges, seam points; or dense alignment of the inverse-perspective-mapped floor) between k and k+1, solved for `(dx, dy, dtheta)` with known K and `h` = 0.2099 m. Metric, no scale ambiguity. Use near floor only (far floor aliases and depth error grows as z^2).
- Relative planar motion from **scan-to-scan ICP** between the virtual scans of k and k+1 (stage 2), with its covariance (e.g. Censi's closed form, or from the Hessian). This covariance exposes the long-corridor degeneracy: motion along a corridor with featureless walls is weakly constrained.
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
- [ ] Scale (variant C): estimated total path length within 1 % of the true path length. Checks `h` end to end.
- [ ] **Collisions (revision 3):** every test-seed event with true rotation > 45 degrees (19 events) and every event > 30 degrees resolves to the correct 90 degree branch: heading error under 5 degrees five frames after the event. A single wrong branch fails the stage.
- [ ] Variant C ATE lower than variant A on every test seed.
- [ ] Filter consistency (variant C): NIS of the fused relative-motion update within the 95 % chi-square bound for 3 degrees of freedom (7.81) on at least 90 % of updates.
- [ ] **Report, no threshold:** variant B vs C. If B's ATE is within 20 % of C's on every test seed, the pipeline does not depend on the action labels (robust to an unknown server `motion_bias`): state this conclusion explicitly. If B is much worse, say where (collisions, long corridors) with figures.
- [ ] Leakage test (rule 10) passes for the stage 4 pipeline.

---

## Stage 5: Loop closures

**Goal:** confirmed revisit pairs with a measured relative pose and covariance, within and across drives.

**Tasks**
1. Candidates: VLAD place recognition (reuse `vlad.VLADExtractor`, but compute SIFT on the **wall-only mask** from stage 2; use a separate cache key). For each frame, top-5 candidates excluding frames within 10 indices of it in the same drive.
2. Verification: SIFT ratio-test matches + RANSAC homography; accept if inliers >= a threshold tuned on dev seeds.
3. Relative pose, two methods, compared on dev seeds:
   - (a) decompose the homography (`cv2.decomposeHomographyMat`), keep the solution with positive depth and a horizontal-plane normal consistent with a vertical wall, set scale from the scan range to that wall;
   - (b) 2-D ICP between the two frames' virtual scans, initialized from (a) or from the identity.
   Keep whichever is more accurate; if both work, require agreement within 0.1 m and 5 degrees as an extra check.
4. Covariance per closure from ICP residuals or an empirical value from dev seeds.
5. Ground-truth revisit definition (for scoring only): true positions within 0.5 m and headings within 30 degrees.

**Deliverables**
- `source/mapping/loops.py`, tests.
- `reports/stage_5.md` with: precision-recall curve over the inlier threshold, chosen threshold, relative-pose error histograms, a plot of accepted closures drawn on the true trajectories.

**Acceptance criteria**
- [ ] Precision >= 0.98 at the chosen threshold (at most 1 false closure in 50).
- [ ] At least one accepted closure between every pair of drives on every test seed (otherwise the drives cannot be joined).
- [ ] Recall >= 0.25 of ground-truth revisit pairs.
- [ ] Relative-pose error of accepted closures: median < 0.08 m and < 2 degrees.

---

## Stage 6: Pose-graph optimization

**Goal:** one globally consistent trajectory for all 3 drives.

**Tasks**
1. Graph: nodes = every frame of every drive (or drive segment, if stage 4 split any); odometry edges from stage 4 variant C; loop-closure edges from stage 5.
2. Spawn constraint (revision 2): stage 0 showed every drive starts at the known spawn (0.2, 2.6, 0) and the first saved frame is 5 ticks later. Add one fixed spawn node at that pose and connect each drive's first frame to it with an edge whose mean and covariance come from the stage 1 action prior for its label. This fixes the gauge and puts the map in the true world frame. Segments that do not start at the spawn float and are tied in only by loop closures.
3. Solver: Levenberg-Marquardt on SE(2) with angle-wrapped residuals and a robust kernel (Cauchy or Huber; or switchable constraints). Use `scipy.optimize.least_squares` with a sparse Jacobian, or GTSAM if installable; document the choice.
4. Robustness test: inject 5 random false closures (random frame pairs with random relative poses) and re-optimize.

**Deliverables**
- `source/mapping/posegraph.py`, tests (toy square loops, see criteria).
- `cache/mapping/<seed>/poses_optimized.json`.
- `reports/stage_6.md` with: before/after trajectories vs truth, ATE table (dead reckoning, EKF, optimized), robustness test results.

**Acceptance criteria**
- [ ] Toy-loop unit tests: (a) a 4-edge square with noise-free odometry and closure, started from a perturbed initial guess, converges to the true square within 1e-6; (b) the same square with 0.1 rad heading drift on one edge and an exact closure ends with the closure residual below 1e-3 and the error spread over all edges (no single edge carries more than half of it).
- [ ] Optimized ATE (all drives jointly, after one rigid alignment) <= 0.08 m RMS on every test seed.
- [ ] Optimized heading RMS <= 2 degrees.
- [ ] Robustness: with 5 injected false closures, ATE increases by no more than 25 %.
- [ ] Runtime for the whole graph < 30 s per maze on a laptop CPU.

---

## Stage 7: Occupancy grid, grid regularization, wall textures

**Goal:** the top-down map itself.

**Tasks**
1. Log-odds occupancy grid at 0.05 m resolution: Bresenham ray casting of every scan at its optimized pose; `l_occ`, `l_free` tuned on dev seeds; clamp at +/-5; down-weight far returns; "no return" columns update free space only up to `r_max`.
2. Lattice detection: histogram occupied cells along the maze axes, estimate cell size L and offset; report them. **Walls lie on lines through cell centres** (stage 0: thin planes through wall-cell centres), so the lattice to detect is the set of cell-centre lines, and a snapped wall is a 0.2 m segment between two adjacent lattice points. Produce the snapped wall map (a set of such segments, plus the equivalent wall-cell grid) as a second output.
3. Wall-texture signatures: for each lattice wall segment observed, store an HSV hue-saturation histogram (16x4 bins, Hellinger-normalized) and the ids of the frames that saw it.
4. Rendering: `map.png` (grid + all three optimized trajectories + start points) and `map_snapped.png`. Draw walls as segments, not as filled 0.2 m squares (the stage 0 trajectory figures used squares, which misrepresents the free space).
5. **Configuration-space map (revision 3):** free space for the robot centre = free cells farther than the robot's collision radius (0.08 m, stage 0) from any wall segment. Save it in `map.npz` for the follow-up planning plan, and check that every true pose on dev seeds lies in it (allowing 0.01 m).

**Deliverables**
- `source/mapping/grid.py`, `source/mapping/build_map.py` (end-to-end CLI: `uv run source/mapping/build_map.py --data <dir>`), tests.
- `cache/mapping/<seed>/map.npz` (log-odds grid, resolution, origin, lattice, wall signatures).
- `reports/stage_7.md` with map images for all test seeds next to the oracle maps.

**Acceptance criteria**
- [ ] Map IoU of occupied cells vs the oracle map (stage 2, true poses) >= 0.75 on every test seed, computed after dilating both by one cell (0.05 m tolerance).
- [ ] IoU of occupied cells vs the true walls, rasterized from the stage 2 reference segments at 0.05 m, >= 0.70, same tolerance (evaluation only, rule 10).
- [ ] Snapped wall map vs the reference: at least 95 % of true wall segments recovered and at most 5 % false segments on every test seed.
- [ ] Detected lattice cell size within 2 % of the true 0.2 m, and the lattice offset within 0.02 m of the true one (cell centres at multiples of 0.2 m).
- [ ] Configuration-space check: at least 99.5 % of true poses on test seeds lie in the configuration-space free region (0.01 m allowance).
- [ ] Leakage test (rule 10) passes for `build_map.py`.
- [ ] End-to-end `build_map.py` runs from raw data to `map.npz` in < 5 minutes per maze on a laptop CPU, with caches cleared.

---

## Stage 8: Goal placement on the map

**Goal:** goal pose `(x_g, y_g, theta_g)` in the map frame, from the 4 goal views.

**Tasks**
1. Virtual scan for each goal view (front, left, back, right); rotate by 0, 90, 180, 270 degrees and merge into one goal scan in the goal's frame.
2. Global search: correlative scan matching over the free cells of the map, coarse-to-fine in `(x, y, theta)`, scored by a blurred likelihood field.
3. Texture term: for the top-N geometric candidates, compare the hue-saturation histograms of the walls seen in the goal views with the map's wall signatures at the predicted walls; combine scores.
4. Refine the winner with ICP. Output pose, a confidence (score margin between best and second-best candidate), and the top-3 candidates.
5. Source of goal views: `target.jpg` from the exploration data (stage 0: present in challenge data too, 240 x 1280, view 0 = the heading to arrive with). Also expose a function that takes the 4 views at runtime (`info.targets`) and runs in < 2 s, as a fallback.
6. The goal views overlap: the horizontal field of view is about 120 degrees, so adjacent views 90 degrees apart share about 30 degrees. Use the overlaps as a consistency check on the merged goal scan.

**Deliverables**
- `source/mapping/goal.py`, tests.
- `reports/stage_8.md` with: goal estimate vs true goal drawn on each test map, error table, confidence vs error plot.

**Acceptance criteria**
- [ ] Goal position error < 0.10 m and heading error < 10 degrees on at least 80 % of test seeds. Increase the number of test seeds to at least 5 for this stage if fewer were recorded.
- [ ] No high-confidence failures: every seed whose error exceeds those bounds has confidence below the threshold chosen on dev seeds.
- [ ] Runtime < 2 s per goal placement given a built map.
- [ ] Leakage test (rule 10) passes.

---

## Stage 9: Challenge data run and agent display

**Goal:** apply the validated pipeline to real challenge data and show the map in the agent's viewer.

**Tasks**
0. Leakage test first (rule 10): on a copy of every test seed's dataset with `gt.json` and all `poses.json` deleted, the full pipeline must produce outputs identical to the original runs. Also grep the pipeline modules for any import of `vis_nav_sim`, `camera_fit.Scene`, or reads of `gt.json` / `poses.json`; there must be none.
1. Run `build_map.py` on the downloaded challenge exploration data (no ground truth; same parameters as frozen after stage 8).
2. In `source/my_agent.py`, load the map in `__init__` (which runs before the server connection; a crash there costs no attempt) and show it via `panel()` with the goal pose marked. No control logic in this stage.
3. Sanity checks without ground truth: loop-closure count per drive pair, final pose-graph cost, goal confidence; compare their ranges with the local test seeds.

**Deliverables**
- `reports/stage_9.md` with the challenge map, goal estimate and sanity-check table.
- `my_agent.py` displaying the map (no attempt needs to be spent: verify with `--local`).

**Acceptance criteria**
- [ ] Leakage test and import check pass.
- [ ] Pipeline runs end to end on challenge data with the frozen parameters, no manual edits.
- [ ] Every sanity-check value lies within the range seen on local test seeds, or the report flags it and explains it.
- [ ] `uv run source/my_agent.py --local <test seed>` shows the map and goal in the panel without errors.

---

## Summary of thresholds

| Stage | Key metric | Threshold |
|---|---|---|
| 0 | done | h = 0.2099 m, offset 0, blind run 1 frame; rotation gate failed (collisions), handled in 1 and 4 |
| 1 | done | prior calibrated (0.934 to 0.948 on test); means within 0.9 mm / 0.7 deg; dead reckoning median ATE 4.0 m, heading RMS 82 deg (reference) |
| 2 | Scan endpoint error vs true walls (<= r_max) | median < 0.03 m, p95 < 0.10 m; range bias within 1 % |
| 3 | VP yaw error | median < 1.5 deg, p95 < 4 deg, >= 70 % valid; >= 95 % of valid frames within 3 deg |
| 3b | Collision component; collision detector | calibration still 0.90 to 0.99 with the component; detector recall 1.0 on > 45 deg events, >= 0.9 on > 30 deg, <= 1 % flagged |
| 4 | Visual odometry (variant C) | heading RMS < 2 deg; 10-frame RPE median < 3 %; path length within 1 %; beats actions-only; every collision event on the correct branch |
| 5 | Loop-closure precision | >= 0.98; >= 1 closure per drive pair |
| 6 | Optimized ATE | <= 0.08 m RMS; heading <= 2 deg |
| 7 | Map IoU vs oracle / true walls | >= 0.75 / >= 0.70 (1-cell tolerance); snapped segments >= 95 % recall, <= 5 % false; C-space holds 99.5 % of true poses |
| 9 | Leakage | outputs identical without `gt.json` / `poses.json`; no ground-truth imports |
| 8 | Goal pose error | < 0.10 m and < 10 deg on >= 80 % of seeds |
