# Stage 0: data audit and ground-truth harness

Status: **all acceptance criteria pass.** Two findings change how Stage 1 must work (see "Findings that affect later stages").

## Built

| file | what |
|---|---|
| `source/mapping/record_gt.py` | records `data/gt-<seed>/`: the student dataset plus `traj_<i>/poses.json` and `gt.json`; `--verify` runs the checks below |
| `source/mapping/seeds.json` | frozen split |
| `source/mapping/stage0_report.py` | writes `cache/mapping/camera.json` and the trajectory figures |
| `tests/mapping/test_record_gt.py` | split sanity, pose/record alignment, student-equality, pose re-render |
| `cache/mapping/camera.json` | K and image size (identical on every seed) |

Dev seeds: 7, 8, 9, 10. Test seeds: 21, 22, 23, 24, 25 (five, so Stage 8 needs no extra). Frozen in `seeds.json`.

## Simulator API (vis_nav_sim 0.1.3), the parts that matter

- `Simulator(textures, seed, motion_noise=DEFAULT_NOISE=(0.3, 0.1), motion_bias=None, motion_seed=0, camera=None)`.
  Noise is a per-tick sigma on the linear/angular speed (30 % and 10 %). `cli.record_exploration_data` uses
  `motion_noise=DEFAULT_NOISE, motion_seed=seed`, no `motion_bias` (`drawn_bias() == (1, 1, 0)`).
- Constants: tick 0.01 s; FORWARD 0.03 m/tick; LEFT/RIGHT 0.06 rad/tick (LEFT = +yaw, RIGHT = -yaw); yaw is CCW from +x.
- `record(out_dir, routes, seed, capture_every, noisy, student_files)`. **`student_files=False` additionally writes
  `traj_<i>/poses.json`, the true `[x, y, yaw]` per saved frame.** So no driving logic had to be reproduced.
  The driving is a deterministic "keypress controller" planned from `seed`; it does not depend on `capture_every`.
- `render(x, y, yaw)` renders any pose; `jpeg(frame)` is the server's encoder.
- Ground truth the server never gives: `grid` / `occupancy` (31x31 cells of 0.2 m, **cell (r, c) is centred at (x, y) = (c*0.2, r*0.2)**,
  walls are whole cells), `spawn`, `target` (x, y, yaw), `shortest_path`. Stage 7's "true layout" criterion is therefore measurable.

## Schema

`data_info.json`: a list of `{"step": k, "image": "k.jpg", "action": [<name>]}`, `step == k`. Names: FORWARD, LEFT, RIGHT, IDLE.
No other fields. `target.jpg` is one image of shape (240, 1280, 3): the four 320-px views side by side (view 0 = heading to arrive with).
Frames are (240, 320, 3) BGR JPEG.

## Unknowns

| unknown | answer | evidence |
|---|---|---|
| `data_info.json` schema, combined actions | above. **Every record in all three data sets has exactly one action** (challenge `ab7efaeb1abf`: 1714/2342/3146 records; local seeds likewise). | scan of all records |
| Challenge data has `target.jpg`? | **Yes**, same 4-view format as local. | `data/ab7efaeb1abf/target.jpg`, shape (240, 1280, 3) |
| Do the 3 drives start at the same pose? | **Yes: the spawn `(0.2, 2.6, 0)`** (x, y, yaw) on every seed; every drive starts from the spawn, and the first saved frame is 5 ticks later. | `record` docstring ("routes from the spawn"); first logged poses are the spawn plus one 5-tick window |
| Can `record()` log true poses? | **Yes**, `student_files=False`. | `poses.json` |
| Intrinsics offline | `Simulator.camera` / `intrinsic_matrix`: fx = fy = 92, cx = 160, cy = 120, 320x240, near 0.02, far 5.0, `height_offset` 0.15. Saved to `cache/mapping/camera.json`. Constant across seeds. | `camera.json` |
| Camera level? | **Level to measurement precision.** Fitting wall top/bottom rows against distance (4 distances) gives principal row cy = 120.3 with max residual 0.2 px (pitch < 0.3 deg). | scratch experiment, `render` at known corridor distances |
| Camera height `h` and offset from rotation centre | **Not cleanly resolved.** The floor line fit gives h = 0.21 m with a 0.10 m offset along the optical axis; these two are confounded with my assumption about where the wall face sits, and the simulator's stated `height_offset` is 0.15. Fallback: estimate `h` (and any axial offset) by least squares from true poses in Stage 2 step 4, as the plan already allows. | same experiment |

## Acceptance

| criterion | result | pass |
|---|---|---|
| Every unknown answered with evidence, or marked unresolvable with a fallback | all but the camera height/offset; that one has a stated fallback | pass |
| Pose alignment: 20 random frames per seed re-rendered at the logged pose, mean abs diff < 2 | mean **0.0**, max **0.0** on all 9 seeds (render passed through the JPEG encoder is bit-exact) | pass |
| Recorder identical to `record_exploration_data` (count, images, actions) | all files byte-identical on all 9 seeds (about 3,100 to 3,500 files each); the only extra files are the `poses.json` and `gt.json` | pass |
| >= 3 dev and >= 3 test seeds, frozen | 4 dev, 5 test | pass |
| `uv run pytest` | 10 passed | pass |

Frames per seed (drives 0/1/2): 7: 723/749/1384, 8: 832/1066/1426, 9: 753/864/1573, 10: 777/736/1607, 21: 766/829/1623,
22: 781/1073/1637, 23: 768/830/1658, 24: 909/831/1496, 25: 827/961/1358. On every seed the three drives together visit
100 % of the free cells and pass through the goal's cell.

Figures: `reports/figures/stage0_true_trajectories_<seed>.png`, all nine seeds.

## Findings that affect later stages

1. **A record's action is the action of the last tick of its 5-tick window, not a 5-tick action.** Comparing the
   `capture_every=1` recording (per-tick actions) with the `capture_every=5` one on seed 7: the record's action equals the
   window's last-tick action in 100 % of records (2,856/2,856), but equals its first-tick action in only 46 %, and **62 % of windows contain
   more than one distinct action** (e.g. a turn ending in a forward). Consistent with that, measured per-record motion
   between true poses is far from the nominal 5 ticks: FORWARD records advance 0.110 +/- 0.046 m (nominal 0.15), LEFT/RIGHT records turn
   about 0.22 rad (nominal 0.30) with 0.04 m of incidental translation, and IDLE records are not idle (yaw std 0.20 rad).
   Dead reckoning from the labels is therefore much coarser than the plan assumed, and the Stage 1 "per-action constants"
   will be a *distribution over window composition*, not a clean (s, omega). The labels also carry no information about
   the first four ticks. This is the main risk to the Stage 4 target (ATE <= 50 % of dead reckoning) being meaningful, and
   to the Stage 1 chi-square criterion; Stage 1 should say so with numbers rather than fit around it silently.
   The challenge data has the same format, so this applies to it too.
2. **The data covers the whole maze**: 100 % of free cells on every seed. Mapping coverage is not the bottleneck; pose accuracy is.
3. Colours differ from the plan's `mask.py` prototype: the floor in these renders is (218, 181, 157) BGR (a light blue), the
   sky is (255, 255, 255). Re-measure in Stage 2 as planned.
4. The challenge data's last record is IDLE while local ones end on a FORWARD; harmless, noted.

## Open issues

- Camera height/axial offset (above), deferred to Stage 2.
- Whether the hidden challenge server uses `motion_bias` (systematic drift) is unknown; the local recordings have none. Stage 1 should
  not assume the noise is zero-mean per tick on the challenge data. If the bias is on there, local dead-reckoning numbers are optimistic.

---

## Addendum (revision 2)

Tasks 6 to 8 only; no dataset was re-recorded. **Seven of eight revision 2 criteria pass; the 45-degree rotation gate fails** (27 collision events, details below). Numbers are on dev seeds for fits (7 to 10) and on all nine seeds for the audit.

New code: `source/mapping/camera_fit.py` (pure fit functions, `Scene` raycaster), `stage0_camera.py` (tasks 6 and 8), `trackability.py` and `stage0_trackability_report.py` (task 7); tests `tests/mapping/test_camera_fit.py`. `cache/mapping/camera.json` gained `height_fit`, `offset_from_rotation_centre`, `walls`, `far_plane_m`; existing fields are unchanged. All figures and fits regenerate with `uv run source/mapping/stage0_camera.py` and `uv run source/mapping/stage0_trackability_report.py`; raw numbers are in `cache/mapping/stage0_camera_results.json` and `stage0_trackability_results.json`.

### Task 6: camera height and offset

**Method.** The calibration renders use the same simulator and textures with a solid green floor (`calibration_textures`), because the real floor is two textures with distance fog and cannot be segmented exactly. Geometry and camera are identical. The boundary row is estimated to sub-pixel from the MSAA blend of the first non-floor pixel.

| quantity | value | 95 % interval | note |
|---|---|---|---|
| camera height `h` above the floor | **0.2099 m** | +/- 0.0003 m formal; +/- 0.001 m with the msaa 1 estimate (0.2093) | 960 renders, 4 dev seeds, 0.25 to 3.0 m from the wall; RMS residual **0.098 px**, max 0.87 px |
| principal row `cy` (fitted, free) | 120.0002 | +/- 0.023 | equals the nominal 120: camera is level and the K in `camera.json` is right |
| wall height `H` | 0.2999 m | +/- 0.0002 (from `H - h = 0.0899`) | wall-top fit RMS 0.083 px |
| axial offset `a` of the camera from the rotation centre | **+0.3 mm** (SIFT parallax); +0.5 mm (boundary rows) | +/- 0.8 mm; +/- 1.3 mm | treat as zero |
| lateral offset `l` | **-0.8 mm** (SIFT); +0.07 mm (boundary rows) | +/- 0.9 mm; +/- 0.9 mm | treat as zero |

**Reconciling 0.15 / 0.21 / 0.2099.** The simulator's `height_offset` of 0.15 m is the camera above the robot origin; the origin sits 0.06 m above the floor, so 0.15 + 0.06 = 0.21 m, which is the fit (`h` = 0.2099 and `H` = 0.2999 are both 0.21 and 0.30 to 1 mm). The stage 0 estimate of 0.21 m was right. What was wrong was the 0.10 m "axial offset" that came with it: it was my assumption that wall faces sit at +/- 0.1 m from the cell centre. They do not (next section). The 0.06 m origin height is inferred from the sum, not read from the simulator.

**The renderer draws walls as thin planes through the wall-cell centres, not as 0.2 m blocks.** Found by back-projecting boundary points from 12 yaws at every free cell of a 13 x 13 cell window of seed 7 and overlaying the grid. Each wall cell contributes a segment of length 0.2 m along x when a left or right neighbour is a wall, and along y when a neighbour above or below is; an isolated wall cell contributes an x segment. The fit puts the wall plane within about 2 mm of the cell-centre plane (a combined axial offset and half-thickness of 2.0 +/- 0.2 mm; the camera offset itself is 0.3 +/- 0.8 mm). Consequences:

- The plan's "wall cell centred at (c * 0.2, r * 0.2) has faces at +/- 0.1 m" is wrong for rendering. Use `camera_fit.Scene(...).segments` as the true wall set for the stage 2 and stage 7 references.
- The open width of a one-cell corridor is 0.4 m between wall planes (0.2 m from its centre line to each), not 0.2 m.
- 6.6 % of boundary columns (over 18,287) still deviate by more than 2 px from this model (median 0.085 px, 92.6 % under 0.5 px). Cause not determined. Hypothesis, unverified: grazing views of planes and segment ends where the cell-to-segment rule is not exactly right. Stage 2's scan accuracy criterion should be measured against `Scene.segments` with that in mind.
- Poses after the large rotation events (task 7) lie 0.080 m from the nearest wall plane (p90 0.082, max 0.090, 88 events), versus a median of 0.152 m for random frames: the robot's collision radius is about 0.08 m.

**Rotate-in-place check.** 24 yaws (15 degree steps) at 4 free cells on each of 2 dev seeds. After compensation (which is a no-op, since the offset is zero):

| residual | median | other |
|---|---|---|
| floor-wall boundary row, all columns | **0.085 px** | p90 0.31 px, 92.6 % under 0.5 px, 6.6 % over 2 px |
| SIFT keypoint reprojection (5,658 matches), no offset | 0.58 px | RMS 3.2 px (mismatches), inliers under 3 px: median 0.51 |
| SIFT keypoint reprojection, fitted offset | 0.57 px | inliers under 3 px: median 0.50 |

The SIFT residual does not move with the offset because it is at SIFT's own localisation noise; the test is sensitive only to about 1 mm of offset, which is what the interval says.

Figures: `stage0_height_fit.png`, `stage0_rotation_parallax.png`.

### Task 7: trackability audit (all 9 seeds, 27 drives, 28,837 consecutive-frame pairs)

The three cues are the plan's, in quick provisional versions: SIFT on the wall mask with a RANSAC homography; fraction of columns with a floor-wall boundary; a vanishing point from LSD segments on the floor outline. Floor segmentation uses the measured palette: blue tile (218, 181, 157) drifting to about (222, 195, 178) with distance, and grey (232, 232, 232); recall 0.995 and no wall pixel accepted over 150 frames checked against exact green-floor renders.

| quantity | value |
|---|---|
| translation between consecutive frames | max 0.174 m, p95 at most 0.155 m per drive |
| rotation between consecutive frames, normal | p95 at most 17.3 degrees (nominal 17.2) |
| rotation above 30 degrees | **88 pairs (0.31 %)** |
| rotation above 45 degrees | **27 pairs (0.09 %)**, max **54.8 degrees**; 19 of them on the test seeds |
| inliers vs previous frame | median 26 to 33 per drive (mean 29.5), 5th percentile 8 to 10, **16.9 % of frames under 15** |
| valid boundary columns | median 0.54 to 0.58, 5th percentile 0.34 to 0.38, only **0.11 % of frames under 0.20** |
| vanishing point found | 87.8 % of frames (85 to 92 % per drive) |
| strongest peak within 3 degrees of a true maze-axis direction | only **38.8 %** of the frames that have one |
| frames with at least two cues failing | 2.3 % |
| **blind frames** (all three fail) | **1** of 28,837 (seed 24, drive 2, frame 1230); longest run **1** |
| blind without the lenient vanishing-point cue | 15 frames, longest run 2 |

**Rotation gate: fails.** The 45-degree criterion is not met: 27 pairs exceed it, on every seed except seed 10. All are wall collisions, not turns: the label is FORWARD or IDLE, the translation is only 0.04 to 0.07 m, and the robot ends 0.08 m from a wall plane (see above). They also are the weak spots for vision: inliers are typically 4 to 10 (median in these events about 8) while valid columns stay high (0.4 to 0.8). The consequences for later stages:

- Stage 3/4 yaw unwrapping by "nearest branch to the prediction" will pick the wrong 90-degree branch on these pairs, because the label prior says FORWARD. Stage 4 needs a branch choice that does not trust the prior at low inlier counts, for example the homography rotation from the surviving inliers, or an explicit second hypothesis kept until the next frame disambiguates.
- The stage 1 prior must carry these events as a heavy tail (about 0.3 % of pairs rotate over 30 degrees under a FORWARD or IDLE label), or the chi-square criterion will fail on them.

**Feasibility gate (blind runs): passes.** The longest blind run is 1 frame (2 without the vanishing-point cue), far below the limit of 3, so no drive needs segmentation. This is the finding that matters for the revision: vision alone almost never loses the robot. It is still a count of frames where a cue exists, not of accuracy; stage 3 and 4 measure that.

**Caveats.**

- The vanishing-point cue is lenient. It reports a vanishing point on 88 % of frames but the strongest peak agrees with the true heading within 3 degrees on only 39 % of those, so it is a poor accuracy indicator as built. The blind count is robust to that (it barely changes without the cue); stage 3 must build a proper validity test (the Manhattan constraint, rejecting segments from oblique wall seams) before trusting yaw.
- Inliers are pairwise to the previous frame only; the 17 % of frames under 15 inliers lean on the other two cues.

Figures: `stage0_trackability_<seed>.png` (nine, per-frame inliers, valid columns and vanishing-point validity, blind frames marked), `stage0_blind_examples.png` (the one blind frame plus the nine weakest by cues then inliers; it is a corner view where the floor boundary is visible but narrow).

### Task 8: far plane

- **The far plane is at 5.0 m of depth along the optical axis, and nothing is drawn beyond it:** the background is white (255, 255, 255), the same as the sky. The floor ends at row `cy + fy * h / 5` = 123.86 (row 123 is a blend, 249; rows above are pure white). Checked in three views: outside the maze looking away across open floor (all 320 columns end in white), the 5.8 m corridor (the far wall at 5.8 m is not drawn: 4 central columns end in white, the other 316 end in side walls), and a 4 m corridor (all 320 columns end in walls; the end wall is inside the far plane).
- **Classification rule for stage 2:** a column is a *wall return* only if the first non-floor pixel above the connected floor is a wall pixel and the pixel two rows above it is not white. If it is white, the floor reaches the far plane: label it **no return** (range beyond the limit, free space up to `r_max` and nothing beyond), never a wall at 5 m. Look two rows up because the boundary pixel is an MSAA blend.
- **`r_max` is set by accuracy, not by the far plane.** Depth from a floor row is `z = fy h / (v - cy)`, so one pixel is worth `z^2 / (fy h)`: 0.05 m at 1 m, 0.21 m at 2 m, 0.47 m at 3 m, 0.83 m at 4 m. With the 0.1 px sub-pixel boundary precision measured here on clean renders, the stage 2 error budget (median 0.03 m, p95 0.10 m) is reachable only to roughly 2 to 3 m; the stage 2 choice of `r_max` should start at 2.5 m and be settled by its own error-vs-range plot, with 5 m as the hard ceiling.

Figure: `stage0_farplane.png` (three views, with floor, sky and wall masks).

### Revision 2 acceptance

| criterion | result | pass |
|---|---|---|
| `h` to +/- 2 mm and `a` to +/- 5 mm (95 %), RMS boundary-row residual below 0.5 px | h = 0.2099 +/- 0.0003 m (+/- 0.001 with estimator spread); a = 0.3 +/- 0.8 mm; RMS 0.098 px | pass |
| Offset model explains rotate-in-place renders, residual below 0.5 px | boundary rows: median 0.085 px (92.6 % under 0.5 px); SIFT: median 0.50 to 0.57 px, at its own noise floor and unaffected by the (zero) offset | pass on boundary rows; **SIFT is not below 0.5 px** |
| Rotation between consecutive saved frames never exceeds 45 degrees | **27 pairs exceed it, max 54.8 degrees** (wall collisions) | **fail** |
| Longest blind run at most 3 on every seed | 1 blind frame in 28,837; longest run 1 (2 without the vanishing-point cue) | pass |
| Far-plane behaviour documented with a classification rule | 5.0 m, white background, rule above | pass |
| Addendum with the figures | `stage0_height_fit.png`, `stage0_rotation_parallax.png`, `stage0_trackability_<seed>.png`, `stage0_blind_examples.png`, `stage0_farplane.png` | pass |
| `uv run pytest` including height-fit tests on synthetic data | 24 passed | pass |

### Open issues carried forward

- Rotation gate failure (27 collision events): decided in stage 4 design; see above. I stopped at this finding rather than relaxing the threshold.
- Camera `h` is known to 1 mm; its remaining risk is the `cy` convention (pixel-centre vs corner), which the free-`cy` fit puts at 120.000, consistent with the nominal K.
- The `Scene` rule for wall segments reproduces 93 % of boundary columns within 0.5 px; the rest is unexplained and should be checked in stage 2 before the scene is used as the exact reference.
- The challenge server's `motion_bias` is still unknown (carried over from the first report).
