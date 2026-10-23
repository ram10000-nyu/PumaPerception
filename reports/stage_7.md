# Stage 7: the top-down map

Plan revision in force: **revision 9**. Implemented by the user's Claude session in a Linux VM on the user's Mac (4 cores; real simulator not needed), from the stage 6 poses and the stage 3 scan caches on `gt4`.

Status: **every criterion passes on all five test seeds.**
- IoU against the oracle map: **0.945 to 0.979** (target ≥ 0.75).
- Snapped walls: **98.7 to 100 % recovered, 0 to 0.22 % false** (targets ≥ 95 %, ≤ 5 %).
- Lattice: scale detected within **0.07 %** of the map's true scale, offset within **1.1 mm**. Applying it fixes stage 6's scale (Umeyama scale 1.0034–1.0085 → 0.9998–1.0007).
- **100 %** of true poses lie in the configuration space.
- End-to-end `build_map.py`, from raw images to `map.npz`: **173 s** (test seed 21; target < 5 min).

Two findings change facts used elsewhere in the plan:
1. **The robot's collision radius is about 0.12 m, not 0.08 m.** Measured against a different geometry; details below.
2. **A post can be missing between two walls.** This happens in about 2 % of junctions in the true layouts, and the gap is visible. The snapping handles it.

## Built

| file | what |
|---|---|
| `source/mapping/grid.py` | **pipeline**: occupancy grid, lattice and scale fit, wall snapping, collision geometry, configuration space, wall-face signatures, `map.npz` writer, rendering |
| `source/mapping/slam.py` | **pipeline**: the stage 5b union-rule closures and the stage 6 pose graph as functions of raw inputs (stage 6's evaluation driver read cached, scored rows). Same frozen parameters. Reproduces stage 6's poses to 5×10⁻⁵ m on the same closures. |
| `source/mapping/build_map.py` | **pipeline CLI**: `--data <dir>` → `map.npz`, `map.png`, `map_snapped.png`. Five resumable steps (inputs, odometry, closures, graph, map), parallel over drives and over candidate pairs; `timing.json` |
| `source/mapping/stage7.py` | evaluation: oracle maps (task 0), scoring against the true layout, dev sweep, figures |
| `tests/mapping/test_grid.py` | 6 tests: synthetic maze (lattice and scale recovery; snapping of a missing post, an isolated cell and a stub; collision geometry), Hellinger distance, two rule-10 leakage tests. All pass, and the other 157 mapping tests still pass. The one test that needs the real simulator (`test_record_gt`) was not run in the VM. |
| `cache/mapping/<seed>/map.npz`, `map.png`, `map_snapped.png` | all 9 seeds, from the stage 6 poses |
| `cache/mapping/<seed>/oracle_map.npz`, `oracle_map_gt4.npz` | regenerated oracle maps (`gt` and `gt4`); the stage 2 version is kept as `oracle_map_stage2.npz` |
| `cache/mapping/build/gt4-{7,21}/` | end-to-end runs |
| `cache/mapping/stage7/*.json` | every number below |
| `reports/figures/stage7_maps_gt4_<seed>.png` | per seed: oracle, estimate, snapped walls over the true layout with the free space |

Reproduce:
```
python3 source/mapping/stage7.py oracle
python3 source/mapping/stage7.py run dev|test      # then: figures test
python3 source/mapping/build_map.py --data data/gt4-21 [--steps inputs,odometry,...]
```

## Task 0: oracle maps regenerated

The same scans at the true poses, now from the current segmentation with wall-top returns, built by `grid.occupancy`. IoU against the full true layout (one-cell tolerance):

| | 7 | 8 | 9 | 10 | 21 | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|---|---|---|---|
| stage 2 oracle (`gt`) | 0.83 | | | | 0.82 | 0.81 | 0.82 | 0.80 | |
| **now, `gt`** | 0.957 | 0.951 | 0.961 | 0.942 | 0.941 | 0.945 | 0.936 | 0.934 | 0.941 |
| **now, `gt4`** | 0.979 | 0.969 | 0.978 | 0.965 | 0.970 | 0.970 | 0.952 | 0.959 | 0.965 |

The ceiling the plan quoted (0.80 to 0.87) is now 0.93 to 0.98. Every true wall cell is reached by some true-pose ray on every seed: the three drives see the whole maze. So "observable" and "full layout" coincide here, and both IoU criteria measure the same thing.

## How the map is made

**1. Lattice and scale.**
- **Wall points and phasors.** The walls lie on lines every 0.4 m: wall cells sit at even indices of the 0.2 m cell grid, true on all 9 layouts. Each scan point on a straight piece whose normal is within 18° of a maze axis is a sample of one wall line's coordinate. A periodogram (the weighted phasor sum of `exp(2πi c / 0.4)`) is largest when all of them sit on lines.
- **Rescale the poses, not the scans.** The scale error is the odometry's, and the scans are metric from the camera height. So the search rescales the poses about the anchor: a point seen from robot coordinate `pc` moves by `(pc − anchor)(1/s − 1)`. It searches 1/s over ±3 %, then shifts the poses so the lines fall on multiples of 0.4 m.
- **The first version was biased.** It rescaled whole points, and a synthetic unit test caught it: 0.43 % error on a 2 m maze where scan ranges are comparable to the maze. On the real 6 m mazes the difference was at most 0.05 %, and it changed no criterion.

**2. Occupancy grid.**
- The grid is log-odds at 0.05 m, with wall lines on cell centres.
- Each column casts one ray. Cells before a return get −0.4. The return's cell gets +0.85 × w, where w = clip(0.01 m / σ, 0.25, 1) and σ is the return's depth noise (1.6 cm at 1 m, so far returns count less).
- A column whose return lies beyond 1.5 m, or with no return (far plane), only clears free space up to 1.5 m. Values are clamped at ±5.
- Parameters were chosen on dev; the sweep is below.

**3. Snapping to wall cells.**
- **Edge cells** (one index odd) are walls when the mean, over 5 samples along their 0.2 m span, of the largest log-odds within one cell across the line is positive, and some free space is known 0.1 m to one side. That side rule removes returns smeared behind the outer wall.
- **Posts** (both indices even) with 3 or 4 wall neighbours are walls. Every other post is decided by its own evidence.
- The true layouts have cases the first version got wrong. Each was counted on all 9 layouts (geometry only, no scoring):
  - A post is **missing between two collinear walls** at 12 of 815 junctions, and between two perpendicular walls at 11 of 451. The rendered stubs each stop 0.1 m short of the post, so the gap is visible. The first rule ("2 wall neighbours → post") put a false wall in every such gap: 2 to 3 per dev seed, which pushed true poses out of the configuration space.
  - **Isolated vertical edge cells** (5 in 9 layouts) render as a horizontal 0.2 m piece (the renderer's rule for a lone cell), so their evidence is also checked along x.
- The output geometry follows stage 2's renderer rule: centres of 4-adjacent wall cells, plus 0.1 m past free ends.

**4. Collision geometry and configuration space.** This is a new measurement, made on dev.
- Measured against the rendered geometry, true poses come as close as 0.047 m, always at a wall's end. Along faces the closest is 0.117 m.
- Against the segments between centres of adjacent wall cells, *without* the free-end extension, the minimum is 0.117 to 0.124 m on all 9 seeds. Test seeds alone: 0.118 to 0.122.
- So the simulator collides the robot, radius about 0.12 m, with the unextended wall-cell centre lines.
- Stage 0's "0.08 m" was measured to the rendered planes after collisions at wall ends.
- The configuration space is the known-free cells farther than **0.115 m** from those collision segments, restricted to the 4-connected components the robot visited.

**5. Wall-face signatures.** Per wall cell and side (N/S/E/W): a 16 × 4 hue-saturation histogram of the wall pixels that saw it (Hellinger-normalised), the pixel count, and the frames that saw it.
- The plan said "per lattice wall segment". I stored per face instead, because stage 5b found that texture changes per 0.2 m cell face and per side.
- Repeatability: for faces seen by two drives, the Hellinger distance between the two drives' signatures has a median of 0.14–0.17. For different faces it is 0.78–0.82. The chance that a same-face pair is closer than a different-face pair (AUC) is 0.98 to 0.99 on every seed. About 1,200 faces per seed.

## Dev tuning (frozen before test)

8 configurations on dev (`sweep_dev.json`). Worst seed for each:

| l_occ | l_free | τ | IoU vs oracle | snap recall | snap false | in config. space |
|---|---|---|---|---|---|---|
| **0.85** | **−0.4** | **0** | **0.820** | **0.985** | **0.0022** | **0.996** |
| 0.85 | −0.2 | 0 | 0.821 | 0.998 | 0.0042 | 0.994 |
| 1.5 | −0.4 | 0 or 1 | 0.822 | 0.998 | 0.0042 | 0.994 |
| 0.85 | −0.4 | 1 | 0.820 | 0.980 | 0.0022 | 0.996 |

The sweep ran before the final lattice fit and the connected-component restriction; neither changes this ranking. The configuration-space criterion decides: a false wall cell in a corridor pushes true poses out. So the defaults (stage 2's l_occ and l_free, τ = 0) were kept. All other constants were set before any test seed was scored.

## Results

From the stage 6 poses, `gt4`:

| seed | IoU vs oracle | IoU vs true layout (oracle's) | snapped recall / false | raw-map cell size | scale detection error | offset error | pose RMSE before → after lattice | Umeyama scale before → after | in config. space |
|---|---|---|---|---|---|---|---|---|---|
| 7 (dev) | 0.958 | 0.942 (0.979) | 0.993 / 0.002 | 0.1994 | −0.004 % | 1.2 mm | 0.025 → 0.019 | 1.0028 → 1.0000 | 0.996 |
| 8 (dev) | 0.821 | 0.813 (0.969) | 0.985 / 0 | 0.1976 | −0.038 % | 1.9 mm | 0.046 → 0.028 | 1.0118 → 0.9996 | 1.000 |
| 9 (dev) | 0.880 | 0.873 (0.978) | 0.992 / 0.002 | 0.1979 | 0.049 % | 2.9 mm | 0.055 → 0.028 | 1.0111 → 1.0005 | 1.000 |
| 10 (dev) | 0.949 | 0.932 (0.965) | 1.000 / 0 | 0.1997 | 0.022 % | 1.6 mm | 0.038 → 0.020 | 1.0015 → 1.0002 | 1.000 |
| **21** | **0.953** | 0.939 (0.970) | 0.998 / 0 | 0.1985 | 0.068 % | 1.1 mm | 0.037 → 0.018 | 1.0085 → 1.0007 | 1.000 |
| **22** | **0.971** | 0.956 (0.970) | 1.000 / 0 | 0.1990 | −0.011 % | 0.2 mm | 0.043 → 0.014 | 1.0049 → 0.9999 | 1.000 |
| **23** | **0.979** | 0.940 (0.952) | 1.000 / 0.002 | 0.1993 | −0.023 % | 0.7 mm | 0.036 → 0.010 | 1.0034 → 0.9998 | 1.000 |
| **24** | **0.945** | 0.924 (0.959) | 0.998 / 0 | 0.1987 | 0.039 % | 0.5 mm | 0.032 → 0.016 | 1.0068 → 1.0004 | 1.000 |
| **25** | **0.967** | 0.942 (0.965) | 0.987 / 0 | 0.1987 | 0.037 % | 0.9 mm | 0.031 → 0.018 | 1.0069 → 1.0004 | 1.000 |

How to read the columns:
- *Raw-map cell size* is 0.2 m × the poses' scale. Detecting it is the point; the corrected map's cells are 0.2 m by construction.
- *Scale detection error* is the detected scale times the true one (Umeyama), minus 1.
- *Offset error* is the mean position error left after the lattice shift. Before the shift, the map was offset by up to 29 mm (dev) and 25 mm (test); that is what the shift removes.

Missing walls (true layout cells not covered by the map):
- On test: 0, 0, 1, 2 and 7 cells out of about 1,900.
- The 2 on seed 24 were never in range of a true-pose ray.
- The rest are pose error: the oracle has them. Seed 25's 7 are one corner (0 to 0.8 m along x = 0), where the map is doubled by about 6 cm (visible in `stage7_maps_gt4_25.png`).

Dev seeds 8 and 9 have the lowest IoU (0.82, 0.88). They have the largest pose error left after stage 6 (2.8 cm RMSE after the lattice), and their walls appear doubled in places. The snapped walls are still 98.5 % and 99.2 % right.

**End to end** (`build_map.py`, caches cleared, 4 workers in the VM):

| step | seed 7 (dev) | seed 21 (test) |
|---|---|---|
| inputs (images → scans, cues, SIFT, features, boundaries) | 68.7 s | 88.4 s |
| odometry (calibrated priors, filter) | 34.1 s | 39.2 s |
| closures (union rule; 363 on seed 21) | 22.8 s | 27.5 s |
| graph | 2.8 s | 6.4 s |
| map (incl. signatures) | 9.9 s | 11.5 s |
| **total** | **138 s** | **173 s** |

The poses agree with the stage 6 evaluation run to 1.2 mm (seed 7) and 1.6 mm (seed 21) at worst; RMSE 0.3 and 0.7 mm. The maps score the same: seed 21 IoU 0.954, snapped recall 0.998, no false cells; seed 7 IoU 0.960, recall 0.993.

## Acceptance

| criterion | result | |
|---|---|---|
| IoU vs oracle ≥ 0.75 on every test seed | 0.945 to 0.979 | pass |
| IoU vs observable true walls ≥ 0.75 (and report vs full layout) | 0.924 to 0.956 (all walls observable, so the same as full) | pass |
| Snapped walls: ≥ 95 % recovered, ≤ 5 % false | 98.7 to 100 % / 0 to 0.22 % (counted per wall cell) | pass |
| Cell size within 2 %, offset within 0.02 m | raw map −0.36 to −0.77 % (detected to ≤ 0.07 %); offset ≤ 1.1 mm | pass |
| ≥ 99.5 % of true poses in the configuration space (0.01 m allowance) | 100 % on test (dev 99.6 to 100 %) | pass, with the corrected radius (below) |
| Leakage test (rule 10) for `build_map.py` | static scan of `grid.py`, `slam.py`, `build_map.py`; map identical with and without `poses.json`/`gt.json` | pass |
| `build_map.py` < 5 min per maze, caches cleared | 173 s | pass |

## Disclosures

- **Radius.** The configuration-space criterion was met by *changing the radius and the collision geometry* (0.08 m to the rendered walls → 0.115 m to the unextended centre lines), on dev evidence. With stage 0's model, even the true walls fail the criterion (99.6 to 99.9 % of true poses), because the robot passes closer than 0.08 m to rendered wall ends. The new model is a measurement of the simulator, not a tuning knob: no true pose on any of the 9 seeds violates it.
- **Snapping rules.** The missing-post and isolated-cell rules came from counting cases in the dev and test *layouts* (geometry, not scores) after the first dev run. The dev scores motivated them. The test seeds were scored once those rules were in.
- **Lattice fit changed after the first test scoring.** A synthetic unit test showed the bias of rescaling whole points; this was not driven by test scores, which already passed. Dev and test were rerun, and every number above is from the final code.

## Not done

- Maps on the `gt` (msaa 1) frames: stage 6 ran on `gt4` only. The `gt` oracle maps are regenerated.
- The challenge download (one frame per tick) is stage 9's.

## For stage 8

`map.npz` holds:

| field | contents |
|---|---|
| `logodds`, `hits`, `origin`, `resolution` | the grid (lattice-aligned frame: walls at multiples of 0.4 m) |
| `wall_cells` (with `cells_origin`), `wall_segments` | rendered geometry, for matching a goal view's scan |
| `collision_segments`, `cspace` | for planning |
| `poses` | lattice-aligned poses (drive, k, x, y, θ) |
| `sig_keys` (r, c, side), `sig_hist`, `sig_pixels`, `sig_frames` | wall-face signatures |
| `lattice_*` | the fitted scale, phase and offset |

`build_map.py` writes the same for a challenge directory, in its own frame (anchor = the first drive's start).

- **Goal placement** can score candidate poses on the lattice: positions on the free cells, headings near multiples of 90° refined by the compass. Score by the scan's fit to `wall_segments` plus the face signatures.
- **Signatures and twins.** Same-face signature distance ≈ 0.15, different faces ≈ 0.8. But 75 % of faces have a texture twin (stage 5b), so a signature confirms a pose hypothesis; it cannot pick one alone.
