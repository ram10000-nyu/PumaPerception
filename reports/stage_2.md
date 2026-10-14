# Stage 2: floor segmentation and the virtual laser scan

Plan revision in force: **revision 4** (read from disk before starting; its stated next stage was stage 2).

Status: **all six acceptance criteria pass.** Chosen `r_max` = **1.5 m**. Two things in the stage 0 findings were wrong or incomplete and are corrected here: the wall-segment rule (task 0) and the floor palette (JPEG ringing, below).

## Built

| file | what |
|---|---|
| `source/mapping/segment.py` | pipeline: floor/sky/wall masks and the sub-pixel floor-wall boundary per column, with the far-plane rule |
| `source/mapping/scan.py` | pipeline: back-projection with `h` from `camera.json`, range and bearing per column, hits within `r_max`, free range; `R_MAX = 1.5` |
| `source/mapping/camera_fit.py` | gained `wall_segments` (the corrected reference rule); `Scene` now uses it |
| `source/mapping/stage2.py` | evaluation only: reference check, accuracy and bias by range, coverage, mask review, oracle maps |
| `cache/mapping/<seed>/oracle_map.npz` | oracle occupancy grid per seed (0.05 m, log-odds, origin, `r_max`) for stage 7 |
| `cache/mapping/stage2_results.json` | every number below |
| `tests/mapping/test_scan.py`, `test_stage2.py` | 15 + 10 tests; `uv run pytest`: 65 passed |

`segment.py` and `scan.py` read only `cache/mapping/camera.json` (measured constants); a test checks that they never mention ground truth, `Scene` or the simulator (rule 10). Reproduce: `uv run source/mapping/stage2.py all`.

## Task 0: the reference check (cause found, rule fixed)

Stage 0's segment rule explained 93 % of boundary columns. The cause of the rest was the rule for **corners**, not grazing views: it gave every wall cell a 0.2 m segment along each neighbour axis, which extends a corner 0.1 m past its centre. Back-projecting boundary points around the L-corner at cell (4, 4) of seed 7 shows the renderer draws nothing there (`stage2_reference_check.png`, left and middle).

**Corrected rule** (`camera_fit.wall_segments`): a segment joins the centres of every pair of 4-adjacent wall cells; the wall extends 0.1 m past the cell centre only at a **free end** (a wall cell with no neighbour: an x segment of 0.2 m; or exactly one neighbour: the stub continues in its own direction). Corners and junctions end at the cell centre. Isolated cells were checked directly on five cells of seed 7: x segments 0.22 m long (0.2 m plus the plane's thickness), on the centre line. The wall half-thickness is **2.2 mm** (swept 1.0 to 3.0 mm on dev seeds; 99.2 % at 2.0 and 2.2, 97.4 % at 1.5, 98.7 % at 2.5).

| rule | columns within 0.5 px | within 2 px | median |error| |
|---|---|---|---|
| stage 0 rule | 90.5 % | 91.4 % | 0.090 px |
| **corrected rule** | **99.21 %** (258,988 columns, 4 dev seeds) | 99.63 % | 0.084 px |

Per dev seed: 99.12, 99.09, 99.33, 99.31 %. The remaining 0.8 % (2,037 columns) are characterised, not excluded: median depth 0.33 m (46 % under 0.3 m, where 1 mm of depth is already 0.5 px), and 46 % within 2 cm of a segment end. By depth the corrected rule scores 98.8 % (0 to 0.25 m), 99.5 %, 99.6 %, 98.4 % (1 to 2 m) and 94.7 % (2 to 5 m, 1,681 columns). Calibration renders (solid green floor) were used, as in stage 0.

## Segmentation (what changed from stage 0's audit code)

The stage 0 palette (a box around the blue tile and the grey tile) worked on clean renders but **failed on the JPEG frames the pipeline actually receives**, and the first version of this stage measured it: errors that grew with range and were one-sided (the boundary too near; median +0.96 px at 1.5 m and beyond, 50 % of columns off by more than 1 px), while clean renders were within ±0.2 px. Two causes, both fixed:

1. **Floor palette = a line, not a box.** Everything the floor looks like (blue tile, grey tile, the distance fog that drifts the blue tile towards grey, and the 1 to 2 px blends where the two tiles meet) lies on the straight line from (218, 181, 157) to (232, 232, 232). Pixels within 5 levels of that line are floor. White (sky and far plane) lies past the grey end and is excluded. On clean renders: recall 1.000, no wall pixel accepted (120 frames, exact green-floor ground truth). The old rule lost the tile seams, which put a boundary inside the floor.
2. **JPEG ringing.** Next to a dark wall the floor pixels in the same 8 x 8 block shift by 10 to 25 levels for 4 to 5 rows, off any strict palette. The boundary is therefore found with a **loose** palette (30 levels) by a **per-column step fit** (the row minimising strictly-floor pixels above it plus not-even-loosely-floor pixels below it, so isolated noise pixels cost one each and do not move it), and only the sub-pixel step uses clean colours: the floor share of the blend pixel is solved against the first strictly floor-coloured pixel below the edge and the wall pixel above it.

Also from the plan: the border-connectivity prior (floor must connect to the bottom row; floor-coloured wall texture higher up stays wall) is in `connected_floor`, and the far-plane rule is implemented as specified: a column is a wall return only if the pixel above the floor is a wall pixel and the pixel two rows above that is not white; white means **FAR** (no return, free up to `r_max`), never a wall at 5 m. A wall closer than about 0.16 m (the bottom pixel is not floor) is **INVALID**.

With these, on JPEG frames and in the final pipeline the pixel error of the boundary against the reference is: median -0.02 to +0.10 px, 5th to 95th percentile about -0.5 to +1.9 px (0 to 1.5 m) and -0.5 to +4 px beyond 1.5 m.

## Mask review (50 random test-seed frames)

Contact sheet `stage2_mask_contact_sheet.png` (floor tinted green, sky orange, boundary in red, per-frame wall-to-floor percentage in each title). I checked it frame by frame, and each frame's wall pixels were also scored exactly against a solid-green-floor render of the same pose (a rim of one pixel around the true floor is excluded from the wall area):

| quantity over the 50 frames | median | max |
|---|---|---|
| wall pixels labelled floor, as a share of wall area | 0.000 % | **0.036 %** |
| floor pixels labelled wall (away from the boundary) | 0.15 % | 0.37 % |
| sky pixels not recognised as sky | | 5.2 % (a 1 px rim of blend pixels along the wall tops; harmless) |

**No failures:** no frame comes close to the 5 % limit.

## Scan accuracy against the true wall segments (test seeds, true poses)

Every third frame of all drives, all columns with a wall return; the endpoint is placed at the true pose and scored by distance to the nearest reference segment. Dev seeds were used to choose `r_max`; the test numbers are reported only.

| metric | dev (4 seeds) | **test (5 seeds)** | threshold | pass |
|---|---|---|---|---|
| returns within `r_max` = 1.5 m | 1,072,400 | 1,397,043 | | |
| endpoint to nearest segment, median | 0.0027 m | **0.0028 m** | < 0.03 m | pass |
| endpoint to nearest segment, 95th percentile | 0.0227 m | **0.0232 m** | < 0.10 m | pass |
| 99th percentile | 0.074 m | 0.077 m | | |
| per test seed, median / p95 | | 0.0027 to 0.0029 m / 0.0217 to 0.0253 m | | pass on all five |

By true range (test), median / p95 distance in metres: 0 to 0.25 m: 0.002 / 0.006; 0.25 to 0.5: 0.002 / 0.009; 0.5 to 0.75: 0.003 / 0.020; 0.75 to 1: 0.006 / 0.034; 1 to 1.25: 0.009 / 0.066; 1.25 to 1.5: 0.015 / 0.083; 1.5 to 1.75: 0.020 / 0.123; 2 to 2.25: 0.038 / 0.162; 2.5 to 2.75: 0.057 / 0.186 (`stage2_range_error.png`).

## No range bias (the `h` cross-check, task 4)

Median signed range error as a share of range, per 0.25 m bin of true range (test): **between -0.09 % and +0.06 % up to 1.5 m**, -0.14 % at 1.5 to 1.75 m, -0.3 % at 1.75 to 2 m, -0.45 % at 2.25 to 2.5 m, -0.6 % at 2.75 to 3 m; all within the ±1 % criterion (dev gives the same to 2.5 m, then noisy bins with under 1,000 returns each). There is no trend proportional to range up to 1.5 m, so `h` = 0.2099 m from stage 0 is right to well under 1 % and nothing goes back to stage 0. The slight negative drift beyond 2 m is the small sample's tail, not a scale error.

## Coverage

Mean fraction of columns per frame with a wall return within 1.5 m: **0.801 test** (0.805 dev); 5th percentile of frames 0.60; any range 0.848; far-plane columns about 0 (corridors in these mazes never reach 5 m of floor except looking along the longest runs); INVALID (wall nearer than 0.16 m) 0.152. Threshold 0.60: pass. 5 % of frames are under 0.60 (the minimum is 0.01 to 0.0, a wall filling the view).

## `r_max`

The plan said to start at 2.5 m and settle it from the error-vs-range plot. Rule used: the largest range such that **every 0.25 m bin up to it** has a dev median under 0.03 m, p95 under 0.10 m and median bias within ±1 %. Dev bins pass through 1.25 to 1.5 m (p95 0.085 m) and fail at 1.5 to 1.75 m (p95 0.118 m), hence **`R_MAX = 1.5` in `scan.py`**.

| `r_max` | returns | median | p95 | p99 | coverage (dev) |
|---|---|---|---|---|---|
| **1.5 m** | 1,072,400 | 0.0027 | 0.023 | 0.074 | 0.805 |
| 2.0 m | 1,106,956 | 0.0028 | 0.028 | 0.086 | 0.831 |
| 2.5 m | 1,116,571 | 0.0028 | 0.030 | 0.092 | 0.838 |
| 3.0 m | 1,119,583 | 0.0028 | 0.030 | 0.095 | 0.840 |
| 5.0 m | 1,121,410 | 0.0028 | 0.031 | 0.099 | 0.842 |

The overall acceptance criteria are met at every `r_max` up to 5 m, because 95 % of returns are within 1.5 m (the mazes are corridors 0.4 m wide: corridors are short). My per-bin rule is stricter than the criteria: returns beyond 1.5 m cost about 4 % of the data but have a p95 of 0.12 to 0.19 m. **Decision for the reader:** 1.5 m trades about 3 points of coverage for sharper endpoints; if stage 5 (scan matching) wants more far points in long corridors, 2.5 m passes the stated criteria too, with a larger per-return variance (depth error per pixel is 0.21 m at 2 m). Keep `R_MAX` configurable in `scan.py`; nothing else in this stage depends on it except the oracle maps.

## Oracle maps (for stage 7)

Occupancy grids built from the scans at **true** poses (0.05 m cells, log-odds with occupied +0.85, free -0.4, clamp ±5; origin -0.225 m so the wall planes, at multiples of 0.2 m, fall on cell centres; a first version with the origin at -0.2 m put the planes on cell boundaries and erased half of them, IoU 0.45). Rasterised true wall segments are the layout reference.

| seed | IoU vs true layout (1 cell tolerance) | (2 cells) |
|---|---|---|
| 7, 8, 9, 10 (dev) | 0.834, 0.844, 0.871, 0.825 | 0.861, 0.865, 0.891, 0.852 |
| 21, 22, 23, 24, 25 (test) | 0.824, 0.808, 0.817, 0.800, 0.815 | 0.848, 0.836, 0.844, 0.823, 0.835 |

This is the ceiling that stage 7's map IoU (>= 0.75 against the oracle, >= 0.70 against the true layout) is measured against: a map built from true poses reaches only about 0.80 to 0.87 against the true layout, so the 0.70 requirement leaves little room for pose error. The occupied-cell count is 88 to 93 % of the rasterised true wall cells (1,607 to 1,841 against 1,782 to 1,990). I did not investigate which walls are missing; likely candidates are corridor ends beyond `r_max` and walls only seen at grazing angles. Files: `cache/mapping/<seed>/oracle_map.npz`, `stage2_oracle_maps.png`.

## Acceptance

| criterion | result | pass |
|---|---|---|
| Mask review: no frame with more than 5 % wall labelled floor | 50 frames, max 0.036 %, no failures | pass |
| Task 0: cause identified, reference explains >= 99 % of columns within 0.5 px on dev | cause: corner rule; 99.21 %; remainder characterised | pass |
| Scan accuracy: median < 0.03 m, p95 < 0.10 m (test, <= `r_max`) | 0.0028 m, 0.0232 m | pass |
| No range bias: median signed error within ±1 % of range in every bin up to `r_max` | within ±0.09 % up to 1.5 m (and within ±1 % up to 3 m) | pass |
| Coverage >= 60 % of columns per frame | 0.801 (test), 0.805 (dev) | pass |
| Unit test: synthetic floor point projected then back-projected recovers (X, Z) to 1e-9 | `test_backproject_recovers_projected_points_to_1e9` | pass |

## Figures

`stage2_reference_check.png` (task 0), `stage2_mask_contact_sheet.png` (50 frames), `stage2_range_error.png` (distance and signed range error vs range, dev and test, returns per bin), `stage2_oracle_maps.png` (nine oracle maps with the true layout in red).

## Open issues and notes for later stages

- **The plan's facts table should change** (I did not edit the plan file): the wall rule is the corrected one above, the wall half-thickness is 2.2 mm, and the floor palette is the blue-grey **line** (strict 5 levels, loose 30 for JPEG). The stage 0 statement "recall 0.995, no wall pixel accepted" was measured on clean renders and is not the JPEG behaviour.
- Stage 0's `trackability.py` still uses the old palette; its cue statistics (valid columns, blind frames) were therefore computed with a floor test that fails near dark walls on JPEG. They are likely pessimistic for the valid-column fraction; the blind-frame conclusion (1 in 28,837) can only improve. Not rerun, per the plan.
- Depth error is dominated by pixel quantisation beyond 1.5 m (0.21 m per pixel at 2 m); the error is heavy-tailed (p99 0.074 m at 1.5 m) and one-sided errors are gone, but stage 5's scan matching should still use a robust loss.
- Stage 3 can reuse `segment.floor` and the boundary rows (the seam lines of the floor outline) for vanishing-point detection; this stage did not produce segments for it.
- Evaluation used every third frame for the accuracy and bias numbers; coverage and the oracle maps use every frame, and the mask review uses the 50 listed frames.
