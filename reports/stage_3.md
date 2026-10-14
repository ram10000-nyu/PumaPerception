# Stage 3: vanishing-point compass (with the revision 5 additions)

Plan revision in force: **revision 5** (read from disk first; its header lists the stage 3 additions, which sit in the section "Stage 3 additions (revision 5)" after stage 3b: a second heading cue from the scan, and wall-top ranging).

Status: **all four acceptance criteria pass on the fused output, on both frame types (below).** The level-camera check passes with a stated qualification. One finding about the data is flagged first because it affects how to read every number from stage 1 on.

## A finding first: the local recordings are not anti-aliased, the server's frames are

`vis_nav_sim.Simulator` defaults to `msaa=1`, and `cli.record_exploration_data` (hence every `data/gt-<seed>/` set) does not override it; the server renders with `msaa=4`. I checked the real challenge frames in `data/ab7efaeb1abf/`: **70 % of their floor-wall boundary columns have a fractional row (anti-aliased blends), against 13 % in the local recordings.** So the local datasets are a pessimistic proxy for the server's frames in one respect: sub-pixel precision of every edge (the boundary row is quantised to whole pixels, noise about 0.29 px).

I did not re-record (the plan forbids it). Instead every number below is reported on two kinds of frames:

- **official**: the recorded local frames, unchanged;
- **msaa4**: frames rendered at the *same true poses* with `msaa=4` and JPEG-encoded with the recorder's settings (quality 90, 4:4:4), the way the server would send them. This is evaluation-only code (`stage3_extract.msaa4_frames`).

Thresholds were tuned on **official dev** only. The msaa4 columns are a robustness check. Both pass.

## Built

| file | what |
|---|---|
| `source/mapping/vp.py` | vanishing-point cue: seam segments (LSD on the floor mask in the seam band; a polyline variant for comparison), horizon angle and noise per segment, robust circular peak, two-family split, implied focal length |
| `source/mapping/scan_heading.py` | scan cue: straight pieces of the virtual scan, wall direction mod 90 |
| `source/mapping/heading.py` | fusion, the plan's vanishing-point validity test, `estimate_heading(img)` for one frame |
| `source/mapping/segment.py`, `scan.py` | wall-top ranging (status `TOP_RETURN`, `Scan.z_top`); boundary sub-pixel now reads the floor share of both pixels around the edge |
| `source/mapping/stage3_extract.py`, `stage3.py` | cached per-frame cues (pipeline path) and the evaluation (reads ground truth) |
| `tests/mapping/test_heading.py` | 22 tests; `uv run pytest`: 87 passed |

Reproduce: `uv run source/mapping/stage3_extract.py && uv run source/mapping/stage3.py`. Results: `cache/mapping/stage3_results.json`. The new pipeline modules never mention ground truth (a test checks it, rule 10).

## How the two cues work

**The key observation:** every wall seam is along a maze axis, so a floor-wall seam line crosses the horizon at `alpha = yaw mod 90` or `alpha - 90` (stage 0's convention: positive clockwise, `alpha = atan((x_h - cx) / fx)`). Both families vote for the **same** number `psi = yaw mod 90`, so no pairing of peaks is needed to get the heading; the two-peak structure is then used to *test* the Manhattan constraint and the focal length.

- **Vanishing-point cue.** LSD on the stage 2 floor mask, keeping segments of at least 8 px with at least 60 % of their samples in the seam band (the wall-return boundary dilated 2 px). Each segment's line gives `alpha = atan2(-A, B)` with `A = a cx + b cy + c`, `B = a fx` (well conditioned for lines parallel to the horizon too), and an angular standard deviation propagated from the line-fit noise (0.25 px per boundary point) plus a 0.3 degree model floor. A robust circular peak (smoothed histogram start, then iteratively reweighted circular means over inliers within max(2 degrees, 3 sigma)) gives `psi` and its sigma.
- **Scan cue** (revision 5). Straight pieces of the stage 2 scan (returns up to 2.5 m), found by line fits over runs of adjacent returns, **longest window first (41, 25, 15, 9 columns)** and accepted if the residual fits the depth noise. A first version used 9-column windows only and was wrong on 75 % of frames: the boundary row is quantised to a pixel on the official frames, so a gently sloped wall is a staircase and short windows read it as a wall facing the camera. Longer windows average the staircase out. A piece of direction `theta` in the robot frame votes `psi = -theta mod 90`; the same circular peak is used.
- **Polyline variant.** The same peak machinery on straight pieces of the boundary polyline (Douglas-Peucker split). The first version had a bug (split point chosen from the total-least-squares line instead of the chord, so it missed corners; fixed). It is still the worse choice on the official frames (below), so **LSD is the vanishing-point cue**, as the plan prescribes.

## Fusion and validity

The cues are independent measurements. **When they agree within `TAU` = 3 degrees the result is their inverse-variance circular mean and it is valid; otherwise the frame is flagged `conflict` and is not valid.** Chosen on official dev:

| `tau` | frames valid | valid frames under 3 degrees |
|---|---|---|
| 1 | 85.5 % | 99.97 % |
| 2 | 95.9 % | 99.96 % |
| **3** | **97.2 %** | **99.95 %** |
| 5 | 97.8 % | 99.93 % |

The alternative, resolving disagreements with the plan's per-cue validity test, does not reach the required precision: on the 346 dev frames where the cues disagree, the vanishing-point cue is right 74 % of the time when it passes its test and the scan cue 76 % when the vanishing-point cue does not (msaa4: 67 % and 89 % on 149 frames). Neither is near the 95 % that a validity flag has to keep, so a conflict stays a conflict. A frame with only one cue (never happened in these data) is valid only if the vanishing-point cue passes the plan's test: two peaks consistent with the Manhattan constraint with the implied focal length within 5 % of fx and at least 25 px per family, **or** one peak within 20 degrees of the centre with at least 40 px of segments on each side of the image and 80 px in total.

Confidence = `1 - disagreement / 3 degrees` (1 for identical cues). It is monotone with error: median fused error 0.62 / 0.50 / 0.33 / 0.21 / 0.15 degrees from the lowest to the highest confidence bin (official test; p95 1.8 to 0.6 degrees), `stage3_error_vs_confidence.png`.

## Results (test seeds)

Error is `diff90(estimate, true yaw)` in degrees, modulo 90. "Valid" is the fused flag.

| | official frames | msaa4 frames | threshold |
|---|---|---|---|
| frames | 16,347 | 16,347 | |
| **valid yaw coverage** | **96.0 %** | **97.8 %** | >= 70 % |
| **median error, valid frames** | **0.21 degrees** | **0.16 degrees** | < 1.5 |
| **95th percentile, valid** | **0.75 degrees** | **0.56 degrees** | < 4 |
| 99th percentile, valid | 1.20 | 0.89 | |
| **valid frames under 3 degrees** | **99.95 %** (8 bad of 15,694) | **99.96 %** (7 bad of 15,981) | >= 95 % |
| conflict frames (not valid) | 4.0 % | 2.2 % | |
| the invalid frames where a cue is under 3 degrees anyway | 88.5 % | 93.4 % | |
| vanishing-point cue alone, all frames: median / p95 / under 3 | 0.39 / 1.64 / 97.6 % | 0.33 / 1.18 / 98.7 % | |
| scan cue alone, all frames: median / p95 / under 3 | 0.17 / 0.96 / 97.9 % | 0.13 / 0.59 / 98.9 % | |
| vanishing-point cue under the plan's validity test | valid 79.5 %: median 0.38, p95 1.33, 99.47 % under 3 | valid 83.4 %: 99.72 % under 3 | |
| the frames the plan's test rejects: under 3 degrees | 90.3 % | 93.6 % | |
| polyline variant, all frames: median / under 3 | 1.46 / 62.5 % | 0.31 / 85.8 % | (not used) |

Dev seeds give the same picture (official: valid 97.2 %, median 0.20, p95 0.72, 99.95 % under 3).

Reading the validity criterion: stage 0's detector agreed with the true heading on 39 % of the frames it called valid; the fused flag is right on 99.95 %, and the 4 % of frames it refuses are the ones where the cues disagree (88 to 93 % of those still have one cue right, which is why the rule is not "pick the better cue": nothing in the features says which).

**Confidently wrong frames.** 8 valid frames on official (7 on msaa4) have an error above 3 degrees, up to 37 degrees (44 on msaa4): both cues agree on the wrong value. 71 to 88 % of them are in-place rotation frames (under 1 cm of translation) in open areas with distant, sparsely seen walls and little segment support; **all of them have an estimated `psi` in the diagonal band of 31 degrees or more**, which is 27 % of all valid frames, so they are not randomly placed. I could not find a feature that separates them (their support lengths overlap the good frames'). Stage 4 must therefore treat a valid heading as a measurement to be gated against the odometry, not as ground truth: 15 frames in 32,000 can cost a 90-degree branch error if trusted blindly.

## Focal length, level camera, axis offset

- **Focal length from vanishing-point pairs** (frames with both families supported by at least 25 px, both horizon directions between 15 and 75 degrees: 8,149 official test frames): median `f` = **90.76** vs `fx` = 92: **-1.3 %** (msaa4: 91.29, -0.8 %). 5th to 95th percentile of the error -8.3 % to +7.1 %; 80 % of frames within 5 %. Criterion (median within 5 %): pass.
- **Axis offset** (task 5): the signed median error of the fused yaw is **-0.08 degrees** on test (-0.09 on dev; msaa4 -0.07): the maze axes are along the world x and y axes, offset zero within 0.1 degrees, as stage 0's convention implied.
- **Level camera** (task 2). Near-vertical LSD segments on the grey frames (dev seeds, every 20th frame), the deviation of their lean from vertical:

| minimum segment length | segments | median | within 1 degree | robust sigma |
|---|---|---|---|---|
| 25 px | 7,167 | 0.00 deg | 58.5 % | 0.97 deg |
| 40 px | 2,844 | 0.00 | 72.8 % | 0.41 |
| 60 px | 1,118 | 0.00 | 81.7 % | 0.23 |
| 80 px | 503 | 0.00 | 88.7 % | 0.13 |
| 100 px | 256 | 0.00 | 91.0 % | 0.09 |

The plan's wording is "near-vertical segments should have angle 90 +/- 1 degree". The median is exactly vertical and the spread **shrinks with segment length** (from 0.97 to 0.09 degrees), which is what LSD localisation noise and texture edges look like, not a tilt: a plane fit of the deviation against image position gives no pitch signature (-0.03 and +0.11 degrees per unit of normalised image coordinate). I judge the level assumption confirmed (roll under 0.1 degrees, no pitch); the literal "all within 1 degree" does not hold for short segments, and I did not stop for it. `stage3_level_camera.png`.

## Wall-top ranging (revision 5)

Wall height is 0.2999 m and `h` 0.2099 m, so `Z = fy (H - h) / (cy - v_top)`. The top edge is the first run of non-sky pixels from the top (sky = every channel at least 232, to survive JPEG ringing); its sub-pixel row is solved against the white sky and the wall pixel two rows below. In `scan.py` it ranges the columns whose floor edge is below the image (status INVALID: a wall nearer than about 0.16 m), as status `TOP_RETURN`, and `Scan.z_top` keeps the top-edge depth wherever it exists, for cross-checks.

**The INVALID columns (test seeds, official frames):** 14.9 % of all columns become top returns; 0.30 % stay invalid. Their true depth is 0.085 to 0.160 m (5th to 95th percentile), and the **top-edge error is 0.45 mm median, 1.3 mm at the 95th percentile, 1.6 mm at the 99th** (msaa4: 0.18 / 0.62 / 1.0 mm).

**Cross-check against the floor edge** where both exist (median / 95th percentile of the absolute depth error, metres, official test, by true depth):

| depth | floor edge | wall top |
|---|---|---|
| 0.15 to 0.3 m | 0.0007 / 0.0062 | 0.0014 / 0.0039 |
| 0.3 to 0.5 m | 0.0020 / 0.0172 | 0.0044 / 0.0112 |
| 0.5 to 1.0 m | 0.0061 / 0.0489 | 0.0134 / 0.0425 |
| 1.0 to 1.5 m | 0.0196 / 0.1727 | 0.0378 / 0.1012 |
| 1.5 to 2.5 m | 0.0501 / 0.5013 | 0.1045 / 0.2929 |

From 0.3 m up the wall top has the larger *median* error but a much shorter tail: its 95th percentile is better than the floor edge's, 0.10 m against 0.17 m at 1 to 1.5 m. (The top edge is finer per pixel by a factor 2.3 at equal depth, so the larger median is probably the harder placement of the edge against the white sky on JPEG; I did not establish it.) It is a useful second opinion for far returns where the floor edge's tail is the weakness. A caveat: 1,599 columns with true depth under 0.15 m where both edges exist show a 0.37 m error for *both* edges (corner geometry; I did not investigate).

**Stage 2's criteria re-checked** (the plan requires they still hold with wall-top ranging on; the segmentation also changed slightly, see below). Test seeds, all returns within `R_MAX` = 1.5 m, every third frame:

| | floor edge only | floor edge + wall top | threshold |
|---|---|---|---|
| returns | 1,397,938 | 1,657,709 | |
| endpoint to nearest segment, median | 0.0028 m | **0.0026 m** | < 0.03 |
| endpoint to nearest segment, 95th percentile | 0.0244 m | **0.0214 m** | < 0.10 |
| largest median signed range error in any 0.25 m bin | 0.19 % | 0.19 % | < 1 % |
| coverage (columns with a return) | 0.801 | **0.950** (5th percentile of frames 0.775) | >= 0.60 |

(msaa4 frames: 0.0023 / 0.0168 m, bias 0.12 %, coverage 0.801 floor only; 0.0022 / 0.0142 m, 0.12 %, 0.951 with the top.) **All stage 2 criteria still hold, with margin, and the top returns lift coverage from 0.80 to 0.95.**

**A change to stage 2's code** that this stage made and re-verified: the boundary row now adds the wall coverage of both pixels around the edge (the first non-floor pixel and the first floor pixel, which the loose palette can accept even when it is a blend) instead of the first only. It changes nothing measurable on the official frames (their edges are not anti-aliased) and removes a quantisation on anti-aliased ones. The stage 2 mask review is unaffected (the floor mask does not change). Stage 2's cached results and oracle maps were not regenerated; the re-check above uses the new code.

## Acceptance

| criterion | result | pass |
|---|---|---|
| Valid yaw on at least 70 % of frames | 96.0 % (official), 97.8 % (msaa4) | pass |
| Accuracy on valid frames: median < 1.5 degrees, p95 < 4 degrees | 0.21 / 0.75 (official), 0.16 / 0.56 (msaa4) | pass |
| Focal length from vanishing-point pairs within 5 % of fx (median) | -1.3 % (official), -0.8 % (msaa4) | pass |
| Among valid frames at least 95 % under 3 degrees | 99.95 % (official), 99.96 % (msaa4) | pass |
| Task 2, level camera ("90 +/- 1 degree, else stop") | median 0.00; spread 0.9 to 0.1 degrees by length, no tilt trend | pass, qualified (above) |
| Task 5, axis offset reported | -0.08 degrees | reported |
| Wall-top ranging added, error vs range reported, stage 2 criteria still hold | done; criteria hold | pass |

## Figures

`stage3_yaw_error_hist.png` (error histograms: each cue alone, fused valid frames, official and msaa4), `stage3_error_vs_confidence.png` (vanishing-point error and fraction-with-a-good-cue against the disagreement of the cues; fused error against confidence), `stage3_focal_length.png`, `stage3_level_camera.png`, `stage3_walltop_error.png`.

## Open issues and notes for later stages

- **The anti-aliasing mismatch** (top of this report). Anything tuned on the official frames is conservative for the server's frames. If the project wants to tune on server-like data it should add msaa-4 recordings (a data change, so I left it to you); until then the in-memory renders at true poses are the evaluation path.
- **Confidently wrong frames** (15 in 32,000, diagonal headings, mostly in-place rotation): stage 4 must gate headings against the odometry. Stage 3b's collision detector does not help here (these are not collisions).
- The vanishing-point cue alone is not what to trust: per frame the scan cue is the better one (median 0.17 vs 0.39 degrees) and needs no vanishing points, which matters for frames looking at a single wall. The vanishing-point cue is what supplies the focal-length cross-check and an independent second opinion.
- Conflict frames (4.0 %, official test) get no heading. For a map built offline from three drives this is fine; their neighbours are valid in almost every case, and stage 4 interpolates with the odometry.
- The scan cue uses returns up to 2.5 m (`RANGE_CAP`), beyond the 1.5 m mapping cut, weighted by depth noise: consistent with the revision 5 note for stages 4 and 5.
- The plan's facts table still lists wall-top ranging and the scan cue as future work; this stage delivered both.
