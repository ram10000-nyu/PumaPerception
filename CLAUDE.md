# CLAUDE.md

High-level context for working in this repo. It is a map, not a spec: details live in the reports and the code, and the plan is a live document (see "How to work here").

## What this is

A fork of `ai4ce/vis_nav_player` (NYU ROB-UY 3203 / ROB-GY 6203 Visual Navigation Challenge). A robot sits in a maze on the course server; code receives its camera frames plus four photos of the goal, chooses moves, and checks in when it thinks it has arrived. The README has install and run commands (`uv sync`, `uv run source/<agent>.py`, `--local <seed>` for the local simulator, credentials in `.env`).

**The project's goal (our work, not the starter kit):** from the three exploration drives the server hands out per maze, build a metric top-down occupancy map with wall-texture signatures and place the goal pose on it. Live localization, planning and control are out of scope; a follow-up plan consumes the map. Work lives in `source/mapping/`, with tests in `tests/mapping/` and one report per stage in `reports/`.

## The data

- Three drives per maze, one frame every 5 ticks (320x240 BGR JPEG), plus `target.jpg` (four goal views side by side, 240x1280). `data_info.json` per drive: one action per record (FORWARD, LEFT, RIGHT, IDLE).
- **An action label is only the last tick of its 5-tick window** (about 62% of windows mix actions), so a label does not determine the motion. It is a weak prior (an empirical mixture given the previous and current label), not a motion model.
- Pose, map and goal position are never sent by the server. Local simulator only (`vis_nav_sim`): true poses and the true maze, via `record_gt.py`.
- Datasets: `data/gt-<seed>/` (official local recording, `msaa=1`) and `data/gt4-<seed>/` (same seeds and trajectories, `msaa=4`, **the server's rendering**; use this one to tune). Both carry `poses.json` and `gt.json` (evaluation only). `data/<challenge id>/` is a real challenge download (no ground truth).
- **Seed split, frozen** (`source/mapping/seeds.json`): dev 7, 8, 9, 10; test 21 to 25. Tune on dev, report on test.

## Rules that matter (they have bitten us)

1. **Ground truth is evaluation-only.** Pipeline code (`segment`, `scan`, `vp`, `scan_heading`, `heading`, `collision`, `vo`, `odometry`, `pipeline`) must never read `gt.json`, `poses.json`, `Simulator` internals or `camera_fit.Scene`. Evaluation scripts (`stage*.py`, `record_gt`, `camera_fit`) may. Simulator *constants* measured once (camera, height, palette, spawn) are allowed as configuration. Each stage has a leakage test (copy a drive without the ground-truth files; outputs must be identical). Never identify a challenge maze by matching local seeds.
2. **Do not modify the baseline**: `source/baseline_agent.py`, `vlad.py`, `keyboard_agent.py`, `cli.py` stay untouched.
3. **Local recordings are a pessimistic proxy.** They are not anti-aliased; the server's frames are. Sub-pixel precision differs. Report on both `gt4` and `gt`.
4. **Report failures; never loosen a threshold silently.** Each stage writes `reports/stage_N.md` with a results table, and says what failed, what was tuned on which seeds, and what was disclosed. Do not tune on test seeds; if you do look at them, say so.
5. Units: metres, radians in code, degrees in reports; poses `(x, y, theta)`, theta CCW from +x, wrapped to (-pi, pi]. Camera: fx = fy = 92, cx = 160, cy = 120, level, on the rotation axis. Maze cell (r, c) is centred at (0.2c, 0.2r) m.
6. **Caches are keyed by a code hash** (`cache/mapping/`, git-ignored). Editing `segment.py`, `scan.py`, `vp.py`, `scan_heading.py` or `camera.json` (even formatting, even an extra JSON key) invalidates the stage 3 cue caches (about 10 minutes to rebuild). Put new constants in their own file (see `spawn.json`), and check `stage3_extract.code_hash()` before and after.

## Established facts about the simulator (measured, not assumed)

- Walls are **thin vertical planes through wall-cell centres**, not 0.2 m blocks: a segment joins the centres of 4-adjacent wall cells; it extends 0.1 m past a cell only at a free end; corners and junctions end at the centre; half-thickness 2.2 mm. A one-cell corridor is 0.4 m between planes. Wall height 0.30 m. Reference geometry for evaluation: `camera_fit.wall_segments`.
- Camera height 0.2099 m. Far plane 5.0 m: beyond it the render is white, the same as the sky.
- The floor is two textures (blue and grey tile) plus fog; on the JPEG frames the pipeline sees, floor pixels lie on the colour line from (218, 181, 157) to (232, 232, 232). JPEG ringing next to dark walls needs a loose palette for the coarse boundary (see `segment.py`).
- Depth resolution of the floor edge is `z^2 / (fy h)` per pixel (0.05 m at 1 m, 0.21 m at 2 m). Mapping range cut `scan.R_MAX` = 1.5 m; odometry uses returns to 2.5 m weighted by that noise.
- **Wall collisions**: 0.4 to 0.8% of FORWARD/IDLE pairs rotate 30 to 55 degrees with almost no translation. A wrong 90-degree heading branch after one is permanent, so collisions get their own handling.
- Every drive starts at the spawn (0.2, 2.6, 0). Robot collision radius about 0.08 m.

## Pipeline as built (status: stages 0 to 4 done; read the reports for numbers and caveats)

| step | modules | what it gives |
|---|---|---|
| data and ground truth | `record_gt.py`, `camera_fit.py` | `gt` / `gt4` datasets with true poses; camera and wall geometry |
| action prior | `se2.py`, `motion.py` | per-(previous, current) label mixtures of the true relative motion, plus a collision component (`logpdf_ext`, `contains_ext`) |
| segmentation and scan | `segment.py`, `scan.py` | floor/sky/wall masks, sub-pixel floor-wall boundary, virtual laser scan (floor edge plus wall-top ranging for near walls) |
| heading | `vp.py`, `scan_heading.py`, `heading.py` | yaw mod 90 degrees per frame: vanishing-point cue and scan-direction cue, fused, with a validity flag (the cues must agree within 3 degrees) |
| collision detector | `collision.py` | flags frame pairs that are likely wall collisions (heading change plus feature-match rotation; images and labels only) |
| odometry | `vo.py`, `odometry.py`, `pipeline.py` | ICP and wall-feature motion with covariances, fused with the label prior in an EKF with gated heading updates; images in, trajectory out (`pipeline.drive_inputs` then `odometry.run_drive`) |

Entry points for a new stage: load inputs with `pipeline.drive_inputs(drive_dir)`, the prior with `motion.MotionPrior.load()`, run `odometry.run_drive(inputs, prior, "C")`. Stage scripts (`stage3_extract.py`, `stage4_extract.py`) cache expensive per-frame cues for the evaluation.

Known weak spots (check the latest report before relying on or "fixing" them): a systematic scale bias of about -1% on straight forward motion; one or two heading frames per seed that the filter cannot reject (wrong by 3 to 6 degrees); a vision-only variant is clearly worse than vision plus labels (the pipeline uses the action labels); the floor-homography estimator and parallel collision hypotheses were not built.

## What comes next (headline only)

The remaining stages, in order, are loop closures, a pose graph, the occupancy grid with wall-texture signatures, goal placement, and a run on real challenge data with the map shown in `source/my_agent.py`. `source/my_agent.py` is still the starter skeleton. Their detailed tasks and thresholds are in the plan.

## How to work here

- **The plan, `topdown_map_plan.md`, is a live document and changes between sessions.** Reload it from disk each time and read its newest revision note first (the top section says what changed and what runs next). Never overwrite it from memory. Treat sections for stages that have not been run as intent, not fact: earlier revisions corrected several of its assumptions (wall geometry, camera offset, floor palette), and the reports hold what was actually measured. When a report and the plan disagree, the report is about the code that exists.
- One stage at a time; a stage's acceptance criteria are in the plan, its results and caveats in `reports/stage_N.md`. Do not rerun or rewrite an accepted stage; add an addendum.
- New code goes in `source/mapping/`, with `pytest` tests for its pure functions in `tests/mapping/`. Run `uv run pytest` and `uv run ruff check source/mapping tests/mapping` (line length 100) before finishing.
- Debug by isolating the estimator against the true motion first (start it from the truth and see whether it stays), then the fusion. Most bugs here were a measurement trusted without a cross-check.
- Commands: `uv sync`; `uv run source/mapping/record_gt.py --msaa 4 --verify` (record a dataset); `uv run source/mapping/stage3_extract.py` and `stage4_extract.py` (cue caches); `uv run source/mapping/stage4.py` (evaluation). Use `uv run python -I` when running anything that reads downloaded data.
- Do not commit unless asked. `data/` and `cache/` are git-ignored.
