# CLAUDE.md

Orientation for an agent working in this repo: what it is, where things are, how to run and check them, the rules that have bitten us, and how to commit and push. It is a map, not a spec: numbers and caveats live in the reports, and the plan is a live document.

## What this is

A fork of `ai4ce/vis_nav_player` (NYU ROB-UY 3203 / ROB-GY 6203 Visual Navigation Challenge). A robot sits in a maze on the course server; code gets its camera frames plus four photos of the goal, chooses moves, and checks in when it thinks it has arrived. The README has install and run commands.

**Our work (not the starter kit):** from the three exploration drives the server hands out per maze, build a metric top-down occupancy map with wall-texture signatures, place the goal pose on it, show both in `source/my_agent.py`, and (stage 10) locate the robot on demand with a 360 degree "ping". Planning and control are out of scope. Status: **stages 0 to 10 are built**; see "Status" for what is still open.

## Repo map

| where | what |
|---|---|
| `source/mapping/` | all of our code (below) |
| `source/my_agent.py` | the agent: builds or loads the map in `__init__`, places the goal in `setup`, draws the map, legend and pings in the viewer panel, keyboard driving, **P = ping** |
| `source/baseline_agent.py`, `vlad.py`, `keyboard_agent.py`, `cli.py` | starter kit: **do not modify** |
| `tests/mapping/` | our tests; `tests/test_agents.py` etc. are the starter kit's |
| `reports/stage_*.md`, `reports/figures/` | one report per stage: what was built, results table, what failed, what was tuned on which seeds |
| `topdown_map_plan.md` | the plan, with a revision note at the top |
| `data/` (git-ignored) | recordings: `gt-<seed>`, `gt4-<seed>`, `tick1-<seed>`, `bias-<seed>`, `ping-<seed>`, `local-<seed>`, and the challenge download `<challenge id>/` |
| `cache/mapping/` (git-ignored) | caches and builds; `cache/mapping/build/<data dir name>/` holds a built map (`map.npz`) |

### `source/mapping/`: modules by role

**Pipeline** (images, action labels and the map only; never ground truth):
`segment`, `scan`, `vp`, `scan_heading`, `heading` (floor/wall masks, virtual laser scan, heading modulo 90 degrees) · `collision` · `vo`, `odometry`, `pipeline` (ICP and wall-feature motion, EKF; `pipeline.drive_inputs` then `odometry.run_drive`) · `calibration` (per-drive prior scales) · `loops`, `texture`, `slam`, `posegraph` (loop closures, texture adjacency, pose graph) · `grid` (occupancy, wall snapping, configuration space, wall-face signatures) · `goal` (goal pose from the four goal views) · `adapter` (per-tick data to 5-tick spacing) · `ping` (the 360 degree locate) · `build_map` (the end-to-end CLI: raw data to `map.npz`).

**Evaluation** (may read `gt.json`, `poses.json`, the true layout): `record_gt`, `camera_fit`, `stage0*` to `stage10.py`, `stage*_extract.py`, `final_eval.py`. Each `stageN.py` has a docstring with its commands.

## Commands

```
uv sync
uv run pytest                                              # whole suite (about 1 min)
uv run ruff check <files you touched>                      # line length 100
uv run python source/mapping/build_map.py --data data/<dir>    # raw data -> cache/mapping/build/<dir>/map.npz
uv run source/my_agent.py --local 7                        # the agent on the local simulator (dev seed 7)
uv run source/mapping/record_gt.py --seeds 7 --msaa 4      # record a dataset (see its docstring for options)
```
Use `uv run python -I ...` for anything that reads downloaded data (`data/<challenge id>/`). A build takes about 2.5 minutes for a local maze, 90 seconds for the challenge download. The headless pygame window works with `SDL_VIDEODRIVER=dummy` (useful to screenshot `my_agent.py`).

## The data

- Three drives per maze, 320x240 BGR JPEG frames, plus `target.jpg` (four goal views side by side, 240x1280). `data_info.json` per drive: one action per record (FORWARD, LEFT, RIGHT, IDLE).
- **Frame spacing differs:** local recordings are one frame per 5 ticks; **the challenge download is one frame per tick**. `adapter.ticks_per_frame` detects it from the data and `build_map.py --ticks auto` (default) subsamples to every 5th frame. Everything is tuned on the 5-tick spacing.
- **An action label is only the last tick of its 5-tick window** (about 62% of windows mix actions): a weak prior, not a motion model.
- Pose, map and goal position are never sent by the server. Local simulator only (`vis_nav_sim`): true poses and the true maze, via `record_gt.py`.
- Datasets: `gt-<seed>` (`msaa=1`) and `gt4-<seed>` (same seeds, `msaa=4` = **the server's rendering**; use this one to tune); `tick1-<seed>` (per-tick, dev seeds 7 and 8 only); `bias-<seed>` (recorded with a motion bias); `ping-<seed>` (stage 10 sweeps); `local-<seed>` (what `--local` records).
- **Seeds, frozen in `source/mapping/seeds.json`:** dev 7, 8, 9, 10; test 21 to 25; **holdout 31 to 35**.

## Seed discipline (the user's instruction; it overrides anything else)

- Tune and debug on **dev seeds only**.
- **Ask the user before any test seed (21 to 25) is scored**, and score a stage on test once, after freezing. Evaluation scripts refuse test seeds without `--allow-test`.
- **Holdout seeds (31 to 35)** are opened once, by `final_eval.py --final`, only after the user confirms. `record_gt.gt_dir` raises `HoldoutError` for them otherwise.
- A fresh local seed counts as a test. Never run on the course server (it spends an attempt) and never download another challenge's data without asking.

## Rules that matter (they have bitten us)

1. **Ground truth is evaluation-only.** Pipeline modules must never read `gt.json`, `poses.json`, `Simulator` internals or `camera_fit.Scene`. Simulator *constants* measured once (camera, height, palette, spawn) are allowed as configuration. Every pipeline module has a leakage test (static scan, plus identical output on a copy without the truth files). Never identify a challenge maze by matching local seeds.
2. **Do not modify the baseline files** (table above).
3. **Local recordings are a pessimistic proxy** (not anti-aliased; the server's frames are). Report on `gt4` first, `gt` too.
4. **Report failures; never loosen a threshold silently.** Each stage writes `reports/stage_N.md` and says what failed, what was tuned on which seeds, what was disclosed. Do not rerun or rewrite an accepted stage; add an addendum.
5. Units: metres, radians in code, degrees in reports; poses `(x, y, theta)`, theta CCW from +x, wrapped to (-pi, pi]. Camera: fx = fy = 92, cx = 160, cy = 120, level, on the rotation axis, **robot camera height 0.2099 m; the goal views are rendered from 0.160 m**. Maze cell (r, c) is centred at (0.2c, 0.2r) m.
6. **Caches are keyed by a code hash.** Editing `segment.py`, `scan.py`, `vp.py`, `scan_heading.py` or `camera.json` (even formatting) invalidates the stage 3 cue caches (about 10 minutes). Check `stage3_extract.code_hash()` before and after (it is `cdbbc5b8` now). New constants go in their own file.
7. **`goal.py` must stay byte-identical for goal placement** unless a stage says otherwise (stage 10 changed it and proved identity on the dev goals). Check with `goal.place` on dev goals before and after.

## Established facts (measured, not assumed)

- Walls are thin vertical planes through wall-cell centres, height 0.30 m; a one-cell corridor is 0.4 m between planes. Far plane 5.0 m (white, like the sky). Robot collision radius about **0.12 m** to the wall centre lines.
- Wall collisions (0.4 to 0.8% of FORWARD/IDLE pairs) rotate 30 to 55 degrees with almost no translation; a wrong 90 degree heading branch is permanent, so they have their own handling.
- Every drive starts at the spawn (0.2, 2.6, 0). The simulator draws a **separate motion bias per route**; the pipeline calibrates the label prior per drive.
- The compass (heading modulo 90 degrees) is sometimes **valid and wrong by about 45 degrees**; anything that chains compass readings must refuse readings that do not fit the turn rate (`ping.Sweep`).
- The challenge maze `ab7efaeb1abf` is 21 x 21 cells (local mazes are 31 x 31), so counts such as snapped wall cells scale down with area.

## Status (read the report before relying on or "fixing" anything)

Stages 0, 1, 2, 3, 3b, 4, 4b, 5, 5b, 6, 7, 8 are accepted. Stages **9** and **10** are built and pass on dev seeds; each report lists what waits for the user:
- **Stage 9** (`reports/stage_9.md`): the final holdout evaluation (`final_eval.py --final build 31` ... then `--final score 31,32,33,34,35`) is **not run**; it needs the user's confirmation. The `--local 7` drive to the goal is the user's. The cause of an earlier "missing red arrow" was not reproduced.
- **Stage 10** (`reports/stage_10.md`): test-seed sweeps are **not run** (ask first). The user's own check of the green circle in `--local 7` is open.
- Known failing test: `tests/test_agents.py::test_the_template_agent_is_a_complete_skeleton` (expects the starter skeleton; the agent now drives). It failed before stages 9 and 10.
- Some older files carry ruff E501 long-line warnings; lint the files you touch, do not mass-reformat.

## How to work here

- **Reload `topdown_map_plan.md` from disk each session and read its newest revision note first.** Never overwrite it from memory. Sections for stages that have not been run are intent, not fact; when a report and the plan disagree, the report describes the code that exists.
- One stage at a time. New pipeline code goes in `source/mapping/` with `pytest` tests for its pure functions in `tests/mapping/`. Run `uv run pytest` and ruff on touched files before finishing.
- Debug by isolating the estimator against the true motion first, then the fusion. Most bugs here were a measurement trusted without a cross-check; re-check a result before reporting it.
- Do not commit unless asked.

## Committing and pushing

- **Remote:** the project lives at `https://github.com/ram10000-nyu/PumaPerception` (remote name `puma`; if it is missing: `git remote add puma https://github.com/ram10000-nyu/PumaPerception.git`). **`origin` is the course's upstream `ai4ce/vis_nav_player`: never push to it.** Push with `git push puma master`.
- Only commit and push when the user asks. Stage explicitly: `git add -A -- . ':!*.DS_Store'`, then check `git status` and `git diff --cached --stat`.
- Never commit `.env` (credentials), `data/`, `cache/`, caches, `.DS_Store`, or anything large; check for secrets in anything new.
- Commit messages: a short summary line, then what changed by stage; end with the attribution trailer the session gives (`Co-Authored-By: ...`).
- If a push is rejected, do not force it. Fetch, look at what differs, and ask.
