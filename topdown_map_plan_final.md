# Final-exam plan: top-down map at scale + live localization

**Revision F2 (2026-11-22).** The **shared** plan for the final exam: what exists, scaling, and the interface between the two parts. Side-specific decisions live in their own files: localization in `auto_ping_plan.md` (Rolando), planning in the partner's own notes; how they fit together in `integration_plan.md`. `topdown_map_plan.md` (revision 10, stages 0 to 10) stays the record of how the map was built.

**Repo:** fork of `ai4ce/vis_nav_player` (NYU ROB-GY 6203 Visual Navigation Challenge); remote `puma` (never `origin`).
**Split of work (final project):**
- **Rolando (branch `auto-ping`, plan `auto_ping_plan.md`):** the map at the final's size, and **live localization** — keep the robot's pose on the map updated while it drives, without stopping for a 360 degree ping.
- **Partner (his own branch and notes; merges and tests):** the automated planning agent (path planning, control, check-in). It consumes the outputs defined in Part D.
- **How the two parts fit together:** `integration_plan.md` (split, interface summary, call order, milestones, git workflow).
- **Branches:** `master` (shared, merged by the partner), `auto-ping` (Rolando), and the partner's own branch. **Partner: create your branch from `master` before merging anything** (`git fetch puma && git checkout -b <your-branch> puma/master && git push -u puma <your-branch>`); merge `auto-ping` into it, test, then merge into `master`. Details: `integration_plan.md`, section 6.

**Final maze size:** not announced; likely **41 x 41 or 51 x 51** grid cells (the midterm was 31 x 31). Plan for 51 x 51.
**Final challenge id:** `70594c50463a` (`VIS_NAV_CHALLENGE` in `.env`, set 2026-11-21). One maze per challenge id: its exploration data and goal are fixed. **Do not start a session or download its data without the user's go-ahead.**

---

## How to work through this plan (read first)

The rules of `topdown_map_plan.md` ("How to work through this plan", items 1 to 10) still apply. The ones that matter most here:

1. **One stage at a time;** stop and report if a criterion fails after reasonable effort. Never loosen a threshold silently.
2. **Do not modify the baseline** (`baseline_agent.py`, `vlad.py`, `keyboard_agent.py`, `cli.py`). New code goes in `source/mapping/`.
3. **Ground truth is evaluation-only (rule 10).** Pipeline code (everything `build_map.py` and the agent import, including the new tracker) never reads `gt.json`, `poses.json`, simulator internals or `camera_fit`. Every new pipeline module gets the leakage test (static scan + identical output without the truth files).
4. **Seed discipline (the user's standing instruction):**
   - tune and debug on **dev** seeds only;
   - **ask the user before any test seed is scored**, and score each stage on test once, after freezing; every evaluation script refuses test seeds without `--allow-test`;
   - a fresh seed is a test: do not open new seeds for tuning beyond the new dev seeds listed in stage 11;
   - holdout seeds 31 to 35: only via `final_eval.py --final`, after the user confirms;
   - never run on the server or download challenge data without asking.
5. **Reports** `reports/stage_N.md` with a results table (metric, value, threshold, pass/fail) and figures in `reports/figures/`; tests in `tests/mapping/`; `uv run pytest` passes at every stage.
6. **Commits:** only when the user asks; push with `git push puma master`; never commit `.env`, `data/`, caches (except the three parameter files) or `.DS_Store`.

---

## Part A: what exists (stages 0 to 10, summary)

### Result so far
- **Midterm (31 x 31): 9 of 10 trials succeeded** with this solution (map + goal placement + display; driving by keyboard/partner).
- Challenge downloads seen: `ab7efaeb1abf` (21 x 21, 4 m), `da90a443e215` (midterm, 31 x 31, 6 m). Both are one frame per tick; the adapter handles it.

### The pipeline (raw exploration data -> `map.npz` -> goal)

| stage | what it does | key result (test seeds unless noted) | report |
|---|---|---|---|
| 0 | data audit, ground-truth harness, camera and motion constants | h = 0.2099 m, offset 0, fx = fy = 92 | `stage_0.md` |
| 1 | action labels as a weak prior (label = last tick of the window) | prior calibrated | `stage_1.md` |
| 2 | floor segmentation -> **virtual laser scan**, one beam per column, $Z = f h / (v - c_y)$; wall-top variant $Z = f (H - h)/(c_y - v_{top})$ | endpoint error median 2.8 mm, p95 23 mm | `stage_2.md` |
| 3 | **compass**: vanishing points + scan directions, heading modulo 90 degrees | 96 % valid, median 0.21 deg | `stage_3.md` |
| 3b | collision handling | detector recall 1.0 on > 45 deg events | `stage_3b.md` |
| 4, 4b | **odometry**: symmetric ICP + wall SIFT features, EKF, per-drive calibrated priors | ATE median 0.094 m per drive | `stage_4.md`, `stage_4b.md` |
| 5, 5b | **loop closures**: VLAD/RootSIFT candidates (baseline `_vlad`), RANSAC homography, relative pose from wall features; texture-boundary matching | precision 0.997, pose error 2.1 mm | `stage_5.md`, `stage_5b.md` |
| 6 | **pose graph** (SE(2)), odometry inflated 16x, heading priors, thinning, Cauchy IRLS on loop edges (weights before the first solve), chi-square 11.34 re-check | joint ATE 0.060-0.120 -> **0.013-0.027 m**; every false closure inactive | `stage_6.md` |
| 7 | **occupancy grid** (log-odds, 5 cm), 0.4 m wall lattice fit and scale correction, wall snapping, collision segments, configuration space (radius 0.115 m), wall-face texture signatures | IoU and C-space criteria pass | `stage_7.md` |
| 8 | **goal placement** from the four goal views (camera 0.160 m), likelihood field + Gauss-Newton + texture term, confidence margin | 5/5 test goals, 2-3 mm, <= 0.2 deg | `stage_8.md` |
| 9 | frame-rate **adapter** (per-tick -> every 5th tick), challenge run, agent display | challenge 21 x 21 built in 88.5 s; tick1 build bit-identical to 5-tick | `stage_9.md` |
| 10 | **ping** (P): 360 degree spin (~105 ticks), merged scan, `goal.place` machinery | 92.5 % of dev pings within 0.10 m / 10 deg; 0 confident failures | `stage_10.md` |

**Still open from revision 10:** stage 9 task 8 (final holdout run, needs the user's confirmation); stage 10 test sweeps (needs agreement); the user's own `--local 7` checks.

### Facts the next stages rely on

| fact | value |
|---|---|
| frames | 240 x 320 BGR; fx = fy = 92, cx = 160, cy = 120; horizontal FOV about 120 deg |
| robot camera height | 0.2099 m (goal views: 0.160 m); wall height 0.2999 m |
| motion per tick | FORWARD 0.03 m; LEFT/RIGHT 0.06 rad (3.44 deg); noise per tick plus a per-run speed/turn bias |
| maze | cells of 0.2 m; corridors 0.4 m between wall planes; side $L = 0.2(n-1)$ m |
| collision | robot centre stays about 0.12 m from the collision segments; C-space radius 0.115 m |
| goal success (course rubric, course site) | the run scores only if distance <= 0.2 m **and heading <= 20 deg** at check-in (the local `rl/env.py` uses 30 deg; the site's rubric governs; check the challenge page's own rubric). Points (10): distance <= 0.10 / 0.15 / 0.20 m -> 3 / 2 / 1; heading <= 10 / 15 / 20 deg -> 3 / 2 / 1; SPL >= 0.95 / 0.85 / 0.75 / 0.5 -> 4 / 3 / 2 / 1 |
| SPL | shortest route's steps (turns included) / steps used: **every tick spent localizing (a ping is ~105) lowers the score** |
| session | 60,000-step budget; 3,600 s clock; 120 s silence timeout; `__init__` runs before the session starts |
| scan range | trusted to 1.5 m; depth error grows as $z^2/(f h)$ per pixel (0.05 m at 1 m) |

### Built artefacts the tracker will reuse

| module | what to reuse |
|---|---|
| `segment.py`, `scan.py` | floor/wall masks and the virtual scan of one frame |
| `heading.py`, `scan_heading.py` | heading modulo 90 deg from one frame |
| `goal.py` | `MapField` (likelihood field of distance to walls, 1 cm), `_score`, `refine` (point-to-wall Gauss-Newton, Cauchy), `texture_score` |
| `ping.py` | `Sweep`, `locate`: global relocalization when lost |
| `grid.py` | configuration space, collision segments |
| `map.npz` keys | `logodds`, `resolution`, `origin`, `wall_segments`, `collision_segments`, `cspace`, `robot_radius`, `poses`, `sig_*`, `lattice_*` |

---

## Part B: what a bigger maze does (analysis, 2026-11-21)

### Measured build cost (4-core VM; the Mac is faster)

| maze | side | raw frames | kept (every 5th) | inputs | odometry | closures | graph | map | total |
|---|---|---|---|---|---|---|---|---|---|
| 21 x 21 (`ab7efaeb1abf`) | 4 m | 7,202 | 1,439 | 26 s | 17 s | 13 s | 3.4 s | 4 s | 89 s (incl. 25 s adapter) |
| 31 x 31 (`da90a443e215`) | 6 m | 16,061 | 3,211 | 60 s | 35 s | 29 s | 5.7 s | 10 s | 164 s (incl. 24 s adapter) |

- **Exploration data scales with area:** 2.23x frames for 2.25x area. Assume $N \propto L^2$.
- **Every step was linear in N** between the two (2.2 to 2.5x).
- **Superlinear parts, still small:** VLAD similarity is $O(N^2)$ dot products (8.9k frames: about 20 GFLOP, seconds); pose-graph iterations grow with loop length (expect < 1 min).

### Projection

| maze | side | area vs 31 | raw frames | kept | build |
|---|---|---|---|---|---|
| 41 x 41 | 8 m | 1.78x | ~28,500 | ~5,700 | **~5-7 min** |
| 51 x 51 | 10 m | 2.78x | ~44,600 | ~8,900 | **~8-11 min** |

Build time is **not on the session clock**: the map is built in `__init__` (before the session) and cached in `cache/mapping/build/<data dir>/`. Build it offline right after downloading. **Unknown: peak RAM** (per-frame features for about 9k frames); measure it (stage 11).

### Risks at 41/51, most serious first
1. **Drift between revisits grows.** Longer drives -> longer loops; the fixed 0.5 m / 10 deg odometry gate for closures will reject more true closures, leaving the texture-boundary closures. Heading stays bounded (compass); position drift grows with distance. Mitigation if needed: a Mahalanobis gate on the propagated covariance, $\nu^\top S^{-1} \nu < 11.34$, instead of a fixed 0.5 m.
2. **More aliases.** Candidate pairs grow as $N^2$, true revisits as $N$: about 2.8x more wrong candidates per true one at 51; TOP_K = 20 fixed, so true-match recall drops. The graph's defences held with all 17 known aliases forced in; the dangerous case is several mutually consistent false closures (the seed 22 mode).
3. **Global searches grow with area.** Goal placement 0.55-1 s -> about 1.5-3 s (once per session, fine); ping `locate` 0.54 s -> about 1.5 s; more rival places, so confidence margins shrink (current 0.31-0.38 vs threshold 0.10). The texture term matters more.
4. **Improves with size:** lattice and scale fit (more walls).
5. **Unchanged:** per-frame costs, adapter, camera model.

**Verdict:** usable; nothing is tied to 31 x 31 except constants to check (the spawn anchor `SPAWN = (0.2, 2.6, 0)` in `slam.py` / `spawn.json`, any evaluation script assuming 31 x 31). **Must be validated on local large mazes** (stage 11). The simulator supports it: `vis_nav_sim.Simulator(..., rows=41, cols=41)` (odd sizes).

---

## Part C: live localization (moved)

The localization design (tracking vs global localization, single frame vs submaps, the tracker architecture) is Rolando's side: see **`auto_ping_plan.md`**, section 1.

---

## Part D: interface with the planner (partner)

This is the contract. The tracker produces the inputs the planner consumes; the planner tells the tracker which action it sends. The short version both of us follow, with milestones and the git workflow, is `integration_plan.md`. Names are proposals; agree on them before stage 12 starts and change them only together.

### D.1 Coordinate conventions
- **Map frame:** the frame of `map.npz`, anchored at the exploration spawn. x, y in metres; $\theta$ in radians, counter-clockwise from +x, wrapped to $(-\pi, \pi]$.
- **Grid indexing** (for `cspace`, `logodds`): cell `[iy, ix]` has centre $x = \text{origin}_0 + (ix + 0.5)\,r$, $y = \text{origin}_1 + (iy + 0.5)\,r$, with $r$ = `resolution` (0.05 m).
- **Robot pose** = camera position (the camera is on the rotation axis) and heading.

### D.2 Static outputs (available after `setup()`; computed in `__init__` / `setup`)

| name | type | meaning |
|---|---|---|
| `map.cspace` | bool array (ny, nx) + `origin`, `resolution` | cells where the robot **centre** may be (walls inflated by 0.115 m, limited to the explored component). **Plan paths on this.** |
| `map.collision_segments` | (M, 4) float, metres | wall centre lines the robot must keep about 0.12 m from (for clearance costs) |
| `map.wall_segments` | (M, 4) float | rendered walls (for display, line of sight) |
| `map.logodds` | float (ny, nx) | occupancy evidence: > 0 wall, < 0 free, 0 unknown |
| `goal.pose` | (3,) float | goal pose in the map frame |
| `goal.confidence`, `goal.reliable` | float, bool | reliable = confidence >= 0.10; if False, treat the goal as uncertain (see `goal.top`) |
| `goal.top` | list of up to 3 (pose, score, ...) | runner-up goal places |
| `start` | `LocState` | initial state (spawn with small covariance; stage 12 verifies the live start pose) |

### D.3 Per-step output: `LocState` (every call to `act`)

```python
@dataclass(frozen=True)
class LocState:
    step: int               # obs.step this estimate refers to (after seeing obs.image)
    pose: np.ndarray        # (3,) x [m], y [m], theta [rad] in the map frame
    cov: np.ndarray         # (3, 3) covariance of pose (m^2, rad^2)
    status: str             # "TRACKING" | "DEGRADED" | "LOST" | "PINGING"
    source: str             # what last corrected it: "scan" | "submap" | "ping" | "init" | "predict"
    confidence: float       # fit quality of the last accepted update, 0..1
    weak_dir: np.ndarray | None  # unit vector (map frame) of the least-observed direction, if degenerate
    last_fix_step: int      # step of the last accepted correction
    wants_ping: bool        # True when the tracker cannot recover by itself
```

Meaning of `status`:
- **TRACKING:** position 2-sigma within about 0.10 m, heading within about 5 deg.
- **DEGRADED:** uncertainty growing (corridor, rejected updates); `weak_dir` gives the poorly observed direction.
- **LOST:** do not trust `pose`; `wants_ping` is True.
- **PINGING:** the tracker is spinning and supplies the actions (about 105 ticks).

How the planner reacts to each status is the partner's decision (`integration_plan.md`).

### D.4 Per-step call order inside `MyAgent.act(obs)`

```
state  = tracker.observe(obs)                 # 1. use the new frame: compass, maybe scan update
if state.status == "PINGING":
    action = tracker.ping_action()            #    tracker drives (LEFT) until the spin ends
else:
    if state.wants_ping and planner.allow_ping(state):
        tracker.start_ping(); action = tracker.ping_action()
    else:
        action = planner.act(state, map, goal)   # 2. planner chooses (Action, ticks)
tracker.commit(action)                        # 3. tracker predicts with the action actually sent
return action
```

- **Planner -> tracker:** only the action actually returned (`tracker.commit(action)`), every step, including CHECKIN or idle. The tracker never guesses the action.
- **Multi-tick actions** `(Action, n)` are allowed; the tracker predicts n ticks. Prefer n <= 5 while localization matters (fewer frames between corrections).
- **Check-in rule (suggested for the planner):** status TRACKING, distance to `goal.pose` + 2 sigma < 0.2 m (aim < 0.10 m for full points), heading error to view 0 + 2 sigma < 20 deg (aim < 10 deg).

### D.5 Guarantees from the tracker
- `observe` + `commit` return within **20 ms median, 60 ms p99** on the Mac (stage 12 measures it); a ping's `locate` (about 0.5-1.5 s) runs once at the end of the spin.
- No step blocks for more than 2 s (120 s silence timeout is far away).
- Uses only images, the actions committed, the map and constants (rule 10).
- The display (`panel`) shows the current pose, its 2-sigma ellipse, the recent track, the goal and any pings.

### D.6 What the tracker needs from the planner
- every action committed (D.4);
- permission for pings (the planner decides when spending about 105 ticks is acceptable: they cost SPL points);

---

## Part E: stages (moved)

- **Rolando's stages** (11: large-maze readiness; 12: live localization tracker): **`auto_ping_plan.md`**, section 2.
- **Integration with the planner** (milestones M1 to M7): **`integration_plan.md`**.

---

## Bibliography
- Thrun, Burgard & Fox, *Probabilistic Robotics*: ch. 6.4 (likelihood field), ch. 7 (EKF localization, gating), ch. 8 (Monte Carlo localization)
- Dellaert, Fox, Burgard & Thrun 1999, *Monte Carlo localization for mobile robots*, ICRA
- Censi 2007, *An accurate closed-form estimate of ICP's covariance*, ICRA
- Zhang, Kaess & Singh 2016, *On degeneracy of optimization-based state estimation problems*, ICRA
- Olson 2009, *Real-time correlative scan matching*, ICRA
- Hess, Kohler, Rapp & Andor 2016, *Real-time loop closure in 2D LIDAR SLAM*, ICRA (Cartographer, scan-to-submap)
- Bar-Shalom, Li & Kirubarajan 2001, *Estimation with Applications to Tracking and Navigation* (chi-square gating)
- Grisetti, Kümmerle, Stachniss & Burgard 2010, *A tutorial on graph-based SLAM*, IEEE ITS Magazine
