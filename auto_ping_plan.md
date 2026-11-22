# Auto-ping plan: live localization (Rolando, branch `auto-ping`)

Decisions and stages for **Rolando's side only**: the localization design and the stages that build it. Shared material lives elsewhere and is not repeated here:
- `topdown_map_plan_final.md`: what exists (stages 0 to 10), scaling to 41/51, the interface (Part D);
- `integration_plan.md`: the split with the planner, call order, milestones, git workflow.

Rules: those of `topdown_map_plan_final.md` ("How to work through this plan"). Any new decision for this side goes **in this file**, not in the shared plan.

---

## 1. Design: live localization without the 360 degree ping (design, 2026-11-21)

### Two different problems
- **Global localization** (no prior: "where am I at all?"). The ping solves it: 92.5 % right, honest confidence, but costs about 105 ticks.
- **Tracking** (prior from the last fix plus the commanded actions: "I was at X, moved about delta; where am I now?"). This is what periodic updates need, and it is much easier.

*Analogy:* finding your seat in a stadium with no ticket (global) vs walking 10 rows down from your seat and glancing at one row number (tracking).

### Single frame vs several frames
- **Single frame, no prior** (stage 10 ablation, dev): 52.5 % correct, 64/120 pass the confidence bar, 1 confident failure. Not usable for global localization.
- **Single frame with a prior:** the search window shrinks from the whole maze to about +/-0.15 m and +/-5 deg around the prediction; distant look-alikes cannot win. **One frame is enough for tracking**, except where geometry is degenerate.
- **Degeneracy:** in a straight corridor, sliding along it does not change the scan (1-D picture: a long featureless hallway fixes lateral position and heading, not the position along it). The registration Hessian $\mathcal{H}$ is then nearly singular in that direction ($\lambda_{\min} \approx 0$). Using $\mathcal{H}$ as the measurement information, the update corrects only what the scan constrains; along-corridor uncertainty grows until a corner, junction or texture boundary pins it.
- **Why not "10 consecutive frames + RANSAC":**
  1. consecutive per-tick frames are 3 cm / 3.4 deg apart, nearly the same view, so their errors are correlated: $\operatorname{Var}(\bar{x}) = \sigma^2[\rho + (1-\rho)/n]$; with $\rho = 0.8$, $n = 10$: $0.82\,\sigma^2$, only about 9 % less standard deviation;
  2. RANSAC assumes random outliers; aliasing is systematic (if frame 1 matches the wrong corridor, so do frames 2 to 10), so RANSAC would return the wrong pose with full consensus.
- **The useful multi-frame version: a local submap.** Stitch keyframes spaced by motion (last 0.5-1 m or 90 deg of turning) with the tracker's own relative poses (drift about 1 cm over 1 m), merge their scans, register the merged scan to the map. This widens the field of view like a partial ping, for free during normal driving. Use it when the single-frame fit is degenerate. RANSAC / Cauchy stays inside each fit, to reject bad scan points.

### Architecture (built in stage 12)
1. **Predict every tick** from the commanded action (motion constants, per-tick noise, an online speed/turn scale) -> pose and covariance $P$. Collision: a FORWARD into a wall does not move the robot; clip the prediction against the C-space and inflate $P$.
2. **Compass every frame**: heading modulo 90 deg, branch nearest the prediction, gated. Cheap and drift-free.
3. **Scan-to-map update at keyframes** (moved >= 0.10 m or turned >= 15 deg since the last update, or every 10 ticks): virtual scan -> local search within the 3-sigma window -> Gauss-Newton refine on the likelihood field -> pose $z$ and Hessian $\mathcal{H}$.
4. **Gate:** accept if $\nu^\top S^{-1} \nu < 11.34$ ($\chi^2_{3,0.99}$) and the fit score clears a dev-chosen threshold; else skip and keep predicting.
5. **Kalman update** with measurement information $\mathcal{H}$: $K = P H^\top (H P H^\top + \mathcal{H}^{-1})^{-1}$ (in information form, so a singular $\mathcal{H}$ is fine).
6. **Submap fallback** when $\lambda_{\min}(\mathcal{H})$ is small for several keyframes.
7. **Lost detection:** position covariance too large, or K consecutive rejections, or fit score too low -> status LOST -> recommend a ping (`ping.py`) to the planner.
8. Alternative considered: Monte Carlo localization (particle filter). More expensive; the ping already covers the multimodal case. Revisit only if the tracker fails on dev.

**Compute budget:** 60,000 steps in 3,600 s = 60 ms per step on average. Scan extraction costs about 7 ms per frame (stage 10); a windowed fit is a few ms. The texture check is the expensive part: keyframes only, and only when needed.

---

## 2. Stages

### Stage 11: large-maze readiness (prerequisite)

**Goal:** know that the mapping pipeline works at 41 x 41 and 51 x 51, how long it takes and how much memory it needs, and have large-maze recordings for stage 12.

**Tasks**
1. **Recording at size.** Add `--rows/--cols` to `record_gt.py` (default 31: nothing existing changes). Output directories `gt4s41-<seed>`, `gt4s51-<seed>` (`msaa=4`, server-like). Check that `record_gt`'s exploration routes cover the maze (`summary.trajectories` cells) and that frames per area match the challenge ratio (about 450 raw per-tick frames per m^2, i.e. ~90 kept).
2. **New seeds** in `seeds.json`, frozen before any is opened:
   - `dev_large`: 101, 102 (41 x 41) and 111, 112 (51 x 51);
   - `test_large`: 121, 122 (41 x 41) and 131, 132 (51 x 51), refused without `--allow-test`.
3. **Spawn and anchor check:** where is the spawn in a 41/51 maze? The pipeline anchors at `SPAWN = (0.2, 2.6, 0)` (`slam.py`, `spawn.json`). The map frame is arbitrary, but the anchor must equal the **live run's start pose** for the tracker's initial state; record whether the live session starts at the exploration spawn (local simulator).
4. **Build on dev large:** `build_map.py` on 101, 102, 111, 112. Record per-step time (`timing.json`) and **peak RAM** (`/usr/bin/time -l` on the Mac). If RAM exceeds 8 GB, chunk the VLAD similarity / feature storage.
5. **Mapping metrics on dev large:** stage 6 joint ATE, closures per kind, false closures active; stage 7 IoU and C-space hold; stage 8 goal error and confidence; ping on 10 random poses per dev seed. Make the evaluation scripts size-agnostic (no 31 x 31 or spawn assumptions).
6. **Fix only if dev fails**, mechanism first: Mahalanobis closure gate (Part B risk 1), TOP_K scaled with N, memory chunking. Report before/after on dev; never tune on test.
7. **Test large**, once, after asking the user.

**Acceptance criteria**
- [ ] Build at 51 x 51 < 12 min on the Mac, peak RAM < 8 GB.
- [ ] Joint ATE <= 0.08 m (rigid) on every large seed; no false closure active.
- [ ] Stage 7 criteria (IoU, C-space holds 99.5 % of true poses) hold.
- [ ] Goal within 0.10 m and 10 deg, confidence >= 0.10, on every large seed.
- [ ] Ping: >= 90 % within 0.10 m and 10 deg on dev large; no confident failure.
- [ ] Test large scored once with the user's agreement.

### Stage 12: live localization tracker (the main stage)

**Goal:** `LocState` every step (Part D) with the pose within 0.10 m and 10 deg almost always, honest status when not, without pings in normal driving.

**Tasks**
1. **Recorded live runs (evaluation data).** `record_live.py` (evaluation): drive the local simulator **per tick** (`capture_every 1`, `msaa=4`, default noise and per-run bias) along routes in free space, saving every frame, the commanded action and the true pose:
   - routes: oracle routes to random goals plus random walks, including deliberate wall bumps (collisions) and long straight corridors;
   - 31 x 31 dev seeds 7 to 10 (maps already built) and dev large 101, 102, 111, 112;
   - about 10 runs per seed, 1,000 to 3,000 ticks each.
2. **`source/mapping/track.py` (pipeline):** `Tracker(map, camera, start)` with `observe(obs) -> LocState`, `commit(action)`, `start_ping()`, `ping_action()`; internals as Part C, "Architecture":
   - motion model per tick with an online scale estimate (state augmentation or a slow filter on accepted corrections);
   - collision handling (C-space clip, covariance inflation; stage 3b detector if useful);
   - compass update every frame (gated);
   - keyframe scan-to-map update: windowed search (3 sigma, capped +/-0.2 m, +/-6 deg) on `MapField`, `goal.refine`, Hessian from the Gauss-Newton normal equations ($J^\top W J / \sigma^2$, sigma 3 cm);
   - chi-square gate and fit-score gate; information-form update;
   - submap fallback (last 0.5-1 m of keyframes);
   - texture check (`texture_score`) only to break a tie between two local optima;
   - status logic (TRACKING / DEGRADED / LOST) and `wants_ping`; ping via `ping.py`, its result reinitializes the filter with the ping's covariance.
3. **Offline replay** `stage12.py run <seeds>`: run the tracker on the recorded runs (frames + committed actions only; truth only for scoring).
4. **Agent integration** (`my_agent.py`): the D.4 call order, with keyboard driving standing in for the planner (`planner.act` = the keys) until the partner's planner is merged. Panel: pose, 2-sigma ellipse, track, status in the HUD. P still forces a ping.
5. **Tests** (`tests/mapping/test_track.py`): prediction against the motion constants; information-form update with a singular Hessian (corridor) changes only the observed directions; chi-square gate; compass branch choice; LocState fields and units; leakage.
6. **Choose thresholds on dev only:** keyframe spacing, gate score, lost criteria, submap length. Report the sweeps.

**Metrics** (per run and pooled):
- % of ticks with error < 0.10 m and < 10 deg; median and p95 error; max error;
- **silent failures:** ticks with error > 0.20 m or > 20 deg while status is TRACKING (must be 0);
- time from a real loss to status LOST; recovery by ping;
- pings requested per 1,000 ticks;
- runtime per step (median, p99), Mac and VM;
- calibration: fraction of ticks where the true pose lies in the 95 % ellipse (target 90-99 %).

**Ablations** (dev): no compass; single frame only (no submap); every frame vs keyframes; no gate; predict-only (dead reckoning, the reference).

**Acceptance criteria**
- [ ] Dev (31 x 31 and large): >= 95 % of ticks within 0.10 m and 10 deg; p95 < 0.06 m.
- [ ] **0 silent failures** (error > 0.20 m or 20 deg with status TRACKING) on dev; on test after freezing.
- [ ] Every real loss flagged (DEGRADED or LOST) within 20 ticks.
- [ ] <= 1 ping request per 2,000 ticks of normal driving.
- [ ] Runtime: median <= 20 ms, p99 <= 60 ms per step on the Mac.
- [ ] Covariance calibration 90-99 %.
- [ ] Leakage (rule 10) passes for `track.py` and the agent's tracking path.
- [ ] `--local 7` and one large dev maze: the user drives (or the planner drives) and the panel's pose follows the robot; report three spot checks against the simulator pose.
- [ ] Test seeds (31 x 31 and large) scored once, after asking the user.

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
