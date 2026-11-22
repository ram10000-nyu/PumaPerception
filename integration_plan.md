# Integration plan: localization (Rolando) + planning (partner)

Brief contract between the two halves of the final-exam agent. Full interface spec: `topdown_map_plan_final.md`, Part D (shared plan). Localization side's own decisions: `auto_ping_plan.md`. This file is the summary both of us follow.

## 1. Split

| part | owner | code | produces |
|---|---|---|---|
| Map from exploration data, goal placement | Rolando | `source/mapping/` (`build_map.py`, `grid.py`, `goal.py`) | map products + goal pose (static, before the run) |
| Live localization (tracker) and ping | Rolando | `source/mapping/track.py`, `ping.py` | `LocState` every step |
| Planning and control, check-in decision | partner | his choice (e.g. `source/planning/`) | the action every step |
| Glue: `MyAgent` in `source/my_agent.py` | partner merges and tests | `source/my_agent.py` | calls both in the order below |
| Shared types | both (change only together) | `source/interface.py` | `MapBundle`, `LocState` |

Baseline files (`baseline_agent.py`, `vlad.py`, `keyboard_agent.py`, `cli.py`) stay untouched.

## 2. Conventions (both sides)
- **Map frame:** the frame of `map.npz`, anchored at the exploration spawn. x, y in m; $\theta$ in rad, CCW from +x, wrapped to $(-\pi, \pi]$.
- **Grid cell** `[iy, ix]` centre: $x = o_x + (ix + 0.5)\,r$, $y = o_y + (iy + 0.5)\,r$, $r = 0.05$ m.
- **Pose** = camera position (on the rotation axis) + heading.
- **Motion per tick:** FORWARD 0.03 m, LEFT/RIGHT 0.06 rad (3.44 deg), with noise and a per-run bias.
- **Scoring:** counts only if distance <= 0.2 m and heading <= 20 deg to view 0 at check-in; full points at <= 0.10 m, <= 10 deg; SPL = shortest steps / steps used (every tick counts, pings included).

## 3. Interface

### Static: `MapBundle` (ready after `MyAgent.setup`)
| field | meaning |
|---|---|
| `cspace`, `origin`, `resolution` | bool grid: where the robot **centre** may be. Plan on this. |
| `collision_segments` | (M, 4) wall centre lines; keep about 0.12 m clearance |
| `wall_segments`, `logodds` | rendered walls; occupancy evidence (display, line of sight) |
| `goal_pose`, `goal_confidence`, `goal_reliable`, `goal_top` | goal (x, y, theta) in the map frame, its confidence, runner-ups |
| `start` | initial `LocState` |

### Per step: `LocState` (tracker -> planner)
`step, pose (3,), cov (3x3), status, source, confidence, weak_dir, last_fix_step, wants_ping`
- `TRACKING`: pose within about 0.10 m / 5 deg (2 sigma).
- `DEGRADED`: uncertainty growing (corridor, rejected updates); `weak_dir` says which way.
- `LOST`: pose untrusted; `wants_ping = True`.
- `PINGING`: the tracker is spinning (about 105 ticks) and supplies the actions.

What the planner does with each status is the planner's decision.

### Per step: action (planner -> tracker)
The action **actually returned** to the simulator, `(Action, n)`, every step, including CHECKIN. That is the only thing the tracker needs from the planner. (Tracker's wish, not a rule: shorter bursts, n <= 5, give it more frames between corrections.)

### Call order in `MyAgent.act(obs)` (the tracker's part is fixed; the middle is the planner's)
```
state = tracker.observe(obs)                      # new frame -> LocState
if state.status == "PINGING":
    action = tracker.ping_action()
elif state.wants_ping and planner.allow_ping(state):
    tracker.start_ping(); action = tracker.ping_action()
else:
    action = planner.act(state)                   # partner's decision (any logic he likes)
tracker.commit(action)                            # tracker predicts with what was sent
return action
```

### Planner side (partner's design)
The tracker only needs two hooks: something that returns the action each step, and a yes/no when the tracker asks for a ping (pings cost about 105 ticks of SPL). Everything else (algorithm, replanning, how to use `cov` and `status`, when to check in, how to treat map errors) is the partner's call. A minimal shape that fits the call order:
```
class Planner:
    def __init__(self, bundle: MapBundle): ...
    def act(self, state: LocState) -> tuple[Action, int]: ...
    def allow_ping(self, state: LocState) -> bool: ...
```

**Suggestion: A\* on `cspace`.** The configuration-space grid (5 cm cells) is already a graph: nodes = free cells, edges to the 8 neighbours, cost = step length, plus optionally a clearance penalty from the distance to `collision_segments`. Heuristic = Euclidean distance to the goal cell (admissible, so the path is shortest). At 51 x 51 the grid is about 200 x 200 = 40k cells: a full A\* takes tens of ms in pure Python (less on a coarser 0.1 m grid or with `heapq` and NumPy neighbours), so it can be rerun every few steps from the tracker's pose. Since every tick (turns included) counts for SPL, a cost that also charges heading changes (state = cell + 8 headings) matches the score better.
- Hart, Nilsson & Raphael 1968, *A formal basis for the heuristic determination of minimum cost paths*, IEEE Trans. SSC
- LaValle, *Planning Algorithms* (2006), ch. 2 and 8

### Budgets
- tracker `observe` + `commit`: median <= 20 ms, p99 <= 60 ms. The whole step budget averages 60 ms (60,000 steps in 1 h); the planner has the rest.
- nothing blocks > 2 s (120 s silence timeout; 1 h session).
- the map is built in `MyAgent.__init__` (before the session); `setup` runs with the clock already going.

## 4. Working in parallel
- **Partner, before the tracker exists:** develop against an **oracle tracker** on the local simulator (true pose + noise, status always `TRACKING`; evaluation-only code, never in the submitted path) and the dev maps already built (`cache/mapping/build/gt4-7` ... `gt4-10`).
- **Rolando, before the planner exists:** keyboard driving stands in for `Planner.act` (already in `my_agent.py`).
- **Ground truth** (`gt.json`, `poses.json`, simulator pose) only in evaluation code, on both sides (plan rule 10).
- **Seeds:** dev only for tuning; ask before scoring test seeds; holdout 31 to 35 untouched; no server session or download of `70594c50463a` without both of us agreeing.

## 5. Milestones
| # | what | who | done when |
|---|---|---|---|
| M1 | `source/interface.py` with `MapBundle`, `LocState` | both | merged; names frozen |
| M2 | large-maze readiness (stage 11) | Rolando | maps of 41/51 dev mazes pass |
| M3 | planner on the oracle tracker | partner | his own criteria |
| M4 | tracker (stage 12) with keyboard driving | Rolando | stage 12 dev criteria pass |
| M5 | real tracker + real planner, local dev mazes (31, 41, 51) | partner merges and tests | >= 9 of 10 dev runs score; timing within budget |
| M6 | local test mazes, once | both | with the user's agreement |
| M7 | challenge `70594c50463a` | both | only after M5/M6 and an explicit go-ahead |

## 6. Git
- One repo (`puma` remote), **two branches**: Rolando's is **`auto-ping`** (tracker, map at scale); the partner has his own. The partner merges into `master` and tests the combined agent.
- Changes to `source/interface.py` are agreed by both before merging.
- Never commit `.env`, `data/`, caches (except the three parameter files) or `.DS_Store`.
