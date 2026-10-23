# Stage 5b: texture adjacency

Plan revision in force: **revision 9**. Implemented by the user's Claude session (not the coding agent), on the Linux workspace with system Python (no simulator: the course package index is not reachable there), from the recorded `gt4` datasets.

Status: **go, and the boundary check works.**
- Ordered texture pairs are near-unique in every maze (98.7 to 100 % of boundaries; every triple unique).
- From images alone, **boundary-verified closures reach 0.998 precision without the odometry gate**. Stage 5's closures without the gate: 0.4 to 0.6.
- **Not met:**
  - the boundary rule alone has a frame recall of 0.086 (target 0.25); the union with the gate keeps 0.344;
  - the veto removes only 4 of the 17 stage 5 aliases, because in the other 13 one frame sees no face edge.

## Built

| file | what |
|---|---|
| `source/mapping/stage5b_measure.py` | task 0, **evaluation only**: rebuilds every wall face from frames at true poses (ray cast against `camera_fit.Scene`), per wall line, side and 1 cm bin; refuses holdout seeds |
| `source/mapping/texture.py` | **pipeline** (image and its scan only): straight scan pieces, face-edge lattice phase from appearance, 8 cm rectified windows either side of each edge, `Boundary(left, right, pos)`, `compare_boundaries` (matches, contradictions) |
| `source/mapping/stage5b.py` | evaluation: boundary extraction for every frame, acceptance rules on the stage 5 scored candidates (`scored_*_131a1038.pkl`), stage 4b odometry for the gate |
| `tests/mapping/test_texture.py` | 3 tests (transform; match, contradiction and alias shift; module never names ground truth) |
| `cache/mapping/stage5b/` | strips, `task0_results.json`, `boundaries_gt4_<seed>.pkl`, `rules_gt4_{dev,test}.json` |

Reproduce:
```
python3 source/mapping/stage5b_measure.py build 7 8 9 10 21 22 23 24 25
python3 source/mapping/stage5b.py extract gt4 7 8 9 10 21 22 23 24 25
python3 source/mapping/stage5b.py rules gt4 dev
python3 source/mapping/stage5b.py rules gt4 test 0.8 0.2 0.05 12
```

## Task 0: go/no-go (evaluation only)

**Method.** Every frame (every 2nd) is ray cast at its true pose. Each image column's wall pixels are sampled at 16 heights for walls 0.18 to 1.4 m away and accumulated per wall line, viewing side and 1 cm bin. This rebuilds every visible face as a rectified strip.

**Layout.**
- The appearance change along a wall peaks sharply at positions 0.1 m from wall-cell centres: a mean change of 28 against 12 to 13 everywhere else.
- **Every wall-cell face (0.2 m) carries one texture image.**
- Adjacent faces are uncorrelated (median correlation 0.01).
- Repeats are exact copies: the median face's best match elsewhere has correlation 0.98.

**Identity.** Faces were clustered at correlation ≥ 0.9. The results are stable from 0.8 to 0.95 (seed 7: unique-pair fraction 0.987 to 0.991).

| seed | faces | texture IDs | faces with a twin | boundaries | **unique pairs** | triples | **unique triples** |
|---|---|---|---|---|---|---|---|
| 7 | 678 | 373 | 72 % | 468 | **0.992** | 338 | **1.000** |
| 8 | 688 | 364 | 76 % | 489 | **0.996** | 355 | **1.000** |
| 9 | 665 | 339 | 76 % | 431 | **1.000** | 309 | **1.000** |
| 10 | 633 | 337 | 76 % | 447 | **0.991** | 315 | **1.000** |
| 21 | 673 | 357 | 76 % | 472 | 0.987 | 341 | 1.000 |
| 22 | 686 | 362 | 77 % | 483 | 0.996 | 351 | 1.000 |
| 23 | 658 | 358 | 72 % | 440 | 1.000 | 311 | 1.000 |
| 24 | 640 | 351 | 73 % | 452 | 0.987 | 317 | 1.000 |
| 25 | 677 | 344 | 78 % | 482 | 0.988 | 353 | 1.000 |

**Go** (threshold: at least 90 % of boundaries unique on dev; measured 99.1 to 100 %).
- Single textures are ambiguous: three quarters of faces have a twin.
- Ordered pairs almost never repeat: at most 3 repeated pairs per maze.
- Triples never repeat.
- The faces counted are those seen within 1.4 m; corner adjacencies (two wall lines) are not counted.

## Tasks 1 to 3: boundaries from images (pipeline)

Per frame, `texture.boundaries(img)`:
1. Takes the stage 2 scan (`r_max` 1.8 m) and samples each column's wall at 16 heights for depths 0.17 to 1.5 m.
2. Splits the scan points into straight pieces.
3. Fits the 0.2 m face-edge lattice phase that maximises the column-to-column appearance change. On seed 7 drive 0, 63 % of pieces are within 1 cm of the true phase and 72 % within 2 cm.
4. Describes the 8 cm of wall either side of each edge as a rectified 8 × 16 × 3 patch, zero-mean and unit-norm.

Two full faces in a row (0.4 m of continuous wall) were tried first, giving 0.09 to 0.24 boundaries per frame. The fixed windows give **1.25 per frame, with 67 % of frames having at least one**, at 8 ms per frame.

**Rules**, each applied to the same stage 5 candidates. The measured relative pose is method (c), and every rule keeps stage 5's revisit envelope.
- **gate:** stage 5 (at least 20 inliers, (b) and (c) agree, odometry gate 0.5 m / 10°).
- **boundary:** at least one co-located boundary (within 5 cm under the measured pose) with both windows correlating above 0.8, at least 12 inliers, **no odometry**.
- **gate_nocontra:** gate, unless a co-located boundary has a side correlating below 0.2 (a veto).
- **union:** gate_nocontra or boundary.

The parameters were **chosen on dev, then frozen** before any test number was computed: correlation 0.8, veto 0.2, position 5 cm, boundary inliers 12.
- On dev the gate already has no false closures, so dev shows only the veto's cost: −0.6 points of recall.
- Dev sweep: correlation 0.6 to 0.9; boundary precision 0.994 to 1.000.

| test `gt4` | accepted | precision (pose within 0.3 m, 10°) | false | frame recall | drive pairs joined |
|---|---|---|---|---|---|
| gate (stage 5) | 6,594 | 0.9974 | 17 | 0.344 | all on 4 / 5 (seed 23: 0–2, 1–2) |
| **boundary, no odometry** | 1,174 | **0.9983** | **2** | 0.086 | all on 3 / 5; seed 23 and 25 two pairs each |
| gate_nocontra | 6,328 | 0.9979 | 13 | 0.337 | as gate |
| **union** | 6,568 | 0.9977 | 15 | **0.344** | as gate |

Dev (same parameters): boundary 882 accepted, 1 false (0.9989), frame recall 0.084; union 0.9998, frame recall 0.352.

## The remaining false closures (disclosed: test seeds, diagnosed after the frozen run; nothing changed)

**The 17 stage 5 aliases** come from **3 places**: seed 22 (one) and seed 24 (two). Each is a run of consecutive frames shifted **0.56 to 0.60 m (three cells)** along a corridor, with rotation under 3.4°.
- The boundary rule accepts **none** of them.
- The veto removes 4. **In 13 of the 17, one of the two frames sees no face edge**, so there is no evidence either way.
- Wherever both frames see an edge at the same place, the veto fires.

These clusters are what stage 6's two-pass Mahalanobis gating has to remove; the plan already requires it.

**The boundary rule's 2 false closures** are long-range: 3.6 m (seed 22) and 4.2 m / 90° (seed 21). They are rare repeated pairs or descriptor confusions. They are 2 of 1,174, but each is metres off. Stage 6 must not trust a boundary-only closure blindly:
- weight it by a robust kernel;
- check it against the first-pass optimised trajectory;
- or require **two adjacent boundaries** (a triple: unique on every seed in task 0). That could not be tuned here without dev evidence.

## Acceptance (plan, revision 9)

| criterion | result | |
|---|---|---|
| Task 0 measurement reported; go/no-go stated | go: 99.1 to 100 % unique on dev (98.7 to 100 % test); triples 100 % | pass |
| Precision ≥ 0.98 without the odometry gate | boundary rule: 0.9983 test, 0.9989 dev | pass |
| None of the stage 5 alias closures accepted | boundary rule: 0 of 17. As used with the gate (union): 15 false, 13 of them aliases from the 3 clusters | **pass for the boundary rule, not for the union** |
| Frame recall ≥ 0.25 | boundary rule 0.086; union 0.344 | **fail for the boundary rule alone**; the union keeps stage 5's recall |
| Leakage (rule 10) | `texture.py` reads only the image and its scan; a test checks it never names ground truth | pass |

## What this changes downstream

- **Stage 6.**
  - Use the **union** closures.
  - Mark boundary closures as **gate-free**: they can re-anchor where odometry drift exceeds the 0.5 m gate, i.e. long or biased drives and the per-tick challenge data.
  - Two-pass Mahalanobis gating is still required for the 3 alias clusters.
  - Treat a boundary-only closure that disagrees with the first-pass trajectory by more than a few sigma as an outlier: the two long-range false closures.
- **Stage 8** (goal placement): match the goal views' boundaries against the map's boundary signatures. A pair is near-unique, so a single boundary seen in the goal views identifies its location.
- **Not done here:**
  - `official` (`gt`) frames were not evaluated;
  - corner adjacencies are not used;
  - triple-based (two-boundary) closures are untested;
  - boundaries at walls beyond 1.5 m are not extracted.
