"""The final evaluation (plan rule 4, end of stage 8): the frozen pipeline, once, on the
holdout seeds, with every stage's key metric. EVALUATION CODE (reads ground truth to score).

    python3 source/mapping/final_eval.py --final build 31 [--steps inputs,odometry,...]
    python3 source/mapping/final_eval.py --final score 31,32,33,34,35

Without ``--final`` the holdout seeds are refused (``record_gt.gt_dir`` raises). The same
commands without ``--final`` work on dev and test seeds, which is how the script is checked
before the one real run. ``build`` runs ``build_map.py`` (pipeline: images and labels only) into
``cache/mapping/final/<prefix>-<seed>``; ``score`` reads those outputs and the truth.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import types
from pathlib import Path

for _m in ("vis_nav_sdk", "vis_nav_sim"):  # only so that imports resolve; nothing is used
    if _m not in sys.modules:
        _s = types.ModuleType(_m)
        _s.__getattr__ = lambda n: type(n, (), {})
        sys.modules[_m] = _s

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_fit as cf  # noqa: E402
import goal  # noqa: E402
import grid  # noqa: E402
import motion as M  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402
import stage7 as S7  # noqa: E402

ROOT = record_gt.ROOT
OUT = ROOT / "cache" / "mapping" / "final"


def data_dir(seed: int, prefix: str = "gt4") -> Path:
    return record_gt.gt_dir(seed, prefix=prefix)  # refuses holdout seeds without --final


def build(seed: int, prefix: str, steps: str) -> None:
    d = data_dir(seed, prefix)
    out = OUT / f"{prefix}-{seed}"
    subprocess.run([sys.executable, str(Path(__file__).with_name("build_map.py")), "--data", str(d), "--out", str(out), "--steps", steps], check=True)


def _fold90(a):
    return (a + 45.0) % 90.0 - 45.0


def score_seed(seed: int, prefix: str = "gt4") -> dict:
    d = data_dir(seed, prefix)
    out = OUT / f"{prefix}-{seed}"
    res = pickle.loads((out / "inputs.pkl").read_bytes())
    odo = pickle.loads((out / "odometry.pkl").read_bytes())
    cl = pickle.loads((out / "closures.pkl").read_bytes())
    P = [np.array(p) for p in json.loads((out / "poses_optimized.json").read_text())["drives"]]
    T = record_gt.load_poses(d)
    gt = json.loads((d / "gt.json").read_text())
    occ = np.array([[c == "#" for c in row] for row in gt["grid"]])
    segs = cf.wall_segments(occ)
    r: dict = {"seed": seed, "prefix": prefix}
    inputs = [x[0] for x in res]
    scans = [[(i.rows[k], i.top[k], i.status[k]) for k in range(len(i.labels))] for i in inputs]
    cam = grid.sc.camera_config()

    # stage 2: scan accuracy at the true poses, and coverage
    errs, cover = [], []
    for dr, inp in enumerate(inputs):
        for k in range(0, len(inp.labels), 3):
            c = grid.columns(inp.rows[k], inp.top[k], inp.status[k], cam)
            cover.append(c.hit.mean())
            if not c.hit.any():
                continue
            x, y, yaw = T[dr][k]
            th = yaw - c.bearing[c.hit]
            pts = np.c_[x + np.cos(th) * c.rng[c.hit], y + np.sin(th) * c.rng[c.hit]]
            errs.append(grid.point_segment_distance(pts, segs))
    errs = np.concatenate(errs)
    r["s2_scan"] = {"median_m": float(np.median(errs)), "p95_m": float(np.quantile(errs, 0.95)), "coverage": float(np.mean(cover))}

    # stage 3: compass (fused heading modulo 90 deg)
    e = np.concatenate([np.abs(_fold90(inp.psi - np.degrees(T[i][:, 2]))) for i, inp in enumerate(inputs)])
    ok = np.isfinite(e)
    r["s3_compass"] = {"median_deg": float(np.median(e[ok])), "p95_deg": float(np.quantile(e[ok], 0.95)), "valid_fraction": float(ok.mean())}

    # stage 3b: collision flags against true rotations > 30 / 45 deg under FORWARD or IDLE
    ev30 = ev45 = hit30 = hit45 = flags = pairs = 0
    for i, inp in enumerate(inputs):
        rot = np.degrees(np.abs(se2.wrap(np.diff(T[i][:, 2]))))
        lab = np.array(inp.labels[1:])
        f = np.asarray(inp.flags[1:], bool)
        ev = np.isin(lab, M.COLLISION_LABELS)
        ev30 += int((ev & (rot > 30)).sum())
        ev45 += int((ev & (rot > 45)).sum())
        hit30 += int((ev & (rot > 30) & f).sum())
        hit45 += int((ev & (rot > 45) & f).sum())
        flags += int(f.sum())
        pairs += len(f)
    r["s3b_collisions"] = {"events_30": ev30, "recall_30": hit30 / max(ev30, 1), "events_45": ev45, "recall_45": hit45 / max(ev45, 1), "flag_rate": flags / max(pairs, 1)}

    # stage 4: odometry (variant C, self-calibrated) per drive, spawn frame
    ate = [float(np.sqrt(((o.poses[:, :2] - T[i][:, :2]) ** 2).sum(1).mean())) for i, o in enumerate(odo)]
    hrms = [float(np.degrees(np.sqrt((se2.wrap(o.poses[:, 2] - T[i][:, 2]) ** 2).mean()))) for i, o in enumerate(odo)]
    r["s4_odometry"] = {"ate_m": ate, "heading_rms_deg": hrms}

    # stages 5 / 5b: closure precision (relative pose within 0.3 m and 10 deg of the truth)
    good = [np.hypot(*(c["z"][:2] - se2.relative(T[c["a"][0]][c["a"][1]], T[c["b"][0]][c["b"][1]])[:2])) < 0.3
            and abs(np.degrees(se2.wrap(c["z"][2] - se2.relative(T[c["a"][0]][c["a"][1]], T[c["b"][0]][c["b"][1]])[2]))) < 10 for c in cl]
    r["s5_closures"] = {"n": len(cl), "precision": float(np.mean(good)) if cl else float("nan"),
                        "by_kind": {k: sum(c["kind"] == k for c in cl) for k in ("boundary", "gate")}}

    # stage 6: joint trajectory
    X0 = np.concatenate([o.poses for o in odo])
    X = np.concatenate(P)
    Tt = np.concatenate(T)
    hr = float(np.degrees(np.sqrt((se2.wrap(X[:, 2] - Tt[:, 2]) ** 2).mean())))
    r["s6_graph"] = {"rigid_ate_before_m": S7_rigid(X0[:, :2], Tt[:, :2]), "rigid_ate_after_m": S7_rigid(X[:, :2], Tt[:, :2]),
                     "heading_rms_deg": hr, "scale_per_drive": [S7.umeyama_scale(P[i][:, :2], T[i][:, :2]) for i in range(len(P))]}

    # stage 7: the map rebuilt on the evaluation grid from the pipeline's poses, scored as in stage 7
    m = grid.build(scans, P, S7.SPAWN, spec=S7.SPEC)
    m.poses_in = P
    olo, ohits = grid.occupancy(scans, T, S7.SPEC)
    s7 = S7.score(seed, m, olo, ohits, T) if prefix == "gt4" else None
    if s7 is not None:
        r["s7_map"] = {"iou_vs_oracle": s7["iou_vs_oracle"], "iou_vs_full_layout": s7["iou_vs_full_layout"], "snap_recall": s7["snap"]["recall_all"],
                       "snap_false": s7["snap"]["false_rate"], "scale_detection_err_pct": s7["lattice"]["scale_detection_err_pct"],
                       "offset_err_m": s7["lattice"]["offset_err"], "in_cspace": s7["cspace"]["in_cspace"]}

    # stage 8: the goal on the pipeline's own map
    mf = goal.load_map(out / "map.npz")
    est = goal.place(mf, cv2.imread(str(d / "target.jpg")))
    tg = np.array(gt["target"])
    e_m = float(np.hypot(*(est.pose[:2] - tg[:2])))
    e_d = float(abs(np.degrees(se2.wrap(est.pose[2] - tg[2]))))
    r["s8_goal"] = {"err_m": e_m, "err_deg": e_d, "ok": bool(e_m < 0.10 and e_d < 10), "confidence": est.confidence,
                    "high_confidence_failure": bool(not (e_m < 0.10 and e_d < 10) and est.reliable)}
    r["runtime_s"] = json.loads((out / "timing.json").read_text())
    return r


def S7_rigid(est, true) -> float:
    mu_e, mu_t = est.mean(0), true.mean(0)
    E, Tc = est - mu_e, true - mu_t
    U, _, Vt = np.linalg.svd(Tc.T @ E)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return float(np.sqrt((((R @ E.T).T + mu_t - true) ** 2).sum(1).mean()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="open the holdout seeds (once, at the end of stage 8)")
    p.add_argument("step", choices=["build", "score"])
    p.add_argument("seeds")
    p.add_argument("--prefix", default="gt4")
    p.add_argument("--steps", default="all")
    a = p.parse_args()
    if a.final:
        record_gt.allow_final()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.step == "build":
        for s in seeds:
            build(s, a.prefix, a.steps)  # data_dir() is the holdout gate
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        allr = [score_seed(s, a.prefix) for s in seeds]
        tag = "holdout" if a.final else "check"
        (OUT / f"results_{tag}_{a.prefix}_{'_'.join(map(str, seeds))}.json").write_text(json.dumps(allr, indent=1, default=float))
        for r in allr:
            print(json.dumps(r, default=lambda v: round(float(v), 4)), flush=True)


if __name__ == "__main__":
    main()
