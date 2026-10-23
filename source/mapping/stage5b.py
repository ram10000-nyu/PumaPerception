"""Stage 5b tasks 1 to 3 evaluation: boundary-verified closures (reads ground truth labels from
the stage 5 scored pairs; the boundaries themselves come from images only via texture.py).

  python3 source/mapping/stage5b.py extract 7 8 9 10 21 22 23 24 25
  python3 source/mapping/stage5b.py evaluate
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan as sc  # noqa: E402
import texture as tx  # noqa: E402
import se2  # noqa: E402
import types as _types  # noqa: E402

GATE_M, GATE_DEG = 0.5, 10.0
AGREE_M, AGREE_DEG = 0.10, 5.0


def odometry(prefix="gt4"):
    """Stage 4b variant C poses (pipeline output). Two simulator-SDK modules are stubbed only
    so the pickle's import chain resolves; nothing from them is used."""
    for m in ("vis_nav_sdk", "vis_nav_sim"):
        if m not in sys.modules:
            mod = _types.ModuleType(m)
            mod.__getattr__ = lambda n: type(n, (), {})
            sys.modules[m] = mod
    sys.path.insert(0, str(ROOT / "source"))
    name = "official" if prefix == "gt" else prefix
    d = pickle.loads((ROOT / "cache" / "mapping" / f"stage4b_results_{name}.pkl").read_bytes())
    return {(k[0], k[1]): v.poses for k, v in d.items() if k[2] == "C"}


def gate_ok(r, P, seed) -> bool:
    p = r["poses"]
    if p.get("b") is not None:
        d = p["b"] - p["c"]
        if np.hypot(d[0], d[1]) > AGREE_M or abs(np.degrees(se2.wrap(d[2]))) > AGREE_DEG:
            return False
    rel = se2.relative(P[(seed, r["a"][0])][r["a"][1]], P[(seed, r["b"][0])][r["b"][1]])
    d = p["c"] - rel
    return bool(np.hypot(d[0], d[1]) < GATE_M and abs(np.degrees(se2.wrap(d[2]))) < GATE_DEG)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache" / "mapping" / "stage5b"
HOLDOUT = set(json.loads((Path(__file__).resolve().parent / "seeds.json").read_text())["holdout"])
DEV, TEST = [7, 8, 9, 10], [21, 22, 23, 24, 25]
SCORED = "131a1038"
POSE_OK_M, POSE_OK_DEG = 0.3, 10.0
MIN_INLIERS = 20
MAX_REL_M, MAX_REL_DEG = 0.45, 27.0


def extract(seed: int, prefix: str = "gt4") -> Path:
    assert seed not in HOLDOUT
    cam = sc.camera_config()
    d = ROOT / "data" / f"{prefix}-{seed}"
    drives = []
    for tdir in sorted(d.glob("traj_*"), key=lambda p: int(p.name.split("_")[1])):
        n = len(json.loads((tdir / "data_info.json").read_text()))
        frames = []
        for k in range(n):
            img = cv2.imread(str(tdir / f"{k}.jpg"))
            frames.append(tx.boundaries(img, cam=cam) if img is not None else [])
        drives.append(frames)
    path = OUT / f"boundaries_{prefix}_{seed}.pkl"
    path.write_bytes(pickle.dumps(drives))
    return path


def in_envelope(p) -> bool:
    return np.hypot(p[0], p[1]) < MAX_REL_M and abs(np.degrees(p[2])) < MAX_REL_DEG


def rows_for(seed, prefix="gt4"):
    rows, gt = pickle.loads((ROOT / "cache" / "mapping" / "stage5" / f"scored_{'official' if prefix == 'gt' else prefix}_{seed}_{SCORED}.pkl").read_bytes())
    return rows, gt


def score(seeds, corr_min, pos_tol, min_match=1, prefix="gt4", use_env=True):
    acc, tot_false_cand = [], 0
    frames_acc, gt_frames, joined = set(), set(), {}
    alias_total, alias_acc = 0, 0
    for s in seeds:
        B = pickle.loads((OUT / f"boundaries_{prefix}_{s}.pkl").read_bytes())
        rows, gt = rows_for(s, prefix)
        gt_frames |= {(s, f) for f in gt["frames"]}
        js = set()
        for r in rows:
            p = r["poses"].get("c") if r.get("poses") else None
            if r["inliers"] < MIN_INLIERS or p is None or (use_env and not in_envelope(p)):
                continue
            ok_pose = "c" in r["err"] and r["err"]["c"][0] < POSE_OK_M and r["err"]["c"][1] < POSE_OK_DEG
            if not ok_pose:
                alias_total += 1
            n = tx.match_boundaries(B[r["a"][0]][r["a"][1]], B[r["b"][0]][r["b"][1]], p, corr_min, pos_tol)
            if n >= min_match:
                acc.append((s, r["revisit"], ok_pose))
                if not ok_pose:
                    alias_acc += 1
                if r["revisit"]:
                    frames_acc |= {(s, r["a"]), (s, r["b"])}
                if r["a"][0] != r["b"][0]:
                    js.add(tuple(sorted((r["a"][0], r["b"][0]))))
        joined[s] = sorted(js)
    n = len(acc)
    return {
        "corr_min": corr_min, "pos_tol": pos_tol, "min_match": min_match,
        "accepted": n,
        "precision_revisit": float(np.mean([a[1] for a in acc])) if n else None,
        "precision_pose": float(np.mean([a[2] for a in acc])) if n else None,
        "false_pose": int(sum(not a[2] for a in acc)),
        "frame_recall": len(frames_acc) / max(len(gt_frames), 1),
        "wrong_pose_candidates_in_envelope": alias_total,
        "wrong_pose_accepted": alias_acc,
        "drive_pairs_joined": {str(k): v for k, v in joined.items()},
    }


def rules(seeds, corr_min=0.8, corr_veto=0.3, pos_tol=0.05, prefix="gt4", min_inl_boundary=MIN_INLIERS):
    """Every acceptance rule on the same candidates: gate (stage 5), boundary, gate and no
    contradiction, union. Returns per-rule metrics."""
    P = odometry(prefix)
    names = ("gate", "boundary", "gate_nocontra", "union")
    acc = {n: [] for n in names}
    fr = {n: set() for n in names}
    jn = {n: {} for n in names}
    gt_frames, ncand_false = set(), 0
    for s in seeds:
        B = pickle.loads((OUT / f"boundaries_{prefix}_{s}.pkl").read_bytes())
        rows, gt = rows_for(s, prefix)
        gt_frames |= {(s, f) for f in gt["frames"]}
        js = {n: set() for n in names}
        for r in rows:
            p = r["poses"].get("c") if r.get("poses") else None
            if p is None or not in_envelope(p) or r["inliers"] < min(MIN_INLIERS, min_inl_boundary):
                continue
            ok_pose = "c" in r["err"] and r["err"]["c"][0] < POSE_OK_M and r["err"]["c"][1] < POSE_OK_DEG
            m, c = tx.compare_boundaries(B[r["a"][0]][r["a"][1]], B[r["b"][0]][r["b"][1]], p, corr_min, corr_veto, pos_tol)
            g = r["inliers"] >= MIN_INLIERS and gate_ok(r, P, s)
            dec = {"gate": g, "boundary": m >= 1 and r["inliers"] >= min_inl_boundary,
                   "gate_nocontra": g and c == 0}
            dec["union"] = dec["gate_nocontra"] or dec["boundary"]
            for n in names:
                if dec[n]:
                    acc[n].append((s, r["revisit"], ok_pose, r["a"], r["b"]))
                    if r["revisit"]:
                        fr[n] |= {(s, r["a"]), (s, r["b"])}
                    if r["a"][0] != r["b"][0]:
                        js[n].add(tuple(sorted((r["a"][0], r["b"][0]))))
        for n in names:
            jn[n][str(s)] = sorted(js[n])
    out = {}
    for n in names:
        a = acc[n]
        out[n] = {"accepted": len(a), "precision_pose": float(np.mean([x[2] for x in a])) if a else None,
                  "precision_revisit": float(np.mean([x[1] for x in a])) if a else None,
                  "false_pose": int(sum(not x[2] for x in a)), "frame_recall": len(fr[n]) / max(len(gt_frames), 1),
                  "drive_pairs_joined": jn[n]}
    out["_params"] = {"corr_min": corr_min, "corr_veto": corr_veto, "pos_tol": pos_tol, "min_inl_boundary": min_inl_boundary}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        OUT.mkdir(parents=True, exist_ok=True)
        pre = "gt4"
        args = sys.argv[2:]
        if args and args[0] in ("gt", "gt4"):
            pre, args = args[0], args[1:]
        for s in [int(a) for a in args]:
            print(s, extract(s, pre), flush=True)
    elif cmd == "evaluate":
        pre = sys.argv[2] if len(sys.argv) > 2 else "gt4"
        res = {"sweep_dev": []}
        for c in (0.6, 0.7, 0.8, 0.85, 0.9):
            for t in (0.03, 0.05, 0.08):
                r = score(DEV, c, t, prefix=pre)
                res["sweep_dev"].append(r)
                print(json.dumps({k: r[k] for k in ("corr_min", "pos_tol", "accepted", "precision_revisit", "precision_pose", "false_pose", "frame_recall", "wrong_pose_candidates_in_envelope", "wrong_pose_accepted")}), flush=True)
        (OUT / f"eval_{pre}.json").write_text(json.dumps(res, indent=1))
    elif cmd == "rules":
        pre, split = sys.argv[2], sys.argv[3]
        seeds = DEV if split == "dev" else TEST
        grid = [(0.8, v, 0.05, mi) for v in (0.2, 0.3, 0.4) for mi in (20, 12)] if split == "dev" else [tuple(float(x) if "." in x else int(x) for x in sys.argv[4:8])]
        allres = []
        for cm, cv, pt, mi in grid:
            r = rules(seeds, cm, cv, pt, pre, mi)
            allres.append(r)
            print(json.dumps(r["_params"]), flush=True)
            for n in ("gate", "boundary", "gate_nocontra", "union"):
                x = r[n]
                print(f"  {n:14s} acc {x['accepted']:6d} prec {x['precision_pose']:.4f} false {x['false_pose']:3d} frame_recall {x['frame_recall']:.3f}", flush=True)
        (OUT / f"rules_{pre}_{split}.json").write_text(json.dumps(allres, indent=1))
