"""Stage 5 evaluation: loop-closure candidates, verification and relative pose against the true
poses. EVALUATION CODE (reads ground truth); the pipeline is ``loops.py``.

    uv run python source/mapping/stage5.py vocab            # fit the vocabulary on dev (gt4)
    uv run python source/mapping/stage5.py verify [gt4|official]
    uv run python source/mapping/stage5.py evaluate [gt4|official]
    uv run python source/mapping/stage5.py plots [gt4|official]
"""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loops  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402
import stage4 as s4  # noqa: E402
import stage5_extract as x5  # noqa: E402
import vo  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping" / "stage5"
FIG = ROOT / "reports" / "figures"
DEV, TEST = s4.DEV, s4.TEST
POSE_FLOOR = 10  # verified pairs with at least this many inliers get relative poses (PR sweep)
REVISIT_M, REVISIT_DEG = 0.5, 30.0  # ground-truth revisit (plan)
POSE_OK_M, POSE_OK_DEG = 0.3, 10.0  # a closure whose pose is this close to the truth is usable


def vocab() -> None:
    feats = []
    for s in DEV:
        for drive in x5.run_seed(s, "gt4"):
            feats += drive[::3]
    voc = loops.fit_vocabulary(feats)
    voc.save(provenance="fitted by stage5.py vocab on dev seeds 7-10 (gt4), every 3rd frame",
             frames=len(feats))
    print("vocabulary from", len(feats), "frames")


def verify_hash() -> str:
    h = hashlib.sha1(loops.VOCAB_PATH.read_bytes())
    for fn in (loops.candidates, loops.verify, loops.root_sift, loops.Vocabulary.encode):
        h.update(inspect.getsource(fn).encode())
    h.update(x5.code_hash().encode())
    h.update(f"{loops.TOP_K},{loops.EXCLUDE},{loops.RATIO},{loops.RANSAC_PX},{POSE_FLOOR}".encode())
    h.update(inspect.getsource(poses_for).encode())
    return h.hexdigest()[:8]


def _verify_seed(args):
    """Candidates and verification for one seed: pipeline steps only (no ground truth)."""
    seed, variant = args
    f = CACHE / f"verify_{variant}_{seed}_{verify_hash()}.pkl"
    if f.exists():
        return seed, pickle.loads(f.read_bytes())
    drives = x5.run_seed(seed, variant)
    flat = [ft for d in drives for ft in d]
    drive = np.concatenate([np.full(len(d), i) for i, d in enumerate(drives)])
    index = np.concatenate([np.arange(len(d)) for d in drives])
    g = loops.Vocabulary.load().encode(flat)
    cands = loops.candidates(g, drive, index)
    bf = cv2.BFMatcher()
    rows = []
    for i, j, sim in cands:
        n, H, m = loops.verify(flat[i], flat[j], bf)
        rows.append({"a": (int(drive[i]), int(index[i])), "b": (int(drive[j]), int(index[j])), "sim": sim, "inliers": n, "H": H, "matches": m})
    f.write_bytes(pickle.dumps(rows))
    return seed, rows


def poses_for(rows, inputs) -> None:
    """Methods (a), (b), (c) for every verified pair above ``POSE_FLOOR`` (pipeline steps)."""
    scans = [[vo.scan_points(inp.rows[k], inp.top[k], inp.status[k]) for k in range(len(inp.labels))] for inp in inputs]
    for r in rows:
        r["poses"] = {}
        if r["inliers"] < POSE_FLOOR:
            continue
        sa, sb = scans[r["a"][0]][r["a"][1]], scans[r["b"][0]][r["b"][1]]
        pa = loops.pose_from_homography(r["H"], r["matches"], sa)
        pc = loops.pose_from_features(r["matches"], sa, sb)
        init = pc.mean if pc is not None else pa
        pb = loops.pose_from_icp(sa, sb, init) if init is not None else None
        r["poses"] = {
            "a": None if pa is None else pa,
            "c": None if pc is None else pc.mean,
            "c_cov": None if pc is None else pc.cov,
            "b": None if pb is None else pb.mean,
            "b_cov": None if pb is None else pb.cov,
        }


def truth(rows, true_poses) -> None:
    for r in rows:
        ta, tb = true_poses[r["a"][0]][r["a"][1]], true_poses[r["b"][0]][r["b"][1]]
        rel = se2.relative(ta, tb)
        r["true_rel"] = rel
        r["revisit"] = bool(np.hypot(*(ta[:2] - tb[:2])) < REVISIT_M and abs(np.degrees(se2.wrap(ta[2] - tb[2]))) < REVISIT_DEG)
        r["err"] = {}
        for k in ("a", "b", "c"):
            p = r["poses"].get(k)
            if p is not None:
                d = p - rel
                r["err"][k] = (float(np.hypot(d[0], d[1])), float(abs(np.degrees(se2.wrap(d[2])))))


def gt_revisit_pairs(true_poses) -> dict:
    """Every ground-truth revisit pair (excluding own-drive neighbours within ``EXCLUDE``)."""
    P = np.concatenate(true_poses)
    drive = np.concatenate([np.full(len(p), i) for i, p in enumerate(true_poses)])
    index = np.concatenate([np.arange(len(p)) for p in true_poses])
    tree = cKDTree(P[:, :2])
    pairs = tree.query_pairs(REVISIT_M, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    ok = np.abs(np.degrees(se2.wrap(P[i, 2] - P[j, 2]))) < REVISIT_DEG
    ok &= ~((drive[i] == drive[j]) & (np.abs(index[i] - index[j]) <= loops.EXCLUDE))
    i, j = i[ok], j[ok]
    keys = {((int(drive[a]), int(index[a])), (int(drive[b]), int(index[b]))) for a, b in zip(i, j, strict=True)}
    frames_with = {k for pr in keys for k in pr}
    return {"pairs": keys, "frames": frames_with, "n_frames": len(P)}


def _key(r):
    a, b = r["a"], r["b"]
    return (a, b) if a <= b else (b, a)


def metrics(rows, gt, thr, method="b") -> dict:
    acc = [r for r in rows if r["inliers"] >= thr]
    n = len(acc)
    tp = sum(r["revisit"] for r in acc)
    usable = [r for r in acc if method in r["err"] and r["err"][method][0] < POSE_OK_M and r["err"][method][1] < POSE_OK_DEG]
    acc_keys = {_key(r) for r in acc}
    gt_pairs = {(a, b) if a <= b else (b, a) for a, b in gt["pairs"]}
    frames_acc = {k for r in acc if r["revisit"] for k in (r["a"], r["b"])}
    drive_pairs = {tuple(sorted((r["a"][0], r["b"][0]))) for r in acc if r["a"][0] != r["b"][0]}
    errs = np.array([r["err"][method] for r in acc if method in r["err"]]).reshape(-1, 2)
    return {
        "threshold": thr,
        "accepted": n,
        "precision_revisit": tp / n if n else None,
        "precision_pose": len(usable) / n if n else None,
        "recall_pairs": len(acc_keys & gt_pairs) / max(len(gt_pairs), 1),
        "recall_frames": len(frames_acc) / max(len(gt["frames"]), 1),
        "drive_pairs_joined": sorted(drive_pairs),
        "pose_err_median_m": float(np.median(errs[:, 0])) if len(errs) else None,
        "pose_err_median_deg": float(np.median(errs[:, 1])) if len(errs) else None,
        "pose_err_p95_m": float(np.percentile(errs[:, 0], 95)) if len(errs) else None,
        "pose_err_p95_deg": float(np.percentile(errs[:, 1], 95)) if len(errs) else None,
    }


def load_seed(seed, variant):
    _, rows = _verify_seed((seed, variant))
    drives = s4.load_inputs([seed], variant)
    poses_for(rows, [d["inp"] for d in drives])
    true_poses = [d["true"] for d in drives]
    truth(rows, true_poses)
    return rows, gt_revisit_pairs(true_poses), drives


def _eval_seed(args):
    seed, variant = args
    f = CACHE / f"scored_{variant}_{seed}_{verify_hash()}.pkl"
    if f.exists():
        return seed, pickle.loads(f.read_bytes())
    rows, gt, _ = load_seed(seed, variant)
    for r in rows:
        r.pop("H", None)
    out = (rows, gt)
    f.write_bytes(pickle.dumps(out))
    return seed, out


THRESHOLDS = (10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100)


def odometry(variant):
    """Variant C trajectories of the final stage 4b run (calibrated priors): pipeline output."""
    return pickle.loads((ROOT / "cache" / "mapping" / f"stage4b_results_{variant}.pkl").read_bytes())


def accept(r, poses, seed, thr) -> bool:
    """The pipeline's acceptance (``loops.measure`` + ``loops.odometry_gate``) on a scored pair."""
    p = r["poses"]
    if r["inliers"] < thr or p.get("c") is None or not loops.in_envelope(p["c"]):
        return False
    if p.get("b") is not None and not loops.agree(p["b"], p["c"], loops.AGREE_M, loops.AGREE_DEG):
        return False
    pa = poses[(seed, r["a"][0], "C")].poses[r["a"][1]]
    pb = poses[(seed, r["b"][0], "C")].poses[r["b"][1]]
    return loops.odometry_gate(p["c"], pa, pb)


def pose_ok(r, m="c") -> bool:
    return m in r["err"] and r["err"][m][0] < POSE_OK_M and r["err"][m][1] < POSE_OK_DEG


def summarise(scored, poses, seeds, thr, gate=True) -> dict:
    acc, acc_keys, gt_pairs, gt_frames, frames_acc, joined = [], set(), set(), set(), set(), {}
    for s in seeds:
        rows, gt = scored[s]
        if gate:
            a = [r for r in rows if accept(r, poses, s, thr)]
        else:
            a = [r for r in rows if r["inliers"] >= thr and r["poses"].get("c") is not None]  # no gate, no envelope
        acc += a
        acc_keys |= {(s, *_key(r)) for r in a}
        gt_pairs |= {(s, *((p, q) if p <= q else (q, p))) for p, q in gt["pairs"]}
        gt_frames |= {(s, f) for f in gt["frames"]}
        frames_acc |= {(s, f) for r in a if r["revisit"] for f in (r["a"], r["b"])}
        joined[s] = sorted({tuple(sorted((r["a"][0], r["b"][0]))) for r in a if r["a"][0] != r["b"][0]})
    n = len(acc)
    errs = {m: np.array([r["err"][m] for r in acc if m in r["err"]]).reshape(-1, 2) for m in ("a", "b", "c")}
    return {
        "threshold": thr,
        "accepted": n,
        "false_pose": int(sum(not pose_ok(r) for r in acc)),
        "precision_pose": float(np.mean([pose_ok(r) for r in acc])) if n else None,
        "precision_revisit": float(np.mean([r["revisit"] for r in acc])) if n else None,
        "recall_pairs": len(acc_keys & gt_pairs) / max(len(gt_pairs), 1),
        "recall_frames": len(frames_acc) / max(len(gt_frames), 1),
        "drive_pairs_joined": {str(k): [list(p) for p in v] for k, v in joined.items()},
        "err": {
            m: {
                "n": int(len(e)),
                "median_m": float(np.median(e[:, 0])) if len(e) else None,
                "median_deg": float(np.median(e[:, 1])) if len(e) else None,
                "p95_m": float(np.percentile(e[:, 0], 95)) if len(e) else None,
                "p95_deg": float(np.percentile(e[:, 1], 95)) if len(e) else None,
            }
            for m, e in errs.items()
        },
    }


def nis_inflation(scored, poses, seeds, thr) -> dict:
    """NIS of method (c)'s error against its covariance on accepted pose-correct closures, and
    the inflation that puts 95 % under the 3-dof bound."""
    nis = []
    for s in seeds:
        for r in scored[s][0]:
            if accept(r, poses, s, thr) and pose_ok(r) and r["poses"].get("c_cov") is not None:
                d = r["poses"]["c"] - r["true_rel"]
                d[2] = se2.wrap(d[2])
                nis.append(float(d @ np.linalg.solve(r["poses"]["c_cov"], d)))
    nis = np.array(nis)
    return {"n": len(nis), "frac_under_7.81": float(np.mean(nis < 7.81)), "inflation_for_95": float(max(1.0, np.percentile(nis, 95) / 7.81))}


def load_scored(variant):
    with ProcessPoolExecutor(max_workers=5) as ex:
        return dict(ex.map(_eval_seed, [(s, variant) for s in DEV + TEST]))


def analyze(variant) -> dict:
    scored = load_scored(variant)
    poses = odometry(variant)
    out = {"variant": variant}
    for split, seeds in (("dev", DEV), ("test", TEST)):
        out[f"pr_{split}"] = [summarise(scored, poses, seeds, t) for t in THRESHOLDS]
        out[f"pr_{split}_ungated"] = [summarise(scored, poses, seeds, t, gate=False) for t in THRESHOLDS]
    out["nis_dev"] = nis_inflation(scored, poses, DEV, loops.MIN_INLIERS)
    out["chosen"] = {split: summarise(scored, poses, seeds, loops.MIN_INLIERS) for split, seeds in (("dev", DEV), ("test", TEST))}
    out["per_seed"] = {str(s): summarise(scored, poses, [s], loops.MIN_INLIERS) for s in DEV + TEST}
    return out


def plots(variant: str = "gt4") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((ROOT / "cache" / "mapping" / f"stage5_results_{variant}.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    # precision-recall over the inlier threshold
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, rec in zip(ax, ("recall_frames", "recall_pairs"), strict=True):
        for split, style in (("dev", "o-"), ("test", "s-")):
            for gated, alpha in ((True, 1.0), (False, 0.4)):
                pr = res[f"pr_{split}" + ("" if gated else "_ungated")]
                lab = f"{split}, {'gate + envelope' if gated else 'inliers only'}"
                a.plot([q[rec] for q in pr], [q["precision_revisit"] for q in pr], style, alpha=alpha, label=lab, ms=4)
                if gated:
                    q = next(q for q in pr if q["threshold"] == loops.MIN_INLIERS)
                    a.plot(q[rec], q["precision_revisit"], "k*", ms=12)
        a.axhline(0.98, color="r", lw=0.7, ls="--")
        a.set_xlabel(rec.replace("_", " ") + " (ground-truth revisits)")
        a.set_ylabel("precision (closure is a ground-truth revisit)")
        a.set_title(f"{variant}: threshold swept {THRESHOLDS[0]} to {THRESHOLDS[-1]} inliers (star: {loops.MIN_INLIERS})")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"stage5_pr_{variant}.png", dpi=130)
    plt.close(fig)
    # pose errors of accepted test closures, three methods
    scored = load_scored(variant)
    poses = odometry(variant)
    acc = [r for sd in TEST for r in scored[sd][0] if accept(r, poses, sd, loops.MIN_INLIERS)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = {"a": "(a) homography", "b": "(b) ICP from (c)", "c": "(c) wall features, two scans"}
    for m in ("a", "b", "c"):
        e = np.array([r["err"][m] for r in acc if m in r["err"]])
        ax[0].hist(np.clip(e[:, 0] * 1000, 0, 100), bins=60, histtype="step", label=f"{names[m]} (n={len(e)})")
        ax[1].hist(np.clip(e[:, 1], 0, 10), bins=60, histtype="step", label=names[m])
    ax[0].set_xlabel("translation error [mm] (clipped at 100)")
    ax[1].set_xlabel("rotation error [deg] (clipped at 10)")
    for a in ax:
        a.set_yscale("log")
        a.legend(fontsize=7)
    fig.suptitle(f"accepted test closures, {variant}")
    fig.tight_layout()
    fig.savefig(FIG / f"stage5_pose_errors_{variant}.png", dpi=130)
    plt.close(fig)
    # accepted closures on the true trajectories
    fig, ax = plt.subplots(1, len(TEST), figsize=(4.2 * len(TEST), 4.4))
    for a, sd in zip(ax, TEST, strict=True):
        tp = [d["true"] for d in s4.load_inputs([sd], variant)]
        for d, P in enumerate(tp):
            a.plot(P[:, 0], P[:, 1], lw=0.5, alpha=0.6, label=f"drive {d}")
        for r in scored[sd][0]:
            if accept(r, poses, sd, loops.MIN_INLIERS):
                pa, pb = tp[r["a"][0]][r["a"][1]], tp[r["b"][0]][r["b"][1]]
                bad = not pose_ok(r)
                a.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color="r" if bad else "k", lw=1.5 if bad else 0.3, alpha=1.0 if bad else 0.3)
        a.set_aspect("equal")
        a.set_title(f"seed {sd}: closures (red: pose wrong)")
        a.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / f"stage5_closures_{variant}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    variant = sys.argv[2] if len(sys.argv) > 2 else "gt4"
    if cmd == "vocab":
        vocab()
    elif cmd == "verify":
        with ProcessPoolExecutor(max_workers=5) as ex:
            for s, rows in ex.map(_verify_seed, [(s, variant) for s in DEV + TEST]):
                print(s, len(rows), "candidates", flush=True)
    elif cmd == "evaluate":
        res = analyze(variant)
        (ROOT / "cache" / "mapping" / f"stage5_results_{variant}.json").write_text(json.dumps(res, indent=1))
        for split in ("dev", "test"):
            for row in res[f"pr_{split}"]:
                print(split, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("drive_pairs_joined", "err")}, "c med", row["err"]["c"]["median_m"])
        print("nis", res["nis_dev"])
    elif cmd == "plots":
        plots(variant)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
