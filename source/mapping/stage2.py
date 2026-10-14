"""Stage 2 evaluation: reference check, scan accuracy against the true wall segments, bias,
coverage, mask review, oracle maps. EVALUATION CODE: it reads ground truth (rule 10); the
pipeline modules ``segment.py`` and ``scan.py`` never do.

    uv run source/mapping/stage2.py scans      # run the pipeline on every frame (cached)
    uv run source/mapping/stage2.py reference  # task 0
    uv run source/mapping/stage2.py evaluate   # dev tuning of r_max, then test numbers
    uv run source/mapping/stage2.py masks      # 50-frame contact sheet + exact mask check
    uv run source/mapping/stage2.py oracle     # oracle occupancy maps for stage 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import vis_nav_sim as sim  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_fit as cf  # noqa: E402
import record_gt  # noqa: E402
import scan as sc  # noqa: E402
import se2  # noqa: E402
import segment as sg  # noqa: E402
import stage0_camera as s0  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
FIG = ROOT / "reports" / "figures"
CFG = json.loads(record_gt.SEEDS.read_text())
DEV, TEST = CFG["dev"], CFG["test"]
CAM = sc.camera_config()
RESULTS = CACHE / "stage2_results.json"


def code_hash() -> str:
    h = hashlib.sha1()
    for f in ("segment.py", "scan.py", "camera.json"):
        p = (CACHE / f) if f.endswith("json") else Path(__file__).with_name(f)
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save_results(update: dict) -> None:
    r = load_results()
    r.update(update)
    RESULTS.write_text(json.dumps(r, indent=1))


def occupancy(seed: int) -> np.ndarray:
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    return np.array([[c == "#" for c in row] for row in gt["grid"]])


# ----------------------------------------------------------------------- pipeline on frames


def _scan_drive(args):
    seed, drive = args
    path = record_gt.gt_dir(seed) / f"traj_{drive}"
    info = json.loads((path / "data_info.json").read_text())
    rows = np.full((len(info), 320), np.nan, np.float32)
    status = np.zeros((len(info), 320), np.int8)
    for k, r in enumerate(info):
        img = cv2.imread(str(path / r["image"]))
        s = sg.segment(img)
        rows[k] = s.rows
        status[k] = s.status
    return seed, drive, rows, status


def run_scans(seeds, force=False) -> dict:
    """``{(seed, drive): (rows, status)}``, cached per seed under the code hash."""
    out = {}
    for seed in seeds:
        f = CACHE / f"stage2_scans_{seed}_{code_hash()}.npz"
        if f.exists() and not force:
            z = np.load(f)
            for d in range(3):
                out[(seed, d)] = (z[f"rows{d}"], z[f"status{d}"])
            continue
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(_scan_drive, [(seed, d) for d in range(3)]))
        np.savez(
            f, **{f"rows{d}": r for _, d, r, _ in res}, **{f"status{d}": s for _, d, _, s in res}
        )
        for _, d, r, s in res:
            out[(seed, d)] = (r, s)
        print("scanned", seed, [len(r) for _, _, r, _ in res], flush=True)
    return out


def columns_xz(rows: np.ndarray):
    """Vectorised back-projection for arrays of boundary rows: ``(X, Z)`` with NaN where the
    boundary is not below the horizon."""
    u = np.arange(rows.shape[-1]) + 0.5
    ok = rows > CAM.cy + 1e-6
    v = np.where(ok, rows, np.nan)
    x, z = sc.backproject(np.broadcast_to(u, rows.shape), v, CAM)
    return x, z


# --------------------------------------------------------------------------------- task 0


def old_wall_segments(occ: np.ndarray) -> np.ndarray:
    """Stage 0's rule, for comparison: every wall cell contributes a 0.2 m segment along x
    if a left/right neighbour is a wall, along y if an up/down one is (isolated: x)."""
    rows, cols = occ.shape
    segs = []
    for r in range(rows):
        for c in range(cols):
            if not occ[r, c]:
                continue
            x, y = c * 0.2, r * 0.2
            has_x = (c > 0 and occ[r, c - 1]) or (c + 1 < cols and occ[r, c + 1])
            has_y = (r > 0 and occ[r - 1, c]) or (r + 1 < rows and occ[r + 1, c])
            if has_x or not has_y:
                segs.append((x - 0.1, y, x + 0.1, y))
            if has_y:
                segs.append((x, y - 0.1, x, y + 0.1))
    return np.array(segs)


class OldScene(cf.Scene):
    def __init__(self, occ, half):
        self.segments = old_wall_segments(occ)
        s = self.segments
        self.boxes = np.c_[
            np.minimum(s[:, 0], s[:, 2]) - half,
            np.maximum(s[:, 0], s[:, 2]) + half,
            np.minimum(s[:, 1], s[:, 3]) - half,
            np.maximum(s[:, 1], s[:, 3]) + half,
        ]


def reference_samples(seeds, per_drive=80):
    """Calibration (solid green floor) renders at true poses of random frames, with the
    measured sub-pixel boundary rows."""
    tex = sim.Textures.find()
    cal = s0.calibration_textures(tex)
    cols = np.arange(320)
    out = []
    for seed in seeds:
        w = sim.Simulator(cal, seed, msaa=4)
        rng = np.random.default_rng(seed)
        for d, P in enumerate(record_gt.load_poses(record_gt.gt_dir(seed))):
            for k in rng.choice(len(P), min(per_drive, len(P)), replace=False):
                x, y, yaw = P[k]
                img = w.render(float(x), float(y), float(yaw))
                rows = cf.boundary_rows(img, cols, s0.green_floor)
                out.append(
                    {
                        "seed": seed,
                        "pos": (x, y),
                        "yaw": float(yaw),
                        "rows": rows,
                        "k": int(k),
                        "d": d,
                    }
                )
    return out


def score_reference(samples, scenes) -> dict:
    cols = np.arange(320)
    err, zs, ends = [], [], []
    for s in samples:
        scene = scenes[s["seed"]]
        z = cf.raycast_depths(scene, s["pos"], s["yaw"], cols, CAM.fx, CAM.cx)
        ok = np.isfinite(s["rows"]) & np.isfinite(z) & (z < 5.0)
        e = s["rows"] - (CAM.cy + CAM.fy * CAM.h / z)
        err.append(e[ok])
        zs.append(z[ok])
        # distance of the predicted hit point to the nearest segment end
        fwd = np.array([np.cos(s["yaw"]), np.sin(s["yaw"])])
        right = np.array([np.sin(s["yaw"]), -np.cos(s["yaw"])])
        d = fwd[None] + ((cols[ok] + 0.5 - CAM.cx) / CAM.fx)[:, None] * right[None]
        p = np.array(s["pos"])[None] + z[ok][:, None] * d
        sg_ = scene.segments
        ends.append(
            np.minimum(
                np.hypot(p[:, None, 0] - sg_[None, :, 0], p[:, None, 1] - sg_[None, :, 1]).min(1),
                np.hypot(p[:, None, 0] - sg_[None, :, 2], p[:, None, 1] - sg_[None, :, 3]).min(1),
            )
        )
    err, zs, ends = map(np.concatenate, (err, zs, ends))
    bad = np.abs(err) > 0.5
    return {
        "columns": int(len(err)),
        "within_0.5px": float(np.mean(~bad)),
        "within_2px": float(np.mean(np.abs(err) <= 2)),
        "median_abs_px": float(np.median(np.abs(err))),
        "bad_columns": int(bad.sum()),
        "bad_median_depth_m": float(np.median(zs[bad])) if bad.any() else None,
        "bad_frac_depth_below_0.3": float(np.mean(zs[bad] < 0.3)) if bad.any() else None,
        "bad_frac_within_2cm_of_segment_end": float(np.mean(ends[bad] < 0.02))
        if bad.any()
        else None,
        "_err": err,
        "_z": zs,
    }


def task0() -> dict:
    samples = reference_samples(DEV)
    occs = {s: occupancy(s) for s in DEV}
    res = {"old_rule": {}, "new_rule": {}}
    halves = [0.0010, 0.0015, 0.0018, 0.0020, 0.0022, 0.0025, 0.0030]
    sweep = []
    for half in halves:
        scenes = {s: cf.Scene(occs[s], half=half) for s in DEV}
        r = score_reference(samples, scenes)
        sweep.append((half, r["within_0.5px"], r["within_2px"]))
    best = max(sweep, key=lambda t: t[1])[0]
    res["half_thickness_sweep"] = [
        {"half_m": h, "within_0.5px": a, "within_2px": b} for h, a, b in sweep
    ]
    res["chosen_half_thickness_m"] = best
    old = score_reference(samples, {s: OldScene(occs[s], 0.002) for s in DEV})
    new = score_reference(samples, {s: cf.Scene(occs[s], half=best) for s in DEV})
    for name, r in (("old_rule", old), ("new_rule", new)):
        res[name] = {k: v for k, v in r.items() if not k.startswith("_")}
    # per-seed numbers for the new rule
    res["new_rule_per_seed"] = {}
    for s in DEV:
        sub = [x for x in samples if x["seed"] == s]
        r = score_reference(sub, {s: cf.Scene(occs[s], half=best)})
        res["new_rule_per_seed"][str(s)] = {k: v for k, v in r.items() if not k.startswith("_")}
    # error vs depth for the new rule (what is left)
    z, e = new["_z"], new["_err"]
    res["new_rule_by_depth"] = []
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 5)):
        m = (z >= lo) & (z < hi)
        res["new_rule_by_depth"].append(
            {
                "depth": [lo, hi],
                "n": int(m.sum()),
                "within_0.5px": float(np.mean(np.abs(e[m]) <= 0.5)),
                "median_abs_px": float(np.median(np.abs(e[m]))),
            }
        )
    plot_reference(samples, occs, best, old, new)
    save_results({"task0": res})
    return res


def plot_reference(samples, occs, half, old, new):
    """Left: boundary points back-projected around an L-corner with both segment rules.
    Right: fraction of columns within 0.5 px, old vs new rule, by depth."""
    seed = 7
    occ = occs[seed]
    tex = sim.Textures.find()
    w = sim.Simulator(s0.calibration_textures(tex), seed, msaa=4)
    cx_, cy_ = 0.8, 0.8  # the corner cell (4, 4) of seed 7
    pts = []
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r, c = 4 + dr, 4 + dc
            if not (0 <= r < 31 and 0 <= c < 31) or occ[r, c]:
                continue
            for yaw in np.arange(0, 2 * np.pi, np.pi / 12):
                img = w.render(c * 0.2, r * 0.2, float(yaw))
                rows = cf.boundary_rows(img, np.arange(320), s0.green_floor)
                ok = np.isfinite(rows) & (rows > CAM.cy + 0.3)
                x, z = sc.backproject(np.arange(320)[ok] + 0.5, rows[ok], CAM)
                fwd = np.array([np.cos(yaw), np.sin(yaw)])
                right = np.array([np.sin(yaw), -np.cos(yaw)])
                p = (
                    np.array([c * 0.2, r * 0.2])[None]
                    + z[:, None] * fwd[None]
                    + x[:, None] * right[None]
                )
                pts.append(p[(abs(p[:, 0] - cx_) < 0.35) & (abs(p[:, 1] - cy_) < 0.35)])
    P = np.concatenate(pts)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    for a, name, segs in (
        (ax[0], "stage 0 rule: corners extend 0.1 m past the centre", old_wall_segments(occ)),
        (ax[1], "new rule: segments join centres; only free ends extend", cf.wall_segments(occ)),
    ):
        a.plot(P[:, 0], P[:, 1], "b.", ms=1.2)
        for s in segs:
            a.plot([s[0], s[2]], [s[1], s[3]], "r-", lw=1.5, alpha=0.8)
        a.set_xlim(cx_ - 0.35, cx_ + 0.35)
        a.set_ylim(cy_ - 0.35, cy_ + 0.35)
        a.set_aspect("equal")
        a.set_title(name, fontsize=9)
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 5)]
    for name, r, off in (("stage 0 rule", old, -0.2), ("new rule", new, 0.2)):
        z, e = r["_z"], r["_err"]
        ax[2].bar(
            np.arange(len(bins)) + off,
            [np.mean(np.abs(e[(z >= lo) & (z < hi)]) <= 0.5) for lo, hi in bins],
            0.4,
            label=name,
        )
    ax[2].set_xticks(range(len(bins)), [f"{lo}-{hi} m" for lo, hi in bins])
    ax[2].set_ylabel("columns within 0.5 px of the reference")
    ax[2].set_xlabel("depth")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage2_reference_check.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------- accuracy evaluation


def point_segment_distance(p: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance from each point (n, 2) to the nearest of the segments (m, 4)."""
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    ap = p[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.hypot(*(p[:, None, :] - q).transpose(2, 0, 1)).min(1)


def evaluate_seed(seed, scans, half, step=3) -> dict:
    """Per-column records for every ``step``-th frame: measured and true range, distance of
    the endpoint to the nearest wall segment, all at the true pose."""
    occ = occupancy(seed)
    scene = cf.Scene(occ, half=half)
    poses = record_gt.load_poses(record_gt.gt_dir(seed))
    cols = np.arange(320)
    rec = {k: [] for k in ("rng", "rng_true", "dist", "status", "frame", "u")}
    for d in range(3):
        rows, status = scans[(seed, d)]
        for k in range(0, len(rows), step):
            x, y, yaw = poses[d][k]
            xx, zz = columns_xz(rows[k][None])
            xx, zz = xx[0], zz[0]
            hit = (status[k] == sg.WALL_RETURN) & np.isfinite(zz)
            ztrue = cf.raycast_depths(scene, (x, y), float(yaw), cols, CAM.fx, CAM.cx)
            rng_true = ztrue * np.sqrt(1 + ((cols + 0.5 - CAM.cx) / CAM.fx) ** 2)
            rng = np.hypot(xx, zz)
            dist = np.full(320, np.nan)
            if hit.any():
                p_robot = np.stack([zz[hit], -xx[hit]], 1)
                pw = se2.compose(np.array([x, y, yaw]), np.c_[p_robot, np.zeros(len(p_robot))])[
                    :, :2
                ]
                dist[hit] = point_segment_distance(pw, scene.segments)
            rec["rng"].append(rng)
            rec["rng_true"].append(rng_true)
            rec["dist"].append(dist)
            rec["status"].append(status[k])
            rec["frame"].append(np.full(320, d * 100000 + k))
            rec["u"].append(cols)
    return {k: np.concatenate(v) for k, v in rec.items()}


def range_bins(r, rec, edges):
    """Per true-range bin: count, median/p95 endpoint-to-segment distance, median signed
    relative range error."""
    ok = (rec["status"] == sg.WALL_RETURN) & np.isfinite(rec["rng"]) & np.isfinite(rec["rng_true"])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = ok & (rec["rng_true"] >= lo) & (rec["rng_true"] < hi)
        if m.sum() < 20:
            out.append({"lo": lo, "hi": hi, "n": int(m.sum())})
            continue
        rel = (rec["rng"][m] - rec["rng_true"][m]) / rec["rng_true"][m]
        d = rec["dist"][m]
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(m.sum()),
                "dist_median_m": float(np.nanmedian(d)),
                "dist_p95_m": float(np.nanpercentile(d, 95)),
                "rel_bias_median": float(np.median(rel)),
                "rel_err_abs_p95": float(np.percentile(np.abs(rel), 95)),
            }
        )
    return out


def choose_r_max(bins, dist_med=0.03, dist_p95=0.10, bias=0.01) -> float:
    """The largest bin edge such that every bin up to it meets the three criteria."""
    r = 0.0
    for b in bins:
        if (
            b["n"] < 20
            or b["dist_median_m"] >= dist_med
            or b["dist_p95_m"] >= dist_p95
            or abs(b["rel_bias_median"]) > bias
        ):
            break
        r = b["hi"]
    return r


def evaluate(force=False) -> dict:
    half = load_results()["task0"]["chosen_half_thickness_m"]
    scans = run_scans(DEV + TEST, force)
    edges = list(np.arange(0.0, 5.01, 0.25))
    recs = {}
    for seed in DEV + TEST:
        f = CACHE / f"stage2_eval_{seed}_{code_hash()}_{half}.npz"
        if f.exists() and not force:
            recs[seed] = dict(np.load(f))
        else:
            recs[seed] = evaluate_seed(seed, scans, half)
            np.savez(f, **recs[seed])
        print("evaluated", seed, flush=True)
    cat = lambda seeds: {k: np.concatenate([recs[s][k] for s in seeds]) for k in recs[seeds[0]]}  # noqa: E731
    dev, test = cat(DEV), cat(TEST)
    dev_bins = range_bins(None, dev, edges)
    r_max = choose_r_max(dev_bins)
    test_bins = range_bins(None, test, edges)
    res = {
        "half_thickness_m": half,
        "r_max_chosen_on_dev": r_max,
        "dev_bins": dev_bins,
        "test_bins": test_bins,
    }

    def summary(rec, r_max):
        hit = (rec["status"] == sg.WALL_RETURN) & np.isfinite(rec["rng"]) & (rec["rng"] <= r_max)
        d = rec["dist"][hit]
        rel = (rec["rng"][hit] - rec["rng_true"][hit]) / rec["rng_true"][hit]
        return {
            "returns": int(hit.sum()),
            "dist_median_m": float(np.nanmedian(d)),
            "dist_p95_m": float(np.nanpercentile(d, 95)),
            "dist_p99_m": float(np.nanpercentile(d, 99)),
            "rel_bias_median": float(np.median(rel)),
        }

    res["dev_summary"] = summary(dev, r_max)
    res["test_summary"] = summary(test, r_max)
    res["per_test_seed"] = {str(s): summary(recs[s], r_max) for s in TEST}
    # coverage: fraction of columns per frame with a hit within r_max, mean over frames
    for name, seeds in (("dev", DEV), ("test", TEST)):
        cov, any_ret, far, inv = [], [], [], []
        for s in seeds:
            for d in range(3):
                rows, status = scans[(s, d)]
                xx, zz = columns_xz(rows)
                rng = np.hypot(xx, zz)
                hit = (status == sg.WALL_RETURN) & (rng <= r_max)
                cov.append(hit.mean(1))
                any_ret.append((status == sg.WALL_RETURN).mean(1))
                far.append((status == sg.FAR).mean(1))
                inv.append((status == sg.INVALID).mean(1))
        c = np.concatenate(cov)
        res[f"coverage_{name}"] = {
            "mean_hits_within_r_max": float(c.mean()),
            "p5": float(np.percentile(c, 5)),
            "min": float(c.min()),
            "mean_wall_returns_any_range": float(np.concatenate(any_ret).mean()),
            "mean_far_plane": float(np.concatenate(far).mean()),
            "mean_invalid": float(np.concatenate(inv).mean()),
        }
    # r_max sweep: coverage and accuracy for a few candidates (dev), for the report
    sweep = []
    for rm in (1.5, 2.0, 2.5, 3.0, 3.5, 5.0):
        s_ = summary(dev, rm)
        cvs = np.concatenate(
            [
                (
                    (scans[(s, d)][1] == sg.WALL_RETURN)
                    & (np.hypot(*columns_xz(scans[(s, d)][0])) <= rm)
                ).mean(1)
                for s in DEV
                for d in range(3)
            ]
        )
        sweep.append({"r_max": rm, **s_, "coverage": float(cvs.mean())})
    res["r_max_sweep_dev"] = sweep
    save_results({"evaluate": res})
    plot_accuracy(dev_bins, test_bins, r_max)
    return res


def plot_accuracy(dev_bins, test_bins, r_max):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for bins, name, c in ((dev_bins, "dev", "tab:blue"), (test_bins, "test", "tab:orange")):
        b = [x for x in bins if x["n"] >= 20]
        mid = [(x["lo"] + x["hi"]) / 2 for x in b]
        ax[0].plot(mid, [x["dist_median_m"] for x in b], "o-", color=c, label=f"{name} median")
        ax[0].plot(mid, [x["dist_p95_m"] for x in b], "s--", color=c, label=f"{name} p95")
        ax[1].plot(mid, [100 * x["rel_bias_median"] for x in b], "o-", color=c, label=name)
        ax[2].semilogy(mid, [x["n"] for x in bins if x["n"] >= 20], "o-", color=c, label=name)
    ax[0].axhline(0.03, color="g", lw=0.8)
    ax[0].axhline(0.10, color="g", lw=0.8)
    ax[0].set_ylabel("distance endpoint to nearest wall segment [m]")
    ax[1].axhline(1, color="g", lw=0.8)
    ax[1].axhline(-1, color="g", lw=0.8)
    ax[1].set_ylabel("median signed range error [% of range]")
    ax[2].set_ylabel("returns per bin")
    for a in ax:
        a.axvline(r_max, color="r", ls=":", label=f"r_max {r_max} m")
        a.set_xlabel("true range [m]")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_range_error.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------------- masks


def masks(n=50) -> dict:
    """50 random test-seed frames: contact sheet, and an exact check against the true floor
    (solid-green render of the same pose): wall pixels labelled floor / wall area."""
    tex = sim.Textures.find()
    cal = s0.calibration_textures(tex)
    rng = np.random.default_rng(2)
    picks = []
    for seed in TEST:
        for d in range(3):
            n_f = len(
                json.loads((record_gt.gt_dir(seed) / f"traj_{d}" / "data_info.json").read_text())
            )
            picks += [(seed, d, int(k)) for k in rng.choice(n_f, 4, replace=False)]
    picks = [picks[i] for i in rng.permutation(len(picks))[:n]]
    worlds = {s: sim.Simulator(cal, s, msaa=4) for s in TEST}
    rows_out = []
    fig, axs = plt.subplots(5, 10, figsize=(30, 12))
    for ax, (seed, d, k) in zip(axs.ravel(), picks, strict=True):
        path = record_gt.gt_dir(seed) / f"traj_{d}"
        img = cv2.imread(str(path / f"{k}.jpg"))
        P = record_gt.load_poses(record_gt.gt_dir(seed))[d][k]
        truth = worlds[seed].render(float(P[0]), float(P[1]), float(P[2]))
        t_floor = s0.green_floor(truth)
        t_sky = sg.sky_mask(truth)
        t_wall = ~t_floor & ~t_sky
        # drop the one-pixel antialiased rim around the true floor from the wall area
        rim = cv2.dilate(t_floor.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        t_wall_in = t_wall & ~rim
        s = sg.segment(img)
        wall_as_floor = float((t_wall_in & s.floor).sum() / max(t_wall_in.sum(), 1))
        floor_as_wall = float(
            (
                t_floor
                & ~s.floor
                & ~cv2.dilate((~t_floor).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ).sum()
            / max(t_floor.sum(), 1)
        )
        sky_err = float((t_sky & ~s.sky).sum() / max(t_sky.sum(), 1))
        rows_out.append(
            {
                "seed": seed,
                "drive": d,
                "k": k,
                "wall_labelled_floor": wall_as_floor,
                "floor_labelled_wall": floor_as_wall,
                "sky_missed": sky_err,
            }
        )
        over = img.copy()
        over[s.floor] = (0.55 * over[s.floor] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        over[s.sky] = (0.7 * over[s.sky] + 0.3 * np.array([255, 120, 0])).astype(np.uint8)
        ax.imshow(over[..., ::-1])
        ok = np.isfinite(s.rows)
        ax.plot(np.arange(320)[ok] + 0.5, s.rows[ok], "r.", ms=1)
        ax.set_title(f"s{seed} d{d} #{k}: wall->floor {100 * wall_as_floor:.1f}%", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "stage2_mask_contact_sheet.png", dpi=60)
    plt.close(fig)
    a = np.array([r["wall_labelled_floor"] for r in rows_out])
    res = {
        "frames": rows_out,
        "max_wall_labelled_floor": float(a.max()),
        "median_wall_labelled_floor": float(np.median(a)),
        "frames_over_5pct": [r for r in rows_out if r["wall_labelled_floor"] > 0.05],
        "median_floor_labelled_wall": float(
            np.median([r["floor_labelled_wall"] for r in rows_out])
        ),
        "max_floor_labelled_wall": float(np.max([r["floor_labelled_wall"] for r in rows_out])),
        "max_sky_missed": float(np.max([r["sky_missed"] for r in rows_out])),
    }
    save_results({"masks": res})
    return res


# ---------------------------------------------------------------------------------- oracle

RES = 0.05
ORIGIN = np.array([-0.225, -0.225])  # wall planes (multiples of 0.2 m) fall on cell centres
GRID = 134  # covers 0 .. 6.5 m with a margin
L_OCC, L_FREE, L_CLAMP = 0.85, -0.4, 5.0


def build_map(seed, scans, r_max, poses=None, step_cols=1) -> np.ndarray:
    """Log-odds occupancy grid from scans placed at ``poses`` (true poses for the oracle)."""
    poses = poses if poses is not None else record_gt.load_poses(record_gt.gt_dir(seed))
    occ_cnt = np.zeros(GRID * GRID)
    free_cnt = np.zeros(GRID * GRID)
    cols = np.arange(0, 320, step_cols)
    for d in range(3):
        rows, status = scans[(seed, d)]
        xx, zz = columns_xz(rows)
        for k in range(len(rows)):
            x, y, yaw = poses[d][k]
            st = status[k][cols]
            rng_ = np.hypot(xx[k][cols], zz[k][cols])
            hit = (st == sg.WALL_RETURN) & (rng_ <= r_max)
            free_len = np.zeros(len(cols))
            free_len[hit] = rng_[hit] - RES  # free strictly before the end cell
            beyond = ((st == sg.WALL_RETURN) & (rng_ > r_max)) | (st == sg.FAR)
            free_len[beyond] = r_max
            bearing = np.arctan2(xx[k][cols], zz[k][cols])  # right positive
            th = yaw - bearing  # world direction of each ray
            nsteps = int(r_max / (RES / 2)) + 1
            t = np.arange(nsteps) * (RES / 2)
            sel = t[None, :] <= free_len[:, None]
            px = x + np.cos(th)[:, None] * t[None, :]
            py = y + np.sin(th)[:, None] * t[None, :]
            ix = np.floor((px - ORIGIN[0]) / RES).astype(int)
            iy = np.floor((py - ORIGIN[1]) / RES).astype(int)
            m = sel & (ix >= 0) & (ix < GRID) & (iy >= 0) & (iy < GRID)
            idx = np.unique((iy * GRID + ix)[m] + (np.nonzero(m)[0] * GRID * GRID)) % (GRID * GRID)
            # one free update per cell per ray; different rays may update the same cell
            free_cnt += np.bincount(idx, minlength=GRID * GRID)
            if hit.any():
                ex = x + np.cos(th[hit]) * rng_[hit]
                ey = y + np.sin(th[hit]) * rng_[hit]
                ei = np.floor((ex - ORIGIN[0]) / RES).astype(int)
                ej = np.floor((ey - ORIGIN[1]) / RES).astype(int)
                mm = (ei >= 0) & (ei < GRID) & (ej >= 0) & (ej < GRID)
                occ_cnt += np.bincount(np.unique(ej[mm] * GRID + ei[mm]), minlength=GRID * GRID)
    lo = np.clip(occ_cnt * L_OCC + free_cnt * L_FREE, -L_CLAMP, L_CLAMP)
    return lo.reshape(GRID, GRID)


def true_layout_raster(seed) -> np.ndarray:
    """Wall segments rasterised into the same grid (evaluation only)."""
    segs = cf.wall_segments(occupancy(seed))
    img = np.zeros((GRID, GRID), np.uint8)
    for s in segs:
        p0 = ((s[:2] - ORIGIN) / RES).astype(float)
        p1 = ((s[2:] - ORIGIN) / RES).astype(float)
        cv2.line(
            img,
            (int(round(p0[0] - 0.5)), int(round(p0[1] - 0.5))),
            (int(round(p1[0] - 0.5)), int(round(p1[1] - 0.5))),
            1,
            1,
        )
    return img > 0


def iou(a, b, dilate=1) -> float:
    k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    da = cv2.dilate(a.astype(np.uint8), k) > 0
    db = cv2.dilate(b.astype(np.uint8), k) > 0
    return float((da & db).sum() / max((da | db).sum(), 1))


def oracle() -> dict:
    r_max = load_results()["evaluate"]["r_max_chosen_on_dev"]
    scans = run_scans(DEV + TEST)
    res = {}
    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    for ax, seed in zip(axs.ravel(), DEV[:1] + TEST + DEV[1:], strict=False):
        lo = build_map(seed, scans, r_max)
        d = CACHE / str(seed)
        d.mkdir(exist_ok=True)
        np.savez(d / "oracle_map.npz", logodds=lo, resolution=RES, origin=ORIGIN, r_max=r_max)
        truth = true_layout_raster(seed)
        occ = lo > 0
        res[str(seed)] = {
            "iou_vs_true_layout_1cell": iou(occ, truth, 1),
            "iou_2cell": iou(occ, truth, 2),
            "occupied_cells": int(occ.sum()),
            "true_cells": int(truth.sum()),
        }
        ax.imshow(
            np.clip(lo, -5, 5),
            cmap="gray_r",
            origin="lower",
            extent=[ORIGIN[0], ORIGIN[0] + GRID * RES, ORIGIN[1], ORIGIN[1] + GRID * RES],
        )
        ty, tx = np.nonzero(truth)
        ax.plot(ORIGIN[0] + (tx + 0.5) * RES, ORIGIN[1] + (ty + 0.5) * RES, "r.", ms=0.8)
        ax.set_title(
            f"oracle map seed {seed} ({'dev' if seed in DEV else 'test'}), "
            f"IoU {res[str(seed)]['iou_vs_true_layout_1cell']:.2f}"
        )
    fig.tight_layout()
    fig.savefig(FIG / "stage2_oracle_maps.png", dpi=70)
    plt.close(fig)
    save_results({"oracle": res})
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["scans", "reference", "evaluate", "masks", "oracle", "all"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.step in ("scans", "all"):
        run_scans(DEV + TEST, a.force)
    if a.step in ("reference", "all"):
        r = task0()
        print(
            json.dumps({k: v for k, v in r.items() if k != "half_thickness_sweep"}, indent=1)[:2500]
        )
    if a.step in ("evaluate", "all"):
        r = evaluate(a.force)
        print(
            json.dumps(
                {k: v for k, v in r.items() if k not in ("dev_bins", "test_bins")}, indent=1
            )[:3500]
        )
    if a.step in ("masks", "all"):
        r = masks()
        print({k: v for k, v in r.items() if k != "frames"})
    if a.step in ("oracle", "all"):
        print(json.dumps(oracle(), indent=1))


if __name__ == "__main__":
    main()
