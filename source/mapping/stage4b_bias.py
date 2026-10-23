"""Stage 4b task 2: where does the forward scale bias come from? EVALUATION CODE (reads ground
truth); it only calls the estimators of ``vo.py``.

Straight pairs = label FORWARD after FORWARD with a true rotation under 1 degree and a true
forward step of at least 5 cm. The bias of an estimator is ``sum(est dx) / sum(true dx) - 1``
over the pairs whose estimate is within 5 cm of the truth (a corridor alias, about 1 % of pairs,
is the gate's job, not a bias); 80 pairs per drive.

    uv run source/mapping/stage4b_bias.py estimators [dev|test]   # before / after, each estimator
    uv run source/mapping/stage4b_bias.py rows                    # scan rows against ray-cast truth
    uv run source/mapping/stage4b_bias.py synthetic               # noise-free vs noisy scans
    uv run source/mapping/stage4b_bias.py arms                    # ICP and wall-feature ablations
"""

# ruff: noqa: E501
from __future__ import annotations

import itertools
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_fit as cf  # noqa: E402
import motion as M  # noqa: E402
import record_gt  # noqa: E402
import scan as sc  # noqa: E402
import se2  # noqa: E402
import segment as sg  # noqa: E402
import stage2 as s2  # noqa: E402
import stage4 as s4  # noqa: E402
import vo  # noqa: E402

CACHE = record_gt.ROOT / "cache" / "mapping"
CAM = sc.camera_config()
PER_DRIVE = 80  # straight pairs sampled per drive
COLS = np.arange(320)


def straight_pairs(dr, rng):
    lab = dr["inp"].labels
    inc = dr["true_inc"]
    ok = [
        k
        for k in range(1, len(lab))
        if lab[k] == "FORWARD"
        and lab[k - 1] == "FORWARD"
        and abs(np.degrees(inc[k, 2])) < 1.0
        and inc[k, 0] >= 0.05
        and inc[k - 1, 0] > 0.05
    ]
    return sorted(rng.choice(ok, min(PER_DRIVE, len(ok)), replace=False)) if ok else []


def summarise(est, true):
    est, true = np.asarray(est), np.asarray(true)
    if len(est) == 0:
        return {}
    idx = [np.random.default_rng(b).integers(0, len(est), len(est)) for b in range(200)]
    boot = [est[i].sum() / true[i].sum() - 1 for i in idx]
    return {
        "n": int(len(est)),
        "bias_pct": float(100 * (est.sum() / true.sum() - 1)),
        "se_pct": float(100 * np.std(boot)),
        "median_ratio_pct": float(100 * (np.median(est / true) - 1)),
    }


def collect(seeds, variant="gt4"):
    """Straight pairs with both scans, the matches and the true increment."""
    rng = np.random.default_rng(0)
    out = []
    for dr in s4.load_inputs(seeds, variant):
        inp = dr["inp"]
        for k in straight_pairs(dr, rng):
            out.append(
                {
                    "p0": vo.scan_points(inp.rows[k - 1], inp.top[k - 1], inp.status[k - 1]),
                    "p1": vo.scan_points(inp.rows[k], inp.top[k], inp.status[k]),
                    "rows": (inp.rows[k - 1 : k + 1], inp.top[k - 1 : k + 1], inp.status[k - 1 : k + 1]),
                    "m": inp.matches[k],
                    "t": dr["true_inc"][k],
                }
            )
    return out


def bias_of(fn, data):
    e, t = [], []
    for d in data:
        r = fn(d)
        if r is not None and abs(r[0] - d["t"][0]) < 0.05:
            e.append(r[0])
            t.append(d["t"][0])
    return {**summarise(e, t), "dropped_frac": float(1 - len(e) / max(len(data), 1))}


def estimators(split="dev"):
    seeds = s4.DEV if split == "dev" else s4.TEST
    data = collect(seeds)
    g = M.MotionPrior.load().get("FORWARD", "FORWARD")
    out = {
        "icp (stage 4), from truth": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp(d["p0"], d["p1"], d["t"])), data),
        "icp (stage 4), from zero": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp(d["p0"], d["p1"], np.zeros(3))), data),
        "icp (stage 4), from label mean": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp(d["p0"], d["p1"], g.mean)), data),
        "icp (stage 4), from truth x0.8": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp(d["p0"], d["p1"], d["t"] * [0.8, 1, 1])), data),
        "icp (stage 4), from truth x1.2": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp(d["p0"], d["p1"], d["t"] * [1.2, 1, 1])), data),
        "icp symmetric (4b), from label mean": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp_symmetric(d["p0"], d["p1"], g.mean)), data),
        "icp symmetric (4b), from zero": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.icp_symmetric(d["p0"], d["p1"], np.zeros(3))), data),
        "wall features, row ratio (stage 4)": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.wall_motion(d["m"], d["p0"])), data),
        "wall features, two scans (4b)": bias_of(lambda d: (lambda m: None if m is None else m.mean)(vo.wall_motion(d["m"], d["p0"], scan_cur=d["p1"])), data),
    }
    return out


# ------------------------------------------------------------------ rows against ray-cast truth


def rows_vs_truth(seeds=None, variant="gt4"):
    """Signed error [px] of the cached floor-edge rows against the ray-cast truth, by range."""
    seeds = seeds or s4.DEV
    D = s4.s3.load(seeds, variant, keep_scans=True)
    res = []
    for s in seeds:
        scene = cf.Scene(s2.occupancy(s), half=0.0022)
        P = record_gt.load_poses(record_gt.gt_dir(s))
        for d in range(3):
            for i in np.where((D["seed"] == s) & (D["drive"] == d))[0][::4]:
                x, y, yaw = P[d][int(D["k"][i])]
                z = cf.raycast_depths(scene, (x, y), float(yaw), COLS, CAM.fx, CAM.cx)
                ok = (D["status"][i] == sg.WALL_RETURN) & np.isfinite(z) & (z < 3) & (z > 0.3)
                res.append(np.stack([z[ok], D["rows"][i][ok] - (CAM.cy + CAM.fy * CAM.h / z[ok])], 1))
    R = np.concatenate(res)
    out = {}
    bins = [0.3, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        e = R[(R[:, 0] >= a) & (R[:, 0] < b), 1]
        e = e[np.abs(e) < 1.0]
        out[f"{a}-{b} m"] = {"n": int(len(e)), "mean_px": float(e.mean()), "median_px": float(np.median(e))}
    return out


def row_offset_sensitivity(data, deltas=(-0.1, -0.05, 0, 0.05, 0.1)):
    """ICP bias when every boundary row is shifted by ``-delta`` pixels (what a row offset does)."""
    out = {}
    for dl in deltas:
        def fn(d, dl=dl):
            (r, tp, st) = d["rows"]
            m = vo.icp(vo.scan_points(r[0] - dl, tp[0], st[0]), vo.scan_points(r[1] - dl, tp[1], st[1]), d["t"])
            return None if m is None else m.mean
        out[str(dl)] = bias_of(fn, data)
    return out


# ------------------------------------------------------------------------- synthetic scans


def synth_rows(scene, pose, noise, rng):
    x, y, yaw = pose
    z = cf.raycast_depths(scene, (x, y), float(yaw), COLS, CAM.fx, CAM.cx)
    ok = np.isfinite(z) & (z < 5) & (z > 0.2)
    rows = np.full(320, np.nan)
    rows[ok] = CAM.cy + CAM.fy * CAM.h / z[ok] + noise * rng.normal(size=ok.sum())
    return rows, np.where(ok, sg.WALL_RETURN, 0)


_COUNT = itertools.count()


def variant_module(replacements):
    """``vo`` with source edits, as a fresh module (ablations without touching vo.py)."""
    src = (Path(__file__).resolve().parent / "vo.py").read_text()
    for a, b in replacements:
        assert a in src, a
        src = src.replace(a, b)
    name = f"vo_ablation_{next(_COUNT)}"
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


W = "w = 1.0 / (1.0 + z**2) / sig**2"
N = "agree = ~np.isfinite(sn[:, 0]) | (np.abs((sn_rot * tn[j[m]]).sum(1)) >= NORMAL_COS)"
ARMS = {
    "stage 4 ICP": [],
    "no Cauchy weights": [(W, "w = 1.0 / sig**2")],
    "Huber weights": [(W, "w = np.where(np.abs(z)<2.0,1.0,2.0/np.abs(z)) / sig**2")],
    "no normal-agreement check": [(N, "agree = np.ones(len(sn),bool)")],
    "no initial-guess prior": [("INIT_SIGMA = np.array([0.15, 0.15, np.radians(12.0)])", "INIT_SIGMA = np.array([5.0, 5.0, 1.0])")],
}


def synthetic(noise=0.3, n_per_drive=25):
    """ICP on scans ray-cast at the true poses: exact rows, then rows with ``noise`` px of noise,
    and the ablations on the noisy ones."""
    seeds = s4.DEV
    scenes = {s: cf.Scene(s2.occupancy(s), half=0.0022) for s in seeds}
    rng = np.random.default_rng(0)
    cases = []
    for dr in s4.load_inputs(seeds, "gt4"):
        for k in straight_pairs(dr, rng)[:n_per_drive]:
            T, t = dr["true"], dr["true_inc"][k]
            r = np.random.default_rng(k)
            cases.append((scenes[dr["seed"]], T[k - 1], T[k], t, r))
    nan = np.full(320, np.nan)
    out = {}

    def run(mod, nz, fn="icp"):
        e, tt = [], []
        for scene, a, b, t, r in cases:
            r = np.random.default_rng(int(abs(t[0]) * 1e6))
            ra, sa = synth_rows(scene, a, nz, r)
            rb, sb = synth_rows(scene, b, nz, r)
            m = getattr(mod, fn)(mod.scan_points(ra, nan, sa), mod.scan_points(rb, nan, sb), t)
            if m is not None and abs(m.mean[0] - t[0]) < 0.05:
                e.append(m.mean[0])
                tt.append(t[0])
        return summarise(e, tt)

    out["exact rows"] = run(vo, 0.0)
    out[f"{noise} px row noise"] = run(vo, noise)
    out[f"{noise} px row noise, symmetric ICP"] = run(vo, noise, "icp_symmetric")
    for name, rep in ARMS.items():
        if rep:
            out[f"{noise} px, {name}"] = run(variant_module(rep), noise)
    return out


def arms(data):
    """ICP ablations (forward and symmetric) on the real straight pairs."""
    out = {}
    for name, rep in ARMS.items():
        mod = variant_module(rep)
        for sym in (False, True):
            e, t, rot = [], [], []
            for d in data:
                a = mod.icp(d["p0"], d["p1"], d["t"])
                if a is None:
                    continue
                est = a.mean
                if sym:
                    b = mod.icp(d["p1"], d["p0"], se2.inverse(d["t"]))
                    if b is None:
                        continue
                    est = (a.mean + se2.inverse(b.mean)) / 2
                if abs(est[0] - d["t"][0]) < 0.05:
                    e.append(est[0])
                    t.append(d["t"][0])
                    rot.append(abs(np.degrees(est[2] - d["t"][2])))
            out[f"{name}{' + symmetric' if sym else ''}"] = {
                **summarise(e, t),
                "rot_median_deg": float(np.median(rot)),
                "dx_rmse_mm": float(1000 * np.sqrt(np.mean((np.array(e) - np.array(t)) ** 2))),
            }
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "estimators"
    if cmd == "estimators":
        split = sys.argv[2] if len(sys.argv) > 2 else "dev"
        res = estimators(split)
        name = f"stage4b_bias_{split}.json"
    elif cmd == "rows":
        res = rows_vs_truth()
        name = "stage4b_rows.json"
    elif cmd == "synthetic":
        res = synthetic()
        name = "stage4b_synthetic.json"
    elif cmd == "arms":
        data = collect(s4.DEV)
        res = {"icp_arms": arms(data), "row_offset": row_offset_sensitivity(data)}
        name = "stage4b_arms.json"
    else:
        raise SystemExit(__doc__)
    for k, v in res.items():
        print(f"{k:40s} {v}")
    (CACHE / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
