"""Stage 4b task 3: label-prior self-calibration. EVALUATION CODE (reads ground truth for the
local seeds and the biased run); the method is ``calibration.py``.

    uv run python source/mapping/stage4b_calib.py local [gt4|official]   # fitted scales, every local seed
    uv run python source/mapping/stage4b_calib.py biased                  # data/bias-7, bias-8: recovery and ATE
    uv run python -I source/mapping/stage4b_calib.py challenge            # fitted scales on the challenge data
    uv run python source/mapping/stage4b_calib.py plot                    # recovered vs true figure
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration as cal  # noqa: E402
import motion as M  # noqa: E402
import odometry as od  # noqa: E402
import pipeline  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402
import stage4 as s4  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
BIAS_SEEDS = (7, 8)  # dev seeds re-recorded with a motion bias (record_gt --motion-bias)
CHALLENGE = ROOT / "data" / "ab7efaeb1abf"


def scales_json(scales) -> list[dict]:
    return [
        {
            "speed": d.speed.value,
            "speed_se": d.speed.se,
            "speed_n": d.speed.n,
            "turn": d.turn.value,
            "turn_se": d.turn.se,
            "turn_n": d.turn.n,
            "pooled": [d.speed.pooled, d.turn.pooled],
        }
        for d in scales
    ]


def true_drive_scales(true_poses: np.ndarray, labels, prior) -> dict:
    """Evaluation only: the same two ratios as ``calibration`` on the true increments of every
    pair (no selection), i.e. the drive's real speed and turn against the prior's means."""
    q = np.vstack([od.SPAWN, true_poses])
    inc = se2.relative(q[:-1], q[1:])
    m = np.array([prior.get(lab, labels[k - 1] if k > 0 else "START").mean for k, lab in enumerate(labels)])
    lab = np.array(labels)
    ok = np.abs(inc[:, 2]) < cal.ROT_MAX
    f = ok & (lab == "FORWARD")
    t = ok & np.isin(lab, cal.LABELS_ROT)
    return {
        "speed": float(inc[f, 0].sum() / m[f, 0].sum()),
        "turn": float((inc[t, 2] * m[t, 2]).sum() / (m[t, 2] ** 2).sum()),
    }


def _fit_seed(args):
    seed, variant = args
    prior = M.MotionPrior.load()
    drives = s4.load_inputs([seed], variant)
    sc = scales_json(cal.fit_drives([d["inp"] for d in drives], prior))
    for d, row in zip(drives, sc, strict=True):
        row["true"] = true_drive_scales(d["true"], d["inp"].labels, prior)
    return seed, sc


def local(variant="gt4") -> dict:
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(_fit_seed, [(s, variant) for s in s4.DEV + s4.TEST]))
    return {str(k): v for k, v in out.items()}


def _drive(path):
    return pipeline.drive_inputs(path)


def dataset_inputs(root: Path, cache_name: str):
    f = CACHE / cache_name
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    paths = sorted(root.glob("traj_*"), key=lambda p: int(p.name.split("_")[1]))
    with ProcessPoolExecutor() as ex:
        inputs = list(ex.map(_drive, paths))
    with open(f, "wb") as fh:
        pickle.dump(inputs, fh)
    return inputs


def _run(args):
    inp, prior = args
    return od.run_drive(inp, prior, "C")


def _ate(res, true) -> float:
    return float(np.sqrt(np.mean(np.hypot(*(res.poses[:, :2] - true[:, :2]).T) ** 2)))


def biased(seed: int) -> dict:
    """``data/bias-<seed>`` against ``data/gt4-<seed>`` (same maze and routes, no bias): the
    fitted per-drive scales against the true ones, and variant C's ATE without and with the
    calibration."""
    root = record_gt.gt_dir(seed, prefix="bias")
    inputs = dataset_inputs(root, f"stage4b_bias{seed}_inputs.pkl")
    prior = M.MotionPrior.load()
    cpriors, sc = cal.calibrated_priors(inputs, prior)
    poses = record_gt.load_poses(root)
    base = s4.load_inputs([seed], "gt4")
    bpriors, bsc = cal.calibrated_priors([d["inp"] for d in base], prior)
    with ProcessPoolExecutor() as ex:
        plain = list(ex.map(_run, [(i, prior) for i in inputs]))
        calib = list(ex.map(_run, [(i, p) for i, p in zip(inputs, cpriors, strict=True)]))
        unb = list(ex.map(_run, [(d["inp"], p) for d, p in zip(base, bpriors, strict=True)]))
    out = {
        "drawn_bias_reported_by_simulator": json.loads((root / "gt.json").read_text())["drawn_bias"],
        "fitted": scales_json(sc),
        "true": [true_drive_scales(p, i.labels, prior) for p, i in zip(poses, inputs, strict=True)],
        "fitted_unbiased_run": scales_json(bsc),
        "ate_per_drive": {
            "unbiased_calibrated": [_ate(r, d["true"]) for r, d in zip(unb, base, strict=True)],
            "biased_uncalibrated": [_ate(r, p) for r, p in zip(plain, poses, strict=True)],
            "biased_calibrated": [_ate(r, p) for r, p in zip(calib, poses, strict=True)],
        },
        "heading_rms_deg": {
            "biased_uncalibrated": [float(np.sqrt(np.mean(np.degrees(se2.wrap(r.poses[:, 2] - p[:, 2])) ** 2))) for r, p in zip(plain, poses, strict=True)],
            "biased_calibrated": [float(np.sqrt(np.mean(np.degrees(se2.wrap(r.poses[:, 2] - p[:, 2])) ** 2))) for r, p in zip(calib, poses, strict=True)],
        },
    }
    out["ate_mean"] = {k: float(np.mean(v)) for k, v in out["ate_per_drive"].items()}
    return out


TICK_M, TICK_RAD = 0.03, 0.06  # simulator constants: one tick of FORWARD, of LEFT/RIGHT


def challenge() -> dict:
    """Fitted scales on the challenge download (no ground truth). Stage 4b found that this
    download holds one frame per tick, not per five, so the 5-tick prior does not apply; the
    raw (unclipped) ratios and the measured motion per frame against one tick are reported."""
    inputs = dataset_inputs(CHALLENGE, "stage4b_challenge_inputs.pkl")
    prior = M.MotionPrior.load()
    out = {"fitted": scales_json(cal.fit_drives(inputs, prior)), "per_drive": []}
    for inp in inputs:
        rows = cal.vision_increments(inp, prior)
        t = cal._terms(rows)
        lab = np.array(inp.labels)
        f = np.array([v for la, _, v in rows if la == "FORWARD"])
        r = np.array([v for la, _, v in rows if la in cal.LABELS_ROT])
        dpsi = [
            abs(od.vp_diff90(a, b))
            for a, b, la, lb in zip(inp.psi[1:], inp.psi[:-1], lab[1:], lab[:-1], strict=True)
            if la == lb and la in cal.LABELS_ROT and np.isfinite(a) and np.isfinite(b)
        ]
        out["per_drive"].append(
            {
                "frames": len(lab),
                "same_label_as_previous": float(np.mean(lab[1:] == lab[:-1])),
                "raw_speed_ratio_vs_5tick_prior": float(t["speed"][0].sum() / t["speed"][1].sum()),
                "raw_turn_ratio_vs_5tick_prior": float(t["turn"][0].sum() / t["turn"][1].sum()),
                "forward_dx_per_frame_mean_m": float(f[:, 0].mean()),
                "forward_dx_per_frame_over_one_tick": float(f[:, 0].mean() / TICK_M),
                "turn_per_frame_median_deg": float(np.degrees(np.median(np.abs(r[:, 2])))),
                "turn_per_frame_over_one_tick": float(np.median(np.abs(r[:, 2])) / TICK_RAD),
                "heading_cue_turn_per_frame_median_deg": float(np.median(dpsi)),
                "n_forward": len(f),
                "n_turn": len(r),
            }
        )
    return out


def plot() -> None:
    """Fitted against true per-drive scale: biased runs and every local drive."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = json.loads((CACHE / "stage4b_calib_biased.json").read_text())
    loc = json.loads((CACHE / "stage4b_calib_local_gt4.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, key in zip(ax, ("speed", "turn"), strict=True):
        xs = [d["true"][key] for ds in loc.values() for d in ds]
        ys = [d[key] for ds in loc.values() for d in ds]
        a.plot(xs, ys, "o", ms=4, color="0.6", label="local drives (no bias; pooled per seed)")
        for sd, v in b.items():
            xt = [t[key] for t in v["true"]]
            yf = [f[key] for f in v["fitted"]]
            ye = [f[f"{key}_se"] for f in v["fitted"]]
            a.errorbar(xt, yf, yerr=ye, fmt="s", ms=6, capsize=3, label=f"bias-{sd} drives")
        lo, hi = 0.4, 1.35
        g = np.linspace(lo, hi, 2)
        a.plot(g, g, "k-", lw=0.7)
        a.fill_between(g, g * 0.99, g * 1.01, color="0.85", zorder=0)
        a.set_xlim(lo, hi)
        a.set_ylim(lo, hi)
        a.set_xlabel(f"true {key} scale of the drive (ground truth)")
        a.set_ylabel(f"fitted {key} scale (images and labels)")
        a.set_title(f"{key} (grey band: +/- 1 %)")
        a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "stage4b_calibration_recovery.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "local"
    if cmd == "local":
        variant = sys.argv[2] if len(sys.argv) > 2 else "gt4"
        res, name = local(variant), f"stage4b_calib_local_{variant}.json"
    elif cmd == "biased":
        res = {str(sd): biased(sd) for sd in BIAS_SEEDS}
        name = "stage4b_calib_biased.json"
    elif cmd == "challenge":
        res, name = challenge(), "stage4b_calib_challenge.json"
    elif cmd == "plot":
        plot()
        return
    else:
        raise SystemExit(__doc__)
    print(json.dumps(res, indent=1))
    (CACHE / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
