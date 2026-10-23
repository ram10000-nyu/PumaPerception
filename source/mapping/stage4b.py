"""Stage 4b evaluation: scale metrics and the full stage 4 criteria with the fixed estimators.
EVALUATION CODE (reads ground truth); the pipeline is ``vo.py`` and ``odometry.py``.

    uv run python source/mapping/stage4b.py [gt4] [official]    # A, B, C on dev and test
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion as M  # noqa: E402
import odometry as od  # noqa: E402
import se2  # noqa: E402
import stage4 as s4  # noqa: E402

CACHE, FIG = s4.CACHE, s4.FIG
DEV, TEST = s4.DEV, s4.TEST
SEG_MIN_PAIRS = 10
SEG_MAX_ROT_DEG = 2.0


def umeyama_scale(est: np.ndarray, true: np.ndarray) -> float:
    """Scale of the similarity alignment (rotation, translation, scale) of ``est`` to ``true``."""
    mu_e, mu_t = est.mean(0), true.mean(0)
    e, t = est - mu_e, true - mu_t
    cov = t.T @ e / len(est)
    u, d, vt = np.linalg.svd(cov)
    s = np.array([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    return float((d * s).sum() / ((e**2).sum() / len(est)))


def straight_segments(true: np.ndarray, labels) -> list[tuple[int, int]]:
    """Maximal runs of at least ``SEG_MIN_PAIRS`` consecutive FORWARD pairs whose true heading
    changes by under ``SEG_MAX_ROT_DEG`` in total: ``(first frame, last frame)``."""
    out, a, n = [], 0, len(true)
    while a < n - 1:
        b = a
        while (
            b + 1 < n
            and labels[b + 1] == "FORWARD"
            and abs(np.degrees(se2.wrap(true[b + 1, 2] - true[a, 2]))) < SEG_MAX_ROT_DEG
        ):
            b += 1
        if b - a >= SEG_MIN_PAIRS:
            out.append((a, b))
            a = b
        else:
            a += 1
    return out


def scale_metrics(res, dr) -> dict:
    true = dr["true"]
    sim = umeyama_scale(res.poses[:, :2], true[:, :2])
    ratios = []
    for a, b in straight_segments(true, dr["inp"].labels):
        td = np.hypot(*(true[b, :2] - true[a, :2]))
        if td > 0.3:
            ratios.append(float(np.hypot(*(res.poses[b, :2] - res.poses[a, :2])) / td))
    return {"sim_scale": sim, "segment_ratios": ratios}


def _task(args):
    key, dr, v, scales = args
    prior = M.MotionPrior.load()
    if v == "A":  # the labels alone, uncalibrated: the stage 1 reference
        return key, s4.run_variant_A(dr, prior)
    if v == "C" and scales is not None:
        prior = prior.scaled({lab: tuple(scales) for lab in M.LABELS})
    return key, od.run_drive(dr["inp"], prior, v)


def run(variant: str, calibrated: bool = True):
    """Variants A, B, C on dev and test. With ``calibrated``, C uses each drive's self-calibrated
    prior (``calibration.fit_drives``, cached by ``stage4b_calib.py local``)."""
    drives = s4.load_inputs(DEV + TEST, variant)
    cal = {}
    if calibrated:
        fits = json.loads((CACHE / f"stage4b_calib_local_{variant}.json").read_text())
        cal = {(int(s), d): (f["speed"], f["turn"]) for s, ds in fits.items() for d, f in enumerate(ds)}
    tasks = [
        ((dr["seed"], dr["drive"], v), dr, v, cal.get((dr["seed"], dr["drive"])))
        for dr in drives
        for v in ("A", "B", "C")
    ]
    with ProcessPoolExecutor() as ex:
        results = dict(ex.map(_task, tasks))
    return drives, results


def summarise(variant, drives, results) -> dict:
    out: dict = {"per_drive": {}, "windows": {}, "scale": {}}
    for dr in drives:
        for v in ("A", "B", "C"):
            key = f"{dr['seed']}-{dr['drive']}-{v}"
            out["per_drive"][key] = {
                **s4.score(results[(dr["seed"], dr["drive"], v)], dr),
                **scale_metrics(results[(dr["seed"], dr["drive"], v)], dr),
            }
    for split, seeds in (("dev", DEV), ("test", TEST)):
        for v in ("B", "C"):
            et, dist, er = s4.pooled_windows(results, drives, v, seeds, 10)
            m = dist >= s4.WINDOW_MIN_DIST
            out["windows"][f"{split}-{v}"] = {
                "windows": int(m.sum()),
                "trans_median_pct": float(np.median(et[m] / dist[m]) * 100),
                "trans_p95_pct": float(np.percentile(et[m] / dist[m], 95) * 100),
                "rot_median_deg": float(np.median(er)),
            }
        rows = [
            out["per_drive"][f"{dr['seed']}-{dr['drive']}-C"]
            for dr in drives
            if dr["seed"] in seeds
        ]
        sims = np.array([r["sim_scale"] for r in rows])
        segs = np.concatenate([r["segment_ratios"] for r in rows])
        pooled = sum(r["path_len_est"] for r in rows) / sum(r["path_len_true"] for r in rows)
        out["scale"][split] = {
            "drives": len(rows),
            "sim_scale_median_dev_pct": float(100 * (np.median(sims) - 1)),
            "sim_scale_worst_abs_dev_pct": float(100 * np.abs(sims - 1).max()),
            "sim_scale_all_pct": [float(100 * (x - 1)) for x in sims],
            "segments": int(len(segs)),
            "segment_ratio_median_dev_pct": float(100 * (np.median(segs) - 1)),
            "segment_ratio_mean_dev_pct": float(100 * (np.mean(segs) - 1)),
            "path_ratio_pooled_dev_pct": float(100 * (pooled - 1)),
        }
        out["scale"][split]["heading_rms_worst_deg"] = float(
            max(r["heading_rms_deg"] for r in rows)
        )
        out["scale"][split]["ate_median_m"] = float(np.median([r["ate_rms_m"] for r in rows]))
        nis = np.concatenate(
            [
                results[(dr["seed"], dr["drive"], "C")].nis[
                    np.isfinite(results[(dr["seed"], dr["drive"], "C")].nis)
                ]
                for dr in drives
                if dr["seed"] in seeds
            ]
        )
        out["scale"][split]["nis_frac_in_bound"] = float(np.mean(nis < 7.81))
    ev_t = s4.event_table([d for d in drives if d["seed"] in TEST], results, "C")
    ev_d = s4.event_table([d for d in drives if d["seed"] in DEV], results, "C")
    for name, ev in (("test", ev_t), ("dev", ev_d)):
        out[f"events_{name}"] = ev
        out["scale"][name]["collisions_ok"] = f"{sum(e['C']['heading_err_5_after_deg'] < 5 for e in ev)} / {len(ev)}"
        out["scale"][name]["collisions_B_ok"] = f"{sum(e['B']['heading_err_5_after_deg'] < 5 for e in ev)} / {len(ev)}"
    out["confidently_wrong_test"] = s4.confidently_wrong(
        [d for d in drives if d["seed"] in TEST], results, "C"
    )
    ate = {
        s: {v: float(np.mean([out["per_drive"][f"{s}-{d}-{v}"]["ate_rms_m"] for d in range(3)])) for v in "ABC"}
        for s in DEV + TEST
    }
    out["ate_by_seed"] = ate
    out["C_beats_A_everywhere_test"] = all(
        out["per_drive"][f"{s}-{d}-C"]["ate_rms_m"] < out["per_drive"][f"{s}-{d}-A"]["ate_rms_m"]
        for s in TEST
        for d in range(3)
    )
    return out


def plot_scale(allres):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, variant in zip(ax, allres, strict=False):
        for i, split in enumerate(("dev", "test")):
            vals = allres[variant]["scale"][split]["sim_scale_all_pct"]
            a.plot(np.full(len(vals), i) + np.linspace(-0.2, 0.2, len(vals)), vals, "o", ms=4)
        a.axhspan(-1, 1, color="0.9", zorder=0)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks([0, 1], ["dev", "test"])
        a.set_ylabel("similarity-alignment scale - 1 [%]")
        a.set_title(f"variant C, {variant} (grey: +/- 1 %)")
    fig.tight_layout()
    fig.savefig(FIG / "stage4b_similarity_scale.png", dpi=130)
    plt.close(fig)


def plot_bias():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [
        ("icp (stage 4), from label mean", "ICP (stage 4)"),
        ("icp symmetric (4b), from label mean", "ICP symmetric (4b)"),
        ("wall features, row ratio (stage 4)", "wall features (stage 4)"),
        ("wall features, two scans (4b)", "wall features (4b)"),
    ]
    w = 0.38
    for i, split in enumerate(("dev", "test")):
        d = json.loads((CACHE / f"stage4b_bias_{split}.json").read_text())
        ax.bar(np.arange(len(names)) + (i - 0.5) * w, [d[k]["bias_pct"] for k, _ in names], w, yerr=[d[k]["se_pct"] for k, _ in names], label=split)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(names)), [n for _, n in names], rotation=12)
    ax.set_ylabel("bias of forward displacement [%]")
    ax.set_title("straight FORWARD pairs, gt4 (error bars: bootstrap s.e.)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage4b_bias_before_after.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / "stage4b_results.json"
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}
    for variant in sys.argv[1:] or s4.VARIANTS:
        drives, results = run(variant)
        with open(CACHE / f"stage4b_results_{variant}.pkl", "wb") as fh:
            pickle.dump(results, fh)
        allres[variant] = summarise(variant, drives, results)
        print(variant, json.dumps(allres[variant]["scale"], indent=1), flush=True)
        print(json.dumps(allres[variant]["windows"], indent=1), flush=True)
    out_path.write_text(json.dumps(allres, indent=1))
    plot_scale(allres)
    plot_bias()


if __name__ == "__main__":
    main()
