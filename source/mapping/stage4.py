"""Stage 4 evaluation: runs the odometry variants on the cached cues and scores them against
the true poses. EVALUATION CODE (reads ground truth); the pipeline is ``vo.py`` and
``odometry.py``.

    uv run source/mapping/stage4.py            # variants A, B, C on dev and test, both frame types
"""

# ruff: noqa: E501  (evaluation tables and plots: long literals read better on one line)
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
import collision as col  # noqa: E402
import motion as M  # noqa: E402
import odometry as od  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402
import stage3 as s3  # noqa: E402
import stage3b as s3b  # noqa: E402
import stage4_extract as x4  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
FIG = ROOT / "reports" / "figures"
DEV, TEST = s3.DEV, s3.TEST
VARIANTS = ("gt4", "official")
WINDOW_MIN_DIST = 0.5  # m: windows shorter than this make a percentage meaningless
WINDOW_LENGTHS = (1, 2, 5, 10, 20, 50)


def load_inputs(seeds, variant):
    """Per-drive filter inputs from the cached cues, plus the true poses for scoring."""
    D = s3.load(seeds, variant, keep_scans=True)
    F = s3.fused(D)
    Pr = M.load_pairs(seeds)
    D2, _, _, psi_best = s3b.drive_arrays(seeds, variant)
    flags, _ = s3b.flags_for(D2, Pr, psi_best)
    out = []
    for s in seeds:
        matches = x4.run_seed(s, variant)
        poses = record_gt.load_poses(record_gt.gt_dir(s))
        for d in range(3):
            idx = np.where((D["seed"] == s) & (D["drive"] == d))[0]
            inp = od.DriveInputs(
                rows=D["rows"][idx],
                top=D["top"][idx],
                status=D["status"][idx],
                psi=np.where(F["valid"][idx], F["psi"][idx], np.nan),
                psi_sigma=np.where(F["valid"][idx], F["sigma"][idx], np.nan),
                labels=list(Pr["label"][idx]),
                flags=flags[idx],
                matches=matches[d],
            )
            out.append(
                {
                    "seed": s,
                    "drive": d,
                    "inp": inp,
                    "true": poses[d],
                    "true_inc": Pr["inc"][idx],
                    "tp": D["tp"][idx],
                    "fused_valid": F["valid"][idx],
                    "fused_psi": F["psi"][idx],
                }
            )
    return out


def run_variant_A(dr, prior):
    """Variant A: the stage 1 dead reckoning from the spawn with the (previous, current)
    prior means."""
    inc = prior.mean_increments(dr["inp"].labels)
    poses = se2.integrate(od.SPAWN, inc)
    return od.DriveResult(
        poses=poses,
        covs=np.zeros((len(poses), 3, 3)),
        inc_mean=inc,
        inc_cov=np.zeros((len(poses), 3, 3)),
        nis=np.full(len(poses), np.nan),
        used_icp=np.zeros(len(poses), bool),
        heading_accepted=np.zeros(len(poses), bool),
        wall_used=np.zeros(len(poses), bool),
        icp_sigma_along=np.full(len(poses), np.nan),
        heading_valid=np.zeros(len(poses), bool),
        heading_innovation_deg=np.full(len(poses), np.nan),
    )


def _task(args):
    key, dr, variant = args
    prior = M.MotionPrior.load()
    if variant == "A":
        return key, run_variant_A(dr, prior)
    return key, od.run_drive(dr["inp"], prior, variant)


def window_errors(res, true, length):
    """Relative translation error [m], true distance [m] and rotation error [deg] over windows
    of ``length`` frames (stride ``max(1, length // 2)``)."""
    n = len(true)
    step = np.hypot(*np.diff(np.vstack([od.SPAWN[:2], true[:, :2]]), axis=0).T)
    et, dist, er = [], [], []
    for a in range(0, n - length, max(1, length // 2)):
        b = a + length
        ti = se2.relative(true[a], true[b])
        ei = se2.relative(res.poses[a], res.poses[b])
        et.append(float(np.hypot(*(ei[:2] - ti[:2]))))
        er.append(abs(np.degrees(se2.wrap(ei[2] - ti[2]))))
        dist.append(float(step[a + 1 : b + 1].sum()))
    return np.array(et), np.array(dist), np.array(er)


def score(res, dr) -> dict:
    true = dr["true"]
    herr = np.degrees(se2.wrap(res.poses[:, 2] - true[:, 2]))
    perr = np.hypot(*(res.poses[:, :2] - true[:, :2]).T)
    step = np.hypot(*np.diff(np.vstack([od.SPAWN[:2], true[:, :2]]), axis=0).T)
    est_step = np.hypot(*res.inc_mean[:, :2].T)
    et, dist, er = window_errors(res, true, 10)
    mv = dist >= WINDOW_MIN_DIST
    nis = res.nis[np.isfinite(res.nis)]
    return {
        "frames": int(len(true)),
        "heading_rms_deg": float(np.sqrt(np.mean(herr**2))),
        "heading_max_deg": float(np.abs(herr).max()),
        "ate_rms_m": float(np.sqrt(np.mean(perr**2))),
        "final_err_m": float(perr[-1]),
        "path_len_true": float(step.sum()),
        "path_len_est": float(est_step.sum()),
        "path_ratio": float(est_step.sum() / step.sum()),
        "windows_10": int(mv.sum()),
        "nis_n": int(len(nis)),
        "nis_frac_under_7.81": float(np.mean(nis < 7.81)) if len(nis) else None,
        "icp_used_frac": float(res.used_icp.mean()),
        "wall_used_frac": float(res.wall_used.mean()),
        "heading_accept_frac": float(res.heading_accepted.sum() / max(res.heading_valid.sum(), 1)),
    }


def pooled_windows(results, drives, variant, split_seeds, length):
    et, dist, er = [], [], []
    for dr in drives:
        if dr["seed"] not in split_seeds:
            continue
        a, b, c = window_errors(results[(dr["seed"], dr["drive"], variant)], dr["true"], length)
        et.append(a)
        dist.append(b)
        er.append(c)
    return np.concatenate(et), np.concatenate(dist), np.concatenate(er)


def event_table(drives, results, variant):
    """Every true collision event (> 30 degrees under FORWARD/IDLE): what each variant did."""
    rows = []
    for dr in drives:
        rot = np.degrees(np.abs(dr["true_inc"][:, 2]))
        fi = np.isin(dr["inp"].labels, col.LABELS)
        for k in np.where(fi & (rot > 30))[0]:
            row = {
                "seed": dr["seed"],
                "drive": dr["drive"],
                "k": int(k),
                "true_rot_deg": float(np.degrees(dr["true_inc"][k, 2])),
                "flagged": bool(dr["inp"].flags[k]),
            }
            for v in ("B", "C"):
                res = results[(dr["seed"], dr["drive"], v)]
                k5 = min(k + 5, len(dr["true"]) - 1)
                info = res.notes.get(k)
                row[v] = {
                    "est_rot_deg": float(np.degrees(res.inc_mean[k, 2])),
                    "rot_err_deg": float(
                        abs(np.degrees(se2.wrap(res.inc_mean[k, 2] - dr["true_inc"][k, 2])))
                    ),
                    "heading_err_5_after_deg": float(
                        abs(np.degrees(se2.wrap(res.poses[k5, 2] - dr["true"][k5, 2])))
                    ),
                    "chosen": None if info is None else info["chosen"],
                    "margin": None if info is None else float(info["margin"]),
                    "hypotheses": None
                    if info is None
                    else [(h[0], h[1]) for h in info["hypotheses"]],
                }
            rows.append(row)
    return rows


def confidently_wrong(drives, results, variant):
    """Frames whose stage 3 fused heading is valid but wrong by more than 3 degrees: what the
    filter's heading gate did with each."""
    rows = []
    for dr in drives:
        err = np.abs(s3.vp.diff90(dr["fused_psi"], dr["tp"]))
        bad = np.where(dr["fused_valid"] & (err > 3.0))[0]
        for k in bad:
            for v in ("C", "B"):
                res = results[(dr["seed"], dr["drive"], v)]
                if v == "C":
                    herr = abs(np.degrees(se2.wrap(res.poses[k, 2] - dr["true"][k, 2])))
                    rows.append(
                        {
                            "seed": dr["seed"],
                            "drive": dr["drive"],
                            "k": int(k),
                            "psi_error_deg": float(err[k]),
                            "gate_accepted": bool(res.heading_accepted[k]),
                            "innovation_deg": float(res.heading_innovation_deg[k]),
                            "filtered_heading_error_deg": float(herr),
                            "flagged_pair": bool(dr["inp"].flags[k]),
                        }
                    )
    return rows


# --------------------------------------------------------------------------------- figures


def plot_trajectories(seed, drives, results, tag):
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    cs = gt["cell_size"]
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for d in range(3):
        dr = next(x for x in drives if x["seed"] == seed and x["drive"] == d)
        for j, v in enumerate(("A", "B", "C")):
            a = ax[d, j]
            for r, row in enumerate(gt["grid"]):
                for c, ch in enumerate(row):
                    if ch == "#":
                        a.add_patch(
                            plt.Rectangle(
                                (c * cs - cs / 2, r * cs - cs / 2), cs, cs, color="0.9", lw=0
                            )
                        )
            res = results[(seed, d, v)]
            a.plot(dr["true"][:, 0], dr["true"][:, 1], "k", lw=0.8, label="true")
            a.plot(res.poses[:, 0], res.poses[:, 1], "tab:red", lw=0.8, label=f"variant {v}")
            a.set_aspect("equal")
            a.set_xlim(-0.3, 6.5)
            a.set_ylim(-0.3, 6.5)
            sc = score(res, dr)
            a.set_title(
                f"seed {seed} drive {d}, {v}: ATE {sc['ate_rms_m']:.2f} m, heading RMS {sc['heading_rms_deg']:.1f} deg",
                fontsize=9,
            )
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"stage4_trajectories_{tag}_{seed}.png", dpi=60)
    plt.close(fig)


def plot_window_error(results, drives, tag):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for v, c in (("B", "tab:orange"), ("C", "tab:blue")):
        med, p95, rmed = [], [], []
        for L in WINDOW_LENGTHS:
            et, dist, er = pooled_windows(results, drives, v, TEST, L)
            m = dist >= max(0.05, 0.05 * L)
            med.append(np.median(et[m] / dist[m]) * 100)
            p95.append(np.percentile(et[m] / dist[m], 95) * 100)
            rmed.append(np.median(er))
        ax[0].plot(WINDOW_LENGTHS, med, "o-", color=c, label=f"{v} median")
        ax[0].plot(WINDOW_LENGTHS, p95, "s--", color=c, label=f"{v} p95")
        ax[1].plot(WINDOW_LENGTHS, rmed, "o-", color=c, label=v)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("window length [frames]")
    ax[0].set_ylabel("relative translation error [% of distance]")
    ax[0].axhline(3, color="g", lw=0.6)
    ax[0].axhline(8, color="g", lw=0.6)
    ax[0].legend(fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("window length [frames]")
    ax[1].set_ylabel("median relative rotation error [deg]")
    ax[1].legend()
    fig.suptitle(f"test seeds, {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"stage4_window_error_{tag}.png", dpi=80)
    plt.close(fig)


def plot_nis_and_corridor(drives, results, tag):
    dr = next(x for x in drives if x["seed"] == TEST[0] and x["drive"] == 2)
    res = results[(dr["seed"], dr["drive"], "C")]
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].semilogy(np.where(np.isfinite(res.nis), res.nis, np.nan), ".", ms=2)
    ax[0].axhline(7.81, color="r", lw=0.8, label="chi-square 95 %, 3 dof")
    ax[0].set_ylabel("NIS of the fused relative-motion update")
    ax[0].legend()
    ax[1].plot(res.icp_sigma_along, lw=0.5)
    ax[1].set_yscale("log")
    ax[1].axhline(0.5, color="0.5", lw=0.6, ls=":")
    ax[1].set_ylabel("ICP standard deviation along the corridor [m]")
    ax[1].set_xlabel("frame")
    fig.suptitle(f"seed {dr['seed']} drive {dr['drive']}, variant C, {tag}")
    fig.tight_layout()
    fig.savefig(FIG / f"stage4_nis_corridor_{tag}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = (
        json.loads((CACHE / "stage4_results.json").read_text())
        if (CACHE / "stage4_results.json").exists()
        else {}
    )
    for variant in sys.argv[1:] or VARIANTS:
        drives = load_inputs(DEV + TEST, variant)
        tasks = [((dr["seed"], dr["drive"], v), dr, v) for dr in drives for v in ("A", "B", "C")]
        with ProcessPoolExecutor() as ex:
            results = dict(ex.map(_task, tasks))
        with open(CACHE / f"stage4_results_{variant}.pkl", "wb") as fh:
            pickle.dump(results, fh)
        per_drive = {
            f"{s}-{d}-{v}": score(results[(s, d, v)], dr)
            for dr in drives
            for (s, d, v) in [(dr["seed"], dr["drive"], v) for v in ("A", "B", "C")]
        }
        out[variant] = {"per_drive": per_drive}
        for split, seeds in (("dev", DEV), ("test", TEST)):
            agg = {}
            for v in ("B", "C"):
                et, dist, er = pooled_windows(results, drives, v, seeds, 10)
                m = dist >= WINDOW_MIN_DIST
                agg[v] = {
                    "windows": int(m.sum()),
                    "rel_trans_pct_median": float(np.median(et[m] / dist[m]) * 100),
                    "rel_trans_pct_p95": float(np.percentile(et[m] / dist[m], 95) * 100),
                    "rel_rot_deg_median": float(np.median(er)),
                    "rel_rot_deg_p95": float(np.percentile(er, 95)),
                }
                m3 = dist >= 0.3
                agg[v]["rel_trans_pct_median_min0.3"] = float(np.median(et[m3] / dist[m3]) * 100)
                agg[v]["rel_trans_pct_p95_min0.3"] = float(
                    np.percentile(et[m3] / dist[m3], 95) * 100
                )
            out[variant][f"windows_{split}"] = agg
        out[variant]["events"] = event_table([d for d in drives if d["seed"] in TEST], results, "C")
        out[variant]["events_dev"] = event_table(
            [d for d in drives if d["seed"] in DEV], results, "C"
        )
        out[variant]["confidently_wrong_test"] = confidently_wrong(
            [d for d in drives if d["seed"] in TEST], results, "C"
        )
        out[variant]["window_length_curve_test"] = {
            v: {
                str(L): {
                    "median_pct": float(np.median(e[m] / dsc[m]) * 100),
                    "p95_pct": float(np.percentile(e[m] / dsc[m], 95) * 100),
                    "rot_median": float(np.median(r)),
                }
                for L in WINDOW_LENGTHS
                for e, dsc, r in [pooled_windows(results, drives, v, TEST, L)]
                for m in [dsc >= max(0.05, 0.05 * L)]
            }
            for v in ("B", "C")
        }
        tag = variant
        for s in TEST:
            plot_trajectories(s, drives, results, tag)
        plot_window_error(results, drives, tag)
        plot_nis_and_corridor(drives, results, tag)
        print(variant, "done", flush=True)
    (CACHE / "stage4_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
