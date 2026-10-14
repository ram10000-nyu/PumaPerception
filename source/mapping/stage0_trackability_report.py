"""Stage 0 revision 2, task 7: statistics and figures from the cached trackability cues.

    uv run source/mapping/stage0_trackability_report.py

Writes ``reports/figures/stage0_trackability_<seed>.png``, ``stage0_blind_examples.png`` and
``cache/mapping/stage0_trackability_results.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_gt  # noqa: E402
import trackability as T  # noqa: E402

ROOT = record_gt.ROOT
FIG = ROOT / "reports" / "figures"


def seed_stats(seed: int, drives: list[dict], motion: list[dict]) -> dict:
    out = {"drives": []}
    poses = record_gt.load_poses(record_gt.gt_dir(seed))
    for d, m, P in zip(drives, motion, poses, strict=True):
        inl, val, vl = d["inliers"], d["valid_cols"], d["vp_len"]
        blind = T.blind_flags(inl, val, vl)
        # provisional VP agreement with the true heading (strongest peak vs either axis VP)
        agree = []
        for k in range(len(P)):
            if vl[k] > 0:
                e = T.expected_vp_angles(P[k, 2])
                agree.append(min(T.angle_diff_deg(d["vp_ang"][k, 0], x) for x in e))
        agree = np.array(agree)
        out["drives"].append(
            {
                "frames": int(len(inl)),
                "trans_max_m": float(m["trans"].max()),
                "trans_p95_m": float(np.percentile(m["trans"], 95)),
                "rot_max_deg": float(np.degrees(np.abs(m["rot"]).max())),
                "rot_p95_deg": float(np.degrees(np.percentile(np.abs(m["rot"]), 95))),
                "frac_rot_gt_45deg": float((np.abs(m["rot"]) > np.radians(45)).mean()),
                "n_rot_gt_45deg": int((np.abs(m["rot"]) > np.radians(45)).sum()),
                "n_rot_gt_30deg": int((np.abs(m["rot"]) > np.radians(30)).sum()),
                "rot_gt_45_indices": np.where(np.abs(m["rot"]) > np.radians(45))[0].tolist(),
                "inliers_p5": float(np.percentile(inl, 5)),
                "inliers_median": float(np.median(inl)),
                "frac_inliers_lt15": float((inl < 15).mean()),
                "valid_cols_p5": float(np.percentile(val, 5)),
                "valid_cols_median": float(np.median(val)),
                "valid_cols_min": float(val.min()),
                "frac_valid_lt20": float((val < 0.2).mean()),
                "frac_vp": float((vl > 0).mean()),
                "vp_within_3deg_of_truth": float((agree < 3).mean()) if len(agree) else 0.0,
                "blind_frames": int(blind.sum()),
                "longest_blind_run": T.longest_run(blind),
                "blind_indices": np.where(blind)[0].tolist(),
                "frac_two_cues_fail": float(
                    (
                        ((inl < 15).astype(int) + (val < 0.2).astype(int) + (vl <= 0).astype(int))
                        >= 2
                    ).mean()
                ),
            }
        )
    return out


def plot_seed(seed: int, drives: list[dict]) -> None:
    fig, ax = plt.subplots(3, 3, figsize=(15, 8), sharex="col")
    for i, d in enumerate(drives):
        n = len(d["inliers"])
        x = np.arange(n)
        blind = T.blind_flags(d["inliers"], d["valid_cols"], d["vp_len"])
        ax[0, i].plot(x, d["inliers"], lw=0.5)
        ax[0, i].axhline(15, color="r", lw=0.8, ls="--")
        ax[0, i].set_title(f"seed {seed} drive {i} ({n} frames)")
        ax[0, i].set_ylabel("RANSAC inliers vs previous" if i == 0 else "")
        ax[1, i].plot(x, d["valid_cols"], lw=0.5, color="tab:green")
        ax[1, i].axhline(0.2, color="r", lw=0.8, ls="--")
        ax[1, i].set_ylabel("valid boundary columns" if i == 0 else "")
        ax[2, i].fill_between(
            x, 0, (d["vp_len"] > 0).astype(float), step="mid", color="tab:purple", alpha=0.6
        )
        for b in np.where(blind)[0]:
            ax[2, i].axvline(b, color="k", lw=1.5)
        ax[2, i].set_yticks([0, 1], ["none", "found"])
        ax[2, i].set_ylabel("vanishing point (black = blind frame)" if i == 0 else "")
        ax[2, i].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(FIG / f"stage0_trackability_{seed}.png", dpi=90)
    plt.close(fig)


def blind_examples(all_drives: dict[int, list[dict]], n: int = 10) -> list[dict]:
    """The blind frames if any, filled with the weakest frames (fewest cues, then fewest
    inliers) so the figure always shows what 'nearly blind' looks like."""
    rows = []
    for seed, drives in all_drives.items():
        for i, d in enumerate(drives):
            blind = T.blind_flags(d["inliers"], d["valid_cols"], d["vp_len"])
            for k in range(len(d["inliers"])):
                cues = (
                    int(d["inliers"][k] >= 15)
                    + int(d["valid_cols"][k] >= 0.2)
                    + int(d["vp_len"][k] > 0)
                )
                rows.append(
                    (
                        not blind[k],
                        cues,
                        int(d["inliers"][k]),
                        float(d["valid_cols"][k]),
                        seed,
                        i,
                        k,
                    )
                )
    rows.sort()
    return [
        {
            "blind": not r[0],
            "cues": r[1],
            "inliers": r[2],
            "valid_cols": r[3],
            "seed": r[4],
            "drive": r[5],
            "k": r[6],
        }
        for r in rows[:n]
    ]


def plot_examples(ex: list[dict]) -> None:
    fig, ax = plt.subplots(2, 5, figsize=(17, 7))
    for a, e in zip(ax.ravel(), ex, strict=True):
        p = record_gt.gt_dir(e["seed"]) / f"traj_{e['drive']}" / f"{e['k']}.jpg"
        a.imshow(cv2.imread(str(p))[..., ::-1])
        a.set_title(
            f"{'BLIND' if e['blind'] else 'weakest'} s{e['seed']} d{e['drive']} #{e['k']}\n"
            f"inl {e['inliers']}, cols {e['valid_cols']:.2f}, cues {e['cues']}/3",
            fontsize=8,
        )
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "stage0_blind_examples.png", dpi=90)
    plt.close(fig)


def main() -> None:
    cfg = json.loads(record_gt.SEEDS.read_text())
    seeds = cfg["dev"] + cfg["test"]
    FIG.mkdir(parents=True, exist_ok=True)
    allres, all_drives = {}, {}
    for s in seeds:
        drives = T.run_seed(s)
        all_drives[s] = drives
        allres[s] = seed_stats(s, drives, T.true_motion(s))
        plot_seed(s, drives)
    ex = blind_examples(all_drives)
    plot_examples(ex)
    allres["examples"] = ex
    allres["totals"] = {
        "pairs": int(sum(d["frames"] for s in seeds for d in allres[s]["drives"])),
        "n_rot_gt_45deg": int(sum(d["n_rot_gt_45deg"] for s in seeds for d in allres[s]["drives"])),
        "n_rot_gt_30deg": int(sum(d["n_rot_gt_30deg"] for s in seeds for d in allres[s]["drives"])),
        "max_rot_deg": float(max(d["rot_max_deg"] for s in seeds for d in allres[s]["drives"])),
        "blind_frames": int(sum(d["blind_frames"] for s in seeds for d in allres[s]["drives"])),
        "longest_blind_run": int(
            max(d["longest_blind_run"] for s in seeds for d in allres[s]["drives"])
        ),
    }
    print("totals", allres["totals"])
    (ROOT / "cache" / "mapping" / "stage0_trackability_results.json").write_text(
        json.dumps(allres, indent=1)
    )
    print("seed drv rotmax >45 inl p5/med cols p5/min vp% vpOK% blind run")
    for s in seeds:
        for i, d in enumerate(allres[s]["drives"]):
            print(
                f"{s:>4} {i:>3} {d['rot_max_deg']:7.1f} {d['n_rot_gt_45deg']:4d} "
                f"{d['inliers_p5']:5.0f}/{d['inliers_median']:<5.0f} "
                f"{d['valid_cols_p5']:6.2f}/{d['valid_cols_min']:<5.2f} "
                f"{100 * d['frac_vp']:5.1f} {100 * d['vp_within_3deg_of_truth']:6.1f} "
                f"{d['blind_frames']:5d} {d['longest_blind_run']:3d}"
            )
    print("examples", ex[:3])


if __name__ == "__main__":
    main()
