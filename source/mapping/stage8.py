"""Stage 8 evaluation: the goal pose from the four goal views on the stage 7 maps. EVALUATION
CODE: reads the true goal (rule 10); ``goal.py`` never does.

    python3 source/mapping/stage8.py run dev                 # real goal views, dev seeds
    python3 source/mapping/stage8.py synthetic dev           # many rendered goal scans (geometry only)
    python3 source/mapping/stage8.py run test                # ONLY after the user agrees (test seeds)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _m in ("vis_nav_sdk", "vis_nav_sim"):  # only so that imports resolve; nothing is used
    if _m not in sys.modules:
        _s = types.ModuleType(_m)
        _s.__getattr__ = lambda n: type(n, (), {})
        sys.modules[_m] = _s

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal  # noqa: E402
import grid  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
OUT = CACHE / "stage8"
CFG = json.loads(record_gt.SEEDS.read_text())
DEV, TEST = CFG["dev"], CFG["test"]
POS_OK, DEG_OK = 0.10, 10.0


def truth(seed, prefix="gt4"):
    return np.array(json.loads((record_gt.gt_dir(seed, prefix=prefix) / "gt.json").read_text())["target"])


def errors(est, true):
    return float(np.hypot(*(est[:2] - true[:2]))), float(abs(np.degrees(se2.wrap(est[2] - true[2]))))


def run_seed(args):
    seed, prefix, use_tex, use = (*args, (0, 1, 2, 3))[:4] if len(args) == 3 else args
    t0 = time.perf_counter()
    mf = goal.load_map(CACHE / str(seed) / "map.npz")
    t_load = time.perf_counter() - t0
    img = cv2.imread(str(record_gt.gt_dir(seed, prefix=prefix) / "target.jpg"))
    t0 = time.perf_counter()
    est = goal.place(mf, img, use_texture=use_tex, use=use)
    dt = time.perf_counter() - t0
    T = truth(seed, prefix)
    e_pos, e_deg = errors(est.pose, T)
    top = [dict(pose=[round(float(v), 4) for v in p], score=s, geo=g, tex=t, faces=n, err=errors(p, T)) for p, s, g, t, n in est.top]
    return (seed, tuple(use)), dict(prefix=prefix, views=list(use), load_s=t_load, pose=est.pose.tolist(), truth=T.tolist(), err_m=e_pos, err_deg=e_deg, ok=bool(e_pos < POS_OK and e_deg < DEG_OK),
                      confidence=est.confidence, camera_height=est.camera_height, overlap_residual=est.overlap_residual,
                      n_points=est.n_points, runtime_s=dt, top=top)


SUBSETS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def run(split, prefix="gt4", use_tex=True, subsets=((0, 1, 2, 3),), serial=False):
    seeds = DEV if split == "dev" else TEST if split == "test" else [int(s) for s in split.split(",")]
    with ProcessPoolExecutor(1 if serial else 4) as ex:
        res = dict(ex.map(run_seed, [(s, prefix, use_tex, u) for s in seeds for u in subsets]))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{split}_{prefix}{'' if use_tex else '_notex'}{'_subsets' if len(subsets) > 1 else ''}"
    (OUT / f"results_{tag}.json").write_text(json.dumps({f"{k[0]}:{''.join(map(str, k[1]))}": v for k, v in res.items()}, indent=1))
    for s, r in res.items():
        print(f"seed {s} {prefix}: err {r['err_m']:.3f} m {r['err_deg']:.1f} deg ok={r['ok']} conf {r['confidence']:.3f} h {r['camera_height']:.3f} "
              f"overlap {r['overlap_residual']:.4f} {r['runtime_s']:.2f} s | top: " + "; ".join(f"{t['score']:.3f} (g {t['geo']:.3f} t {t['tex']:.3f} n {t['faces']}) err {t['err'][0]:.2f}" for t in r['top']), flush=True)
    return res


def figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = ROOT / "reports" / "figures"
    test = json.loads((OUT / "results_test_gt4.json").read_text())
    fig, axs = plt.subplots(1, 5, figsize=(26, 5.6))
    for ax, (k, r) in zip(axs, test.items(), strict=False):
        seed = int(k.split(":")[0])
        z = np.load(CACHE / str(seed) / "map.npz", allow_pickle=True)
        ax.imshow(np.where(z["cspace"], 1.0, np.nan), cmap="Greens", vmin=0, vmax=3, origin="lower",
                  extent=grid.GridSpec(z["origin"], z["logodds"].shape, float(z["resolution"])).extent(), alpha=0.5)
        for sg_ in z["wall_segments"]:
            ax.plot([sg_[0], sg_[2]], [sg_[1], sg_[3]], "k-", lw=1)
        for t in r["top"][1:]:
            x, y, th = t["pose"]
            ax.plot(x, y, "o", color="tab:orange", ms=7, mfc="none")
        x, y, th = r["truth"]
        ax.arrow(x, y, 0.35 * np.cos(th), 0.35 * np.sin(th), color="tab:blue", width=0.03, length_includes_head=True)
        x, y, th = r["pose"]
        ax.arrow(x, y, 0.25 * np.cos(th), 0.25 * np.sin(th), color="tab:red", width=0.012, length_includes_head=True)
        ax.set_title(f"seed {seed}: error {1000 * r['err_m']:.0f} mm, {r['err_deg']:.1f} deg, confidence {r['confidence']:.2f}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-0.25, 6.45)
        ax.set_ylim(-0.25, 6.45)
    fig.suptitle("Stage 8, test seeds (gt4): true goal (blue arrow), estimate (red arrow), runner-up places (orange circles)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "stage8_goals_test_gt4.png", dpi=75)
    plt.close(fig)
    # confidence vs error: dev stress subsets, dev, test, both renderings
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sets = [("results_dev_gt4_subsets.json", "dev, 2 or 3 of the 4 views", "tab:gray", "x"), ("results_dev_gt4.json", "dev, 4 views", "tab:green", "o"),
            ("results_dev_gt.json", "dev, msaa 1", "tab:olive", "s"), ("results_test_gt4.json", "test, 4 views", "tab:red", "o"), ("results_test_gt.json", "test, msaa 1", "tab:purple", "s")]
    for f, lab, col, mk in sets:
        if not (OUT / f).exists():
            continue
        d = json.loads((OUT / f).read_text())
        e = np.array([max(v["err_m"], 1e-3) for v in d.values()])
        c = np.array([v["confidence"] for v in d.values()])
        ax.scatter(c, e, label=lab, color=col, marker=mk, s=30)
    ax.axvline(goal.CONF_MIN, color="k", ls="--", lw=1, label=f"threshold {goal.CONF_MIN} (dev)")
    ax.axhline(POS_OK, color="tab:red", ls=":", lw=1, label="0.10 m bound")
    ax.set_yscale("log")
    ax.set_xlabel("confidence (score margin to the best other place)")
    ax.set_ylabel("position error (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage8_confidence_vs_error.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "figures"])
    p.add_argument("split", nargs="?", default="dev")
    p.add_argument("--prefix", default="gt4")
    p.add_argument("--notex", action="store_true")
    p.add_argument("--subsets", action="store_true", help="stress test: every 3- and 2-view subset of the goal views")
    p.add_argument("--serial", action="store_true", help="one process (for runtime)")
    p.add_argument("--allow-test", action="store_true", help="score test seeds (only with the user's agreement)")
    a = p.parse_args()
    if a.step == "figures":
        figures()
        return
    seeds = DEV if a.split == "dev" else TEST if a.split == "test" else [int(s) for s in a.split.split(",")]
    if set(seeds) & set(TEST) and not a.allow_test:
        sys.exit("refusing to score test seeds without --allow-test (ask the user first)")
    run(a.split, a.prefix, not a.notex, SUBSETS if a.subsets else ((0, 1, 2, 3),), a.serial)


if __name__ == "__main__":
    main()
