"""Stage 9 evaluation: the frame-rate adapter's validation, the transparency check, and the
challenge-data sanity checks. EVALUATION CODE (the dev checks read ground-truth files only to
compare recordings; the challenge checks read nothing but the download and its build).

    uv run python source/mapping/stage9.py detect                  # detector on dev, tick1, gt
    uv run python -I source/mapping/stage9.py detect-challenge     # ... on the challenge download
    uv run python source/mapping/stage9.py subsample               # tick1-7/8 vs gt4-7/8
    uv run python source/mapping/stage9.py transparency            # build(tick1-7) vs build(gt4-7)
    uv run python -I source/mapping/stage9.py sanity <build dir> [--name n]
    uv run python -I source/mapping/stage9.py figure <build dir>   # the challenge map and goal

Test seeds are never read here. Holdout seeds are refused (``record_gt``).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import filecmp
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapter  # noqa: E402
import goal  # noqa: E402
import slam  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "mapping" / "stage9"
FIG = ROOT / "reports" / "figures"
BUILD = ROOT / "cache" / "mapping" / "build"
DATA = ROOT / "data"
CHALLENGE = DATA / "ab7efaeb1abf"


def detect(names: list[str], out: str) -> dict:
    res = {}
    for n in names:
        k, ev = adapter.ticks_per_frame(DATA / n)
        res[n] = ev.as_dict()
        print(n, "k =", k, {a: round(b, 3) if isinstance(b, float) else b for a, b in ev.as_dict().items()}, flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / out).write_text(json.dumps(res, indent=1))
    return res


def compare_subsample(seed: int) -> dict:
    """The subsampled per-tick recording against the native 5-tick one (same maze, routes, noise)."""
    out = BUILD.parent / "adapted" / f"tick1-{seed}"
    if out.exists():
        import shutil

        shutil.rmtree(out)
    adapter.subsample_dataset(DATA / f"tick1-{seed}", 5, out)
    n = n_lab = n_img = n_comp = 0
    max_diff = 0.0
    for d in range(3):
        a = json.loads((DATA / f"gt4-{seed}" / f"traj_{d}" / "data_info.json").read_text())
        b = json.loads((out / f"traj_{d}" / "data_info.json").read_text())
        assert len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b, strict=True):
            n += 1
            n_lab += x["action"] == y["action"]
            n_comp += sum(y["composition"].values()) == 5
            pa, pb = DATA / f"gt4-{seed}" / f"traj_{d}" / x["image"], out / f"traj_{d}" / y["image"]
            if filecmp.cmp(pa, pb, shallow=False):
                n_img += 1
            else:
                max_diff = max(max_diff, float(np.abs(cv2.imread(str(pa)).astype(int) - cv2.imread(str(pb)).astype(int)).max()))
    return {"seed": seed, "frames": n, "labels_equal": n_lab, "images_identical": n_img, "max_pixel_diff_if_not_identical": max_diff, "composition_sums_to_5": n_comp}


def _same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype == object:
        return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(x.ravel(), y.ravel(), strict=True))
    return bool(np.array_equal(x, y))


def transparency(reference: str = "gt4-7-rebuild") -> dict:
    """``build_map`` on tick1-7 (detected, subsampled) against the native 5-tick build of the
    same maze made with the same code (``gt4-7-rebuild``; the older ``gt4-7`` build predates a
    change of ``grid.py``'s output keys)."""
    a, b = BUILD / reference, BUILD / "tick1-7"
    pa = json.loads((a / "poses_optimized.json").read_text())["drives"]
    pb = json.loads((b / "poses_optimized.json").read_text())["drives"]
    d = np.concatenate([np.hypot(*(np.array(x)[:, :2] - np.array(y)[:, :2]).T) for x, y in zip(pa, pb, strict=True)])
    za, zb = np.load(a / "map.npz", allow_pickle=True), np.load(b / "map.npz", allow_pickle=True)
    cells_equal = bool(np.array_equal(za["wall_cells"], zb["wall_cells"]))
    ta, tb = cv2.imread(str(DATA / "gt4-7" / "target.jpg")), cv2.imread(str(DATA / "tick1-7" / "target.jpg"))
    ga, gb = goal.place(goal.load_map(za), ta), goal.place(goal.load_map(zb), tb)
    return {
        "pose_diff_max_m": float(d.max()), "pose_diff_median_m": float(np.median(d)), "poses": int(len(d)),
        "wall_cells_identical": cells_equal, "wall_cells": [int(za["wall_cells"].sum()), int(zb["wall_cells"].sum())],
        "reference": reference,
        "keys_equal": sorted(za.files) == sorted(zb.files),
        "arrays_identical": [k for k in za.files if k in zb.files and _same(za[k], zb[k])],
        "arrays_differ": [k for k in za.files if k in zb.files and not _same(za[k], zb[k])],
        "target_identical": bool(np.array_equal(ta, tb)),
        "goal_reference": [float(v) for v in ga.pose], "goal_tick1_7": [float(v) for v in gb.pose],
        "goal_position_diff_m": float(np.hypot(*(ga.pose[:2] - gb.pose[:2]))),
        "goal_heading_diff_deg": float(abs(np.degrees((ga.pose[2] - gb.pose[2] + np.pi) % (2 * np.pi) - np.pi))),
        "adapter": json.loads((b / "adapter.json").read_text()),
        "timing_reference": json.loads((a / "timing.json").read_text()), "timing_tick1_7": json.loads((b / "timing.json").read_text()),
    }


def graph_cost(build: Path) -> dict:
    odo = pickle.loads((build / "odometry.pkl").read_bytes())
    cl = pickle.loads((build / "closures.pkl").read_bytes())
    cl = slam.thin(cl)
    g, x0, _, _ = slam.build(odo, cl, slam.SPAWN)
    _, X, dropped = slam.two_pass(g, x0)
    r = g.residuals(np.asarray(X).ravel())
    n_edges = max(len(r) // 3, 1)
    return {"edges": int(n_edges), "chi2_per_edge": float((r**2).sum() / n_edges), "dropped": int(dropped)}


def sanity(build: Path, name: str) -> dict:
    """Every pipeline value of a build that needs no ground truth, plus the goal placement."""
    z = np.load(build / "map.npz", allow_pickle=True)
    cl = pickle.loads((build / "closures.pkl").read_bytes())
    ad = json.loads((build / "adapter.json").read_text()) if (build / "adapter.json").exists() else {"k": 5, "evidence": None}
    cal = json.loads((build / "calibration.json").read_text())
    pairs: dict[str, int] = {}
    for c in cl:
        key = "-".join(map(str, sorted({c["a"][0], c["b"][0]})))
        pairs[key] = pairs.get(key, 0) + 1
    wc = np.asarray(z["wall_cells"])
    rr, cc = np.where(wc)
    box = wc[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]  # the maze's bounding box
    ring = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
    outer = bool(ring.all())
    timing = json.loads((build / "timing.json").read_text())
    data = Path(ad["source"]) if ad.get("source") else None
    est = None
    for cand in (data / "target.jpg" if data else None, build.parent.parent.parent.parent / "data" / build.name / "target.jpg"):
        if cand is not None and cand.exists():
            est = goal.place(goal.load_map(z), cv2.imread(str(cand)))
            break
    out = {
        "name": name,
        "ticks_per_frame": ad["k"], "evidence": ad.get("evidence"),
        "calibration": {k: cal[0][k] for k in ("FORWARD",)} if isinstance(cal, list) and cal else cal,
        "calibration_per_drive": cal,
        "closures": len(cl), "closures_by_pair": pairs,
        "closures_by_kind": {k: sum(c["kind"] == k for c in cl) for k in sorted({c["kind"] for c in cl})},
        "graph": graph_cost(build),
        "lattice_scale": float(z["lattice_scale"]), "lattice_strength": float(z["lattice_strength"]),
        "cell_size": float(z["cell_size"]),
        "snapped_wall_cells": int(wc.sum()), "outer_wall_closed": outer, "outer_wall_fill": float(ring.mean()),
        "maze_cells": [int(box.shape[0]), int(box.shape[1])],
        "cspace_cells": int(np.asarray(z["cspace"]).sum()),
        "timing_s": timing, "total_s": float(sum(timing.get(s, 0) for s in ("adapter", "inputs", "odometry", "closures", "graph", "map"))),
    }
    if est is not None:
        out["goal"] = {"pose": [float(v) for v in est.pose], "confidence": float(est.confidence), "reliable": bool(est.reliable),
                       "camera_height": float(est.camera_height), "overlap_residual": float(est.overlap_residual), "n_points": int(est.n_points)}
    return out


def figure(build: Path, target: Path | None, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(build / "map.npz", allow_pickle=True)
    est = goal.place(goal.load_map(z), cv2.imread(str(target))) if target is not None else None
    fig, ax = plt.subplots(figsize=(7, 7))
    res, org = float(z["resolution"]), z["origin"]
    cs = np.asarray(z["cspace"])
    ax.imshow(np.ma.masked_where(~cs.astype(bool), np.ones(cs.shape)), cmap="Greens", vmin=0, vmax=3, origin="lower",
              extent=(org[0], org[0] + cs.shape[1] * res, org[1], org[1] + cs.shape[0] * res), alpha=0.5)
    P = z["poses"]
    for d in range(int(P[:, 0].max()) + 1):
        q = P[P[:, 0] == d]
        ax.plot(q[:, 2], q[:, 3], lw=0.4, alpha=0.6, label=f"drive {d}")
    for s in z["wall_segments"]:
        ax.plot([s[0], s[2]], [s[1], s[3]], "k-", lw=1.5)
    if est is not None:
        for t in est.top[1:]:
            ax.plot(*t[0][:2], "o", mfc="none", mec="orange", ms=10, label="runner-up")
        x, y, th = est.pose
        ax.annotate("", xy=(x + 0.35 * np.cos(th), y + 0.35 * np.sin(th)), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="red" if est.reliable else "orange", lw=2.5))
        ax.plot(x, y, "o", color="red" if est.reliable else "orange")
        title += f"\ngoal ({x:.2f}, {y:.2f}) m, {np.degrees(th):.0f} deg, confidence {est.confidence:.2f}"
    ax.set_aspect("equal")
    pts = np.asarray(z["wall_segments"]).reshape(-1, 2)
    ax.set_xlim(pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3)
    ax.set_ylim(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3)
    ax.set_title(title, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="upper right")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["detect", "detect-challenge", "subsample", "transparency", "sanity", "figure"])
    p.add_argument("build", nargs="?")
    p.add_argument("--name")
    a = p.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.cmd == "detect":
        detect(["gt4-7", "gt4-8", "gt4-9", "gt4-10", "gt-7", "gt-8", "gt-9", "gt-10", "tick1-7", "tick1-8"], "detect_local.json")
    elif a.cmd == "detect-challenge":
        detect([CHALLENGE.name], "detect_challenge.json")
    elif a.cmd == "subsample":
        res = [compare_subsample(7), compare_subsample(8)]
        print(json.dumps(res, indent=1))
        (CACHE / "subsample.json").write_text(json.dumps(res, indent=1))
    elif a.cmd == "transparency":
        res = transparency()
        print(json.dumps(res, indent=1))
        (CACHE / "transparency.json").write_text(json.dumps(res, indent=1))
    elif a.cmd == "sanity":
        build = Path(a.build).resolve()
        res = sanity(build, a.name or build.name)
        print(json.dumps(res, indent=1))
        (CACHE / f"sanity_{res['name']}.json").write_text(json.dumps(res, indent=1))
    else:
        build = Path(a.build).resolve()
        target = (json.loads((build / "adapter.json").read_text())["source"] + "/target.jpg") if (build / "adapter.json").exists() else None
        figure(build, Path(target) if target else None, FIG / f"stage9_{a.name or build.name}.png", f"{a.name or build.name}: map from the exploration data, goal from target.jpg")


if __name__ == "__main__":
    main()
