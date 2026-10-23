"""Stage 7 evaluation: oracle maps (task 0), the map from the stage 6 poses, and its scores
against the true layout. EVALUATION CODE: it reads ground truth (rule 10); ``grid.py`` and
``build_map.py`` never do.

    python3 source/mapping/stage7.py oracle            # task 0: oracle maps, gt and gt4
    python3 source/mapping/stage7.py run dev|test      # maps from the stage 6 poses + scores
    python3 source/mapping/stage7.py sweep             # dev tuning of the grid parameters
    python3 source/mapping/stage7.py figures           # map images for the test seeds
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_fit as cf  # noqa: E402
import grid  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402
import stage3_extract as x3  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
OUT = CACHE / "stage7"
FIG = ROOT / "reports" / "figures"
CFG = json.loads(record_gt.SEEDS.read_text())
DEV, TEST = CFG["dev"], CFG["test"]
SPEC = grid.GridSpec(np.array([-0.225, -0.225]), (134, 134))  # stage 2's oracle grid
SPAWN = np.array([0.2, 2.6])
VARIANT_DIR = {"gt4": "gt4", "official": "gt"}  # stage 3 cache name -> data prefix


def scans(seed, variant="gt4"):
    R = x3.run_seed(seed, variant=variant)
    return [[(R[d]["rows"][k], R[d]["top"][k], R[d]["status"][k]) for k in range(len(R[d]["rows"]))] for d in range(3)]


def true_poses(seed, variant="gt4"):
    return record_gt.load_poses(record_gt.gt_dir(seed, prefix=VARIANT_DIR[variant]))


def est_poses(seed, source="stage6"):
    """Optimised poses: stage 6's evaluation run, or ``build_map.py``'s end-to-end run."""
    f = CACHE / str(seed) / "poses_optimized.json" if source == "stage6" else CACHE / "build" / f"gt4-{seed}" / "poses_optimized.json"
    return [np.array(p) for p in json.loads(f.read_text())["drives"]]


def truth_grid(seed):
    gt = json.loads((record_gt.gt_dir(seed, prefix="gt4") / "gt.json").read_text())
    return np.array([[c == "#" for c in row] for row in gt["grid"]])


def truth_raster(segs) -> np.ndarray:
    img = np.zeros(SPEC.shape, np.uint8)
    for s in segs:
        p0 = (s[:2] - SPEC.origin) / SPEC.res
        p1 = (s[2:] - SPEC.origin) / SPEC.res
        cv2.line(img, (int(round(p0[0] - 0.5)), int(round(p0[1] - 0.5))), (int(round(p1[0] - 0.5)), int(round(p1[1] - 0.5))), 1, 1)
    return img > 0


def dil(a, n=1):
    k = np.ones((2 * n + 1, 2 * n + 1), np.uint8)
    return cv2.dilate(a.astype(np.uint8), k) > 0


def iou(a, b, n=1) -> float:
    da, db = dil(a, n), dil(b, n)
    return float((da & db).sum() / max((da | db).sum(), 1))


# ------------------------------------------------------------------------------ task 0


def _oracle(args):
    seed, variant = args
    lo, hits = grid.occupancy(scans(seed, variant), true_poses(seed, variant), SPEC)
    d = CACHE / str(seed)
    d.mkdir(exist_ok=True)
    name = "oracle_map.npz" if variant == "official" else "oracle_map_gt4.npz"
    np.savez(d / name, logodds=lo, hits=hits, resolution=SPEC.res, origin=SPEC.origin, r_max=grid.R_MAX,
             note="stage 7 task 0: current segment/scan (wall-top returns), grid.occupancy at true poses")
    truth = truth_raster(cf.wall_segments(truth_grid(seed)))
    return seed, variant, {"iou_vs_true_layout_1cell": iou(lo > 0, truth, 1), "iou_2cell": iou(lo > 0, truth, 2), "occupied_cells": int((lo > 0).sum())}


def oracle() -> dict:
    for s in DEV + TEST:  # keep the stage 2 oracle for reference
        old = CACHE / str(s) / "oracle_map.npz"
        bak = CACHE / str(s) / "oracle_map_stage2.npz"
        if old.exists() and not bak.exists():
            old.rename(bak)
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_oracle, [(s, v) for v in ("gt4", "official") for s in DEV + TEST]))
    out = {v: {str(s): r for s, vv, r in res if vv == v} for v in ("gt4", "official")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oracle.json").write_text(json.dumps(out, indent=1))
    return out


def load_oracle(seed, variant="gt4"):
    z = np.load(CACHE / str(seed) / ("oracle_map_gt4.npz" if variant == "gt4" else "oracle_map.npz"))
    return z["logodds"], z["hits"]


# ------------------------------------------------------------------------------ scoring


def true_cells(seed, cells: grid.CellGrid):
    """The true wall cells on ``cells``' index range (the maze grid starts at cell (0, 0))."""
    T = truth_grid(seed)
    out = np.zeros_like(cells.walls)
    for r in range(cells.walls.shape[0]):
        for c in range(cells.walls.shape[1]):
            R, C = cells.r0 + r, cells.c0 + c
            if 0 <= R < T.shape[0] and 0 <= C < T.shape[1]:
                out[r, c] = T[R, C]
    return out


def observable_cells(cells: grid.CellGrid, hits) -> np.ndarray:
    """Cells that some true-pose return reached (oracle hit weight on the cell's span)."""
    ev = grid.snap(np.where(hits > 0, 1.0, 0.0) * 5.0, SPEC, 0.0, 0.0)
    return ev.observed & (np.nan_to_num(ev.score) > 0)


def signature_frames(seed, poses_ok, variant="gt4", every=1):
    path = record_gt.gt_dir(seed, prefix=VARIANT_DIR[variant])
    R = x3.run_seed(seed, variant=variant)
    for d in range(3):
        info = json.loads((path / f"traj_{d}" / "data_info.json").read_text())
        for k in range(0, len(info), every):
            img = cv2.imread(str(path / f"traj_{d}" / info[k]["image"]))
            yield d, k, img, R[d]["rows"][k], R[d]["top"][k], R[d]["status"][k]


def signature_repeatability(m: grid.Map, seed) -> dict:
    """Faces seen by two drives: Hellinger distance of the two drives' signatures of the same
    face, against the same drive pair's signatures of other faces (AUC: the probability that a
    same-face pair is closer than a different-face pair)."""
    builders = [grid.SignatureBuilder(m.cells) for _ in range(3)]
    for d, k, img, rows, top, status in signature_frames(seed, None):
        builders[d].add(img, rows, top, status, m.poses[d][k], d * 100000 + k)
    per = [b.result() for b in builders]
    same, diff = [], []
    rng = np.random.default_rng(0)
    for a in range(3):
        for b in range(a + 1, 3):
            ka = {k: i for i, k in enumerate(per[a].keys) if per[a].pixels[i] >= 200}
            kb = {k: i for i, k in enumerate(per[b].keys) if per[b].pixels[i] >= 200}
            common = sorted(set(ka) & set(kb))
            if len(common) < 3:
                continue
            A = per[a].hist[[ka[k] for k in common]]
            B = per[b].hist[[kb[k] for k in common]]
            same += list(grid.hellinger(A, B))
            perm = rng.permutation(len(common))
            perm = np.where(perm == np.arange(len(common)), (perm + 1) % len(common), perm)
            diff += list(grid.hellinger(A, B[perm]))
    same, diff = np.array(same), np.array(diff)
    if len(same) == 0:
        return {"faces_two_drives": 0}
    auc = float((same[:, None] < diff[None, :]).mean())
    return {"faces_two_drives": int(len(same)), "same_median": float(np.median(same)), "diff_median": float(np.median(diff)), "auc": auc}


def score(seed, m: grid.Map, oracle_lo, oracle_hits, P_true) -> dict:
    T_segs = cf.wall_segments(truth_grid(seed))
    truth = truth_raster(T_segs)
    occ, ocl = m.logodds > 0, oracle_lo > 0
    observable_r = truth & dil(oracle_hits > 0, 1)
    out = {}
    out["iou_vs_oracle"] = iou(occ, ocl, 1)
    out["iou_vs_observable"] = iou(occ, observable_r, 1)
    out["iou_vs_full_layout"] = iou(occ, truth, 1)
    out["oracle_iou_vs_full_layout"] = iou(ocl, truth, 1)
    # missing true walls (layout raster cells not covered by the dilated map)
    miss = truth & ~dil(occ, 1)
    unseen = miss & ~dil(oracle_hits > 0, 1)
    weak = miss & dil(oracle_hits > 0, 1) & ~dil(oracle_hits > 1.0, 1)
    out["missing_wall_cells"] = {"total": int(miss.sum()), "of_layout": int(truth.sum()), "never_in_range": int(unseen.sum()),
                                 "grazing_only": int(weak.sum()), "seen_by_oracle": int((miss & ~unseen & ~weak).sum())}
    # snapped wall cells
    tc = true_cells(seed, m.cells)
    obs = observable_cells(m.cells, oracle_hits)
    est = m.cells.walls
    out["snap"] = {
        "true_cells": int(tc.sum()), "observable_true_cells": int((tc & obs).sum()), "est_cells": int(est.sum()),
        "recall_all": float((est & tc).sum() / max(tc.sum(), 1)),
        "recall_observable": float((est & tc & obs).sum() / max((tc & obs).sum(), 1)),
        "false_rate": float((est & ~tc).sum() / max(est.sum(), 1)),
        "false_cells": int((est & ~tc).sum()),
        "missed_observable": int((tc & obs & ~est).sum()),
    }
    # lattice (truth: wall lines at multiples of 0.4 m, cell size 0.2 m)
    lat = m.lattice
    out["lattice"] = {"cell_size_raw_map": lat.period / 2, "cell_size_err_pct": 100 * (lat.period / 2 / grid.CELL - 1),
                      "scale": lat.scale, "offset_x": float(lat.offset[0]), "offset_y": float(lat.offset[1]), "strength": lat.strength}
    # poses: before and after the lattice correction
    E0 = np.concatenate(m.poses_in if hasattr(m, "poses_in") else m.poses)[:, :2]
    E1 = np.concatenate(m.poses)[:, :2]
    Tt = np.concatenate(P_true)[:, :2]
    sb = umeyama_scale(E0, Tt)
    out["poses"] = {"raw_rmse_before": float(np.sqrt(((E0 - Tt) ** 2).sum(1).mean())), "raw_rmse_after": float(np.sqrt(((E1 - Tt) ** 2).sum(1).mean())),
                    "scale_before": sb, "scale_after": umeyama_scale(E1, Tt),
                    "bias_before": (E0 - Tt).mean(0).tolist(), "bias_after": (E1 - Tt).mean(0).tolist()}
    # detection errors: the detected scale vs the map's true scale (1 / Umeyama), and the
    # translation left after the lattice shift (the offset's error)
    out["lattice"]["scale_detection_err_pct"] = 100 * (lat.scale * sb - 1)
    out["lattice"]["offset_err"] = float(np.abs((E1 - Tt).mean(0)).max())
    # configuration space
    pts = Tt
    d_est = grid.point_segment_distance(pts, m.collision)
    tcells = grid.CellGrid(0, 0, truth_grid(seed), np.ones_like(truth_grid(seed)), np.zeros(truth_grid(seed).shape))
    d_true = grid.point_segment_distance(pts, grid.collision_segments(tcells))
    out["true_pose_clearance"] = {"min_to_true_collision": float(d_true.min()), "min_to_true_rendered": float(grid.point_segment_distance(pts, T_segs).min())}
    iy, ix = SPEC.index(pts[:, 0], pts[:, 1])
    known_free = m.logodds[iy, ix] < 0
    tol = grid.ROBOT_RADIUS - 0.01
    out["cspace"] = {"in_cspace": float(((d_est >= tol) & known_free).mean()), "clear_of_snapped_walls": float((d_est >= tol).mean()),
                     "in_known_free": float(known_free.mean()), "true_walls_check": float((d_true >= tol).mean()),
                     "cspace_cells": int(m.cspace.sum())}
    return out


def umeyama_scale(est, true):
    mu_e, mu_t = est.mean(0), true.mean(0)
    E, T = est - mu_e, true - mu_t
    U, D, Vt = np.linalg.svd(T.T @ E / len(est))
    S = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        S[1, 1] = -1
    return float(np.trace(np.diag(D) @ S) / ((E ** 2).sum() / len(est)))


def run_seed(args):
    seed, params, with_sig = args
    t0 = time.perf_counter()
    S = scans(seed)
    P = est_poses(seed, params.get("poses", "stage6"))
    m = grid.build(S, P, SPAWN if params.get("anchor_spawn", True) else P[0][0], spec=SPEC, **params.get("build", {}))
    m.poses_in = P
    t1 = time.perf_counter()
    lo, hits = load_oracle(seed)
    r = score(seed, m, lo, hits, true_poses(seed))
    r["runtime_map_s"] = t1 - t0
    if params.get("poses", "stage6") != "stage6":
        ref = np.concatenate(est_poses(seed))
        X = np.concatenate(P)
        Tt = np.concatenate(true_poses(seed))
        r["e2e"] = {"max_pos_diff_vs_stage6": float(np.abs(X[:, :2] - ref[:, :2]).max()),
                    "rmse_vs_stage6": float(np.sqrt(((X[:, :2] - ref[:, :2]) ** 2).sum(1).mean())),
                    "raw_rmse_vs_truth": float(np.sqrt(((X[:, :2] - Tt[:, :2]) ** 2).sum(1).mean())),
                    "stage6_raw_rmse_vs_truth": float(np.sqrt(((ref[:, :2] - Tt[:, :2]) ** 2).sum(1).mean()))}
        return seed, r
    if with_sig:
        grid.add_signatures(m, signature_frames(seed, None))
        r["signatures"] = {"faces": len(m.signatures.keys), "repeatability": signature_repeatability(m, seed)}
        d = CACHE / str(seed)
        grid.save(m, d / "map.npz")
        grid.render(m, d / "map.png", f"seed {seed}: occupancy (log-odds) and optimised trajectories")
        grid.render(m, d / "map_snapped.png", f"seed {seed}: snapped walls (black), true walls (orange), free space for the robot centre (green)",
                    snapped=True, truth_segments=cf.wall_segments(truth_grid(seed)))
    return seed, r


def run(split: str, params=None, with_sig=True, tag="") -> dict:
    seeds = DEV if split == "dev" else TEST if split == "test" else [int(x) for x in split.split(",")]
    params = params or {}
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(run_seed, [(s, params, with_sig) for s in seeds]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{split}{tag}.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
    return res


def summary(res) -> dict:
    keys = {
        "iou_vs_oracle": lambda r: r["iou_vs_oracle"],
        "iou_vs_observable": lambda r: r["iou_vs_observable"],
        "iou_vs_full": lambda r: r["iou_vs_full_layout"],
        "snap_recall_obs": lambda r: r["snap"]["recall_observable"],
        "snap_recall_all": lambda r: r["snap"]["recall_all"],
        "snap_false": lambda r: r["snap"]["false_rate"],
        "scale_det_err_pct": lambda r: r["lattice"]["scale_detection_err_pct"],
        "offset_err": lambda r: r["lattice"]["offset_err"],
        "in_cspace": lambda r: r["cspace"]["in_cspace"],
    }
    return {k: [round(f(r), 4) for r in res.values()] for k, f in keys.items()}


def figures(split="test"):
    """Per seed: the oracle map (true poses), the map from the stage 6 poses, and the snapped
    walls with the true layout and the configuration space."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((OUT / f"results_{split}.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    for seed in DEV if split == "dev" else TEST:
        z = np.load(CACHE / str(seed) / "map.npz", allow_pickle=True)
        lo, _ = load_oracle(seed)
        r = res[str(seed)]
        fig, axs = plt.subplots(1, 3, figsize=(21, 7.4))
        ext = SPEC.extent()
        axs[0].imshow(np.clip(lo, -5, 5), cmap="gray_r", vmin=-5, vmax=5, origin="lower", extent=ext)
        grid.draw_trajectories(axs[0], true_poses(seed))
        axs[0].set_title(f"oracle: same scans at the TRUE poses", fontsize=10)
        axs[1].imshow(np.clip(z["logodds"], -5, 5), cmap="gray_r", vmin=-5, vmax=5, origin="lower", extent=ext)
        P = [z["poses"][z["poses"][:, 0] == d][:, 2:] for d in range(3)]
        grid.draw_trajectories(axs[1], P)
        axs[1].set_title(f"estimate: stage 6 poses, lattice-corrected; IoU vs oracle {r['iou_vs_oracle']:.2f}", fontsize=10)
        free = np.where(z["cspace"], 1.0, np.nan)
        axs[2].imshow(free, cmap="Greens", vmin=0, vmax=3, origin="lower", extent=ext, alpha=0.6)
        for s in cf.wall_segments(truth_grid(seed)):
            axs[2].plot([s[0], s[2]], [s[1], s[3]], "-", color="tab:orange", lw=4, alpha=0.6, solid_capstyle="butt")
        for s in z["wall_segments"]:
            axs[2].plot([s[0], s[2]], [s[1], s[3]], "k-", lw=1.4)
        sn = r["snap"]
        axs[2].set_title(f"snapped walls (black) over true walls (orange); green = robot-centre free space\n"
                         f"recall {sn['recall_all']:.3f}, false {sn['false_rate']:.3f}, true poses in free space {r['cspace']['in_cspace']:.3f}", fontsize=10)
        for ax in axs:
            ax.set_aspect("equal")
            ax.set_xlim(-0.25, 6.45)
            ax.set_ylim(-0.25, 6.45)
        fig.suptitle(f"seed {seed} ({split}), gt4 frames", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / f"stage7_maps_gt4_{seed}.png", dpi=80)
        plt.close(fig)


def sweep():
    grid_ = []
    for l_occ in (0.85, 1.5):
        for l_free in (-0.2, -0.4):
            for tau in (0.0, 1.0):
                grid_.append({"build": {"l_occ": l_occ, "l_free": l_free, "edge_tau": tau, "post_tau": tau}})
    out = []
    for p in grid_:
        res = run("dev", p, with_sig=False, tag="_sweep")
        s = summary(res)
        out.append({"params": p["build"], **{k: (float(np.max(np.abs(v))) if k in ("snap_false", "scale_det_err_pct", "offset_err") else float(np.min(v))) for k, v in s.items()}})
        print(json.dumps(out[-1]), flush=True)
    (OUT / "sweep_dev.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["oracle", "run", "sweep", "figures"])
    p.add_argument("split", nargs="?", default="dev")
    p.add_argument("--params", default="{}")
    p.add_argument("--nosig", action="store_true")
    a = p.parse_args()
    if a.step == "oracle":
        print(json.dumps(oracle(), indent=1))
    elif a.step == "run":
        prm = json.loads(a.params)
        res = run(a.split, prm, not a.nosig, "_e2e" if prm.get("poses") == "build" else "")
        print(json.dumps(summary(res)))
    elif a.step == "figures":
        figures(a.split)
    else:
        sweep()


if __name__ == "__main__":
    main()
