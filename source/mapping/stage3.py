"""Stage 3 evaluation: heading accuracy per cue and fused, focal length, level-camera check,
axis offset, wall-top ranging, and the stage 2 criteria re-checked with wall-top ranging on.
EVALUATION CODE: it reads ground truth (rule 10); the pipeline modules never do.

    uv run source/mapping/stage3.py            # needs the cached cues: stage3_extract.py
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
import camera_fit as cf  # noqa: E402
import heading as hd  # noqa: E402
import record_gt  # noqa: E402
import scan as sc  # noqa: E402
import scan_heading as sh  # noqa: E402
import se2  # noqa: E402
import segment as sg  # noqa: E402
import stage2 as s2  # noqa: E402
import stage3_extract as ex  # noqa: E402
import vp  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
FIG = ROOT / "reports" / "figures"
DEV, TEST = ex.CFG["dev"], ex.CFG["test"]
CAM = sc.camera_config()
RESULTS = CACHE / "stage3_results.json"
VARIANTS = ("official", "msaa4")
THRESH = 3.0  # degrees


def load(seeds, variant="official", keep_scans=False) -> dict:
    d = {k: [] for k in ("lsd", "poly", "sh", "tp", "seed", "drive", "k")}
    scans = {"rows": [], "top": [], "status": []}
    for s in seeds:
        res = ex.run_seed(s, variant=variant)
        poses = record_gt.load_poses(record_gt.gt_dir(s))
        for dr in range(3):
            n = len(res[dr]["rows"])
            d["lsd"].append(res[dr]["vp_lsd"])
            d["poly"].append(res[dr]["vp_poly"])
            d["sh"].append(res[dr]["sh"])
            d["tp"].append(vp.true_psi(poses[dr][:, 2]))
            d["seed"].append(np.full(n, s))
            d["drive"].append(np.full(n, dr))
            d["k"].append(np.arange(n))
            if keep_scans:
                for k in scans:
                    scans[k].append(res[dr][k])
    out = {k: np.concatenate(v) for k, v in d.items()}
    if keep_scans:
        out.update({k: np.concatenate(v) for k, v in scans.items()})
    return out


def vp_result(row) -> vp.VPResult:
    psi, sig, sup, nin, fa, fb, le, ri, fi, nseg = row
    return vp.VPResult(
        psi=psi,
        sigma=sig,
        support=sup,
        n_inliers=int(nin) if np.isfinite(nin) else 0,
        family_len=(fa, fb),
        left_len=le,
        right_len=ri,
        f_implied=fi,
        alpha_a=psi,
        n_segments=int(nseg) if np.isfinite(nseg) else 0,
    )


def sh_result(row) -> sh.ScanHeading:
    psi, sig, sup, npc, nin, fx, fy = row
    return sh.ScanHeading(
        psi,
        sig,
        sup,
        int(npc) if np.isfinite(npc) else 0,
        int(nin) if np.isfinite(nin) else 0,
        (fx, fy),
    )


def fused(D) -> dict:
    out = {k: [] for k in ("psi", "sigma", "valid", "conf", "source", "delta", "vp_valid")}
    for a, b in zip(D["lsd"], D["sh"], strict=True):
        h = hd.fuse(vp_result(a), sh_result(b), CAM.fx)
        out["psi"].append(h.psi)
        out["sigma"].append(h.sigma)
        out["valid"].append(h.valid)
        out["conf"].append(h.confidence)
        out["source"].append(h.source)
        out["delta"].append(h.delta)
        out["vp_valid"].append(h.vp_valid)
    return {k: np.array(v) for k, v in out.items()}


def stats(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"n": 0}
    return {
        "n": int(len(err)),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(err.max()),
        "frac_under_3deg": float(np.mean(err < THRESH)),
    }


def heading_table(D, F) -> dict:
    tp = D["tp"]
    out = {"frames": int(len(tp))}
    for name in ("lsd", "poly", "sh"):
        e = np.abs(vp.diff90(D[name][:, 0], tp))
        has = np.isfinite(e)
        out[name] = {"has_estimate": float(has.mean()), **stats(e[has])}
    # the vanishing-point cue under its own validity test
    e = np.abs(vp.diff90(D["lsd"][:, 0], tp))
    vv = F["vp_valid"] & np.isfinite(e)
    out["lsd_with_validity_test"] = {
        "valid_fraction": float(vv.mean()),
        **stats(e[vv]),
        "invalid_frac_under_3deg": float(np.mean(e[~vv & np.isfinite(e)] < THRESH)),
    }
    ef = np.abs(vp.diff90(F["psi"], tp))
    valid = F["valid"]
    out["fused"] = {
        "valid_fraction": float(valid.mean()),
        **stats(ef[valid]),
        "conflict_fraction": float(np.mean(F["source"] == "conflict")),
        "signed_median_error": float(np.median(vp.diff90(F["psi"][valid], tp[valid]))),
        "signed_mean_error": float(np.mean(vp.diff90(F["psi"][valid], tp[valid]))),
    }
    # what the unvalidated frames would have scored (validity must separate good from bad)
    ev = np.minimum(np.abs(vp.diff90(D["lsd"][:, 0], tp)), np.abs(vp.diff90(D["sh"][:, 0], tp)))
    out["fused"]["invalid_frames"] = int((~valid).sum())
    out["fused"]["invalid_frac_where_a_cue_is_under_3deg"] = (
        float(np.mean(ev[~valid] < THRESH)) if (~valid).any() else None
    )
    # focal length from two vanishing points
    fi = D["lsd"][:, 8]
    two = np.isfinite(fi) & (np.minimum(D["lsd"][:, 4], D["lsd"][:, 5]) >= hd.VP_TWO_MIN_LEN)
    rel = fi[two] / CAM.fx - 1.0
    out["focal"] = {
        "frames_with_two_vps": int(two.sum()),
        "median_f": float(np.median(fi[two])) if two.any() else None,
        "median_rel_error": float(np.median(rel)) if two.any() else None,
        "p5_p95_rel_error": [float(np.percentile(rel, 5)), float(np.percentile(rel, 95))]
        if two.any()
        else None,
        "frac_within_5pct": float(np.mean(np.abs(rel) < 0.05)) if two.any() else None,
    }
    return out


# ----------------------------------------------------------------------- level camera


def level_check(seeds=DEV, every=20) -> dict:
    """Near-vertical LSD segments on the grey image: with a level camera (no pitch or roll)
    vertical 3-D lines stay vertical in the image, so their angle should be 90 +/- 1 degree.
    Short segments are dominated by texture edges and LSD quantisation, so the spread is
    reported by segment length, and a plane fit of the deviation against image position tests
    for pitch (which tilts vertical lines in proportion to their distance from the centre)."""
    lsd = vp.make_lsd()
    rows = []
    for s in seeds:
        for d in range(3):
            path = record_gt.gt_dir(s) / f"traj_{d}"
            info = json.loads((path / "data_info.json").read_text())
            for k in range(0, len(info), every):
                g = cv2.imread(str(path / info[k]["image"]), cv2.IMREAD_GRAYSCALE)
                lines = lsd.detect(g)[0]
                if lines is None:
                    continue
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    length = np.hypot(x2 - x1, y2 - y1)
                    if length < 25:
                        continue
                    top, bot = ((x1, y1), (x2, y2)) if y1 < y2 else ((x2, y2), (x1, y1))
                    dev = np.degrees(np.arctan2(top[0] - bot[0], bot[1] - top[1]))
                    if abs(dev) < 10.0:  # signed lean of the top: positive to the right
                        rows.append((dev, length, (x1 + x2) / 2, (y1 + y2) / 2))
    a = np.array(rows)
    table = []
    for lmin in (25, 40, 60, 80, 100):
        m = a[:, 1] >= lmin
        d = a[m, 0]
        table.append(
            {
                "min_length_px": lmin,
                "segments": int(m.sum()),
                "median_deg": float(np.median(d)),
                "within_1deg": float(np.mean(np.abs(d) < 1.0)),
                "robust_sigma_deg": float(1.4826 * np.median(np.abs(d - np.median(d)))),
                "p5_p95_deg": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
            }
        )
    m = a[:, 1] >= 60
    u = (a[m, 2] - CAM.cx) / CAM.fx
    v = (a[m, 3] - CAM.cy) / CAM.fy
    coef = np.linalg.lstsq(np.c_[u, v, np.ones(m.sum())], a[m, 0], rcond=None)[0]
    return {
        "segments": int(len(a)),
        "by_length": table,
        "plane_fit_deg_per_unit_u_v_const": [float(c) for c in coef],
        "deviations": a[a[:, 1] >= 60, 0],
    }


# ------------------------------------------------------------ wall top and stage 2 re-check


def xz_from_arrays(rows, top, status):
    """Vectorised scan columns for floor-edge returns and wall-top returns: (X, Z)."""
    u = np.arange(rows.shape[-1]) + 0.5
    z = np.full(rows.shape, np.nan)
    ok = (status == sg.WALL_RETURN) & (rows > CAM.cy + 1e-6)
    z[ok] = (CAM.fy * CAM.h / (rows - CAM.cy))[ok]
    tp_ = status == sg.TOP_RETURN
    z[tp_] = sc.top_depth(top[tp_], CAM)
    x = z * (u - CAM.cx) / CAM.fx
    return x, z


def walltop_and_stage2(seeds, variant, half, step=3, r_max=sc.R_MAX) -> dict:
    """Wall-top accuracy by range, and the stage 2 criteria with and without wall-top returns."""
    rec = {
        k: []
        for k in (
            "z_true",
            "z_floor",
            "z_top",
            "status",
            "dist_floor",
            "dist_both",
            "rng_floor",
            "rng_both",
        )
    }
    for s in seeds:
        res = ex.run_seed(s, variant=variant)
        occ = s2.occupancy(s)
        scene = cf.Scene(occ, half=half)
        poses = record_gt.load_poses(record_gt.gt_dir(s))
        cols = np.arange(320)
        for d in range(3):
            r_, t_, st_ = res[d]["rows"], res[d]["top"], res[d]["status"]
            for k in range(0, len(r_), step):
                x, y, yaw = poses[d][k]
                ztrue = cf.raycast_depths(scene, (x, y), float(yaw), cols, CAM.fx, CAM.cx)
                xf, zf = xz_from_arrays(
                    r_[k], t_[k], np.where(st_[k] == sg.TOP_RETURN, sg.INVALID, st_[k])
                )
                xb, zb = xz_from_arrays(r_[k], t_[k], st_[k])
                zt = np.full(320, np.nan)
                okt = np.isfinite(t_[k]) & (t_[k] < CAM.cy - 0.5)
                zt[okt] = sc.top_depth(t_[k][okt], CAM)
                dist = {}
                for name, xx, zz in (("floor", xf, zf), ("both", xb, zb)):
                    dd = np.full(320, np.nan)
                    h = np.isfinite(zz) & (np.hypot(xx, zz) <= r_max)
                    if h.any():
                        pr = np.stack([zz[h], -xx[h]], 1)
                        pw = se2.compose(np.array([x, y, yaw]), np.c_[pr, np.zeros(len(pr))])[:, :2]
                        dd[h] = s2.point_segment_distance(pw, scene.segments)
                    dist[name] = dd
                rec["z_true"].append(ztrue)
                rec["z_floor"].append(zf)
                rec["z_top"].append(zt)
                rec["status"].append(st_[k])
                rec["dist_floor"].append(dist["floor"])
                rec["dist_both"].append(dist["both"])
                rec["rng_floor"].append(np.hypot(xf, zf))
                rec["rng_both"].append(np.hypot(xb, zb))
    R = {k: np.concatenate(v) for k, v in rec.items()}
    out = {}
    # (1) depth error of the two edges by true depth, where both exist
    both = np.isfinite(R["z_floor"]) & np.isfinite(R["z_top"]) & np.isfinite(R["z_true"])
    bins = [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)]
    rows = []
    for lo, hi in bins:
        m = both & (R["z_true"] >= lo) & (R["z_true"] < hi)
        if m.sum() < 30:
            continue
        ef, et = R["z_floor"][m] - R["z_true"][m], R["z_top"][m] - R["z_true"][m]
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(m.sum()),
                "floor_median_abs": float(np.median(np.abs(ef))),
                "floor_p95_abs": float(np.percentile(np.abs(ef), 95)),
                "top_median_abs": float(np.median(np.abs(et))),
                "top_p95_abs": float(np.percentile(np.abs(et), 95)),
                "top_rel_bias": float(np.median(et / R["z_true"][m])),
            }
        )
    out["edge_error_by_depth"] = rows
    # (2) the wall-top returns that fill the INVALID columns
    tops = R["status"] == sg.TOP_RETURN
    inv_true = tops & np.isfinite(R["z_true"])
    et = R["z_top"][inv_true] - R["z_true"][inv_true]
    out["top_returns"] = {
        "fraction_of_columns": float(tops.mean()),
        "columns_still_invalid": float((R["status"] == sg.INVALID).mean()),
        "n_with_truth": int(inv_true.sum()),
        "true_depth_pct_5_50_95": [
            float(v) for v in np.percentile(R["z_true"][inv_true], [5, 50, 95])
        ],
        "abs_error_median_m": float(np.median(np.abs(et))),
        "abs_error_p95_m": float(np.percentile(np.abs(et), 95)),
        "abs_error_p99_m": float(np.percentile(np.abs(et), 99)),
    }
    # (3) stage 2 criteria, with and without the wall-top returns
    for name in ("floor", "both"):
        d = R[f"dist_{name}"]
        hit = np.isfinite(d)
        # signed range error by range bin, as stage 2
        rng_true = R["z_true"] * np.sqrt(
            1 + ((np.tile(np.arange(320) + 0.5, len(R["z_true"]) // 320) - CAM.cx) / CAM.fx) ** 2
        )
        rel = (R[f"rng_{name}"] - rng_true) / rng_true
        bias = []
        for lo in np.arange(0.0, r_max, 0.25):
            m = hit & (rng_true >= lo) & (rng_true < lo + 0.25) & np.isfinite(rel)
            if m.sum() >= 20:
                bias.append(float(np.median(rel[m])))
        cover = np.mean(hit.reshape(-1, 320), axis=1)
        out[f"stage2_check_{name}"] = {
            "returns": int(hit.sum()),
            "dist_median_m": float(np.median(d[hit])),
            "dist_p95_m": float(np.percentile(d[hit], 95)),
            "dist_p99_m": float(np.percentile(d[hit], 99)),
            "max_abs_bin_bias": float(np.max(np.abs(bias))),
            "coverage_mean": float(cover.mean()),
            "coverage_p5": float(np.percentile(cover, 5)),
        }
    out["_R"] = R
    return out


# --------------------------------------------------------------------------------- figures


def plot_hist(tabs):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for a, variant in zip(ax, VARIANTS, strict=True):
        D, F = tabs[variant]["D"], tabs[variant]["F"]
        tp = D["tp"]
        bins = np.linspace(-6, 6, 121)
        for name, label, c in (
            ("lsd", "vanishing point", "tab:blue"),
            ("sh", "scan", "tab:orange"),
        ):
            e = vp.diff90(D[name][:, 0], tp)
            a.hist(
                np.clip(e[np.isfinite(e)], -6, 6),
                bins,
                histtype="step",
                color=c,
                label=f"{label} alone (all frames)",
            )
        e = vp.diff90(F["psi"][F["valid"]], tp[F["valid"]])
        a.hist(np.clip(e, -6, 6), bins, color="tab:green", alpha=0.5, label="fused, valid frames")
        a.set_yscale("log")
        a.set_xlabel("yaw error mod 90 [deg] (clipped at +-6)")
        a.set_ylabel("frames")
        a.set_title(f"test seeds, {variant} frames")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage3_yaw_error_hist.png", dpi=90)
    plt.close(fig)


def plot_confidence(tabs):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for variant, c in zip(VARIANTS, ("tab:blue", "tab:orange"), strict=True):
        D, F = tabs[variant]["D"], tabs[variant]["F"]
        tp = D["tp"]
        both = np.isfinite(F["delta"])
        ef = np.abs(vp.diff90(F["psi"], tp))
        # error of each cue against the disagreement between the cues
        edges = [0, 0.25, 0.5, 1, 2, 3, 5, 10, 30, 45.1]
        mids, med, p95, frac = [], [], [], []
        el = np.abs(vp.diff90(D["lsd"][:, 0], tp))
        es = np.abs(vp.diff90(D["sh"][:, 0], tp))
        ev = np.minimum(el, es)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = both & (F["delta"] >= lo) & (F["delta"] < hi)
            if m.sum() < 20:
                continue
            mids.append(f"{lo}-{hi}")
            frac.append(np.mean(ev[m] < THRESH))
            med.append(np.median(el[m]))
            p95.append(np.percentile(el[m], 95))
        ax[0].plot(range(len(mids)), med, "o-", color=c, label=f"{variant}: VP median")
        ax[0].plot(range(len(mids)), p95, "s--", color=c, label=f"{variant}: VP p95")
        ax[0].set_xticks(range(len(mids)), mids, rotation=45, fontsize=7)
        ax[1].plot(range(len(mids)), frac, "o-", color=c, label=variant)
        ax[1].set_xticks(range(len(mids)), mids, rotation=45, fontsize=7)
        # fused error against confidence (valid frames)
        v = F["valid"]
        cb = [0, 0.25, 0.5, 0.75, 0.9, 1.0001]
        xs, m50, m95 = [], [], []
        for lo, hi in zip(cb[:-1], cb[1:], strict=True):
            m = v & (F["conf"] >= lo) & (F["conf"] < hi)
            if m.sum() < 20:
                continue
            xs.append(f"{lo}-{min(hi, 1.0)}")
            m50.append(np.median(ef[m]))
            m95.append(np.percentile(ef[m], 95))
        ax[2].plot(range(len(xs)), m50, "o-", color=c, label=f"{variant}: median")
        ax[2].plot(range(len(xs)), m95, "s--", color=c, label=f"{variant}: p95")
        ax[2].set_xticks(range(len(xs)), xs, fontsize=8)
    ax[0].set_xlabel("disagreement between the two cues [deg]")
    ax[0].set_ylabel("vanishing-point error [deg]")
    ax[0].set_yscale("log")
    ax[1].set_xlabel("disagreement between the two cues [deg]")
    ax[1].set_ylabel("fraction of frames where a cue is under 3 deg")
    ax[2].set_xlabel("fused confidence = 1 - disagreement / 3 deg (valid frames)")
    ax[2].set_ylabel("fused yaw error [deg]")
    ax[2].axhline(THRESH, color="g", lw=0.8)
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle("test seeds")
    fig.tight_layout()
    fig.savefig(FIG / "stage3_error_vs_confidence.png", dpi=90)
    plt.close(fig)


def plot_focal(tabs):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, c in zip(VARIANTS, ("tab:blue", "tab:orange"), strict=True):
        D = tabs[variant]["D"]
        fi = D["lsd"][:, 8]
        two = np.isfinite(fi) & (np.minimum(D["lsd"][:, 4], D["lsd"][:, 5]) >= hd.VP_TWO_MIN_LEN)
        ax.hist(
            np.clip(fi[two] / CAM.fx - 1, -0.3, 0.3),
            np.linspace(-0.3, 0.3, 80),
            histtype="step",
            color=c,
            label=f"{variant}: median {100 * np.median(fi[two] / CAM.fx - 1):+.1f} %, "
            f"n={two.sum()}",
        )
    ax.axvline(0, color="k", lw=0.5)
    ax.axvspan(-0.05, 0.05, color="g", alpha=0.1)
    ax.set_xlabel("focal length implied by two vanishing points / fx - 1")
    ax.set_ylabel("frames (test seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage3_focal_length.png", dpi=90)
    plt.close(fig)


def plot_level(lv):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    a = lv["deviations"]
    ax.hist(np.clip(a, -5, 5), np.linspace(-5, 5, 101))
    ax.axvspan(-1, 1, color="g", alpha=0.1)
    ax.set_xlabel("deviation of near-vertical segments from vertical [deg]")
    ax.set_ylabel("segments (dev seeds)")
    r = next(t for t in lv["by_length"] if t["min_length_px"] == 60)
    ax.set_title(
        f"segments >= 60 px: median {r['median_deg']:+.2f} deg, "
        f"{100 * r['within_1deg']:.0f} % within 1 deg"
    )
    fig.tight_layout()
    fig.savefig(FIG / "stage3_level_camera.png", dpi=90)
    plt.close(fig)


def plot_walltop(wt):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    rows = wt["official"]["edge_error_by_depth"]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["floor_p95_abs"] for r in rows], 0.4, label="floor edge, p95")
    ax[0].bar(x + 0.2, [r["top_p95_abs"] for r in rows], 0.4, label="wall top, p95")
    ax[0].plot(x - 0.2, [r["floor_median_abs"] for r in rows], "k.", label="floor edge, median")
    ax[0].plot(x + 0.2, [r["top_median_abs"] for r in rows], "r.", label="wall top, median")
    ax[0].set_xticks(x, [f"{r['lo']}-{r['hi']}" for r in rows])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("true depth [m]")
    ax[0].set_ylabel("absolute depth error [m]")
    ax[0].set_title("test seeds, official frames, columns with both edges")
    ax[0].legend(fontsize=8)
    R = wt["official"]["_R"]
    inv = R["status"] == sg.TOP_RETURN
    ok = inv & np.isfinite(R["z_true"])
    ax[1].hist(
        np.clip((R["z_top"][ok] - R["z_true"][ok]) * 1000, -10, 10), np.linspace(-10, 10, 81)
    )
    ax[1].set_xlabel("wall-top depth error for columns the floor edge could not range [mm]")
    ax[1].set_ylabel("columns")
    fig.tight_layout()
    fig.savefig(FIG / "stage3_walltop_error.png", dpi=90)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if "--level-only" in sys.argv:
        lv = level_check()
        plot_level(lv)
        res = json.loads(RESULTS.read_text())
        res["level_camera"] = {k: v for k, v in lv.items() if k != "deviations"}
        RESULTS.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["level_camera"], indent=1))
        return
    half = json.loads((CACHE / "stage2_results.json").read_text())["task0"][
        "chosen_half_thickness_m"
    ]
    res: dict = {"tau_deg": hd.TAU}
    tabs = {}
    for variant in VARIANTS:
        for split, seeds in (("dev", DEV), ("test", TEST)):
            D = load(seeds, variant)
            F = fused(D)
            res.setdefault(variant, {})[split] = heading_table(D, F)
            if split == "test":
                tabs[variant] = {"D": D, "F": F}
        print(variant, json.dumps(res[variant]["test"]["fused"], indent=0)[:600], flush=True)
    plot_hist(tabs)
    plot_confidence(tabs)
    plot_focal(tabs)
    lv = level_check()
    plot_level(lv)
    res["level_camera"] = {k: v for k, v in lv.items() if k != "deviations"}
    print("level", res["level_camera"], flush=True)
    wt = {}
    for variant in VARIANTS:
        wt[variant] = walltop_and_stage2(TEST, variant, half)
    plot_walltop(wt)
    res["walltop"] = {v: {k: x for k, x in w.items() if k != "_R"} for v, w in wt.items()}
    RESULTS.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["walltop"]["official"], indent=1)[:3000])


if __name__ == "__main__":
    main()
