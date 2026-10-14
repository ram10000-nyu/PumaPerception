"""Stage 0 revision 2, tasks 6 and 8: camera height, camera offset, far plane.

    uv run source/mapping/stage0_camera.py

Writes ``reports/figures/stage0_height_fit.png``, ``stage0_rotation_parallax.png``,
``stage0_farplane.png`` and the camera fields of ``cache/mapping/camera.json``; prints the
numbers the report quotes (also saved to ``cache/mapping/stage0_camera_results.json``).
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
import vis_nav_sim as sim  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_fit as cf  # noqa: E402
import record_gt  # noqa: E402

ROOT = record_gt.ROOT
FIG = ROOT / "reports" / "figures"
CAM_JSON = ROOT / "cache" / "mapping" / "camera.json"
FX = FY = 92.0
CX, CY = 160.0, 120.0
DIRS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # yaw = k * 90 deg


def calibration_textures(tex: sim.Textures) -> sim.Textures:
    """The same pack with a solid green floor, so the floor segments exactly (the real
    floor is two textures with distance fog). Geometry and camera are unchanged."""
    d = ROOT / "cache" / "mapping" / "calibration_floor"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "floor.png"), np.full((256, 256, 3), (0, 255, 0), np.uint8))
    return sim.Textures(tex.directory, floor=d / "floor.png")


def green_floor(img: np.ndarray) -> np.ndarray:
    """Exact calibration floor, (0, 232, 0) BGR. Strict on purpose: a pixel that blends floor
    and wall (MSAA) must not count as floor, or the sub-pixel boundary estimate is biased."""
    p = img.astype(np.int16)
    return (p[..., 0] <= 4) & (p[..., 2] <= 4) & (np.abs(p[..., 1] - 232) <= 4)


def corridor_lines(occ: np.ndarray):
    """Every (start cell, direction k, n free cells ahead) whose ray ends at a wall face."""
    rows, cols = occ.shape
    out = []
    for r in range(rows):
        for c in range(cols):
            if occ[r, c]:
                continue
            for k, (dx, dy) in DIRS.items():
                n = 0
                rr, cc = r, c
                while 0 <= rr < rows and 0 <= cc < cols and not occ[rr, cc]:
                    n += 1
                    rr += dy
                    cc += dx
                if 0 <= rr < rows and 0 <= cc < cols and occ[rr, cc]:
                    out.append((r, c, k, n))
    return out


# ------------------------------------------------------------------------------ task 6a: height


def height_samples(world: sim.Simulator, rng, per_line=6, min_run=1.0, max_d=3.0):
    """Render the camera facing a wall face squarely at distances 0.25 .. min(run, 3) m."""
    occ = world.occupancy
    lines = corridor_lines(occ)
    # keep only lines starting at the far end of a corridor (a wall behind), so each
    # corridor/direction is used once, with the longest span
    best = {}
    for r, c, k, n in lines:
        key = (
            k,
            r if k in (0, 2) else c,
            (c + n * DIRS[k][0]) if k in (0, 2) else (r + n * DIRS[k][1]),
        )
        if key not in best or n > best[key][3]:
            best[key] = (r, c, k, n)
    cand = [v for v in best.values() if v[3] * 0.2 >= min_run]
    cand.sort(key=lambda v: -v[3])
    out = []
    for r, c, k, n in cand[:40]:
        dx, dy = DIRS[k]
        face = np.array([(c + n * dx) * 0.2, (r + n * dy) * 0.2])  # wall plane (cell centre)
        run = n * 0.2 - 0.1
        for d in rng.uniform(0.25, min(run, max_d), per_line):
            pos = face - d * np.array([dx, dy])
            out.append((pos, k * np.pi / 2, float(d)))
    return out


def measure_height_samples(world, samples):
    ds, rows, tops = [], [], []
    scene = cf.Scene(world.occupancy)
    for pos, yaw, d in samples:
        img = world.render(float(pos[0]), float(pos[1]), float(yaw))
        # columns where the ray reaches the facing wall squarely (within 1 px of its edge)
        half = max(0.0, 0.2 / d * FX - 1.5)  # the end wall spans +/- 0.2 m between side walls
        cols = np.arange(int(np.ceil(CX - half - 0.5)), int(np.floor(CX + half - 0.5)) + 1)
        if len(cols) == 0:
            cols = np.array([159, 160])
        z = cf.raycast_depths(scene, pos, yaw, cols, FX, CX)
        ok = np.isfinite(z) & (np.abs(z - d) < 0.003 * max(d, 1))
        if ok.sum() == 0:
            continue
        b = cf.boundary_rows(img, cols[ok], green_floor)
        b = b[np.isfinite(b)]
        if len(b) == 0:
            continue
        # wall top: first non-sky pixel from the top, blended with sky
        top = np.nan
        c0 = cols[ok]
        tvals = []
        for u in c0:
            sm = cf.sky_mask(img[:, u : u + 1]).ravel()
            j = 0
            while j < 240 and sm[j]:
                j += 1
            if j == 0 or j >= 238:
                continue
            wall_c = img[j + 1, u].astype(float)
            d_ = cf.SKY.astype(float) - wall_c
            share = np.clip((img[j, u].astype(float) - wall_c) @ d_ / (d_ @ d_), 0, 1)
            tvals.append(j + share)
        top = float(np.mean(tvals)) if tvals else np.nan
        ds.append(d)
        rows.append(float(np.mean(b)))
        tops.append(top)
    return np.array(ds), np.array(rows), np.array(tops)


def task_height(tex, seeds, out):
    res = {}
    allds, allrows = [], []
    for msaa in (4, 1):
        D, R, T = [], [], []
        for seed in seeds:
            world = sim.Simulator(calibration_textures(tex), seed, msaa=msaa)
            rng = np.random.default_rng(seed)
            s = height_samples(world, rng)
            d, r, t = measure_height_samples(world, s)
            D.append(d)
            R.append(r)
            T.append(t)
        D, R, T = map(np.concatenate, (D, R, T))
        fit_a = cf.fit_height(D, R, FY, cy=CY)
        fit_b = cf.fit_height(D, R, FY, cy=None)
        fit_c = cf.fit_height(D, R, FY, cy=CY, offset=False)
        res[f"msaa{msaa}"] = {
            "fit_h_a_cy120": fit_a,
            "fit_h_a_cyfree": fit_b,
            "fit_h_only_cy120": fit_c,
        }
        if msaa == 4:
            allds, allrows = D, R
            ok = np.isfinite(T)
            if ok.sum() > 5:
                top_fit = cf.fit_height(
                    D[ok], 2 * CY - T[ok], FY, cy=CY, offset=True
                )  # (H - h) and axial offset
                res["msaa4"]["wall_top"] = {
                    "H_minus_h": top_fit["h"],
                    "H_minus_h_ci95": top_fit["h_ci95"],
                    "a": top_fit["a"],
                    "rms_px": top_fit["rms_px"],
                    "n": top_fit["n"],
                }
    fit = res["msaa4"]["fit_h_a_cy120"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    inv = 1 / (allds - fit["a"])
    ax[0].plot(inv, allrows, ".", ms=4, label=f"renders (n={len(allds)})")
    xs = np.linspace(inv.min(), inv.max(), 50)
    ax[0].plot(
        xs,
        CY + FY * fit["h"] * xs,
        "r-",
        label=f"h={fit['h'] * 1000:.1f} mm, a={fit['a'] * 1000:.1f} mm",
    )
    ax[0].set_xlabel("1 / (d - a)  [1/m]")
    ax[0].set_ylabel("floor-wall boundary row [px]")
    ax[0].legend()
    ax[0].set_title("height fit")
    pred = CY + FY * fit["h"] / (allds - fit["a"])
    ax[1].plot(allds, allrows - pred, ".", ms=4)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("true distance to face d [m]")
    ax[1].set_ylabel("residual [px]")
    ax[1].set_title(f"RMS {fit['rms_px']:.3f} px")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    res["samples"] = {"d": allds.tolist(), "rows": allrows.tolist()}
    return res


# --------------------------------------------------------------------- task 6b: rotation offset


def rotate_in_place_samples(world, rng, n_pos=6, step_deg=15):
    """Renders at a few free cells over a full turn in ``step_deg`` steps."""
    free = np.argwhere(~world.occupancy[2:-2, 2:-2]) + 2
    pick = free[rng.choice(len(free), n_pos, replace=False)]
    out = []
    for r, c in pick:
        pos = np.array([c * 0.2, r * 0.2])
        yaws = np.deg2rad(np.arange(0, 360, step_deg))
        imgs = [world.render(float(pos[0]), float(pos[1]), float(y)) for y in yaws]
        out.append((pos, yaws, imgs))
    return out


def boundary_samples_for_offset(world, rot):
    scene = cf.Scene(world.occupancy)
    cols = np.arange(0, 320, 2)
    samples = []
    for pos, yaws, imgs in rot:
        for yaw, img in zip(yaws, imgs, strict=True):
            rows = cf.boundary_rows(img, cols, green_floor)
            z = cf.raycast_depths(scene, pos, float(yaw), cols, FX, CX)
            # keep columns away from wall corners (neighbouring columns agree on depth)
            zn = np.r_[z[1:], z[-1]]
            zp = np.r_[z[0], z[:-1]]
            smooth = (np.abs(zn - z) < 0.04 * z) & (np.abs(zp - z) < 0.04 * z)
            keep = np.isfinite(rows) & np.isfinite(z) & smooth & (z > 0.25) & (z < 4.0)
            samples.append(
                {
                    "scene": scene,
                    "pos": pos,
                    "yaw": float(yaw),
                    "cols": cols,
                    "rows": rows,
                    "keep": keep,
                }
            )
    return samples


def sift_pair_residuals(world, pos, yaws, imgs, offset, h_unused=None):
    """Reprojection residuals of SIFT matches between yaw-adjacent renders, predicted with
    the true grid depth and the camera at ``pos + offset`` in the robot frame."""
    sift = cv2.SIFT_create(nfeatures=300)
    bf = cv2.BFMatcher()
    rows = []
    n = len(imgs)
    for i in range(n):
        j = (i + 1) % n
        g1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2GRAY)
        k1, d1 = sift.detectAndCompute(g1, None)
        k2, d2 = sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            continue
        m = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.8 * b.distance]
        if not m:
            continue
        p1 = np.array([k1[a.queryIdx].pt for a in m])
        p2 = np.array([k2[a.trainIdx].pt for a in m])
        rows.append((yaws[i], yaws[j], p1, p2))
    return rows


def predict_second_view(scene, pos, yaw1, yaw2, p1, off, fx=FX, fy=FY, cx=CX, cy=CY):
    """Where keypoints of view 1 land in view 2 given the true grid and a camera offset."""
    a, l_ = off

    def cam_at(yaw):
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        return np.asarray(pos) + a * fwd + l_ * left

    c1, c2 = cam_at(yaw1), cam_at(yaw2)
    fwd1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    right1 = np.array([np.sin(yaw1), -np.cos(yaw1)])
    fwd2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    right2 = np.array([np.sin(yaw2), -np.cos(yaw2)])
    out = np.full_like(p1, np.nan)
    for n, (u, v) in enumerate(p1):
        d = fwd1 + (u - cx) / fx * right1
        t = scene.cast(c1, d[None])[0]
        if not np.isfinite(t) or t < 0.1:
            continue
        hit = c1 + t * d
        y_down = (v - cy) / fy * t
        w = hit - c2
        z2 = w @ fwd2
        if z2 < 0.05:
            continue
        out[n] = (cx + fx * (w @ right2) / z2, cy + fy * y_down / z2)
    return out


def task_offset(tex, seeds, out):
    res = {}
    # (i) boundary-row fit
    samples = []
    worlds = {}
    rot_all = {}
    cal = calibration_textures(tex)
    for seed in seeds:
        world = sim.Simulator(tex, seed, msaa=4)
        worlds[seed] = world
        rng = np.random.default_rng(100 + seed)
        rot = rotate_in_place_samples(world, rng, n_pos=4)  # real textures: SIFT needs texture
        rot_all[seed] = rot
        calw = sim.Simulator(cal, seed, msaa=4)
        rot_cal = [
            (p, y, [calw.render(float(p[0]), float(p[1]), float(a)) for a in y]) for p, y, _ in rot
        ]
        samples += boundary_samples_for_offset(calw, rot_cal)
    res["boundary_fit"] = cf.fit_offset(samples, FY, FX, CX)
    res["boundary_fit_cyfree"] = cf.fit_offset(samples, FY, FX, CX, free_cy=True)
    # (ii) SIFT parallax: residual of predicted keypoint positions, with and without offset
    pairs = []
    scenes = {seed: cf.Scene(worlds[seed].occupancy) for seed in seeds}
    for seed in seeds:
        for pos, yaws, imgs in rot_all[seed]:
            for y1, y2, p1, p2 in sift_pair_residuals(worlds[seed], pos, yaws, imgs, None):
                pairs.append((scenes[seed], pos, y1, y2, p1, p2))
    rng = np.random.default_rng(0)

    def gather(off, sub):
        r = []
        for scene, pos, y1, y2, p1, p2 in sub:
            pr = predict_second_view(scene, pos, y1, y2, p1, off)
            r.append(pr - p2)
        return np.concatenate(r)

    # gate gross mismatches using the no-offset prediction, then fit (a, l)
    sub = [pairs[i] for i in rng.choice(len(pairs), min(len(pairs), 120), replace=False)]
    gated = []
    for scene, pos, y1, y2, p1, p2 in sub:
        pr = predict_second_view(scene, pos, y1, y2, p1, (0.0, 0.0))
        e = np.linalg.norm(pr - p2, axis=1)
        k = np.isfinite(e) & (e < 25)
        if k.sum() >= 5:
            gated.append((scene, pos, y1, y2, p1[k], p2[k]))
    r0 = gather((0.0, 0.0), gated)
    r0n = np.linalg.norm(r0[np.isfinite(r0).all(1)], axis=1)

    def f(p):
        r = gather((p[0], p[1]), gated)
        r = np.nan_to_num(r, nan=0.0)
        return r.ravel()

    s = least_squares(
        f, [0.0, 0.0], loss="soft_l1", f_scale=1.0, diff_step=1e-2, x_scale=[0.05, 0.05]
    )
    r1 = gather(tuple(s.x), gated)
    r1n = np.linalg.norm(r1[np.isfinite(r1).all(1)], axis=1)
    cov = np.linalg.inv(s.jac.T @ s.jac) * float(s.fun @ s.fun) / max(len(s.fun) - 2, 1)
    res["sift_fit"] = {
        "a": float(s.x[0]),
        "l": float(s.x[1]),
        "a_ci95": float(1.96 * np.sqrt(cov[0, 0])),
        "l_ci95": float(1.96 * np.sqrt(cov[1, 1])),
        "n_matches": int(len(r1n)),
        "median_px_no_offset": float(np.median(r0n)),
        "rms_px_no_offset": float(np.sqrt(np.mean(r0n**2))),
        "median_px_with_offset": float(np.median(r1n)),
        "rms_px_with_offset": float(np.sqrt(np.mean(r1n**2))),
        "p90_px_with_offset": float(np.percentile(r1n, 90)),
        "median_px_with_offset_inliers_lt3": float(np.median(r1n[r1n < 3])),
        "median_px_zero_offset_inliers_lt3": float(np.median(r0n[r0n < 3])),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 8, 60)
    ax[0].hist(
        np.clip(r0n, 0, 8), bins, alpha=0.6, label=f"no offset (median {np.median(r0n):.2f} px)"
    )
    ax[0].hist(
        np.clip(r1n, 0, 8), bins, alpha=0.6, label=f"fitted offset (median {np.median(r1n):.2f} px)"
    )
    ax[0].set_xlabel("keypoint reprojection error [px]")
    ax[0].legend()
    ax[0].set_title("rotate in place, 15 deg steps, SIFT matches")
    ab = res["boundary_fit"]
    ax[1].bar(
        ["a (axial)", "l (lateral)"],
        [ab["a"] * 1000, ab["l"] * 1000],
        yerr=[ab["a_ci95"] * 1000, ab["l_ci95"] * 1000],
        label="boundary rows",
    )
    sf = res["sift_fit"]
    ax[1].bar(
        ["a (axial)", "l (lateral)"],
        [sf["a"] * 1000, sf["l"] * 1000],
        yerr=[sf["a_ci95"] * 1000, sf["l_ci95"] * 1000],
        width=0.3,
        label="SIFT parallax",
    )
    ax[1].set_ylabel("offset [mm]")
    ax[1].legend()
    ax[1].set_title("camera offset from the rotation centre")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- task 8: far plane


def task_farplane(tex, out):
    """What the renderer draws beyond the 5 m far plane, and how such columns classify."""
    world = sim.Simulator(tex, 7, msaa=4)
    calw = sim.Simulator(calibration_textures(tex), 7, msaa=4)
    occ = world.occupancy
    r, c, k, n = max(corridor_lines(occ), key=lambda v: v[3])  # longest run: 5.8 m
    views = {
        "inside, along the longest corridor (5.8 m)": ((c * 0.2, r * 0.2), k * np.pi / 2),
        "outside the maze, looking away (open floor)": ((-1.0, 3.0), np.pi),
        "inside, 4 m of corridor": ((c * 0.2, r * 0.2), k * np.pi / 2),
    }
    # the third: stand 4 m before the far wall of the same corridor
    dx, dy = DIRS[k]
    far = np.array([(c + n * dx) * 0.2, (r + n * dy) * 0.2])
    views["inside, 4 m of corridor"] = (tuple(far - 4.0 * np.array([dx, dy])), k * np.pi / 2)
    info = {"longest_run_m": n * 0.2, "views": {}}
    fig, ax = plt.subplots(2, len(views), figsize=(5 * len(views), 7.5))
    for i, (name, (p, y)) in enumerate(views.items()):
        img = world.render(float(p[0]), float(p[1]), float(y))
        cal = calw.render(float(p[0]), float(p[1]), float(y))
        ax[0, i].imshow(img[..., ::-1])
        ax[0, i].set_title(name, fontsize=8)
        ax[0, i].axis("off")
        gm = green_floor(cal)
        sm = cf.sky_mask(cal)
        ax[1, i].imshow(np.where(gm, 1.0, np.where(sm, 0.5, 0.0)), cmap="gray", vmin=0, vmax=1)
        ax[1, i].set_title("floor white, sky grey, wall black", fontsize=8)
        ax[1, i].axis("off")
        # per column: the topmost row of the floor region that connects to the bottom
        tops = []
        for u in range(320):
            j = 239
            while j >= 0 and gm[j, u]:
                j -= 1
            tops.append(j + 1)
        tops = np.array(tops)
        # the pixel directly above the floor top can be an MSAA blend; look two rows up
        above_sky = np.array(
            [(t - 2 >= 0) and bool(sm[t - 2, u]) for u, t in enumerate(tops)], dtype=bool
        )
        # depth of the floor top row (flat floor, h from the fit) for the columns that end in a wall
        hh = json.loads(CAM_JSON.read_text()).get("height_fit", {}).get("h", 0.15)
        info["views"][name] = {
            "columns_floor_ends_in_sky": int(above_sky.sum()),
            "columns_floor_ends_in_wall": int((~above_sky & (tops > 0)).sum()),
            "min_top_floor_row": int(tops.min()),
            "row_at_5m": float(CY + FY * hh / 5.0),
            "row_at_5.8m": float(CY + FY * hh / 5.8),
            "center_column_depth_m": float(FY * hh / (tops[160] - CY)) if tops[160] > CY else None,
        }
    # sky/background colour past the far plane in the real render
    img = world.render(*[float(v) for v in (-1.0, 3.0, np.pi)])
    info["pixel_colours_row_121_to_125_center"] = [img[i, 160].tolist() for i in range(121, 126)]
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return info


def main() -> None:
    cfg = json.loads(record_gt.SEEDS.read_text())
    tex = sim.Textures.find()
    FIG.mkdir(parents=True, exist_ok=True)
    results = {}
    results["height"] = task_height(tex, cfg["dev"], FIG / "stage0_height_fit.png")
    h = results["height"]["msaa4"]["fit_h_a_cy120"]
    print("height (msaa 4):", {k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()})
    print("height (msaa 1):", results["height"]["msaa1"]["fit_h_a_cy120"])
    print("cy free:", results["height"]["msaa4"]["fit_h_a_cyfree"])
    print("wall top:", results["height"]["msaa4"].get("wall_top"))
    results["offset"] = task_offset(tex, cfg["dev"][:2], FIG / "stage0_rotation_parallax.png")
    print("offset boundary fit:", results["offset"]["boundary_fit"])
    print("offset boundary fit cy free:", results["offset"]["boundary_fit_cyfree"])
    print("offset sift:", results["offset"]["sift_fit"])
    results["farplane"] = task_farplane(tex, FIG / "stage0_farplane.png")
    print("far plane:", results["farplane"])
    results["height"].pop("samples", None)
    cam = json.loads(CAM_JSON.read_text())
    h4, h1 = (
        results["height"]["msaa4"]["fit_h_a_cy120"],
        results["height"]["msaa1"]["fit_h_a_cy120"],
    )
    top = results["height"]["msaa4"]["wall_top"]
    sf = results["offset"]["sift_fit"]
    cam["height_fit"] = {
        "h": h4["h"],
        "h_msaa4": h4["h"],
        "h_msaa1": h1["h"],
        "h_ci95_formal": h4["h_ci95"],
        "h_uncertainty_95": 0.001,
        "wall_height_H": h4["h"] + top["H_minus_h"],
        "fit_rms_px_msaa4": h4["rms_px"],
        "principal_row_free_fit": results["height"]["msaa4"]["fit_h_a_cyfree"]["cy"],
        "note": "msaa=4 fit, 960 renders; msaa=1 gives h_msaa1; metres above the floor; "
        "0.15 height_offset + 0.06 robot origin = 0.21",
    }
    cam["offset_from_rotation_centre"] = {
        "axial_m": sf["a"],
        "axial_ci95_m": sf["a_ci95"],
        "lateral_m": sf["l"],
        "lateral_ci95_m": sf["l_ci95"],
        "boundary_fit_axial_m": results["offset"]["boundary_fit"]["a"],
        "boundary_fit_axial_ci95_m": results["offset"]["boundary_fit"]["a_ci95"],
        "decision": "treat as zero: the camera sits on the rotation axis within 5 mm",
    }
    cam["walls"] = {
        "model": "thin planes through the wall cell centres (half thickness < 5 mm)",
        "height_m": cam["height_fit"]["wall_height_H"],
    }
    cam["far_plane_m"] = 5.0
    CAM_JSON.write_text(json.dumps(cam, indent=1))
    (ROOT / "cache" / "mapping" / "stage0_camera_results.json").write_text(
        json.dumps(results, indent=1)
    )


if __name__ == "__main__":
    main()
