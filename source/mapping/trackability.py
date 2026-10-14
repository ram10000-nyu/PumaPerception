"""Stage 0 revision 2, task 7: can vision alone track the robot frame to frame?

Per frame, three cues (all provisional, quick versions of stages 2, 3 and 4):

- ``inliers``: RANSAC-homography inliers between SIFT matches (ratio 0.8) of a frame and its
  predecessor, features restricted to the wall mask;
- ``valid_cols``: fraction of image columns with a floor-wall boundary (floor connected to
  the bottom row, wall pixels directly above it, not sky);
- ``vp``: a vanishing point exists (a cluster of floor-seam line segments that meet on the
  horizon, or are parallel to it).

A frame is *blind* when inliers < 15, valid_cols < 0.20 and there is no vanishing point.

    uv run source/mapping/trackability.py            # all seeds, cached
    uv run source/mapping/trackability.py --seeds 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_gt  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
FX = 92.0
CX, CY = 160.0, 120.0
CONFIG = {
    "version": 1,
    "sift_features": 400,
    "ratio": 0.8,
    "ransac_px": 3.0,
    "min_seg_px": 8,
    "vp_bin_deg": 1.0,
    "vp_min_len_px": 40.0,
    "blind": {"inliers": 15, "valid_cols": 0.20},
}


def config_hash() -> str:
    return hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------- pure functions


def real_floor_mask(img: np.ndarray) -> np.ndarray:
    """The two floor textures (blue tile and grey) including their slight distance fog.
    Measured on 150 frames against exact green-floor renders: recall 0.995, no wall pixel
    accepted."""
    p = img.astype(np.int16)
    B, G, R = p[..., 0], p[..., 1], p[..., 2]
    gray = (np.abs(B - G) <= 3) & (np.abs(G - R) <= 3) & (G >= 225)
    blue = (B >= 216) & (B <= 226) & (G >= 178) & (G <= 215) & (R >= 155) & (R <= 195)
    blue &= (B > G) & (G > R)
    return gray | blue


def sky_mask(img: np.ndarray) -> np.ndarray:
    return (img == 255).all(-1)


def valid_boundary_columns(img: np.ndarray, floor: np.ndarray | None = None) -> np.ndarray:
    """Boolean per column: floor reaches the bottom row, then a wall pixel (not sky, not
    floor) lies directly above the floor's top."""
    floor = real_floor_mask(img) if floor is None else floor
    sky = sky_mask(img)
    h, w = floor.shape
    out = np.zeros(w, bool)
    # first non-floor row going up from the bottom, per column
    nonfloor = ~floor[::-1]  # row 0 is the image bottom
    ok_bottom = floor[h - 1]
    first = np.argmax(nonfloor, axis=0)  # distance up from the bottom to the first non-floor
    has = nonfloor.any(0)
    j = h - 1 - first  # row of the first non-floor pixel
    for u in range(w):
        if not (ok_bottom[u] and has[u]) or j[u] < 1:
            continue
        out[u] = not sky[j[u], u] and not sky[j[u] - 1, u] and not floor[j[u] - 1, u]
    return out


def segment_horizon_angle(seg, fx: float = FX, cx: float = CX, cy: float = CY) -> float:
    """Direction (degrees, in (-90, 90]) of the vanishing point of the line through the
    segment, as the angle of its horizon crossing from the optical axis:
    ``atan((x_h - cx) / fx)``. Lines parallel to the horizon give +-90 (the same direction
    at infinity). A ground line at heading ``beta`` from the camera's forward direction
    crosses the horizon at ``cx - fx * tan(beta)``, so this equals ``-beta``."""
    x1, y1, x2, y2 = seg
    dy = y2 - y1
    if abs(dy) < 1e-9:
        return 90.0
    xh = x1 + (cy - y1) * (x2 - x1) / dy
    return float(np.degrees(np.arctan2(xh - cx, fx)))


def vp_clusters(
    angles: np.ndarray, lengths: np.ndarray, bin_deg: float = 1.0, min_len: float = 40.0
) -> list[tuple[float, float]]:
    """Length-weighted peaks of segment horizon angles: ``[(angle_deg, total_length), ...]``,
    strongest first. The angle axis wraps at +-90 (infinity), so those ends are merged."""
    angles = np.asarray(angles, float)
    lengths = np.asarray(lengths, float)
    if len(angles) == 0:
        return []
    n = int(round(180.0 / bin_deg))
    idx = np.floor((angles + 90.0) / bin_deg).astype(int) % n
    hist = np.bincount(idx, weights=lengths, minlength=n)
    # a +-1 bin window, circular
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peaks = []
    used = np.zeros(n, bool)
    for i in np.argsort(-smooth):
        if smooth[i] < min_len:
            break
        if used[i]:
            continue
        window = [(i + k) % n for k in (-1, 0, 1)]
        centers = (np.array(window) + 0.5) * bin_deg - 90.0
        w = hist[window]
        # circular mean of the angle in doubled-angle space (period 180 degrees)
        th = np.radians(2.0 * centers)
        a = np.degrees(np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum())) / 2.0
        peaks.append((float(a), float(smooth[i])))
        for k in range(-3, 4):
            used[(i + k) % n] = True
    return peaks


def expected_vp_angles(yaw: float) -> tuple[float, float]:
    """Horizon angles of the two maze-axis vanishing points for a camera heading ``yaw``
    (radians CCW from +x), maze axes along x and y: ``-beta`` for ``beta = k*90 - yaw``."""
    a = (np.degrees(yaw) + 45.0) % 90.0 - 45.0
    other = a - 90.0 if a > 0 else a + 90.0
    return float(a), float(other)


def angle_diff_deg(a: float, b: float) -> float:
    """Difference of two horizon angles on the 180-degree circle."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


# --------------------------------------------------------------------------- per-frame cues


def frame_cues(img: np.ndarray, lsd) -> dict:
    floor = real_floor_mask(img)
    sky = sky_mask(img)
    valid = valid_boundary_columns(img, floor)
    # segments on the floor mask outline
    m = floor.astype(np.uint8) * 255
    segs = lsd.detect(m)[0]
    angles, lengths = [], []
    if segs is not None:
        for s in segs.reshape(-1, 4):
            x1, y1, x2, y2 = map(float, s)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < CONFIG["min_seg_px"]:
                continue
            # drop the image frame and the far-plane/horizon edge
            if (max(x1, x2) < 2 or min(x1, x2) > 317 or min(y1, y2) > 237) or (max(y1, y2) < 127):
                continue
            if abs(x2 - x1) < 1.0 and abs(y2 - y1) > 4:  # (near-)vertical in the image: not a seam
                continue
            angles.append(segment_horizon_angle((x1, y1, x2, y2)))
            lengths.append(length)
    peaks = vp_clusters(
        np.array(angles), np.array(lengths), CONFIG["vp_bin_deg"], CONFIG["vp_min_len_px"]
    )
    return {
        "valid_cols": float(valid.mean()),
        "floor_frac": float(floor.mean()),
        "sky_frac": float(sky.mean()),
        "vp_peaks": peaks,
        "wall_mask": (~floor & ~sky),
    }


def sift_features(img: np.ndarray, wall: np.ndarray, sift):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.erode(wall.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    k, d = sift.detectAndCompute(g, m)
    return np.array([p.pt for p in k], np.float32), d


def pair_inliers(f1, f2, bf) -> int:
    (p1, d1), (p2, d2) = f1, f2
    if d1 is None or d2 is None or len(p1) < 8 or len(p2) < 8:
        return 0
    good = [
        a
        for a, b in (m for m in bf.knnMatch(d1, d2, k=2) if len(m) == 2)
        if a.distance < CONFIG["ratio"] * b.distance
    ]
    if len(good) < 4:
        return 0
    src = np.float32([p1[g.queryIdx] for g in good])
    dst = np.float32([p2[g.trainIdx] for g in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, CONFIG["ransac_px"])
    return int(inl.sum()) if inl is not None else 0


def process_drive(args) -> dict:
    seed, drive = args
    path = record_gt.gt_dir(seed) / f"traj_{drive}"
    info = json.loads((path / "data_info.json").read_text())
    sift = cv2.SIFT_create(nfeatures=CONFIG["sift_features"])
    bf = cv2.BFMatcher()
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    n = len(info)
    inliers = np.zeros(n, int)
    valid = np.zeros(n)
    vp_len = np.zeros(n)
    vp_ang = np.full((n, 2), np.nan)
    prev = None
    for k in range(n):
        img = cv2.imread(str(path / info[k]["image"]))
        c = frame_cues(img, lsd)
        valid[k] = c["valid_cols"]
        if c["vp_peaks"]:
            vp_len[k] = c["vp_peaks"][0][1]
            for i, (a, _) in enumerate(c["vp_peaks"][:2]):
                vp_ang[k, i] = a
        feats = sift_features(img, c["wall_mask"], sift)
        if prev is not None:
            inliers[k] = pair_inliers(prev, feats, bf)
        prev = feats
    inliers[0] = inliers[1] if n > 1 else 0
    return {
        "seed": seed,
        "drive": drive,
        "inliers": inliers,
        "valid_cols": valid,
        "vp_len": vp_len,
        "vp_ang": vp_ang,
    }


# ---------------------------------------------------------------------------- aggregation


def true_motion(seed: int) -> list[dict]:
    """Per drive, true relative translation [m] and rotation [rad] between consecutive saved
    frames (the first from the spawn)."""
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    out = []
    for p in record_gt.load_poses(record_gt.gt_dir(seed)):
        q = np.vstack([np.array(gt["spawn"])[None], p])
        d = q[1:, :2] - q[:-1, :2]
        trans = np.hypot(d[:, 0], d[:, 1])
        rot = (q[1:, 2] - q[:-1, 2] + np.pi) % (2 * np.pi) - np.pi
        out.append({"trans": trans, "rot": rot})
    return out


def blind_flags(inliers, valid, vp_len) -> np.ndarray:
    b = CONFIG["blind"]
    return (inliers < b["inliers"]) & (valid < b["valid_cols"]) & (vp_len <= 0)


def longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def run_seed(seed: int, workers: int | None = None, force: bool = False) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"stage0_trackability_{seed}_{config_hash()}.npz"
    if f.exists() and not force:
        z = np.load(f, allow_pickle=True)
        return list(z["drives"])
    with ProcessPoolExecutor(workers) as ex:
        drives = list(ex.map(process_drive, [(seed, d) for d in range(record_gt.ROUTES)]))
    np.savez(f, drives=np.array(drives, dtype=object))
    return drives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = json.loads(record_gt.SEEDS.read_text())
    seeds = args.seeds or cfg["dev"] + cfg["test"]
    for s in seeds:
        d = run_seed(s, force=args.force)
        print(s, [int(blind_flags(x["inliers"], x["valid_cols"], x["vp_len"]).sum()) for x in d])


if __name__ == "__main__":
    main()
