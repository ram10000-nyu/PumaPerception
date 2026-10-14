"""Vanishing-point compass (stage 3): yaw modulo 90 degrees from one frame.

The maze is axis aligned, so every floor-wall seam is a ground line along one of two
perpendicular directions. A ground line at heading ``beta`` from the optical axis crosses the
horizon row at ``x = cx - fx * tan(beta)``; with ``alpha = atan((x - cx) / fx) = -beta`` the
two families sit at ``alpha`` and ``alpha -+ 90`` degrees, and for a camera heading ``yaw``
(CCW from +x, maze axes along x and y) ``alpha = yaw`` modulo 90. **Every segment therefore
votes for the same number, ``psi = yaw mod 90`` in (-45, 45]**, whichever family it belongs to.

Seam segments come from the floor-wall boundary (sub-pixel, one point per image column),
either by splitting its polyline into straight pieces (``polyline_segments``) or, as the plan
specifies, with LSD on the floor mask restricted to the seam band (``lsd_segments``).
Each segment gets an angular standard deviation propagated from its line-fit noise, a robust
circular peak is found over ``psi`` and refined with inverse-variance weights, and the segments
are split into the two families to test the Manhattan constraint and to imply the focal length.

Pipeline code: it uses only the frame and ``camera.json`` constants, never ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import scan as sc
import segment as sg

PERIOD = 90.0  # degrees: the maze axes repeat every quarter turn


# ----------------------------------------------------------------------------- angle helpers


def fold90(a):
    """Fold an angle in degrees into (-45, 45]."""
    a = np.asarray(a, float)
    return 45.0 - np.mod(45.0 - a, PERIOD)


def diff90(a, b):
    """Signed difference ``a - b`` of two angles on the 90-degree circle, in (-45, 45]."""
    return fold90(np.asarray(a, float) - np.asarray(b, float))


def diff180(a, b):
    """Signed difference on the 180-degree circle, in (-90, 90]."""
    return 90.0 - np.mod(90.0 - (np.asarray(a, float) - np.asarray(b, float)), 180.0)


def true_psi(yaw_rad):
    """``yaw mod 90`` in degrees, in (-45, 45], from a heading in radians."""
    return fold90(np.degrees(yaw_rad))


# -------------------------------------------------------------------------------- segments


def _fit_line(pts: np.ndarray):
    """Total-least-squares line through points (n, 2): centroid, unit direction, rms."""
    c = pts.mean(0)
    q = pts - c
    cov = q.T @ q / len(pts)
    w, v = np.linalg.eigh(cov)
    return c, v[:, 1], float(np.sqrt(max(w[0], 0.0)))


def _split(
    pts: np.ndarray, tol: float, min_len: int, out: list, lo: int = 0, hi: int | None = None
):
    """Douglas-Peucker style recursive split of an ordered polyline into straight pieces."""
    hi = len(pts) if hi is None else hi
    seg = pts[lo:hi]
    if len(seg) < min_len:
        return
    # Douglas-Peucker: the point farthest from the chord joining the ends is the split candidate
    chord = seg[-1] - seg[0]
    norm = np.hypot(*chord)
    nrm = np.array([-chord[1], chord[0]]) / max(norm, 1e-9)
    dist = np.abs((seg - seg[0]) @ nrm)
    k = 1 + int(dist[1:-1].argmax()) if len(seg) > 2 else 0
    if dist[k] <= tol or len(seg) < 2 * min_len:
        out.append((lo, hi))
        return
    _split(pts, tol, min_len, out, lo, lo + k + 1)
    _split(pts, tol, min_len, out, lo + k, hi)


def polyline_segments(
    rows: np.ndarray, status: np.ndarray, tol: float = 0.45, min_len: int = 8
) -> np.ndarray:
    """Straight pieces of the floor-wall boundary: ``(n, 4)`` segments ``x1, y1, x2, y2`` in
    continuous pixel coordinates, each at least ``min_len`` columns long. Runs of adjacent wall-
    return columns are split where the boundary bends or jumps (wall corners, depth steps)."""
    ok = status == sg.WALL_RETURN
    out = []
    u = np.arange(len(rows)) + 0.5
    i = 0
    while i < len(rows):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and ok[j + 1]:
            j += 1
        pts = np.stack([u[i : j + 1], rows[i : j + 1]], 1)
        pieces: list = []
        _split(pts, tol, min_len, pieces)
        for lo, hi in pieces:
            c, d, rms = _fit_line(pts[lo:hi])
            t = (pts[lo:hi] - c) @ d
            p1, p2 = c + d * t.min(), c + d * t.max()
            out.append([*p1, *p2])
        i = j + 1
    return np.array(out).reshape(-1, 4)


def lsd_segments(seg: sg.Segmentation, lsd, min_len: float = 8.0) -> np.ndarray:
    """LSD on the floor mask, keeping segments that lie on the seam band (the wall-return
    boundary dilated by 2 px; at least 60 % of the segment's samples inside it)."""
    h, w = seg.floor.shape
    lines = lsd.detect(seg.floor.astype(np.uint8) * 255)[0]
    if lines is None:
        return np.zeros((0, 4))
    band = np.zeros((h, w), np.uint8)
    ok = np.isfinite(seg.rows) & (seg.status == sg.WALL_RETURN)
    uu = np.where(ok)[0]
    band[np.clip(np.round(seg.rows[ok] - 0.5).astype(int), 0, h - 1), uu] = 1
    band = cv2.dilate(band, np.ones((5, 5), np.uint8)) > 0
    keep = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        n = int(max(length, 2))
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        inside = band[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)]
        if inside.mean() > 0.6:
            keep.append([x1, y1, x2, y2])
    return np.array(keep).reshape(-1, 4)


def make_lsd():
    return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)


# ----------------------------------------------------------------- per-segment angle and noise


@dataclass
class SegmentVotes:
    alpha: np.ndarray  # degrees in (-90, 90]: horizon direction of each segment's line
    psi: np.ndarray  # degrees in (-45, 45]: alpha folded: the vote for yaw mod 90
    sigma: np.ndarray  # degrees: standard deviation of alpha
    length: np.ndarray  # pixels
    mid: np.ndarray  # (n, 2) segment midpoints
    a: np.ndarray  # normalised line coefficients a x + b y + c = 0
    b: np.ndarray
    c: np.ndarray


SIGMA_POINT = 0.25  # px: noise of one boundary point perpendicular to the line
SIGMA_FLOOR = 0.3  # degrees: model error added to every segment


def votes(segs: np.ndarray, cam: sc.Camera | None = None) -> SegmentVotes:
    """Horizon angle and its standard deviation for each ``(x1, y1, x2, y2)`` segment.

    A line ``a x + b y + c = 0`` (normalised) passes through the horizon point ``(cx + f
    tan(alpha), cy)`` iff ``A cos(alpha) + B sin(alpha) = 0`` with ``A = a cx + b cy + c`` and
    ``B = a f``, so ``alpha = atan2(-A, B)`` (folded to 180 degrees) is well conditioned for
    every direction, including lines parallel to the horizon. Its noise follows from the line fit:
    angle error ``sqrt(12) s / (L sqrt(n))`` and offset error ``s / sqrt(n)`` for ``n``
    points at spacing 1 px, carried to the horizon crossing and scaled by ``cos^2(alpha) / f``.
    """
    cam = cam or sc.camera_config()
    if len(segs) == 0:
        z = np.zeros(0)
        return SegmentVotes(z, z, z, z, np.zeros((0, 2)), z, z, z)
    x1, y1, x2, y2 = segs.T
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    a, b = dy / length, -dx / length  # unit normal of the line
    c = -(a * x1 + b * y1)
    A = a * cam.cx + b * cam.cy + c
    B = a * cam.fx
    s = np.where(B >= 0, 1.0, -1.0)
    alpha = np.degrees(np.arctan2(-A * s, B * s))  # (-90, 90]
    alpha = np.where(np.abs(alpha + 90) < 1e-9, 90.0, alpha)
    mid = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], 1)
    # horizon crossing and its distance t from the segment midpoint along the line
    sin_h = np.maximum(np.abs(dy) / length, 1e-3)  # sine of the angle to the horizontal
    t = np.abs(cam.cy - mid[:, 1]) / sin_h
    n = np.maximum(length, 2.0)
    s_ang = np.sqrt(12.0) * SIGMA_POINT / (length * np.sqrt(n))
    s_off = SIGMA_POINT / np.sqrt(n)
    dxh = np.sqrt(s_off**2 + (t * s_ang) ** 2) / sin_h
    cos2 = np.cos(np.radians(alpha)) ** 2
    sigma = np.degrees(cos2 * dxh / cam.fx)
    sigma = np.sqrt(sigma**2 + SIGMA_FLOOR**2)
    return SegmentVotes(alpha, fold90(alpha), sigma, length, mid, a, b, c)


# ----------------------------------------------------------------------------------- peak


def circ_mean90(psi: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean on the 90-degree circle, in (-45, 45]."""
    th = np.radians(psi * 4.0)
    return float(
        fold90(np.degrees(np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum())) / 4.0)
    )


@dataclass
class VPResult:
    psi: float = np.nan  # degrees, yaw mod 90 in (-45, 45]
    sigma: float = np.nan  # degrees: 1 / sqrt(total inlier weight)
    support: float = 0.0  # total inlier length (px)
    n_inliers: int = 0
    family_len: tuple = (0.0, 0.0)  # inlier length of the family at alpha and at alpha -+ 90
    left_len: float = 0.0  # inlier length with midpoint left of the principal point
    right_len: float = 0.0
    f_implied: float = np.nan  # focal length implied by the two vanishing points
    two_vp: bool = False
    alpha_a: float = np.nan  # horizon direction of the family at psi
    n_segments: int = 0
    inlier_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


W_CAP = 1.0 / 0.4**2  # a single segment never counts for more than a 0.4 degree measurement


def circular_peak(
    psi: np.ndarray, sigma: np.ndarray, tol_deg: float = 2.0, iters: int = 4
) -> tuple[float, float, np.ndarray]:
    """Robust ``yaw mod 90`` from votes ``psi`` (degrees) with standard deviations ``sigma``:
    a smoothed circular histogram gives the start, then iteratively reweighted circular means
    over the inliers (within the larger of ``tol_deg`` and 3 sigma of the current estimate).
    Returns ``(psi, sigma_of_the_estimate, inlier_mask)``; NaN if there are no votes."""
    psi = np.asarray(psi, float)
    if len(psi) == 0:
        return np.nan, np.nan, np.zeros(0, bool)
    w = np.minimum(1.0 / sigma**2, W_CAP)
    grid = np.arange(-45.0, 45.0, 0.5)
    d = diff90(grid[:, None], psi[None, :])
    score = (w[None, :] * np.exp(-0.5 * (d / 1.5) ** 2)).sum(1)
    est = float(grid[score.argmax()])
    inl = np.zeros(len(psi), bool)
    for _ in range(iters):
        inl = np.abs(diff90(psi, est)) < np.maximum(tol_deg, 3.0 * sigma)
        if not inl.any():
            return np.nan, np.nan, inl
        est = circ_mean90(psi[inl], w[inl])
    return est, float(1.0 / np.sqrt(w[inl].sum())), inl


def find_peak(v: SegmentVotes, tol_deg: float = 2.0, iters: int = 4) -> VPResult:
    """Peak of the segment votes, plus the two-family split used for the Manhattan test."""
    res = VPResult(n_segments=len(v.psi))
    psi, sig, inl = circular_peak(v.psi, v.sigma, tol_deg, iters)
    if not np.isfinite(psi):
        return res
    w = np.minimum(1.0 / v.sigma**2, W_CAP)
    res.psi = psi
    res.n_inliers = int(inl.sum())
    res.sigma = sig
    res.support = float(v.length[inl].sum())
    res.inlier_idx = np.where(inl)[0]
    alpha_a = psi
    alpha_b = psi - 90.0 if psi > 0 else psi + 90.0
    res.alpha_a = alpha_a
    in_a = np.abs(diff180(v.alpha, alpha_a)) < np.abs(diff180(v.alpha, alpha_b))
    la = float(v.length[inl & in_a].sum())
    lb = float(v.length[inl & ~in_a].sum())
    res.family_len = (la, lb)
    res.left_len = float(v.length[inl & (v.mid[:, 0] < 160.0)].sum())
    res.right_len = float(v.length[inl & (v.mid[:, 0] >= 160.0)].sum())
    del w
    return res


def horizon_x(v: SegmentVotes, idx: np.ndarray, cam: sc.Camera) -> float:
    """Least-squares horizon crossing of the lines ``idx`` in pixel units: minimises the sum of
    squared distances from ``(x, cy)`` to the lines (valid when the crossing is finite)."""
    a, b, c = v.a[idx], v.b[idx], v.c[idx]
    w = np.minimum(1.0 / v.sigma[idx] ** 2, W_CAP) * v.length[idx]
    den = (w * a * a).sum()
    if den < 1e-9:
        return np.nan
    return float(-(w * a * (b * cam.cy + c)).sum() / den)


def implied_focal(v: SegmentVotes, res: VPResult, cam: sc.Camera | None = None) -> float:
    """Focal length from the two vanishing points: ``f^2 = -(x1 - cx)(x2 - cx)`` (the Manhattan
    constraint). Needs both families with finite crossings, so only well away from axis
    alignment (both horizon directions between 15 and 75 degrees)."""
    cam = cam or sc.camera_config()
    if len(res.inlier_idx) == 0:
        return np.nan
    a, b = res.alpha_a, res.alpha_a - 90.0 if res.alpha_a > 0 else res.alpha_a + 90.0
    if not (15.0 < abs(a) < 75.0 and 15.0 < abs(b) < 75.0):
        return np.nan
    inl = res.inlier_idx
    in_a = np.abs(diff180(v.alpha[inl], a)) < np.abs(diff180(v.alpha[inl], b))
    ia, ib = inl[in_a], inl[~in_a]
    if len(ia) == 0 or len(ib) == 0:
        return np.nan
    xa, xb = horizon_x(v, ia, cam), horizon_x(v, ib, cam)
    prod = -(xa - cam.cx) * (xb - cam.cx)
    return float(np.sqrt(prod)) if np.isfinite(prod) and prod > 0 else np.nan


def estimate(seg: sg.Segmentation, source: str = "polyline", lsd=None, cam=None) -> VPResult:
    """Yaw mod 90 from one segmentation. ``source``: ``polyline`` or ``lsd``."""
    cam = cam or sc.camera_config()
    if source == "lsd":
        segs = lsd_segments(seg, lsd or make_lsd())
    else:
        segs = polyline_segments(seg.rows, seg.status)
    v = votes(segs, cam)
    res = find_peak(v)
    if res.n_inliers:
        res.f_implied = implied_focal(v, res, cam)
        res.two_vp = bool(min(res.family_len) > 0 and np.isfinite(res.f_implied))
    return res
