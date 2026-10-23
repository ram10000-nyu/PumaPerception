"""Relative-motion estimators between consecutive frames (stage 4).

Each estimator returns a ``Measurement`` of the relative pose ``(dx, dy, dtheta)`` of frame
``k+1`` in the frame of ``k`` (robot frame: x forward, y left, theta CCW) with a 3 x 3 covariance,
or ``None`` ("no measurement"). Camera offset from the rotation centre is zero (stage 0), so
camera motion is robot motion.

* ``icp``: scan-to-scan point-to-line ICP on the virtual scans of stage 2 (floor edge and, for
  walls nearer than 0.16 m, wall-top returns), returns up to ``RANGE_CAP`` weighted by the depth
  noise ``sigma(z)`` that grows as ``z^2 / (fy h)``, with a robust loss. Its covariance comes from
  the Gauss-Newton Hessian, so the along-corridor degeneracy shows as a large variance.
* ``vp_rotation``: rotation from the two frames' stage 3 headings (modulo 90, unwrapped to the
  branch nearest a prediction).

Pipeline code: scans, headings and constants only (rule 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scan as sc
import se2
import segment as sg
from scipy.spatial import cKDTree

RANGE_CAP = 2.5  # m: returns beyond are too coarse (one pixel is z^2 / (fy h))
SIGMA_PIXEL = 0.3  # px: noise of one boundary row
SIGMA_TOP = 0.002  # m: noise of a wall-top return (it is 2.3 times finer per pixel; stage 3)
MODEL_FLOOR = 0.01  # m: added to every residual's standard deviation (near walls are not exact)
NORMAL_COS = 0.85  # a correspondence must join lines within about 30 degrees of each other
PIECE_MIN, PIECE_STEP, PIECE_MAX = 7, 3, 120  # columns: straight-piece window, grown in steps
PIECE_K = 2.5  # accepted residual of a piece in units of the point noise
PIECE_EXTENT = 0.04  # m: shorter pieces carry no direction
COV_SCALE = 6.0  # inflation of the Hessian covariance: neighbouring columns are correlated
COV_CAP = 0.5**2  # m^2: an unconstrained direction is no tighter than this
INIT_SIGMA = np.array([0.15, 0.15, np.radians(12.0)])  # weak prior around the initial guess


@dataclass
class Measurement:
    mean: np.ndarray  # (3,) dx, dy, dtheta
    cov: np.ndarray  # (3, 3)
    info: dict | None = None  # diagnostics (iterations, inliers, cost, ...)


# ------------------------------------------------------------------------------- scan points


@dataclass
class ScanPoints:
    xy: np.ndarray  # (n, 2) robot frame: (forward, left)
    sigma: np.ndarray  # (n,) metres
    col: np.ndarray  # (n,) image column
    normal: np.ndarray  # (n, 2) unit normal of the point's straight piece (NaN where none)
    offset: np.ndarray  # (n,) offset of that line: n . p = offset on the piece


def scan_points(
    rows: np.ndarray, top: np.ndarray, status: np.ndarray, cam: sc.Camera | None = None
):
    """Points of one frame's scan from the cached floor-edge rows, wall-top rows and status."""
    cam = cam or sc.camera_config()
    u = np.arange(len(rows)) + 0.5
    z = np.full(len(rows), np.nan)
    sig = np.full(len(rows), np.nan)
    wall = (status == sg.WALL_RETURN) & (rows > cam.cy + 1e-6)
    z[wall] = cam.fy * cam.h / (rows[wall] - cam.cy)
    sig[wall] = np.maximum(0.002, SIGMA_PIXEL * z[wall] ** 2 / (cam.fy * cam.h))
    near = status == sg.TOP_RETURN
    z[near] = sc.top_depth(top[near], cam)
    sig[near] = SIGMA_TOP
    x = z * (u - cam.cx) / cam.fx
    rng = np.hypot(x, z)
    ok = np.isfinite(z) & (rng <= RANGE_CAP)
    xy = np.stack([z, -x], 1)
    normals, offsets = _normals(xy, ok)
    idx = np.where(ok)[0]
    return ScanPoints(xy[idx], sig[idx], idx, normals[idx], offsets[idx])


def _normals(xy: np.ndarray, ok: np.ndarray):
    """Straight pieces of the scan: for every run of adjacent valid columns, greedily the longest
    window (grown in steps) whose total-least-squares line fits within the depth noise. Returns
    per-point ``(normal, offset)`` of its piece's line (NaN where no piece fits: corners, depth
    steps, short runs). A long window averages out the one-pixel quantisation of the boundary row,
    which local normals from a few columns cannot (at 0.25 m they tilt by 10 degrees)."""
    cam = sc.camera_config()
    n = len(xy)
    normal = np.full((n, 2), np.nan)
    offset = np.full(n, np.nan)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        a = i
        while a <= j:
            best = None
            m = PIECE_MIN
            while a + m <= j + 1 and m <= PIECE_MAX:
                q = xy[a : a + m]
                c = q.mean(0)
                w, v = np.linalg.eigh((q - c).T @ (q - c) / m)
                tol = PIECE_K * max(0.002, SIGMA_PIXEL * c[0] ** 2 / (cam.fy * cam.h)) + 0.002
                if np.sqrt(max(w[0], 0.0)) > tol:
                    break
                best = (m, c, v[:, 1], np.sqrt(12.0 * max(w[1], 0.0)))
                m += PIECE_STEP
            if best is None or best[3] < PIECE_EXTENT:
                a += 1
                continue
            m, c, d, _ = best
            nrm = np.array([-d[1], d[0]])
            normal[a : a + m] = nrm
            offset[a : a + m] = nrm @ c
            a += m
        i = j + 1
    return normal, offset


# ---------------------------------------------------------------------------------------- ICP


def _transform(pts: np.ndarray, pose) -> np.ndarray:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def icp(
    target: ScanPoints,
    source: ScanPoints,
    init,
    iters: int = 30,
    dmax_start: float = 0.20,
    dmax_end: float = 0.05,
    init_sigma: np.ndarray | None = None,
) -> Measurement | None:
    """Point-to-line ICP aligning ``source`` (frame k+1) to ``target`` (frame k): returns the
    pose of frame k+1 in frame k. ``init`` is a first guess ``(dx, dy, dtheta)``; the weak
    Gaussian prior around it has standard deviations ``init_sigma`` (default ``INIT_SIGMA``)."""
    has_n = np.isfinite(target.normal[:, 0])
    if has_n.sum() < 10 or len(source.xy) < 10:
        return None
    tx, tn, ts = target.xy[has_n], target.normal[has_n], target.sigma[has_n]
    to = target.offset[has_n]
    tree = cKDTree(tx)
    pose = np.array(init, float)
    info = {}
    H = np.zeros((3, 3))
    for it in range(iters):
        dmax = dmax_end + (dmax_start - dmax_end) * max(0.0, 1.0 - it / (0.6 * iters))
        moved = _transform(source.xy, pose)
        d, j = tree.query(moved, distance_upper_bound=dmax)
        m = np.isfinite(d)
        if m.sum() < 8:
            return None
        # the source normal, rotated into frame k, must agree with the target line (a point of a
        # different wall, or the end of a plane seen edge-on, is rejected)
        c_, s_ = np.cos(pose[2]), np.sin(pose[2])
        sn = source.normal[m]
        sn_rot = np.stack([c_ * sn[:, 0] - s_ * sn[:, 1], s_ * sn[:, 0] + c_ * sn[:, 1]], 1)
        agree = ~np.isfinite(sn[:, 0]) | (np.abs((sn_rot * tn[j[m]]).sum(1)) >= NORMAL_COS)
        idx_m = np.where(m)[0][agree]
        m = np.zeros_like(m)
        m[idx_m] = True
        if m.sum() < 8:
            return None
        p, nrm = moved[m], tn[j[m]]
        r = (p * nrm).sum(1) - to[j[m]]
        sig = np.sqrt(source.sigma[m] ** 2 + ts[j[m]] ** 2 + MODEL_FLOOR**2)
        z = r / sig
        w = 1.0 / (1.0 + z**2) / sig**2  # Cauchy weights on the normalised residual
        # d residual / d (dx, dy, dtheta): n.T, and n . (J (p - t)) with J the 90-degree rotation
        rel = p - pose[:2]
        jt = nrm[:, 0] * (-rel[:, 1]) + nrm[:, 1] * rel[:, 0]
        Jm = np.column_stack([nrm[:, 0], nrm[:, 1], jt])
        H = (Jm * w[:, None]).T @ Jm
        g = (Jm * w[:, None]).T @ r
        # MAP step: a weak Gaussian prior around the initial guess keeps directions the scan
        # does not constrain (along a corridor) where they started instead of letting noise
        # throw them far away
        P0inv = np.diag(1.0 / (INIT_SIGMA if init_sigma is None else np.asarray(init_sigma)) ** 2)
        step = -np.linalg.solve(H + P0inv, g + P0inv @ (pose - np.asarray(init)))
        pose = pose + step
        pose[2] = float(se2.wrap(pose[2]))
        if np.abs(step[:2]).max() < 1e-5 and abs(step[2]) < 1e-5:
            break
    info = {
        "iterations": it + 1,
        "inliers": int(m.sum()),
        "dmax": dmax,
        "rms_z": float(np.sqrt(np.mean(z**2))),
    }
    # covariance from the information matrix (unweighted by the robust factor: inliers only)
    sig_in = sig
    Jw = Jm / sig_in[:, None]
    inl = np.abs(z) < 3.0
    info["inlier_fraction"] = float(inl.mean())
    Hc = (Jw[inl]).T @ Jw[inl] / COV_SCALE
    cov = np.linalg.inv(Hc + np.diag([1.0 / COV_CAP, 1.0 / COV_CAP, 1.0 / (np.radians(30.0)) ** 2]))
    return Measurement(pose, cov, info)


def icp_symmetric(
    target: ScanPoints, source: ScanPoints, init, **kw
) -> Measurement | None:
    """``icp`` run in both directions and averaged. One run fits the lines of one scan to the
    noisy points of the other, which shrinks the forward motion (regression dilution: -0.7 % on
    straight pairs, stage 4b); the average of the forward run and the inverted reverse run
    cancels the asymmetry (+0.2 % +/- 0.2) and halves the median rotation error. The
    covariance and diagnostics are the forward run's. If the reverse run fails, the forward
    result stands."""
    fwd = icp(target, source, init, **kw)
    if fwd is None:
        return None
    rev = icp(source, target, se2.inverse(np.asarray(init, float)), **kw)
    if rev is None:
        return fwd
    back = se2.inverse(rev.mean)
    mean = fwd.mean.copy()
    mean[:2] = 0.5 * (fwd.mean[:2] + back[:2])
    mean[2] = fwd.mean[2] + 0.5 * float(se2.wrap(back[2] - fwd.mean[2]))
    return Measurement(mean, fwd.cov, fwd.info)


# --------------------------------------------------------------------------- vp rotation


def vp_rotation(
    psi_prev: float, psi_cur: float, sigma_prev: float, sigma_cur: float, predicted: float
):
    """Relative rotation (radians) from two headings modulo 90 (degrees), unwrapped to the branch
    nearest ``predicted`` (radians). Returns ``(dtheta, sigma)``."""
    d = float(np.radians(psi_cur - psi_prev))
    branch = np.round((predicted - d) / (np.pi / 2))
    return d + branch * np.pi / 2, float(np.radians(np.hypot(sigma_prev, sigma_cur)))


# ----------------------------------------------------------------- global rotation search

FIELD_RES = 0.01  # m
FIELD_EXTENT = (-0.7, 3.0, -2.4, 2.4)  # x0, x1, y0, y1 in the target frame
FIELD_SIGMA = 0.02  # m


def likelihood_field(target: ScanPoints) -> np.ndarray:
    """Occupancy likelihood of a scan: ``exp(-d^2 / 2 sigma^2)`` of the distance to the nearest
    point, on a ``FIELD_RES`` grid (row = y, column = x)."""
    import cv2

    x0, x1, y0, y1 = FIELD_EXTENT
    h, w = int((y1 - y0) / FIELD_RES), int((x1 - x0) / FIELD_RES)
    img = np.full((h, w), 255, np.uint8)
    ix = ((target.xy[:, 0] - x0) / FIELD_RES).astype(int)
    iy = ((target.xy[:, 1] - y0) / FIELD_RES).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    img[iy[ok], ix[ok]] = 0
    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3) * FIELD_RES
    return np.exp(-0.5 * (dt / FIELD_SIGMA) ** 2)


def correlative_scores(field: np.ndarray, source_xy: np.ndarray, thetas, tmax: float = 0.15):
    """For each rotation, the best translation (within ``tmax``) by correlation with the field.
    Returns ``(scores, translations)`` with shapes ``(len(thetas),)`` and ``(len(thetas), 2)``."""
    x0, _, y0, _ = FIELD_EXTENT
    h, w = field.shape
    cells = int(round(tmax / FIELD_RES))
    sh = np.arange(-cells, cells + 1)
    sx, sy = np.meshgrid(sh, sh)
    scores, trans = [], []
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        p = source_xy @ np.array([[c, s], [-s, c]])
        ix = np.round((p[:, 0] - x0) / FIELD_RES).astype(int)[:, None, None] + sx[None]
        iy = np.round((p[:, 1] - y0) / FIELD_RES).astype(int)[:, None, None] + sy[None]
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        val = np.where(inside, field[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)], 0.0)
        total = val.sum(0)
        k = np.unravel_index(total.argmax(), total.shape)
        scores.append(float(total[k]))
        trans.append([sx[k] * FIELD_RES, sy[k] * FIELD_RES])
    return np.array(scores), np.array(trans)


def robust_cost(
    target: ScanPoints, source: ScanPoints, pose, dmax: float = 0.08
) -> tuple[float, int]:
    """Robust alignment cost of ``source`` at ``pose``: sum of ``log(1 + z^2)`` over matched
    points, with every unmatched point costing as a residual of ``dmax``. Lower is better."""
    has = np.isfinite(target.normal[:, 0])
    if has.sum() < 5:
        return float("inf"), 0
    tree = cKDTree(target.xy[has])
    moved = _transform(source.xy, pose)
    d, j = tree.query(moved, distance_upper_bound=dmax)
    m = np.isfinite(d)
    nrm = target.normal[has][j[m]]
    r = (moved[m] * nrm).sum(1) - target.offset[has][j[m]]
    sig = np.sqrt(source.sigma[m] ** 2 + target.sigma[has][j[m]] ** 2 + MODEL_FLOOR**2)
    z = r / sig
    unmatched = (~m).sum() * np.log1p((dmax / 0.02) ** 2)
    return float(np.sum(np.log1p(z**2)) + unmatched), int(m.sum())


# ------------------------------------------------------------------ wall-feature motion

FEATURE_MIN_DROW = 8.0  # px: a feature this close to the horizon row has no usable depth ratio
FEATURE_MAX_Z = 2.0  # m: scan depth of the feature's wall beyond which the position is too coarse
WALL_ROT_SD_MAX_DEG = 10.0  # a wall fit with a wider rotation sd is rejected (typical: 1; p99: 4)
FEATURE_RATIO_TOL = (0.05, 0.10)  # m, relative: scan depth vs row-ratio depth in frame k must agree


def wall_features(img: np.ndarray, seg, sift):
    """SIFT keypoints on the wall mask (continuous pixel coordinates) and descriptors."""
    import cv2

    mask = cv2.erode(seg.wall.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    kp, desc = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    pts = np.array([[k.pt[0] + 0.5, k.pt[1] + 0.5] for k in kp], np.float64).reshape(-1, 2)
    return pts, desc


def match_features(f0, f1, matcher, ratio: float = 0.8) -> np.ndarray:
    """Ratio-test matches between two frames as rows ``(u0, v0, u1, v1)``."""
    (p0, d0), (p1, d1) = f0, f1
    if d0 is None or d1 is None or len(p0) < 4 or len(p1) < 4:
        return np.zeros((0, 4))
    good = [
        m
        for m, n in (x for x in matcher.knnMatch(d0, d1, k=2) if len(x) == 2)
        if m.distance < ratio * n.distance
    ]
    if not good:
        return np.zeros((0, 4))
    return np.array([[*p0[m.queryIdx], *p1[m.trainIdx]] for m in good])


def feature_points(
    matches: np.ndarray,
    scan_prev: ScanPoints,
    cam: sc.Camera | None = None,
    scan_cur: ScanPoints | None = None,
):
    """Metric ground-plane position of each matched feature in both frames (robot frames,
    forward/left). Frame k-1: the depth is the scan's wall depth in the feature's column. Frame
    k: with ``scan_cur`` the same (the scan's wall depth in the feature's column there); without
    it the feature keeps its height, so ``Z2 = Z1 * (v1 - cy) / (v2 - cy)``. The ratio version
    is biased (forward motion about 2.7 % short, stage 4b), the two-scan version is not."""
    cam = cam or sc.camera_config()
    if len(matches) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    zcol = np.full(cam.width, np.nan)
    zcol[scan_prev.col] = scan_prev.xy[:, 0]
    u0, v0, u1, v1 = matches.T
    z0 = zcol[np.clip(np.floor(u0).astype(int), 0, cam.width - 1)]
    d0, d1 = v0 - cam.cy, v1 - cam.cy
    ok = (
        np.isfinite(z0)
        & (z0 <= FEATURE_MAX_Z)
        & (np.abs(d0) >= FEATURE_MIN_DROW)
        & (np.abs(d1) >= FEATURE_MIN_DROW)
    )
    if scan_cur is None:
        ok &= d0 * d1 > 0
        z1 = z0 * d0 / np.where(ok, d1, 1.0)
        ok &= np.isfinite(z1) & (z1 > 0.05) & (z1 < 2.0 * FEATURE_MAX_Z)
    else:
        zcol1 = np.full(cam.width, np.nan)
        zcol1[scan_cur.col] = scan_cur.xy[:, 0]
        z1 = zcol1[np.clip(np.floor(u1).astype(int), 0, cam.width - 1)]
        ok &= np.isfinite(z1) & (z1 > 0.05) & (z1 <= FEATURE_MAX_Z)
        # the scan column in frame k can belong to another wall (a corner, a depth step, a large
        # rotation): the row ratio, noisy and biased but local to the feature, vetoes those
        ok &= d0 * d1 > 0
        z1r = z0 * d0 / np.where(ok, d1, 1.0)
        ok &= np.abs(z1 - z1r) <= FEATURE_RATIO_TOL[0] + FEATURE_RATIO_TOL[1] * z1
    p0 = np.stack([z0, -z0 * (u0 - cam.cx) / cam.fx], 1)[ok]
    p1 = np.stack([z1, -z1 * (u1 - cam.cx) / cam.fx], 1)[ok]
    return p0, p1, z0[ok]


def _rigid(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None):
    """Weighted least-squares ``a = R b + t`` in 2-D: returns ``(t, theta)``."""
    w = np.ones(len(a)) if w is None else w
    ca, cb = (w[:, None] * a).sum(0) / w.sum(), (w[:, None] * b).sum(0) / w.sum()
    qa, qb = a - ca, b - cb
    s = (w * (qb[:, 0] * qa[:, 1] - qb[:, 1] * qa[:, 0])).sum()
    c = (w * (qb[:, 0] * qa[:, 0] + qb[:, 1] * qa[:, 1])).sum()
    th = float(np.arctan2(s, c))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return ca - R @ cb, th


def wall_motion(
    matches: np.ndarray,
    scan_prev: ScanPoints,
    seed: int = 0,
    cam=None,
    scan_cur: ScanPoints | None = None,
) -> Measurement | None:
    """Relative motion of frame k in frame k-1 from texture features on the walls, by RANSAC over
    two-point rigid fits then a weighted refit. Constrains motion along a corridor, where scan ICP
    cannot, whenever the walls carry texture. ``None`` with fewer than 6 inliers."""
    a, b, z = feature_points(matches, scan_prev, cam, scan_cur)
    # SIFT can return several keypoints (orientations) at one spot: count that spot once
    _, keep = np.unique(np.round(np.c_[a, b] / 0.005).astype(int), axis=0, return_index=True)
    a, b, z = a[np.sort(keep)], b[np.sort(keep)], z[np.sort(keep)]
    n = len(a)
    if n < 6:
        return None
    rng = np.random.default_rng(seed)
    sig = np.sqrt(
        (0.003 + 0.5 * z**2 / (92 * 0.21)) ** 2 + (0.012 * z) ** 2
    )  # depth noise + ratio noise
    if scan_cur is not None:  # both depths come from scans: two depth noises, no ratio noise
        z1 = b[:, 0]
        sig = np.hypot(0.003 + 0.5 * z**2 / (92 * 0.21), 0.003 + 0.5 * z1**2 / (92 * 0.21))
    best = None
    for _ in range(150):
        i, j = rng.choice(n, 2, replace=False)
        t, th = _rigid(a[[i, j]], b[[i, j]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        r = np.linalg.norm(a - (b @ R.T + t), axis=1) / sig
        inl = r < 3.0
        if best is None or inl.sum() > best[0].sum():
            best = (inl, t, th)
    inl = best[0]
    if inl.sum() < 6:
        return None
    for _ in range(3):
        t, th = _rigid(a[inl], b[inl], 1.0 / sig[inl] ** 2)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        r = np.linalg.norm(a - (b @ R.T + t), axis=1) / sig
        inl = r < 3.0
        if inl.sum() < 6:
            return None
    # covariance from the point-to-point Hessian, scaled by the residual scatter
    bi, si = b[inl], sig[inl]
    Rm = R
    J = []
    for p, s_ in zip(bi, si, strict=True):
        rot = np.array(
            [-np.sin(th) * p[0] - np.cos(th) * p[1], np.cos(th) * p[0] - np.sin(th) * p[1]]
        )
        J.append((np.array([[1, 0, rot[0]], [0, 1, rot[1]]]), s_))
    H = sum(Jk.T @ Jk / s_**2 for Jk, s_ in J)
    scale = max(1.0, float(np.mean(r[inl] ** 2))) * 1.5
    cov = scale * np.linalg.inv(H + np.diag([1e-6, 1e-6, 1e-6]))
    del Rm
    if np.sqrt(cov[2, 2]) > np.radians(WALL_ROT_SD_MAX_DEG):
        return None  # features at one place: the rotation is not measured (a fit like that
        # still "agrees" with any rotation source at its width)
    return Measurement(
        np.array([t[0], t[1], float(th)]), cov, {"inliers": int(inl.sum()), "matches": int(n)}
    )


def match_rotation(matches: np.ndarray, consensus_deg: float = 3.0, cam=None):
    """Signed rotation (degrees, CCW positive) between two frames from the bearing shifts of
    matched features: the dominant cluster (within ``consensus_deg``) of
    ``bearing(frame k) - bearing(frame k-1)``. Needs no scan and has no 90-degree ambiguity;
    returns ``(rotation_deg, consistent_matches)`` or ``(nan, 0)``."""
    cam = cam or sc.camera_config()
    if len(matches) < 3:
        return np.nan, 0
    u0, u1 = matches[:, 0], matches[:, 2]
    shift = np.degrees(np.arctan((u1 - cam.cx) / cam.fx) - np.arctan((u0 - cam.cx) / cam.fx))
    best_n, best_c = 0, 0.0
    for c in shift:
        n = int((np.abs(shift - c) < consensus_deg).sum())
        if n > best_n or (n == best_n and abs(c) > abs(best_c)):
            best_n, best_c = n, float(c)
    inl = np.abs(shift - best_c) < consensus_deg
    return float(np.median(shift[inl])), int(inl.sum())
