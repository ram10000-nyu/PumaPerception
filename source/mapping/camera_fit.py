"""Camera height and mounting offset from renders at known poses (stage 0, tasks 6 and 8).

The simulator's true grid fixes where every wall face is (cell (r, c) is centred at
(c * 0.2, r * 0.2), faces at +/- 0.1 m). A floor point at depth ``z`` along the optical axis
projects to row ``cy + fy * h / z``, so the sub-pixel row of the floor-wall boundary in a
render, with the true depth from the grid, gives ``h``. The same model with the camera
displaced from the pose by ``(a, l)`` (forward, left) in the robot frame gives the offset.

Pure functions (no simulator) are at the top and unit-tested; the render drivers are below.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

FLOOR_COLORS = np.array([[218, 181, 157], [232, 232, 232]], dtype=np.int16)  # BGR, two textures
SKY = np.array([255, 255, 255], dtype=np.int16)
CELL = 0.2


# ----------------------------------------------------------------------------- pure functions


def floor_mask(img: np.ndarray, tol: int = 2) -> np.ndarray:
    """True where a pixel is exactly one of the two floor colours (within ``tol``)."""
    p = img.astype(np.int16)
    m = np.zeros(img.shape[:2], bool)
    for c in FLOOR_COLORS:
        m |= np.abs(p - c).max(-1) <= tol
    return m


def sky_mask(img: np.ndarray, tol: int = 0) -> np.ndarray:
    return np.abs(img.astype(np.int16) - SKY).max(-1) <= tol


def boundary_rows(img: np.ndarray, cols=None, floor_fn=None) -> np.ndarray:
    """Sub-pixel row (continuous coordinates, pixel ``j`` spans ``[j, j+1)``) of the
    floor-wall boundary per column, NaN where there is none.

    Scans up from the bottom row through floor pixels. The first non-floor pixel ``j`` may be
    a blend of floor and wall (MSAA); its floor share is solved against the pixel above, and
    the boundary is ``j + 1 - share``. A column is invalid when the bottom pixel is not floor,
    or what lies above the floor is sky (the floor reaches the horizon: no wall in range) or
    floor again.
    """
    h, w = img.shape[:2]
    fm = (floor_fn or floor_mask)(img)
    sm = sky_mask(img)
    cols = np.arange(w) if cols is None else np.asarray(cols)
    out = np.full(len(cols), np.nan)
    for n, u in enumerate(cols):
        if not fm[h - 1, u]:
            continue
        j = h - 1
        while j >= 0 and fm[j, u]:
            j -= 1
        if j < 1:
            continue
        above = img[j - 1, u].astype(float)
        if sm[j - 1, u] or fm[j - 1, u] or sm[j, u]:
            continue
        floor_c = img[j + 1, u].astype(float)
        wall_c = above
        d = floor_c - wall_c
        den = float(d @ d)
        if den < 1.0:
            continue
        share = float(np.clip((img[j, u].astype(float) - wall_c) @ d / den, 0.0, 1.0))
        out[n] = j + 1 - share
    return out


def fit_height(
    z: np.ndarray, rows: np.ndarray, fy: float, cy: float | None = 120.0, offset: bool = True
) -> dict:
    """Fit ``row = cy + fy * h / (d - a)`` for height ``h`` and axial offset ``a``.

    ``z`` is the true distance from the pose (rotation centre) to the wall face; ``a`` is how
    far ahead of the pose the camera sits (so the camera's depth is ``d - a``). ``cy=None``
    fits the principal row too. Returns the estimates, 95 % intervals from the Jacobian, and
    the RMS and max residual in pixels.
    """
    z = np.asarray(z, float)
    rows = np.asarray(rows, float)
    names = (["h"] + (["a"] if offset else [])) + ([] if cy is not None else ["cy"])

    def unpack(p):
        it = iter(p)
        h = next(it)
        a = next(it) if offset else 0.0
        c = cy if cy is not None else next(it)
        return h, a, c

    def res(p):
        h, a, c = unpack(p)
        return c + fy * h / (z - a) - rows

    p0 = [0.15] + ([0.0] if offset else []) + ([] if cy is not None else [120.0])
    s = least_squares(res, p0, method="lm")
    dof = max(len(z) - len(p0), 1)
    sigma2 = float(s.fun @ s.fun) / dof
    cov = np.linalg.inv(s.jac.T @ s.jac) * sigma2
    ci = 1.96 * np.sqrt(np.diag(cov))
    return {
        **{n: float(v) for n, v in zip(names, s.x, strict=True)},
        **{f"{n}_ci95": float(c) for n, c in zip(names, ci, strict=True)},
        "rms_px": float(np.sqrt(np.mean(s.fun**2))),
        "max_px": float(np.abs(s.fun).max()),
        "n": int(len(z)),
    }


WALL_HALF = 2e-3  # half thickness of a rendered wall [m]; stage 0 found walls are thin planes


def wall_segments(occ: np.ndarray) -> np.ndarray:
    """Rendered wall segments ``(x0, y0, x1, y1)`` for an occupancy grid (stage 2, task 0).

    A segment joins the centres of every pair of 4-adjacent wall cells. At a **free end**
    (a wall cell with no neighbour, or exactly one) the wall extends 0.1 m past the cell
    centre: an isolated cell is an x segment 0.2 m long; a stub end continues in its own
    direction. Corners and junctions end at the cell centre. (Stage 0's earlier rule, a 0.2 m
    segment per wall cell along each neighbour axis, wrongly extended corners past their centre.)
    """
    rows, cols = occ.shape
    segs = []
    h = CELL / 2

    def wall(r, c):
        return 0 <= r < rows and 0 <= c < cols and bool(occ[r, c])

    for r in range(rows):
        for c in range(cols):
            if not occ[r, c]:
                continue
            x, y = c * CELL, r * CELL
            if wall(r, c + 1):
                segs.append((x, y, x + CELL, y))
            if wall(r + 1, c):
                segs.append((x, y, x, y + CELL))
            nb = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            present = [(dr, dc) for dr, dc in nb if wall(r + dr, c + dc)]
            if len(present) == 0:
                segs.append((x - h, y, x + h, y))
            elif len(present) == 1:
                dr, dc = present[0]
                segs.append((x, y, x - dc * h, y - dr * h))
    return np.array(segs)


class Scene:
    """The maze as the renderer draws it: thin planes (``wall_segments``), stored as thin
    axis-aligned boxes for ray casting; ``segments`` lists them as ``(x0, y0, x1, y1)``."""

    def __init__(self, occ: np.ndarray, half: float = WALL_HALF) -> None:
        self.segments = wall_segments(occ)
        s = self.segments
        self.boxes = np.c_[
            np.minimum(s[:, 0], s[:, 2]) - half,
            np.maximum(s[:, 0], s[:, 2]) + half,
            np.minimum(s[:, 1], s[:, 3]) - half,
            np.maximum(s[:, 1], s[:, 3]) + half,
        ]

    def cast(self, origin, dirs: np.ndarray) -> np.ndarray:
        """First-hit parameter ``t`` along ``origin + t * d`` for each row of ``dirs``;
        NaN for a miss. Slab test against every box, vectorised over rays."""
        o = np.asarray(origin, float)
        d = np.asarray(dirs, float)[:, None, :]  # (R, 1, 2)
        b = self.boxes
        with np.errstate(divide="ignore", invalid="ignore"):
            tx0 = (b[None, :, 0] - o[0]) / d[..., 0]
            tx1 = (b[None, :, 1] - o[0]) / d[..., 0]
            ty0 = (b[None, :, 2] - o[1]) / d[..., 1]
            ty1 = (b[None, :, 3] - o[1]) / d[..., 1]
        # a ray parallel to a slab never enters it unless it starts inside
        inx = (o[0] >= b[:, 0]) & (o[0] <= b[:, 1])
        iny = (o[1] >= b[:, 2]) & (o[1] <= b[:, 3])
        tx0 = np.where(d[..., 0] == 0, np.where(inx, -np.inf, np.inf), tx0)
        tx1 = np.where(d[..., 0] == 0, np.where(inx, np.inf, -np.inf), tx1)
        ty0 = np.where(d[..., 1] == 0, np.where(iny, -np.inf, np.inf), ty0)
        ty1 = np.where(d[..., 1] == 0, np.where(iny, np.inf, -np.inf), ty1)
        tn = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        tf = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        hit = (tn <= tf) & (tf > 1e-9)
        tn = np.where(hit, np.maximum(tn, 0.0), np.inf)
        t = tn.min(axis=1)
        return np.where(np.isfinite(t), t, np.nan)


def column_dirs(yaw: float, cols: np.ndarray, fx: float, cx: float) -> np.ndarray:
    """Unnormalised ground-plane ray per image column: ``forward + ((u + 0.5 - cx) / fx) *
    right``, so the hit parameter ``t`` is the depth along the optical axis."""
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    return fwd[None] + ((np.asarray(cols) + 0.5 - cx) / fx)[:, None] * right[None]


def raycast_depths(scene: Scene, pos, yaw: float, cols: np.ndarray, fx: float, cx: float):
    return scene.cast(pos, column_dirs(yaw, cols, fx, cx))


def fit_offset(samples: list[dict], fy: float, fx: float, cx: float, free_cy: bool = False) -> dict:
    """Joint fit of height, camera offset ``(a, l)`` and (optionally) principal row from many
    renders at many yaws. Each sample: ``scene``, ``pos`` and ``yaw`` of the *pose*, the
    measured ``rows`` per ``cols`` and a ``keep`` mask. The camera sits at
    ``pos + a * fwd + l * left``.
    """

    def residuals(q):
        # parameters are scaled to order 1 so finite-difference steps are meaningful
        h, a, l_ = q[0] * 0.2, q[1] * 0.01, q[2] * 0.01
        c = q[3] * 100.0 if free_cy else 120.0
        out = []
        for s in samples:
            fwd = np.array([np.cos(s["yaw"]), np.sin(s["yaw"])])
            left = np.array([-np.sin(s["yaw"]), np.cos(s["yaw"])])
            cam = np.asarray(s["pos"]) + a * fwd + l_ * left
            z = raycast_depths(s["scene"], cam, s["yaw"], s["cols"], fx, cx)
            pred = c + fy * h / z
            out.append(pred[s["keep"]] - s["rows"][s["keep"]])
        return np.concatenate(out)

    q0 = [1.05, 0.01, 0.01] + ([1.2] if free_cy else [])
    s = least_squares(residuals, q0, loss="soft_l1", f_scale=0.5, diff_step=1e-4)
    r = s.fun
    names = ["h", "a", "l"] + (["cy"] if free_cy else [])
    scale = np.array([0.2, 0.01, 0.01, 100.0][: len(q0)])
    cov = np.linalg.inv(s.jac.T @ s.jac) * float(r @ r) / max(len(r) - len(q0), 1)
    ci = 1.96 * np.sqrt(np.diag(cov)) * scale
    return {
        **{n: float(v) for n, v in zip(names, s.x * scale, strict=True)},
        **{f"{n}_ci95": float(c) for n, c in zip(names, ci, strict=True)},
        "rms_px": float(np.sqrt(np.mean(r**2))),
        "median_abs_px": float(np.median(np.abs(r))),
        "p90_abs_px": float(np.percentile(np.abs(r), 90)),
        "frac_below_0.5px": float(np.mean(np.abs(r) < 0.5)),
        "frac_above_2px": float(np.mean(np.abs(r) > 2.0)),
        "n": int(len(r)),
    }


def predicted_rows(z: np.ndarray, fy: float, cy: float, h: float) -> np.ndarray:
    return cy + fy * h / z
