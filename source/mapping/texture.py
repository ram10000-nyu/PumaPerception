"""Texture adjacency (stage 5b): wall faces, their appearance, and the boundaries between them.

Stage 5b task 0 measured (from ground truth) that every wall-cell face carries one texture
image, faces change at cell edges (0.2 m apart), about 75 % of faces have an identical twin
elsewhere in the maze, but 98.7 to 100 % of ordered neighbour pairs (A|B) are unique in their
maze. A boundary is therefore a near-unique landmark, and a single physical point.

Per frame (pipeline code, rule 10: the image and its scan only):

1. **Straight pieces** of the scan (continuous depth, split where the points leave a line).
2. **Face edges.** Along a piece, the wall's appearance changes at the cell edges, which are
   0.2 m apart; the phase of that lattice is the one that maximises the column-to-column
   appearance change on it.
3. **Faces.** Each lattice interval fully seen is resampled to a rectified ``NS x NZ`` colour
   patch (positions along the wall x heights on the wall), zero-mean and unit-norm.
4. **Boundaries.** At every face edge with ``WIN`` of wall seen on both sides: descriptors of
   the window left and right of the edge (as seen in the image) and the edge's position in the
   robot frame (forward, left). Fixed windows cover the same physical patch however much of
   each face is visible (two full faces in a row need 0.4 m of continuous wall, which is rare).

``match_boundaries`` compares two frames' boundaries under a relative pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scan as sc
import segment as sg

CELL = 0.2
NZ = 16  # samples up the wall
NS = 10  # samples along a face
WIN = 0.08  # window on each side of a face edge [m]
NW = 8  # samples along each window
Z_MIN, Z_MAX = 0.17, 1.5  # depth range used for appearance (floor edge in the image; resolution)
PHASE_STEP = 0.005
MIN_PHASE_RATIO = 1.4  # lattice change / mean change needed to trust a phase
CORR_MIN = 0.80  # descriptor correlation for "same texture" (stage 5b, chosen on dev)
POS_TOL = 0.05  # boundary positions must agree within this under the relative pose [m]


@dataclass
class Boundary:
    left: np.ndarray  # descriptor of the face on the left in the image (NS*NZ*3,)
    right: np.ndarray
    pos: np.ndarray  # robot frame (forward, left) [m]
    depth: float


def _strips(img: np.ndarray, s: sc.Scan, cam: sc.Camera):
    """Per column: wall colours at NZ heights (NaN where unusable)."""
    zk = (np.arange(NZ) + 0.5) * cam.wall_height / NZ
    out = np.full((len(s.u), NZ, 3), np.nan)
    ok = s.hit & np.isfinite(s.z) & (s.z > Z_MIN) & (s.z < Z_MAX)
    for u in np.nonzero(ok)[0]:
        v = np.round(cam.cy + cam.fy * (cam.h - zk) / s.z[u] - 0.5).astype(int)
        if v.min() < 0 or v.max() >= img.shape[0]:
            continue
        out[u] = img[v, u]
    return out


def _pieces(p: np.ndarray, ok: np.ndarray, tol0: float = 0.008, tol1: float = 0.03):
    """Straight pieces as lists of column indices: continuous runs split recursively where a
    point is farther than ``tol0 + tol1 * depth`` from the chord."""
    runs, cur = [], []
    for u in range(len(ok)):
        if ok[u] and cur and np.hypot(*(p[u] - p[cur[-1]])) > 0.03 + 0.04 * p[u][0]:
            runs.append(cur)
            cur = []
        if ok[u]:
            cur.append(u)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []

    def split(idx):
        if len(idx) < 6:
            return
        a, b = p[idx[0]], p[idx[-1]]
        d = b - a
        n = np.hypot(*d)
        if n < 1e-6:
            return
        nrm = np.array([-d[1], d[0]]) / n
        dev = np.abs((p[idx] - a) @ nrm)
        k = int(np.argmax(dev))
        if dev[k] > tol0 + tol1 * p[idx[k]][0] and 2 <= k <= len(idx) - 3:
            split(idx[: k + 1])
            split(idx[k:])
        else:
            out.append(idx)

    for r in runs:
        split(r)
    return out


def _desc(patch: np.ndarray) -> np.ndarray:
    d = patch.reshape(-1).astype(float)
    d = d - d.mean()
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def boundaries(img: np.ndarray, s: sc.Scan | None = None, cam: sc.Camera | None = None) -> list[Boundary]:
    cam = cam or sc.camera_config()
    s = s if s is not None else sc.scan_from_image(img, r_max=Z_MAX * 1.2, cam=cam)
    strips = _strips(img, s, cam)
    ok = np.isfinite(strips).all(axis=(1, 2))
    p = np.stack([s.z, -s.x], axis=1)  # robot frame (forward, left)
    out: list[Boundary] = []
    for idx in _pieces(p, ok):
        idx = np.array(idx)
        a, b = p[idx[0]], p[idx[-1]]
        d = (b - a) / max(np.hypot(*(b - a)), 1e-9)  # left-to-right in the image
        sp = (p[idx] - a) @ d
        if sp[-1] - sp[0] < 1.5 * CELL:
            continue
        col = strips[idx]
        diff = np.abs(np.diff(col, axis=0)).mean(axis=(1, 2))
        mid = 0.5 * (sp[1:] + sp[:-1])
        gap = np.diff(sp)
        best, best_r = None, 0.0
        base = diff.mean() + 1e-9
        for ph in np.arange(0, CELL, PHASE_STEP):
            w = np.abs(((mid - ph + CELL / 2) % CELL) - CELL / 2) <= np.maximum(gap / 2, 0.004)
            if w.sum() == 0:
                continue
            r = diff[w].mean() / base
            if r > best_r:
                best, best_r = ph, r
        if best is None or best_r < MIN_PHASE_RATIO:
            continue
        edges = np.arange(best + CELL * np.ceil((sp[0] - best) / CELL), sp[-1] + 1e-9, CELL)
        for e in edges:
            side = []
            for lo, hi in ((e - WIN, e), (e, e + WIN)):
                m = (sp >= lo - 0.01) & (sp <= hi + 0.01)
                if m.sum() < 3 or sp[m].min() > lo + 0.012 or sp[m].max() < hi - 0.012 or np.diff(sp[m]).max() > 0.025:
                    side = None
                    break
                t = lo + (np.arange(NW) + 0.5) * WIN / NW
                patch = np.stack([[np.interp(t, sp[m], col[m][:, r, ch]) for r in range(NZ)] for ch in range(3)])
                side.append(_desc(patch))
            if side is None:
                continue
            pos = a + e * d
            out.append(Boundary(side[0], side[1], pos, float(pos[0])))
    return out


def transform(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points in frame b to frame a, ``pose`` = b's pose in a (forward, left, yaw)."""
    c, s_ = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s_], [-s_, c]]) + pose[:2]


def match_boundaries(ba: list[Boundary], bb: list[Boundary], rel: np.ndarray,
                     corr_min: float = CORR_MIN, pos_tol: float = POS_TOL) -> int:
    """Number of boundaries of frame b that match one of frame a: both faces' descriptors
    correlate above ``corr_min`` and the positions agree within ``pos_tol`` under ``rel``."""
    if not ba or not bb:
        return 0
    pa = np.stack([x.pos for x in ba])
    pb = transform(rel, np.stack([x.pos for x in bb]))
    La = np.stack([x.left for x in ba]); Ra = np.stack([x.right for x in ba])
    Lb = np.stack([x.left for x in bb]); Rb = np.stack([x.right for x in bb])
    cl = La @ Lb.T
    cr = Ra @ Rb.T
    near = np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1)) < pos_tol
    return int(((cl > corr_min) & (cr > corr_min) & near).any(axis=0).sum())


def compare_boundaries(ba: list[Boundary], bb: list[Boundary], rel: np.ndarray,
                       corr_min: float = CORR_MIN, corr_veto: float = 0.3, pos_tol: float = POS_TOL):
    """(matches, contradictions) between two frames' boundaries under ``rel``: a *match* is a
    co-located pair whose left and right descriptors both correlate above ``corr_min``; a
    *contradiction* is a co-located pair where either side correlates below ``corr_veto``
    (both frames see a face edge at the same physical spot, with different textures)."""
    if not ba or not bb:
        return 0, 0
    pa = np.stack([x.pos for x in ba])
    pb = transform(rel, np.stack([x.pos for x in bb]))
    cl = np.stack([x.left for x in ba]) @ np.stack([x.left for x in bb]).T
    cr = np.stack([x.right for x in ba]) @ np.stack([x.right for x in bb]).T
    near = np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1)) < pos_tol
    match = near & (cl > corr_min) & (cr > corr_min)
    contra = near & ((cl < corr_veto) | (cr < corr_veto)) & ~match.any(axis=0)[None, :] & ~match.any(axis=1)[:, None]
    return int(match.any(axis=0).sum()), int(contra.any(axis=0).sum())
