"""Stage 8: the goal pose on the map from the four goal views (pipeline code, rule 10: the map
and the goal images only).

1. **Goal scan.** ``target.jpg`` (or ``info.targets``) holds four 320-px views from the goal,
   facing front, left, back and right, i.e. at goal headings theta + k * 90 deg (counter-
   clockwise; measured on dev). **The goal views are rendered from a lower camera than the
   robot's:** the ratio of floor-edge to wall-top offsets from the horizon gives h = 0.160 m
   on every dev seed and on the challenge download, against 0.210 m for exploration frames
   (stage 8). The height is measured from the views themselves, then each view gives a virtual
   scan (stage 2) and the four are rotated into the goal frame. Adjacent views overlap by about
   30 deg; their disagreement there is reported as a consistency check.
2. **Heading.** The scan's straight pieces give the maze axes in the goal frame (stage 3's scan
   compass): theta is known modulo 90 deg, so four headings are searched, each +/- 2 deg.
3. **Global search.** Every candidate position on the map's free cells (log-odds < 0, at
   least ``CLEAR`` from the collision segments), at the four headings, scored by the mean of a
   likelihood field (Gaussian of the distance to the nearest rendered wall) at the scan
   points; then a fine grid around the best ``N_REFINE`` distinct candidates, and a point-to-
   wall Gauss-Newton refinement.
4. **Texture term.** For each refined candidate, the goal views' wall pixels are assigned to
   the map's wall faces at that pose and compared with the faces' signatures (stage 7,
   Hellinger). Textures repeat, so this term only separates candidates that geometry leaves
   close.
5. **Output.** The best pose, a confidence (score margin to the best candidate that is a
   different place: more than ``DISTINCT_M`` away or a different heading branch), and the top
   three.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

import grid
import scan as sc
import segment as sg
import se2
import vo

VIEW_W = 320
SIGMA_FIELD = 0.03  # m, likelihood-field width
FIELD_RES = 0.01  # m
CLEAR = 0.08  # m from the collision segments for a candidate position
COARSE_PTS = 240  # scan points used in the global search
N_REFINE = 12  # distinct candidates refined
DISTINCT_M = 0.25  # m: candidates closer than this (same heading branch) are the same place
FINE_STEP, FINE_HALF = 0.01, 0.05  # m
FINE_DEG = (-2.0, -1.0, 0.0, 1.0, 2.0)
TEX_WEIGHT = 0.5
TEX_MIN_PIXELS = 100
CONF_MIN = 0.10  # below this margin the placement is flagged unreliable (dev: 39 of 40 correct
#                  stress queries had >= 0.18, the one failure 0.02; stage 8)
H_RANGE = (0.10, 0.30)


@dataclass
class GoalScan:
    xy: np.ndarray  # (n, 2) goal frame: (forward, left)
    sigma: np.ndarray
    view: np.ndarray  # which view each point came from
    normal: np.ndarray  # (n, 2) goal frame, NaN where none
    cam: sc.Camera
    overlap_residual: float  # median disagreement of adjacent views in their overlap (m)
    views: list = field(default_factory=list)  # per view: (k, img, rows, top, status[, angle])


def split_views(target: np.ndarray) -> list[np.ndarray]:
    """``target.jpg`` (240 x 1280) or a list of four views."""
    if isinstance(target, (list, tuple)):
        return [np.asarray(v) for v in target]
    return [target[:, VIEW_W * k: VIEW_W * (k + 1)] for k in range(target.shape[1] // VIEW_W)]


def camera_height(segs: list, cam: sc.Camera) -> float:
    """Camera height from the walls' bottom and top edges: (v_b - cy) / (cy - v_t) = h / (H - h)."""
    r = []
    for s in segs:
        ok = np.isfinite(s.rows) & np.isfinite(s.top_rows) & (s.status == sg.WALL_RETURN) & (s.top_rows < cam.cy - 2)
        r.append((s.rows[ok] - cam.cy) / (cam.cy - s.top_rows[ok]))
    r = np.concatenate(r) if r else np.zeros(0)
    if len(r) < 50:
        return cam.h
    q = float(np.median(r))
    return float(np.clip(cam.wall_height * q / (1 + q), *H_RANGE))


def goal_scan(target, use=(0, 1, 2, 3)) -> GoalScan:
    """The merged scan of the views ``use`` (all four by default; fewer for stress tests)."""
    base = sc.camera_config()
    allv = split_views(target)
    views = {k: allv[k] for k in use}
    segs = {k: sg.segment(v) for k, v in views.items()}
    cam = replace(base, h=camera_height(list(segs.values()), base))
    XY, SIG, V, N, VW = [], [], [], [], []
    for k in views:
        v, s = views[k], segs[k]
        status = sc.scan_from_segmentation(s, cam).status
        p = vo.scan_points(s.rows, s.top_rows, status, cam)
        a = k * np.pi / 2
        c, sn = np.cos(a), np.sin(a)
        R = np.array([[c, -sn], [sn, c]])
        XY.append(p.xy @ R.T)
        N.append(p.normal @ R.T)
        SIG.append(p.sigma)
        V.append(np.full(len(p.xy), k))
        VW.append((k, v, s.rows, s.top_rows, status))
    xy, normal = np.concatenate(XY), np.concatenate(N)
    view = np.concatenate(V)
    # adjacent views overlap by ~30 deg: nearest-point disagreement there
    d = []
    for k in views:
        if (k + 1) % 4 not in views:
            continue
        a, b = xy[view == k], xy[view == (k + 1) % 4]
        if len(a) and len(b):
            dd = np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)).min(1)
            d.append(dd[dd < 0.10])
    d = np.concatenate(d) if d else np.zeros(0)
    return GoalScan(xy, np.concatenate(SIG), view, normal, cam, float(np.median(d)) if len(d) else np.nan, VW)


def axis_heading(gs: GoalScan) -> float:
    """theta modulo 90 deg: the rotation that puts the scan's wall normals on the maze axes."""
    ok = np.isfinite(gs.normal[:, 0])
    phi = np.arctan2(gs.normal[ok, 1], gs.normal[ok, 0])
    w = 1.0 / np.maximum(gs.sigma[ok], 0.003) ** 2
    m = np.angle((w * np.exp(4j * phi)).sum()) / 4
    return float(-m)


# ------------------------------------------------------------------------------ the map


@dataclass
class MapField:
    segments: np.ndarray
    collision: np.ndarray
    logodds: np.ndarray
    spec: grid.GridSpec
    origin: np.ndarray
    field: np.ndarray  # likelihood field at FIELD_RES
    candidates: np.ndarray  # (n, 2) candidate positions
    cells_origin: np.ndarray
    wall_cells: np.ndarray
    sig_index: dict
    sig_hist: np.ndarray


def load_map(path_or_npz) -> MapField:
    z = np.load(path_or_npz, allow_pickle=True) if not hasattr(path_or_npz, "files") else path_or_npz
    segs = z["wall_segments"]
    col = z["collision_segments"]
    spec = grid.GridSpec(z["origin"], z["logodds"].shape, float(z["resolution"]))
    x0, x1, y0, y1 = spec.extent()
    n = (int(np.ceil((x1 - x0) / FIELD_RES)), int(np.ceil((y1 - y0) / FIELD_RES)))
    img = np.ones((n[1], n[0]), np.uint8)
    for s in segs:
        p0 = tuple(int(round(v)) for v in (np.array(s[:2]) - (x0, y0)) / FIELD_RES)
        p1 = tuple(int(round(v)) for v in (np.array(s[2:]) - (x0, y0)) / FIELD_RES)
        cv2.line(img, p0, p1, 0, 1)
    D = cv2.distanceTransform(img, cv2.DIST_L2, 5) * FIELD_RES
    L = np.exp(-0.5 * (D / SIGMA_FIELD) ** 2).astype(np.float32)
    lo = z["logodds"]
    ny, nx = lo.shape
    xs = spec.origin[0] + (np.arange(nx) + 0.5) * spec.res
    ys = spec.origin[1] + (np.arange(ny) + 0.5) * spec.res
    X, Y = np.meshgrid(xs, ys)
    free = lo < 0
    P = np.c_[X[free], Y[free]]
    P = P[grid.point_segment_distance(P, col) > CLEAR]
    keys = [tuple(k) for k in z["sig_keys"]]
    return MapField(segs, col, lo, spec, np.array([x0, y0]), L, P, z["cells_origin"], z["wall_cells"],
                    {(r, c, grid.SIDES[s]): i for i, (r, c, s) in enumerate(keys)}, z["sig_hist"])


def _score(mf: MapField, xy, w, poses) -> np.ndarray:
    """Mean likelihood of the scan points ``xy`` (goal frame) at each pose (n, 3)."""
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    wx = poses[:, None, 0] + c[:, None] * xy[None, :, 0] - s[:, None] * xy[None, :, 1]
    wy = poses[:, None, 1] + s[:, None] * xy[None, :, 0] + c[:, None] * xy[None, :, 1]
    ix = np.floor((wx - mf.origin[0]) / FIELD_RES).astype(int)
    iy = np.floor((wy - mf.origin[1]) / FIELD_RES).astype(int)
    ok = (ix >= 0) & (ix < mf.field.shape[1]) & (iy >= 0) & (iy < mf.field.shape[0])
    v = np.where(ok, mf.field[np.clip(iy, 0, mf.field.shape[0] - 1), np.clip(ix, 0, mf.field.shape[1] - 1)], 0.0)
    return (v * w[None, :]).sum(1) / w.sum()


def _weights(gs: GoalScan, idx=None):
    w = 1.0 / np.maximum(gs.sigma, 0.005) ** 2
    w = w / w.max()
    return w if idx is None else w[idx]


def refine(mf: MapField, gs: GoalScan, pose, iters: int = 15, dmax: float = 0.08):
    """Point-to-wall Gauss-Newton: each point's residual is its signed distance to the
    nearest rendered wall segment's line (walls are axis-aligned)."""
    p = np.array(pose, float)
    w = _weights(gs)
    reach = float(np.hypot(*gs.xy.T).max()) + 0.3 if len(gs.xy) else 3.0
    mid = (mf.segments[:, :2] + mf.segments[:, 2:]) / 2
    segs = mf.segments[np.hypot(*(mid - p[:2]).T) < reach]
    if len(segs) == 0:
        return p
    for _ in range(iters):
        c, s = np.cos(p[2]), np.sin(p[2])
        wx = p[0] + c * gs.xy[:, 0] - s * gs.xy[:, 1]
        wy = p[1] + s * gs.xy[:, 0] + c * gs.xy[:, 1]
        a, b = segs[:, :2], segs[:, 2:]
        ab = b - a
        q = np.c_[wx, wy][:, None, :]
        t = np.clip(((q - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        pr = a + t[..., None] * ab
        d = np.hypot(*(q - pr).transpose(2, 0, 1))
        j = d.argmin(1)
        dist = d[np.arange(len(j)), j]
        nrm = (q[:, 0, :] - pr[np.arange(len(j)), j])
        nrm = nrm / np.maximum(np.hypot(*nrm.T), 1e-9)[:, None]
        keep = dist < dmax
        if keep.sum() < 10:
            break
        r = dist[keep]
        n = nrm[keep]
        # d r / d (x, y, theta) for a point p_w = R(theta) p_g + t
        px, py = gs.xy[keep, 0], gs.xy[keep, 1]
        dth = np.c_[-s * px - c * py, c * px - s * py]
        J = np.c_[n[:, 0], n[:, 1], (n * dth).sum(1)]
        W = w[keep] / (1 + (r / 0.02) ** 2)
        H = J.T @ (J * W[:, None]) + np.diag([1e-6, 1e-6, 1e-6])
        delta = -np.linalg.solve(H, J.T @ (W * r))
        p += delta
        if np.abs(delta[:2]).max() < 1e-4 and abs(delta[2]) < 1e-4:
            break
    p[2] = se2.wrap(p[2])
    return p


def texture_score(mf: MapField, gs: GoalScan, pose) -> tuple[float, int]:
    """1 - mean Hellinger distance between the goal views' wall-face histograms at ``pose`` and
    the map's face signatures (faces with at least ``TEX_MIN_PIXELS`` pixels); and the number
    of faces compared."""
    cells = grid.CellGrid(int(mf.cells_origin[0]), int(mf.cells_origin[1]), mf.wall_cells, mf.wall_cells, np.zeros(mf.wall_cells.shape))
    b = grid.SignatureBuilder(cells, cam=gs.cam)
    for view in gs.views:
        k, img, rows, top, status = view[:5]
        # a view may carry its own angle (stage 10: a spin's frames are not k * 90 deg apart);
        # without one it is the goal view's k * 90 deg, with the expression unchanged
        angle = view[5] if len(view) > 5 else k * np.pi / 2
        b.add(img, rows, top, status, (pose[0], pose[1], pose[2] + angle), k, col_step=1)
    sig = b.result()
    d, wsum = 0.0, 0.0
    n = 0
    for key, h, px in zip(sig.keys, sig.hist, sig.pixels, strict=True):
        i = mf.sig_index.get(key)
        if i is None or px < TEX_MIN_PIXELS:
            continue
        d += px * float(grid.hellinger(h, mf.sig_hist[i]))
        wsum += px
        n += 1
    return (1.0 - d / wsum, n) if wsum else (0.0, 0)


@dataclass
class GoalEstimate:
    pose: np.ndarray
    confidence: float
    top: list  # [(pose, score, geo, tex, faces)] best first, distinct places
    heading_mod90: float
    camera_height: float
    overlap_residual: float
    n_points: int

    @property
    def reliable(self) -> bool:
        return self.confidence >= CONF_MIN


def place(mf: MapField, target, use_texture: bool = True, rng=None, use=(0, 1, 2, 3)) -> GoalEstimate:
    """The goal pose from ``target`` (``target.jpg``, the list ``info.targets``, or a
    ``GoalScan``) on the map ``mf`` (``load_map``)."""
    gs = target if isinstance(target, GoalScan) else goal_scan(target, use)
    rng = rng or np.random.default_rng(0)
    th0 = axis_heading(gs)
    heads = th0 + np.arange(4) * np.pi / 2
    idx = rng.choice(len(gs.xy), min(COARSE_PTS, len(gs.xy)), replace=False) if len(gs.xy) else np.zeros(0, int)
    xy_c, w_c = gs.xy[idx], _weights(gs, idx)
    # coarse: every candidate cell x 4 headings x 3 heading offsets
    cand = []
    for h in heads:
        for dh in np.radians((-1.5, 0.0, 1.5)):
            P = np.c_[mf.candidates, np.full(len(mf.candidates), h + dh)]
            sc_ = np.concatenate([_score(mf, xy_c, w_c, P[j:j + 2000]) for j in range(0, len(P), 2000)])
            cand.append(np.c_[P, sc_])
    cand = np.concatenate(cand)
    cand = cand[np.argsort(-cand[:, 3])]
    picked = []
    for c in cand:
        if all(np.hypot(*(c[:2] - p[:2])) > DISTINCT_M or abs(se2.wrap(c[2] - p[2])) > np.radians(45) for p in picked):
            picked.append(c)
        if len(picked) >= N_REFINE:
            break
    # fine grid and Gauss-Newton on every picked candidate, all points
    w_all = _weights(gs)
    out = []
    g = np.arange(-FINE_HALF, FINE_HALF + 1e-9, FINE_STEP)
    for c in picked:
        GX, GY, GT = np.meshgrid(c[0] + g, c[1] + g, c[2] + np.radians(FINE_DEG), indexing="ij")
        P = np.c_[GX.ravel(), GY.ravel(), GT.ravel()]
        s_ = _score(mf, gs.xy, w_all, P)
        p = refine(mf, gs, P[np.argmax(s_)])
        geo = float(_score(mf, gs.xy, w_all, p[None])[0])
        tex, nf = texture_score(mf, gs, p) if use_texture else (0.0, 0)
        out.append((p, geo + TEX_WEIGHT * tex, geo, tex, nf))
    out.sort(key=lambda r: -r[1])
    best = out[0]
    rival = next((r for r in out[1:] if np.hypot(*(r[0][:2] - best[0][:2])) > DISTINCT_M or abs(se2.wrap(r[0][2] - best[0][2])) > np.radians(45)), None)
    conf = best[1] - (rival[1] if rival is not None else 0.0)
    distinct = [best] + [r for r in out[1:] if all(np.hypot(*(r[0][:2] - q[0][:2])) > DISTINCT_M or abs(se2.wrap(r[0][2] - q[0][2])) > np.radians(45) for q in [best])]
    return GoalEstimate(best[0], float(conf), distinct[:3], float(np.mod(th0, np.pi / 2)), gs.cam.h, gs.overlap_residual, len(gs.xy))
