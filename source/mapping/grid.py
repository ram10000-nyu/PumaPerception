"""Stage 7: the top-down map from scans at optimised poses (pipeline code, rule 10: it reads
scans, poses and images only; never ground truth).

1. **Lattice and scale.** Wall points (scan returns on straight pieces, stage 4) are split by
   their world normal into points on x = const and y = const walls. The maze's walls lie on
   lines every ``WALL_PERIOD`` = 0.4 m (wall cells sit at even grid indices of the 0.2 m cell
   lattice), so the circular mean of ``exp(2 pi i c / 0.4)`` over the wall coordinates c peaks
   when the poses are at the right scale. The scale error is the odometry's (stage 6: poses
   about 0.5 % small; the scans are metric), so the search rescales the poses about the anchor,
   not the scans; then the poses are shifted so the wall lines fall on multiples of 0.4 m.
   Rotation is left alone (stage 6 heading priors: residual 0.12 deg).
2. **Occupancy grid** (log-odds, ``RES`` = 0.05 m, wall lines on cell centres): every column of
   every frame casts a ray; cells before a return get ``L_FREE``, the return's cell ``L_OCC``
   times a weight that falls with the return's depth noise; columns without a return (far
   plane, or a return beyond ``R_MAX``) clear free space up to ``R_MAX``; clamp at +/-5.
3. **Snapped walls.** The maze's wall cells: posts (even, even) and edge cells (one index odd);
   odd-odd cells are never walls. An edge cell is a wall when the grid's evidence along its
   0.2 m span is positive; a post is a wall when two or more of its edge cells are, otherwise by
   its own evidence (a stub's free end stops 0.1 m short of the next post). The wall geometry is
   the renderer's rule (centres of 4-adjacent wall cells, 0.1 m past free ends).
4. **Wall-face signatures:** per wall cell and side, a 16 x 4 hue-saturation histogram of the
   wall pixels that saw it (Hellinger-normalised: square root of the frequencies), and the
   frames that saw it.
5. **Configuration space:** free cells (log-odds < 0) farther than ``ROBOT_RADIUS`` from every
   **collision segment**. Measured on dev (stage 7): the simulator collides the robot with the
   segments joining the centres of 4-adjacent wall cells, without the rendered 0.1 m free-end
   extension, at a radius of about 0.12 m (true poses never come closer than 0.117 m to them; to
   the rendered geometry they come as close as 0.047 m at wall ends and 0.117 m along faces).
   Stage 0's 0.08 m was measured to the rendered planes at wall ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import scan as sc
import segment as sg
import vo

CELL = 0.2  # m, the maze's cell lattice
WALL_PERIOD = 2 * CELL  # walls lie on every other lattice line
RES = 0.05  # m, occupancy grid resolution
R_MAX = sc.R_MAX  # 1.5 m (stage 2)
L_OCC, L_FREE, L_CLAMP = 0.85, -0.4, 5.0
SIGMA_REF = 0.01  # m: a return this precise (or better) votes with full weight
W_MIN = 0.25  # floor of the depth-noise weight of a return
EDGE_TAU = 0.0  # an edge cell is a wall when its mean evidence exceeds this
POST_TAU = 0.0
SIDE_D = 0.1  # m: where a wall cell looks for known free space on either side
OBSERVED_MIN = 1.0  # |log-odds| somewhere on a cell for it to count as observed
ROBOT_RADIUS = 0.115  # m, to the collision segments (dev minimum 0.117 m; stage 7)
PERIOD_SEARCH = (0.97, 1.03)  # scale range searched by the lattice fit
H_BINS, S_BINS = 16, 4
SIDES = ("N", "S", "E", "W")


@dataclass
class GridSpec:
    origin: np.ndarray  # world position of cell (0, 0)'s lower-left corner
    shape: tuple[int, int]  # (ny, nx)
    res: float = RES

    def index(self, x, y):
        return (
            np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(int),
            np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int),
        )

    def extent(self):
        return [
            self.origin[0],
            self.origin[0] + self.shape[1] * self.res,
            self.origin[1],
            self.origin[1] + self.shape[0] * self.res,
        ]


def spec_for(poses: list[np.ndarray], margin: float = R_MAX + 0.3) -> GridSpec:
    """A grid covering every pose plus ``margin``, with lattice lines (multiples of 0.05 m) on
    cell centres."""
    P = np.concatenate([p[:, :2] for p in poses])
    lo = np.floor((P.min(0) - margin) / CELL) * CELL - RES / 2
    hi = np.ceil((P.max(0) + margin) / CELL) * CELL + RES / 2
    n = np.round((hi - lo) / RES).astype(int)
    return GridSpec(lo, (int(n[1]), int(n[0])))


# ------------------------------------------------------------------------------ columns


@dataclass
class Columns:
    """One frame's columns in the robot frame: range, bearing (right positive), noise, and
    whether the column ends in a wall (``hit``) or only clears free space to ``free_len``."""

    rng: np.ndarray
    bearing: np.ndarray
    sigma: np.ndarray
    hit: np.ndarray
    free_len: np.ndarray


def columns(rows, top, status, cam: sc.Camera, r_max: float = R_MAX) -> Columns:
    n = len(rows)
    u = np.arange(n) + 0.5  # full-width columns
    z = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    wall = (status == sg.WALL_RETURN) & (rows > cam.cy + 1e-6)
    z[wall] = cam.fy * cam.h / (rows[wall] - cam.cy)
    sig[wall] = np.maximum(0.002, vo.SIGMA_PIXEL * z[wall] ** 2 / (cam.fy * cam.h))
    near = status == sg.TOP_RETURN
    z[near] = sc.top_depth(top[near], cam)
    sig[near] = vo.SIGMA_TOP
    x = z * (u - cam.cx) / cam.fx
    rng = np.hypot(x, z)
    bearing = np.where(np.isfinite(z), np.arctan2(x, z), np.arctan2(u - cam.cx, cam.fx))
    ret = np.isfinite(rng)
    hit = ret & (rng <= r_max)
    free_len = np.zeros(n)
    free_len[hit] = rng[hit] - RES  # free strictly before the end cell
    beyond = (ret & (rng > r_max)) | (status == sg.FAR)
    free_len[beyond] = r_max
    return Columns(np.where(ret, rng, np.nan), bearing, sig, hit, free_len)


def hit_weight(sigma) -> np.ndarray:
    return np.clip(SIGMA_REF / np.asarray(sigma), W_MIN, 1.0)


# ------------------------------------------------------------------------------ occupancy


def occupancy(scans, poses, spec: GridSpec, r_max: float = R_MAX, l_occ: float = L_OCC,
              l_free: float = L_FREE, step_cols: int = 1, cam: sc.Camera | None = None):
    """Log-odds grid and per-cell hit weight from ``scans[d][k] = (rows, top, status)`` placed
    at ``poses[d][k]``. Each ray updates a cell at most once."""
    cam = cam or sc.camera_config()
    ny, nx = spec.shape
    N = ny * nx
    occ = np.zeros(N)
    free = np.zeros(N)
    nsteps = int(r_max / (spec.res / 2)) + 1
    t = np.arange(nsteps) * (spec.res / 2)
    cols = np.arange(0, 320, step_cols)
    for d, drive in enumerate(scans):
        for k, (rows, top, status) in enumerate(drive):
            c = columns(rows, top, status, cam, r_max)
            if step_cols > 1:
                c = Columns(c.rng[cols], c.bearing[cols], c.sigma[cols], c.hit[cols], c.free_len[cols])
            x, y, yaw = poses[d][k]
            th = yaw - c.bearing
            sel = t[None, :] <= c.free_len[:, None]
            px = x + np.cos(th)[:, None] * t[None, :]
            py = y + np.sin(th)[:, None] * t[None, :]
            iy, ix = spec.index(px, py)
            m = sel & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            ray = np.nonzero(m)[0]
            idx = np.unique((iy * nx + ix)[m] + ray.astype(np.int64) * N) % N
            free += np.bincount(idx, minlength=N)
            if c.hit.any():
                ex = x + np.cos(th[c.hit]) * c.rng[c.hit]
                ey = y + np.sin(th[c.hit]) * c.rng[c.hit]
                ej, ei = spec.index(ex, ey)
                mm = (ei >= 0) & (ei < nx) & (ej >= 0) & (ej < ny)
                occ += np.bincount((ej * nx + ei)[mm], weights=hit_weight(c.sigma[c.hit][mm]), minlength=N)
    lo = np.clip(occ * l_occ + free * l_free, -L_CLAMP, L_CLAMP)
    return lo.reshape(ny, nx), occ.reshape(ny, nx)


# ------------------------------------------------------------------------------ lattice


def wall_points(scans, poses, r_max: float = R_MAX, cam: sc.Camera | None = None):
    """World wall points from the scan pieces whose normal is within 18 degrees of a maze axis:
    the coordinate across the wall ``c`` (x for an x = const wall, axis 0; y otherwise, axis 1),
    the axis, a weight, and the same coordinate of the robot that saw it ``pc``."""
    cam = cam or sc.camera_config()
    C, A, W, PC = [], [], [], []
    for d, drive in enumerate(scans):
        for k, (rows, top, status) in enumerate(drive):
            s = vo.scan_points(rows, top, status, cam)
            if len(s.xy) == 0:
                continue
            ok = np.isfinite(s.normal[:, 0]) & (np.hypot(*s.xy.T) <= r_max)
            if not ok.any():
                continue
            x, y, yaw = poses[d][k]
            c, sn = np.cos(yaw), np.sin(yaw)
            f, l = s.xy[ok, 0], s.xy[ok, 1]
            wx, wy = x + c * f - sn * l, y + sn * f + c * l
            nx_, ny_ = c * s.normal[ok, 0] - sn * s.normal[ok, 1], sn * s.normal[ok, 0] + c * s.normal[ok, 1]
            ax = np.where(np.abs(nx_) > 0.95, 0, np.where(np.abs(ny_) > 0.95, 1, -1))
            keep = ax >= 0
            C.append(np.where(ax[keep] == 0, wx[keep], wy[keep]))
            A.append(ax[keep])
            W.append(np.minimum((SIGMA_REF / s.sigma[ok][keep]) ** 2, 4.0))
            PC.append(np.where(ax[keep] == 0, x, y))
    if not C:
        return np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0)
    return np.concatenate(C), np.concatenate(A), np.concatenate(W), np.concatenate(PC)


@dataclass
class Lattice:
    scale: float  # the input poses' scale relative to the maze (stage 6: about 0.995)
    phase: np.ndarray  # (2,) wall-line position mod 0.4 m (x, y) after rescaling, before the shift
    offset: np.ndarray  # (2,) the shift applied after rescaling (phase wrapped to +/-0.2 m)
    anchor: np.ndarray  # (2,) the fixed point of the rescaling
    strength: float  # |mean phasor| at the peak, 0..1 (1 = every wall point on a line)

    @property
    def period(self) -> float:
        """The wall-line period as the input map shows it (0.4 m x scale)."""
        return WALL_PERIOD * self.scale

    def apply(self, poses: np.ndarray) -> np.ndarray:
        """Input-map poses to lattice-aligned poses (positions only; headings unchanged)."""
        out = np.array(poses, float, copy=True)
        out[..., :2] = self.anchor + (out[..., :2] - self.anchor) / self.scale - self.offset
        return out


def fit_lattice(c, axis, w, anchor, pc=None, n_max: int = 200_000, seed: int = 0) -> Lattice:
    """The pose scale s and the shift that put the wall points on lines every 0.4 m.

    The scale error is in the poses (odometry), not in the scans (metric from the camera
    height), so a wall point seen from robot coordinate ``pc`` moves to
    ``c + (pc - anchor) (1 / s - 1)`` when the poses are rescaled about the anchor. The
    periodogram at the 0.4 m period is searched over 1 / s (coarse, then fine); the phases at
    the peak give the shift. With ``pc`` omitted, the whole point is taken to scale."""
    rng = np.random.default_rng(seed)
    pc = c if pc is None else pc
    if len(c) > n_max:
        i = rng.choice(len(c), n_max, replace=False)
        c, axis, w, pc = c[i], axis[i], w[i], pc[i]
    terms = []
    for a in (0, 1):
        m = axis == a
        B = pc[m] - anchor[a]
        terms.append((c[m] - B, B, w[m]))  # c'(k) = A + B k with k = 1 / s

    def phasors(ks):
        out = []
        for A, B, ww in terms:
            z = np.zeros(len(ks), complex)
            for j in range(0, len(A), 20000):
                arg = (A[j:j + 20000, None] + B[j:j + 20000, None] * ks[None, :]) / WALL_PERIOD
                z += (ww[j:j + 20000, None] * np.exp(2j * np.pi * arg)).sum(0)
            out.append(z)
        return out

    def power(ks):
        return sum(np.abs(z) ** 2 for z in phasors(ks))

    coarse = 1.0 / np.linspace(*PERIOD_SEARCH, 121)
    k = coarse[np.argmax(power(coarse))]
    fine = np.linspace(k - 0.0006, k + 0.0006, 61)
    k = float(fine[np.argmax(power(fine))])
    zx, zy = (z[0] for z in phasors(np.array([k])))
    phase = np.mod(np.array([np.angle(zx), np.angle(zy)]) * WALL_PERIOD / (2 * np.pi), WALL_PERIOD)
    offset = (phase + WALL_PERIOD / 2) % WALL_PERIOD - WALL_PERIOD / 2
    strength = float((abs(zx) + abs(zy)) / max(sum(t[2].sum() for t in terms), 1e-9))
    return Lattice(1.0 / k, phase, offset, np.asarray(anchor, float), strength)


# ------------------------------------------------------------------------------ snapping


@dataclass
class CellGrid:
    """The maze's wall-cell grid: ``walls[r, c]`` for the cell centred at
    ``(c0 + c) * 0.2, (r0 + r) * 0.2``; ``observed`` where the occupancy grid had evidence."""

    r0: int
    c0: int
    walls: np.ndarray
    observed: np.ndarray
    score: np.ndarray

    def centre(self, r, c):
        return np.array([(self.c0 + c) * CELL, (self.r0 + r) * CELL])


def _lo_at(lo, spec: GridSpec, x, y):
    iy, ix = spec.index(x, y)
    ok = (ix >= 0) & (ix < spec.shape[1]) & (iy >= 0) & (iy < spec.shape[0])
    out = np.zeros(np.shape(x))
    out[ok] = lo[iy[ok], ix[ok]]
    return out


def _evidence(lo, spec, px, py, along_x: bool):
    """Mean over the sample points of the largest log-odds within one cell across the line;
    and the largest |log-odds| seen."""
    d = spec.res
    offs = (-d, 0.0, d)
    if along_x:
        v = np.max([_lo_at(lo, spec, px, py + o) for o in offs], 0)
    else:
        v = np.max([_lo_at(lo, spec, px + o, py) for o in offs], 0)
    return float(v.mean()), float(np.abs(v).max())


def snap(lo, spec: GridSpec, edge_tau: float = EDGE_TAU, post_tau: float = POST_TAU) -> CellGrid:
    """Wall cells from the lattice-aligned occupancy grid.

    Edge cells (one index odd) are decided by the evidence along their 0.2 m span; a vertical
    edge cell may also render as a horizontal 0.2 m piece (an isolated cell, the renderer's
    rule), so its evidence along x counts too. Posts (both indices even) with three or four
    wall neighbours are walls; every other post is decided by its own evidence, along the axis
    of its neighbours (both axes at a corner, x when it has none): in the true layouts a post
    is missing between 1.5 % of collinear and 2.4 % of perpendicular wall pairs, and these gaps
    are visible (stage 7)."""
    x0, x1, y0, y1 = spec.extent()
    c0 = int(np.ceil(x0 / WALL_PERIOD)) * 2
    r0 = int(np.ceil(y0 / WALL_PERIOD)) * 2
    nc = int(np.floor(x1 / WALL_PERIOD)) * 2 - c0 + 1
    nr = int(np.floor(y1 / WALL_PERIOD)) * 2 - r0 + 1
    walls = np.zeros((nr, nc), bool)
    observed = np.zeros((nr, nc), bool)
    score = np.full((nr, nc), np.nan)

    def ev(cx, cy, offs, along_x):
        offs = np.asarray(offs, float)
        n = len(offs)
        px, py = (cx + offs, np.full(n, cy)) if along_x else (np.full(n, cx), cy + offs)
        return _evidence(lo, spec, px, py, along_x)

    span = (-0.08, -0.04, 0.0, 0.04, 0.08)
    for r in range(nr):
        for c in range(nc):
            R, C = r0 + r, c0 + c
            if (R % 2) == (C % 2):
                continue
            cx, cy = C * CELL, R * CELL
            along_x = R % 2 == 0  # a horizontal wall (y = const) runs along x
            m, a = ev(cx, cy, span, along_x)
            if not along_x:
                mx, ax_ = ev(cx, cy, span, True)
                m, a = max(m, mx), max(a, ax_)
            score[r, c] = m
            observed[r, c] = a >= OBSERVED_MIN
            # a wall is seen from free space: known free on one side (rules out returns
            # smeared behind an outer wall)
            if along_x:
                side = min(_lo_at(lo, spec, np.array([cx]), np.array([cy - SIDE_D]))[0], _lo_at(lo, spec, np.array([cx]), np.array([cy + SIDE_D]))[0])
            else:
                side = min(_lo_at(lo, spec, np.array([cx - SIDE_D]), np.array([cy]))[0], _lo_at(lo, spec, np.array([cx + SIDE_D]), np.array([cy]))[0],
                           _lo_at(lo, spec, np.array([cx]), np.array([cy - SIDE_D]))[0], _lo_at(lo, spec, np.array([cx]), np.array([cy + SIDE_D]))[0])
            walls[r, c] = observed[r, c] and m > edge_tau and side < 0
    near = (-0.04, 0.0, 0.04)
    for r in range(nr):
        for c in range(nc):
            R, C = r0 + r, c0 + c
            if R % 2 or C % 2:
                continue
            nb = [(dr, dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                  if 0 <= r + dr < nr and 0 <= c + dc < nc and walls[r + dr, c + dc]]
            cx, cy = C * CELL, R * CELL
            if len(nb) >= 3:
                walls[r, c] = observed[r, c] = True
                score[r, c] = np.inf
                continue
            axes = {dr == 0 for dr, dc in nb} or {True}
            res = [ev(cx, cy, near, ax_) for ax_ in axes]
            m = max(v[0] for v in res)
            a = max(v[1] for v in res)
            score[r, c] = m
            observed[r, c] = a >= OBSERVED_MIN
            walls[r, c] = observed[r, c] and m > post_tau
    return CellGrid(r0, c0, walls, observed, score)


def wall_segments(cells: CellGrid) -> np.ndarray:
    """The renderer's wall geometry for a wall-cell grid (the same rule as stage 2's reference):
    a segment joins the centres of 4-adjacent wall cells; at a free end (no neighbour or one)
    the wall continues 0.1 m past the centre; an isolated cell is a 0.2 m x segment."""
    occ = cells.walls
    rows, cols = occ.shape
    segs = []
    h = CELL / 2

    def wall(r, c):
        return 0 <= r < rows and 0 <= c < cols and bool(occ[r, c])

    for r in range(rows):
        for c in range(cols):
            if not occ[r, c]:
                continue
            x, y = (cells.c0 + c) * CELL, (cells.r0 + r) * CELL
            if wall(r, c + 1):
                segs.append((x, y, x + CELL, y))
            if wall(r + 1, c):
                segs.append((x, y, x, y + CELL))
            present = [(dr, dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)) if wall(r + dr, c + dc)]
            if not present:
                segs.append((x - h, y, x + h, y))
            elif len(present) == 1:
                dr, dc = present[0]
                segs.append((x, y, x - dc * h, y - dr * h))
    return np.array(segs).reshape(-1, 4)


def collision_segments(cells: CellGrid) -> np.ndarray:
    """What the simulator collides with (measured, stage 7): segments between the centres of
    4-adjacent wall cells, and a point for a wall cell without neighbours; no free-end
    extension."""
    occ = cells.walls
    rows, cols = occ.shape
    segs = []
    for r in range(rows):
        for c in range(cols):
            if not occ[r, c]:
                continue
            x, y = (cells.c0 + c) * CELL, (cells.r0 + r) * CELL
            right = c + 1 < cols and occ[r, c + 1]
            up = r + 1 < rows and occ[r + 1, c]
            if right:
                segs.append((x, y, x + CELL, y))
            if up:
                segs.append((x, y, x, y + CELL))
            left = c > 0 and occ[r, c - 1]
            down = r > 0 and occ[r - 1, c]
            if not (right or up or left or down):
                segs.append((x, y, x, y))
    return np.array(segs).reshape(-1, 4)


def point_segment_distance(p: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance from each point (n, 2) to the nearest segment (m, 4)."""
    p = np.atleast_2d(p)
    if len(segs) == 0:
        return np.full(len(p), np.inf)
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    out = np.full(len(p), np.inf)
    for j in range(0, len(p), 4096):
        q = p[j:j + 4096, None, :]
        t = np.clip(((q - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        d = np.hypot(*(q - (a + t[..., None] * ab)).transpose(2, 0, 1))
        out[j:j + 4096] = d.min(1)
    return out


def cspace(lo, spec: GridSpec, segs: np.ndarray, radius: float = ROBOT_RADIUS, seeds=None) -> np.ndarray:
    """Cells the robot's centre may occupy: known free, farther than ``radius`` from the
    collision segments ``segs``, and (with ``seeds``, positions the robot is known to have
    occupied) in a 4-connected component that contains one of them."""
    ny, nx = spec.shape
    xs = spec.origin[0] + (np.arange(nx) + 0.5) * spec.res
    ys = spec.origin[1] + (np.arange(ny) + 0.5) * spec.res
    X, Y = np.meshgrid(xs, ys)
    d = point_segment_distance(np.c_[X.ravel(), Y.ravel()], segs).reshape(ny, nx)
    free = (lo < 0) & (d > radius)
    if seeds is None:
        return free
    _, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    iy, ix = spec.index(seeds[:, 0], seeds[:, 1])
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    keep = np.unique(lab[iy[ok], ix[ok]])
    keep = keep[keep > 0]
    return free & np.isin(lab, keep)


# ------------------------------------------------------------------------------ textures


@dataclass
class Signatures:
    """Per wall face ``(r, c, side)`` (wall-cell indices in the lattice frame, side N/S/E/W =
    the face looks towards +y/-y/+x/-x): a Hellinger-normalised hue-saturation histogram, the
    pixel count, and the frames that saw it (``drive * 100000 + k``)."""

    keys: list = field(default_factory=list)
    hist: np.ndarray = field(default_factory=lambda: np.zeros((0, H_BINS * S_BINS)))
    pixels: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    frames: list = field(default_factory=list)


class SignatureBuilder:
    def __init__(self, cells: CellGrid, cam: sc.Camera | None = None, r_max: float = R_MAX):
        self.cells, self.cam, self.r_max = cells, cam or sc.camera_config(), r_max
        self.index: dict = {}
        self.counts: list[np.ndarray] = []
        self.frames: list[set] = []

    def _face(self, key):
        i = self.index.get(key)
        if i is None:
            i = self.index[key] = len(self.counts)
            self.counts.append(np.zeros(H_BINS * S_BINS))
            self.frames.append(set())
        return i

    def add(self, img: np.ndarray, rows, top, status, pose, frame_id: int, col_step: int = 2, row_step: int = 2):
        cam = self.cam
        c = columns(rows, top, status, cam, self.r_max)
        x, y, yaw = pose
        th = yaw - c.bearing
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hb = (hsv[..., 0].astype(int) * H_BINS) // 180
        sb = (hsv[..., 1].astype(int) * S_BINS) // 256
        bins = hb * S_BINS + sb
        H = img.shape[0]
        cells = self.cells
        for u in range(0, len(rows), col_step):
            if not c.hit[u]:
                continue
            wx = x + np.cos(th[u]) * c.rng[u]
            wy = y + np.sin(th[u]) * c.rng[u]
            # the nearer wall line (x = 0.4 m k or y = 0.4 m k)
            lx = WALL_PERIOD * np.round(wx / WALL_PERIOD)
            ly = WALL_PERIOD * np.round(wy / WALL_PERIOD)
            dx, dy = abs(wx - lx), abs(wy - ly)
            if min(dx, dy) > 0.06:
                continue
            if dx < dy:  # on a vertical wall x = lx
                C, R = int(round(lx / CELL)), int(round(wy / CELL))
                side = "E" if x > lx else "W"
            else:
                R, C = int(round(ly / CELL)), int(round(wx / CELL))
                side = "N" if y > ly else "S"
            r, cc = R - cells.r0, C - cells.c0
            if not (0 <= r < cells.walls.shape[0] and 0 <= cc < cells.walls.shape[1]) or not cells.walls[r, cc]:
                continue
            t_row = top[u] if np.isfinite(top[u]) else -1.0
            b_row = rows[u] if (status[u] == sg.WALL_RETURN and np.isfinite(rows[u])) else H
            lo_r, hi_r = int(max(t_row, -1.0)) + 2, int(min(b_row, H)) - 2
            if hi_r <= lo_r:
                continue
            i = self._face((R, C, side))
            self.counts[i] += np.bincount(bins[lo_r:hi_r:row_step, u], minlength=H_BINS * S_BINS)
            self.frames[i].add(frame_id)

    def result(self) -> Signatures:
        keys = sorted(self.index, key=self.index.get)
        cnt = np.array(self.counts).reshape(-1, H_BINS * S_BINS)
        tot = cnt.sum(1, keepdims=True)
        hist = np.sqrt(cnt / np.maximum(tot, 1))
        return Signatures(keys, hist, tot.ravel().astype(int), [sorted(self.frames[self.index[k]]) for k in keys])


def hellinger(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hellinger distance between Hellinger-normalised histograms (rows)."""
    return np.sqrt(np.maximum(0.0, 1.0 - (a * b).sum(-1)))


# ------------------------------------------------------------------------------ the map


@dataclass
class Map:
    logodds: np.ndarray
    hits: np.ndarray
    spec: GridSpec
    lattice: Lattice
    cells: CellGrid
    segments: np.ndarray  # rendered geometry (drawing, matching)
    collision: np.ndarray  # collision geometry (configuration space)
    cspace: np.ndarray
    poses: list  # lattice-aligned poses per drive
    signatures: Signatures | None = None


def build(scans, poses, anchor, r_max: float = R_MAX, spec: GridSpec | None = None,
          l_occ: float = L_OCC, l_free: float = L_FREE, edge_tau: float = EDGE_TAU,
          post_tau: float = POST_TAU, correct_scale: bool = True) -> Map:
    """Scans and optimised poses (any frame whose walls run along its axes) to the map."""
    c, ax, w, pc = wall_points(scans, poses, r_max)
    lat = fit_lattice(c, ax, w, np.asarray(anchor, float)[:2], pc)
    if not correct_scale:
        lat = Lattice(1.0, lat.phase, np.zeros(2), lat.anchor, lat.strength)
    P = [lat.apply(p) for p in poses]
    spec = spec or spec_for(P)
    lo, hits = occupancy(scans, P, spec, r_max, l_occ, l_free)
    cells = snap(lo, spec, edge_tau, post_tau)
    segs = wall_segments(cells)
    col = collision_segments(cells)
    seeds = np.concatenate([p[:, :2] for p in P])
    return Map(lo, hits, spec, lat, cells, segs, col, cspace(lo, spec, col, seeds=seeds), P)


def add_signatures(m: Map, frames) -> Map:
    """``frames`` yields ``(drive, k, img, rows, top, status)`` for the frames to use."""
    b = SignatureBuilder(m.cells)
    for d, k, img, rows, top, status in frames:
        b.add(img, rows, top, status, m.poses[d][k], d * 100000 + k)
    m.signatures = b.result()
    return m


def save(m: Map, path) -> None:
    sig = m.signatures or Signatures()
    np.savez_compressed(
        path,
        logodds=m.logodds.astype(np.float32),
        hits=m.hits.astype(np.float32),
        resolution=m.spec.res,
        origin=m.spec.origin,
        lattice_period=m.lattice.period,
        lattice_scale=m.lattice.scale,
        lattice_phase=m.lattice.phase,
        lattice_offset=m.lattice.offset,
        lattice_anchor=m.lattice.anchor,
        lattice_strength=m.lattice.strength,
        cell_size=CELL,
        cells_origin=np.array([m.cells.r0, m.cells.c0]),
        wall_cells=m.cells.walls,
        observed_cells=m.cells.observed,
        wall_segments=m.segments,
        collision_segments=m.collision,
        cspace=m.cspace,
        robot_radius=ROBOT_RADIUS,
        poses=np.array([np.r_[d, k, p] for d, P in enumerate(m.poses) for k, p in enumerate(P)]),
        sig_keys=np.array([(r, c, SIDES.index(s)) for r, c, s in sig.keys], int).reshape(-1, 3),
        sig_hist=sig.hist.astype(np.float32),
        sig_pixels=sig.pixels,
        sig_frames=np.array([np.array(f, np.int64) for f in sig.frames] + [None], dtype=object)[:-1],
    )


DRIVE_COLOURS = ("tab:red", "tab:green", "tab:blue")


def draw_trajectories(ax, poses, legend: bool = True) -> None:
    from matplotlib.lines import Line2D

    for P, col in zip(poses, DRIVE_COLOURS, strict=False):
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=0.6)
    ax.plot(poses[0][0, 0], poses[0][0, 1], "o", color="k", ms=5)
    if legend:
        h = [Line2D([], [], color=c, lw=1.5) for c in DRIVE_COLOURS[: len(poses)]] + [Line2D([], [], color="k", marker="o", lw=0)]
        ax.legend(h, [f"drive {d}" for d in range(len(poses))] + ["spawn"], loc="upper right", fontsize=7, framealpha=0.9)


def draw(ax, m: Map, snapped: bool = False, truth_segments=None, legend: bool = True) -> None:
    if snapped:
        free = np.where(m.cspace, 1.0, np.nan)
        ax.imshow(free, cmap="Greens", vmin=0, vmax=3, origin="lower", extent=m.spec.extent(), alpha=0.6)
        for s in m.segments:
            ax.plot([s[0], s[2]], [s[1], s[3]], "k-", lw=2)
    else:
        ax.imshow(np.clip(m.logodds, -5, 5), cmap="gray_r", vmin=-5, vmax=5, origin="lower", extent=m.spec.extent())
    if truth_segments is not None:
        for s in truth_segments:
            ax.plot([s[0], s[2]], [s[1], s[3]], "-", color="tab:orange", lw=0.8, alpha=0.8)
    draw_trajectories(ax, m.poses, legend)
    ax.set_aspect("equal")


def render(m: Map, path, title: str = "", snapped: bool = False, truth_segments=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    draw(ax, m, snapped, truth_segments)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
