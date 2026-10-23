"""Stage 10: locate the robot on the map on demand, from an in-place 360 degree spin (pipeline
code, rule 10: camera frames, the stage 3 compass and the map only).

A single 120 degree view is often ambiguous (corridors look alike), but a spin yields the same
kind of panoramic scan the four goal views give, and ``goal.place`` places that. The robot
turns about its camera (stage 0), so the position does not move during the spin. The cost is
steps, not attempts: 360 / 3.44 = about 105 ticks.

* ``Sweep``: ``start()``, then ``add(image)`` for the frame before the spin and after every LEFT
  tick. Each frame is segmented and given a compass reading as it arrives, so little is left to
  do when the spin ends. The sweep is done once the compass says the robot has turned 360 degrees.
* **Relative angles** come from the compass, not from the tick count. The compass (heading
  modulo 90 degrees) is absolute, so its unwrapped series telescopes: frame ``i`` is turned by
  ``psi_i - psi_0 + 90 n`` degrees with no accumulated drift, ``n`` counted from the small
  frame-to-frame steps (3.4 degrees, far below the 45 degree ambiguity). The tick count
  (3.44 degrees per LEFT tick, with motion noise and a possible per-route turn bias) is only a
  consistency check.
* ``locate(map, sweep)``: each kept frame's virtual scan (robot camera height 0.21 m, stage 2) is
  rotated by its angle into the first frame's frame and the slices are merged; the merged scan,
  with the frames as angled views for the texture term, goes through ``goal.place``. The robot's
  pose at the end of the spin is the first frame's pose plus the total turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import goal
import heading as hd
import numpy as np
import odometry as od
import scan as sc
import segment as sg
import vo
import vp

TICK_DEG = float(np.degrees(0.06))  # one LEFT tick (simulator constant, stage 0)
TARGET_DEG = 360.0  # the sweep is done once the compass has turned this far
MAX_TICKS = 220  # ... or when this many ticks have gone by (the robot is not turning)
KEEP_EVERY = 3  # ticks between the frames whose scans are merged (about 35 frames)
SLICE_DEG = 12.0  # bearing slice taken from each kept frame's scan (kept frames: 10.3 deg apart)
TEX_EVERY = 6  # every TEX_EVERY-th kept frame (about 60 degrees apart) enters the texture term
OUTLIER_BASE_DEG = 10.0  # a compass step may differ from the turn rate by this ...
OUTLIER_FRAC = 0.3  # ... plus this fraction of the expected step, else the reading is refused
REANCHOR = 5  # consecutive refused readings after which the next one is taken
MISMATCH_WARN_DEG = 10.0  # compass total vs ticks x 3.44 degrees: report beyond this
MISMATCH_FAIL_DEG = 40.0  # ... and do not trust the sweep beyond this (robot stuck or pushed)
# below this margin the ping is flagged not reliable. Dev evidence, 120 recorded sweeps: failures
# <= 0.033, correct >= 0.194; the threshold is the geometric middle of that gap
PING_CONF_MIN = 0.10
MIN_VALID = 20  # frames with a compass reading needed


@dataclass
class SweepFrame:
    image: np.ndarray
    rows: np.ndarray
    top: np.ndarray
    status: np.ndarray
    psi: float  # compass, degrees modulo 90; NaN where invalid


class Sweep:
    """The frames of one spin and their compass readings."""

    def __init__(self, max_ticks: int = MAX_TICKS) -> None:
        self.max_ticks = max_ticks
        self.active = False
        self.frames: list[SweepFrame] = []
        self._lsd = None

    def start(self) -> None:
        self.frames = []
        self.active = True
        if self._lsd is None:
            self._lsd = vp.make_lsd()

    def add(self, image: np.ndarray) -> None:
        """The frame after ``len(self.frames)`` LEFT ticks (the first one: before the spin)."""
        seg = sg.segment(image)
        status = sc.scan_from_segmentation(seg).status
        h = hd.heading_from_segmentation(seg, self._lsd)
        psi = h.psi if h.valid else np.nan
        self.frames.append(SweepFrame(image, seg.rows, seg.top_rows, status, psi))
        if self.done:
            self.active = False

    # ---------------------------------------------------------------- angles

    @staticmethod
    def _unwrap(psi: np.ndarray, ok: np.ndarray):
        """The accepted frames and their unwrapped compass angles. A reading is accepted if its
        step from the last accepted one (modulo 90 degrees, the multiple of 90 chosen by the
        turn rate seen so far) fits that rate: the compass is sometimes valid and wrong by
        about 45 degrees (stage 3), and one such reading would otherwise corrupt every angle
        after it. After ``REANCHOR`` readings in a row are refused the next one is taken (the
        rate estimate was wrong, not the readings)."""
        idx_all = np.where(ok)[0]
        idx, un = [int(idx_all[0])], [float(psi[idx_all[0]])]
        rate, refused = TICK_DEG, 0
        for j in idx_all[1:]:
            i = idx[-1]
            gap = (j - i) * rate
            d = od.vp_diff90(psi[j], psi[i])  # the step modulo 90, in (-45, 45]
            d += 90.0 * round((gap - d) / 90.0)  # the multiple of 90 that fits the turn rate
            if abs(d - gap) > OUTLIER_BASE_DEG + OUTLIER_FRAC * abs(gap) and refused < REANCHOR:
                refused += 1
                continue
            refused = 0
            idx.append(int(j))
            un.append(un[-1] + d)
            if len(idx) > 6:  # follow the turn rate (a robot rubbing a wall turns slower)
                r = (un[-1] - un[-7]) / (idx[-1] - idx[-7])
                rate = float(np.clip(r, 0.3 * TICK_DEG, 1.5 * TICK_DEG))
        return np.array(idx), np.array(un)

    @property
    def ticks(self) -> int:
        return max(0, len(self.frames) - 1)

    def relative_angles(self) -> np.ndarray:
        """Per frame, degrees turned since the first frame (CCW positive), from the compass.
        Frames without a reading are interpolated; the ends are extrapolated at the median
        per-frame step."""
        psi = np.array([f.psi for f in self.frames])
        ok = np.isfinite(psi)
        n = len(psi)
        if ok.sum() < 2:
            return np.arange(n) * TICK_DEG  # no compass: the nominal ticks (see ``problems``)
        idx, un = self._unwrap(psi, ok)
        if len(idx) < 2:
            return np.arange(n) * TICK_DEG
        steps = np.diff(un) / np.maximum(np.diff(idx), 1)
        rate = float(np.median(steps)) if len(steps) else TICK_DEG
        rel = np.interp(np.arange(n), idx, un - un[0])
        rel[: idx[0]] = -rate * (idx[0] - np.arange(idx[0]))
        rel[idx[-1] + 1 :] = (un[-1] - un[0]) + rate * (np.arange(idx[-1] + 1, n) - idx[-1])
        return rel - rel[0]  # relative to the first frame, whether or not it has a reading

    @property
    def turned_deg(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return float(self.relative_angles()[-1])

    @property
    def done(self) -> bool:
        return len(self.frames) > self.max_ticks or (
            len(self.frames) > 4 and self.turned_deg >= TARGET_DEG
        )

    def problems(self) -> list[str]:
        """What is wrong with this sweep (empty: nothing)."""
        out = []
        valid = sum(np.isfinite(f.psi) for f in self.frames)
        if valid < MIN_VALID:
            out.append(f"compass valid on only {valid} of {len(self.frames)} frames")
        turned = self.turned_deg
        if turned < TARGET_DEG - 5.0:
            out.append(f"turned only {turned:.0f} deg in {self.ticks} ticks")
        mismatch = turned - self.ticks * TICK_DEG
        if abs(mismatch) > MISMATCH_FAIL_DEG:
            nominal = self.ticks * TICK_DEG
            out.append(f"compass {turned:.0f} deg vs {nominal:.0f} deg from the ticks")
        return out

    def mismatch_deg(self) -> float:
        return self.turned_deg - self.ticks * TICK_DEG


@dataclass
class PingResult:
    pose: np.ndarray  # (x, y, theta) of the robot at the end of the spin, map frame
    first_pose: np.ndarray  # ... at the first frame
    confidence: float
    reliable: bool
    top: list  # [(pose at the end of the spin, score)], best first, distinct places
    seconds: float  # compute time of ``locate``
    turned_deg: float
    ticks: int
    mismatch_deg: float
    warnings: list = field(default_factory=list)
    n_points: int = 0
    n_frames: int = 0


def merged_scan(
    sweep: Sweep,
    rel_deg: np.ndarray | None = None,
    keep_every: int = KEEP_EVERY,
    slice_deg: float = SLICE_DEG,
    tex_every: int = TEX_EVERY,
    max_deg: float | None = None,
    frames: list[int] | None = None,
) -> goal.GoalScan:
    """The sweep as one scan in the first frame's frame, with its frames as angled views.
    ``max_deg`` / ``frames`` restrict it (the half-spin and single-frame ablations)."""
    cam = sc.camera_config()
    rel = sweep.relative_angles() if rel_deg is None else rel_deg
    pick = list(range(0, len(sweep.frames), keep_every)) if frames is None else list(frames)
    if max_deg is not None:
        pick = [i for i in pick if rel[i] <= max_deg]
    half = np.radians(slice_deg) / 2
    XY, SIG, V, N, views = [], [], [], [], []
    for n, i in enumerate(pick):
        f = sweep.frames[i]
        p = vo.scan_points(f.rows, f.top, f.status, cam)
        if len(p.xy) == 0:
            continue
        bearing = np.arctan2(p.xy[:, 1], p.xy[:, 0])
        keep = np.abs(bearing) <= half
        a = np.radians(rel[i])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        XY.append(p.xy[keep] @ R.T)
        N.append(p.normal[keep] @ R.T)
        SIG.append(p.sigma[keep])
        V.append(np.full(int(keep.sum()), i))
        if n % tex_every == 0:
            views.append((i, f.image, f.rows, f.top, f.status, a))
    if not XY:
        empty = np.zeros((0, 2))
        return goal.GoalScan(empty, np.zeros(0), np.zeros(0, int), empty, cam, np.nan, [])
    xy, sig, view, normal = (np.concatenate(a) for a in (XY, SIG, V, N))
    return goal.GoalScan(xy, sig, view, normal, cam, float("nan"), views)


def locate(
    mf: goal.MapField,
    sweep: Sweep,
    use_texture: bool = True,
    conf_min: float | None = None,
    **scan_kwargs,
) -> PingResult:
    """The robot's pose at the end of the spin on the map ``mf`` (``goal.load_map``)."""
    t0 = time.perf_counter()
    rel = sweep.relative_angles()
    gs = merged_scan(sweep, rel, **scan_kwargs)
    if len(gs.xy) < 50:
        raise ValueError(f"the sweep gives only {len(gs.xy)} scan points")
    est = goal.place(mf, gs, use_texture=use_texture)
    turn = np.radians(float(rel[-1]))

    def at_end(p):
        q = np.array(p, float)
        q[2] = float(np.angle(np.exp(1j * (q[2] + turn))))
        return q

    cmin = PING_CONF_MIN if conf_min is None else conf_min
    warnings = list(sweep.problems())
    mismatch = sweep.mismatch_deg()
    if abs(mismatch) > MISMATCH_WARN_DEG and not any("compass" in w for w in warnings):
        nominal = sweep.ticks * TICK_DEG
        warnings.append(f"compass {sweep.turned_deg:.0f} deg vs {nominal:.0f} deg from the ticks")
    return PingResult(
        pose=at_end(est.pose),
        first_pose=est.pose,
        confidence=est.confidence,
        reliable=est.confidence >= cmin and not warnings,
        top=[(at_end(t[0]), t[1]) for t in est.top],
        seconds=time.perf_counter() - t0,
        turned_deg=sweep.turned_deg,
        ticks=sweep.ticks,
        mismatch_deg=mismatch,
        warnings=warnings,
        n_points=len(gs.xy),
        n_frames=len(gs.views),
    )


def encode_jpeg(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """The recordings' frames went through JPEG; this mimics it for a raw frame."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img

