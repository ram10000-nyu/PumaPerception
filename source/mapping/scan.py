"""Virtual laser scan from one frame (stage 2): range and bearing per image column.

Floor points project to ``v = cy + fy * h / Z`` and ``u = cx + fx * X / Z`` (``Z`` depth along
the optical axis, ``X`` to the right), with the camera level, ``h`` above the floor and on the
rotation axis (stage 0: offset zero within 1 mm). Pixel coordinates are continuous: pixel
``(i, j)`` spans ``[i, i + 1) x [j, j + 1)`` and its centre is ``(i + 0.5, j + 0.5)``, which is
what makes ``cx = 160`` and the fitted principal row 120.000 consistent.

Constants come from ``cache/mapping/camera.json`` (measured once on the simulator, provenance
recorded there); nothing here reads ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import segment as sg

CAMERA_JSON = Path(__file__).resolve().parents[2] / "cache" / "mapping" / "camera.json"
R_MAX = 1.5  # metres; stage 2: the largest range whose per-bin p95 error on dev is under 0.10 m


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    h: float
    far: float
    width: int
    height: int
    wall_height: float = 0.2999  # metres above the floor (stage 0: 0.2999 +/- 0.0002)


@lru_cache(maxsize=1)
def camera_config(path: Path = CAMERA_JSON) -> Camera:
    c = json.loads(path.read_text())
    cam = c["camera"]
    return Camera(
        cam["fx"],
        cam["fy"],
        cam["cx"],
        cam["cy"],
        c["height_fit"]["h"],
        c.get("far_plane_m", 5.0),
        cam["width"],
        cam["height"],
        c["height_fit"]["wall_height_H"],
    )


def backproject(u, v, cam: Camera | None = None):
    """Ground point ``(X, Z)`` of a continuous pixel ``(u, v)`` below the horizon."""
    cam = cam or camera_config()
    z = cam.fy * cam.h / (np.asarray(v, float) - cam.cy)
    x = z * (np.asarray(u, float) - cam.cx) / cam.fx
    return x, z


def project(x, z, cam: Camera | None = None):
    """Continuous pixel ``(u, v)`` of a ground point ``(X, Z)`` (inverse of ``backproject``)."""
    cam = cam or camera_config()
    z = np.asarray(z, float)
    return cam.cx + cam.fx * np.asarray(x, float) / z, cam.cy + cam.fy * cam.h / z


@dataclass
class Scan:
    """One scan, in the robot frame (x forward, y left). Arrays have one entry per column."""

    u: np.ndarray  # column centre, continuous
    status: np.ndarray  # segment.WALL_RETURN / FAR / INVALID
    z: np.ndarray  # depth along the optical axis, NaN without a return
    x: np.ndarray  # lateral position (right positive), NaN without a return
    rng: np.ndarray  # range from the robot, NaN without a return
    bearing: np.ndarray  # atan2(X, Z), radians, right positive
    r_max: float
    z_top: np.ndarray | None = None  # depth from the wall-top edge, NaN where not visible

    @property
    def hit(self) -> np.ndarray:
        """Wall returns within ``r_max`` (floor edge, or the wall top for walls too near for
        the floor edge to be in the image): the occupied endpoints."""
        return ((self.status == sg.WALL_RETURN) | (self.status == sg.TOP_RETURN)) & (
            self.rng <= self.r_max
        )

    @property
    def free_range(self) -> np.ndarray:
        """How far each column's ray is known to be free: up to the hit, else ``r_max`` for
        far-plane and beyond-``r_max`` columns, 0 for invalid columns."""
        out = np.zeros_like(self.rng)
        out[self.hit] = self.rng[self.hit]
        beyond = ((self.status == sg.WALL_RETURN) & (self.rng > self.r_max)) | (
            self.status == sg.FAR
        )
        out[beyond] = self.r_max
        return out

    def points_robot(self) -> np.ndarray:
        """Endpoints of the hits, shape (n, 2), in the robot frame: (forward, left)."""
        m = self.hit
        return np.stack([self.z[m], -self.x[m]], axis=1)


def top_depth(v_top, cam: Camera):
    """Depth of a wall from the row of its top edge: ``Z = fy (H - h) / (cy - v_top)``."""
    return cam.fy * (cam.wall_height - cam.h) / (cam.cy - np.asarray(v_top, float))


def scan_from_segmentation(
    seg: sg.Segmentation, cam: Camera | None = None, r_max: float = R_MAX, use_top: bool = True
) -> Scan:
    """Scan of one frame. Columns whose floor edge is below the image (a wall nearer than about
    0.16 m, status INVALID) are ranged from the wall's top edge when ``use_top`` and the
    top is visible (status TOP_RETURN). ``z_top`` holds the top-edge depth wherever it exists,
    for cross-checks."""
    cam = cam or camera_config()
    u = np.arange(len(seg.rows)) + 0.5
    valid = seg.status == sg.WALL_RETURN
    x = np.full(len(u), np.nan)
    z = np.full(len(u), np.nan)
    if valid.any():
        # a boundary above the horizon is not a floor point
        below = valid & (seg.rows > cam.cy + 1e-6)
        x[below], z[below] = backproject(u[below], seg.rows[below], cam)
    status = seg.status.copy()
    status[valid & ~np.isfinite(z)] = sg.INVALID
    z_top = np.full(len(u), np.nan)
    if seg.top_rows is not None:
        ok = np.isfinite(seg.top_rows) & (seg.top_rows < cam.cy - 0.5)
        z_top[ok] = top_depth(seg.top_rows[ok], cam)
        if use_top:
            near = ok & (status == sg.INVALID)
            z[near] = z_top[near]
            x[near] = z_top[near] * (u[near] - cam.cx) / cam.fx
            status[near] = sg.TOP_RETURN
    rng = np.hypot(x, z)
    return Scan(u, status, z, x, rng, np.arctan2(x, z), r_max, z_top)


def scan_from_image(img: np.ndarray, r_max: float = R_MAX, cam: Camera | None = None) -> Scan:
    return scan_from_segmentation(sg.segment(img), cam, r_max)
