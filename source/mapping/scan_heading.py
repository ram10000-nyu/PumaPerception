"""Second heading cue (stage 3, revision 5): the dominant wall direction in the virtual scan.

Every wall is along a maze axis, so the local direction of the scan polyline is, modulo 90
degrees, the camera's heading with the sign flipped: a wall running along world ``x`` seen from
a robot at heading ``yaw`` has direction ``-yaw`` in the robot frame, so ``yaw = -theta mod 90``.
Straight pieces are found with sliding total-least-squares line fits over runs of adjacent
returns; a piece is kept if its residual is consistent with the depth noise (this rejects
wall corners and depth steps), and votes with an angular standard deviation from its length
and the noise. The votes go through the same circular peak as the vanishing-point cue.

Pipeline code: scan and camera constants only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scan as sc
import segment as sg
import vp

WINDOWS = (41, 25, 15, 9)  # adjacent columns per line fit, longest first (multi-scale)
STRIDE = 3
MIN_EXTENT = 0.05  # m: a piece shorter than this carries no direction
RANGE_CAP = 2.5  # m: farther returns are too coarse (depth error per pixel grows as z^2)
SIGMA_PIXEL = 0.3  # px: noise of one boundary row (stage 2: sub-pixel on JPEG)
SIGMA_FLOOR = 0.5  # degrees: model error added to every vote
RESID_K = 2.5  # accepted residual in units of the point noise


@dataclass
class ScanHeading:
    psi: float = np.nan  # degrees, yaw mod 90 in (-45, 45]
    sigma: float = np.nan
    support: float = 0.0  # total inlier length of the pieces (metres)
    n_pieces: int = 0
    n_inliers: int = 0
    family_support: tuple = (0.0, 0.0)  # inlier length along each of the two wall directions


def point_sigma(z: np.ndarray, cam: sc.Camera) -> np.ndarray:
    """Standard deviation (m) of a floor-edge point at depth ``z``: one pixel is ``z^2 /
    (fy h)``."""
    return np.maximum(0.002, SIGMA_PIXEL * z**2 / (cam.fy * cam.h))


def pieces(scan: sc.Scan, cam: sc.Camera | None = None):
    """Straight pieces of the scan: ``(theta_deg, sigma_deg, length_m)`` arrays, with ``theta``
    the direction of each piece in the robot frame (x forward, y left), in (-90, 90].

    For every start column the longest window of adjacent returns that fits a line within the
    depth noise is kept (long windows average out the one-pixel quantisation of the boundary
    row, which on a gently sloped wall makes short windows read a staircase as a flat wall)."""
    cam = cam or sc.camera_config()
    ok = (scan.status == sg.WALL_RETURN) & np.isfinite(scan.z) & (scan.rng <= RANGE_CAP)
    P = np.stack([scan.z, -scan.x], 1)
    th, sg_, ln = [], [], []
    n = len(ok)
    # run lengths of consecutive valid columns, to skip impossible windows quickly
    run = np.zeros(n + 1, int)
    for i in range(n - 1, -1, -1):
        run[i] = run[i + 1] + 1 if ok[i] else 0
    i = 0
    while i < n:
        if run[i] < WINDOWS[-1]:
            i += 1
            continue
        for m in WINDOWS:
            if run[i] < m:
                continue
            q = P[i : i + m]
            c = q.mean(0)
            d = q - c
            w, vecs = np.linalg.eigh(d.T @ d / m)
            dirv = vecs[:, 1]
            t = d @ dirv
            extent = float(t.max() - t.min())
            sig_pt = float(point_sigma(np.array([c[0]]), cam)[0])
            rms = float(np.sqrt(max(w[0], 0.0)))
            if extent >= MIN_EXTENT and rms <= RESID_K * sig_pt + 0.002:
                ang = float(vp.diff180(np.degrees(np.arctan2(dirv[1], dirv[0])), 0.0))
                s_ang = np.degrees(np.sqrt(12.0) * max(sig_pt, rms) / (extent * np.sqrt(m)))
                th.append(ang)
                sg_.append(np.hypot(s_ang, SIGMA_FLOOR))
                ln.append(extent)
                break
        i += STRIDE
    return np.array(th), np.array(sg_), np.array(ln)


def estimate(scan: sc.Scan, cam: sc.Camera | None = None) -> ScanHeading:
    theta, sigma, length = pieces(scan, cam)
    out = ScanHeading(n_pieces=len(theta))
    if len(theta) == 0:
        return out
    psi_votes = vp.fold90(-theta)
    psi, sig, inl = vp.circular_peak(psi_votes, sigma)
    if not np.isfinite(psi):
        return out
    out.psi, out.sigma = psi, sig
    out.n_inliers = int(inl.sum())
    out.support = float(length[inl].sum())
    # along the two wall directions: theta = -psi (x walls) or -psi + 90 (y walls)
    along_x = np.abs(vp.diff180(theta, -psi)) < 45.0
    out.family_support = (float(length[inl & along_x].sum()), float(length[inl & ~along_x].sum()))
    return out
