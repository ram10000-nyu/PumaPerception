"""Stage 3: angle helpers, vanishing-point voting, scan heading, fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
import heading as hd  # noqa: E402
import scan as sc  # noqa: E402
import scan_heading as sh  # noqa: E402
import segment as sg  # noqa: E402
import vp  # noqa: E402

CAM = sc.Camera(92.0, 92.0, 160.0, 120.0, 0.2099, 5.0, 320, 240)


# ---------------------------------------------------------------------------- angle helpers


def test_fold_and_diff_on_the_90_degree_circle():
    assert vp.fold90(45.0) == pytest.approx(45.0)
    assert vp.fold90(-45.0) == pytest.approx(45.0)
    assert vp.fold90(100.0) == pytest.approx(10.0)
    assert vp.fold90(-130.0) == pytest.approx(-40.0)
    assert vp.diff90(44.0, -44.0) == pytest.approx(-2.0)
    assert vp.diff180(89.0, -89.0) == pytest.approx(-2.0)
    assert vp.true_psi(np.radians(135.0)) == pytest.approx(45.0)


def test_circular_mean_across_the_cut():
    assert vp.circ_mean90(np.array([44.0, -44.0]), np.ones(2)) == pytest.approx(45.0, abs=1e-6)


# -------------------------------------------------------------- vanishing points, synthetic


def ground_line_segment(beta_deg, z0, cam=CAM, length=0.8):
    """Image segment of a ground line at heading ``beta`` (clockwise from the optical axis, to
    the right) through the ground point (X=0, Z=z0): project two of its points."""
    b = np.radians(beta_deg)
    d = np.array([np.sin(b), np.cos(b)])  # (X, Z) direction
    p = []
    for t in (0.0, length):
        X, Z = d[0] * t, z0 + d[1] * t
        p.append(sc.project(X, Z, cam))
    return np.array([p[0][0], p[0][1], p[1][0], p[1][1]])


def test_horizon_angle_of_a_ground_line_is_its_heading():
    for beta in (-60, -30, -5, 12, 40, 75):
        v = vp.votes(ground_line_segment(beta, 1.0)[None], CAM)
        assert v.alpha[0] == pytest.approx(beta, abs=0.05), beta


def test_every_family_votes_for_the_same_yaw_mod_90():
    yaw = 23.0  # camera heading (CCW): the axis families are at yaw and yaw - 90 clockwise
    segs = np.array(
        [ground_line_segment(yaw, 0.6), ground_line_segment(yaw, 1.4)]
        + [ground_line_segment(yaw - 90, 0.7), ground_line_segment(yaw - 90, 1.5)]
    )
    v = vp.votes(segs, CAM)
    assert np.allclose(v.psi, yaw, atol=0.1)
    res = vp.find_peak(v)
    assert res.psi == pytest.approx(yaw, abs=0.1)
    assert min(res.family_len) > 0


def test_peak_rejects_outlier_segments():
    yaw = -18.0
    good = [ground_line_segment(yaw, z) for z in (0.5, 0.8, 1.1, 1.4)]
    bad = [ground_line_segment(30.0, 0.9), ground_line_segment(-50.0, 1.2)]
    res = vp.find_peak(vp.votes(np.array(good + bad), CAM))
    assert res.psi == pytest.approx(yaw, abs=0.2)
    assert res.n_inliers == 4


def test_implied_focal_length_from_two_vanishing_points():
    yaw = 28.0
    segs = np.array(
        [ground_line_segment(yaw, z) for z in (0.5, 0.9, 1.3)]
        + [ground_line_segment(yaw - 90, z) for z in (0.6, 1.0, 1.4)]
    )
    v = vp.votes(segs, CAM)
    res = vp.find_peak(v)
    assert vp.implied_focal(v, res, CAM) == pytest.approx(92.0, rel=0.01)


def test_implied_focal_is_nan_near_axis_alignment():
    segs = np.array(
        [ground_line_segment(-3.0, z) for z in (0.5, 1.0)]
        + [ground_line_segment(87.0, z) for z in (0.7, 1.2)]
    )
    v = vp.votes(segs, CAM)
    assert np.isnan(vp.implied_focal(v, vp.find_peak(v), CAM))


def test_segment_noise_is_larger_for_short_segments():
    long = ground_line_segment(20, 0.6, length=1.2)
    short = ground_line_segment(20, 0.6, length=0.15)
    v = vp.votes(np.array([long, short]), CAM)
    assert v.sigma[1] > v.sigma[0]


def test_polyline_segments_split_at_a_corner():
    u = np.arange(100) + 0.5
    rows = np.where(u < 50, 150.0 + 0.4 * (u - 50), 150.0 - 0.9 * (u - 50))
    status = np.zeros(100, np.int8)
    segs = vp.polyline_segments(rows, status)
    assert len(segs) == 2


# ----------------------------------------------------------------------- scan heading


def scan_of_walls(yaw_deg, cam=CAM, r_max=2.5):
    """A scan of the axis-aligned corner x = 1.0 (a wall along y) and y = 0.6 (along x) seen
    from the origin at heading ``yaw``: columns hit the nearer wall."""
    yaw = np.radians(yaw_deg)
    u = np.arange(320) + 0.5
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    z = np.full(320, np.nan)
    for i, uu in enumerate(u):
        d = fwd + (uu - cam.cx) / cam.fx * right
        ts = []
        if d[0] > 1e-9:
            ts.append(1.0 / d[0])
        if d[1] > 1e-9:
            ts.append(0.6 / d[1])
        if ts:
            z[i] = min(ts)  # parameter along the unnormalised ray = optical-axis depth
    x = z * (u - cam.cx) / cam.fx
    status = np.where(np.isfinite(z) & (z < 2.4), sg.WALL_RETURN, sg.FAR).astype(np.int8)
    return sc.Scan(u, status, z, x, np.hypot(x, z), np.arctan2(x, z), r_max)


@pytest.mark.parametrize("yaw", [-40.0, -12.0, 0.0, 17.0, 33.0, 44.0])
def test_scan_heading_recovers_yaw_mod_90(yaw):
    r = sh.estimate(scan_of_walls(yaw), CAM)
    assert r.psi == pytest.approx(float(vp.fold90(yaw)), abs=0.5)
    assert r.support > 1.0


def test_scan_heading_with_no_returns_is_nan():
    s = scan_of_walls(10.0)
    s.status[:] = sg.FAR
    assert np.isnan(sh.estimate(s, CAM).psi)


# ----------------------------------------------------------------------------------- fusion


def res_vp(psi, sigma=0.4, **kw):
    r = vp.VPResult(psi=psi, sigma=sigma, support=200.0, n_inliers=8)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def res_sc(psi, sigma=0.3):
    return sh.ScanHeading(psi=psi, sigma=sigma, support=5.0, n_pieces=20, n_inliers=15)


def test_agreeing_cues_are_fused_and_valid():
    h = hd.fuse(res_vp(10.0), res_sc(10.8))
    assert h.valid and h.source == "both"
    assert 10.0 < h.psi < 10.8
    assert h.psi == pytest.approx(10.51, abs=0.02)  # inverse-variance: nearer the scan cue
    assert 0.5 < h.confidence < 1.0


def test_agreement_across_the_wrap_cut():
    h = hd.fuse(res_vp(44.5), res_sc(-44.8))
    assert h.valid and abs(abs(h.psi) - 45.0) < 0.6


def test_conflicting_cues_are_not_valid():
    h = hd.fuse(res_vp(10.0), res_sc(20.0))
    assert not h.valid and h.source == "conflict" and np.isnan(h.psi)


def test_single_cue_needs_the_vp_validity_test():
    weak = res_vp(5.0, alpha_a=5.0, family_len=(100.0, 0.0), left_len=10.0, right_len=90.0)
    assert not hd.fuse(weak, sh.ScanHeading()).valid
    strong = res_vp(
        5.0, alpha_a=5.0, family_len=(100.0, 0.0), left_len=60.0, right_len=60.0, support=120.0
    )
    h = hd.fuse(strong, sh.ScanHeading())
    assert h.valid and h.source == "vp"


def test_vp_validity_two_peaks_needs_the_focal_length():
    two = res_vp(25.0, alpha_a=25.0, family_len=(60.0, 50.0), f_implied=93.0)
    assert hd.vp_valid(two)
    assert not hd.vp_valid(res_vp(25.0, alpha_a=25.0, family_len=(60.0, 50.0), f_implied=80.0))
    assert not hd.vp_valid(res_vp(25.0, alpha_a=25.0, family_len=(60.0, 5.0), f_implied=92.0))


def test_pipeline_modules_do_not_touch_ground_truth():
    import re

    forbidden = re.compile(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene|occupancy")
    for name in ("vp.py", "scan_heading.py", "heading.py"):
        src = (SRC / name).read_text()
        body = src.split('"""', 2)[-1]
        code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
        assert not forbidden.search(code), name
