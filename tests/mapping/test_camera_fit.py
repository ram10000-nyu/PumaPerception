"""Stage 0 revision 2: height fit, thin-wall scene, boundary rows, vanishing-point helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import camera_fit as cf  # noqa: E402
import trackability as T  # noqa: E402

FY = 92.0


def test_fit_height_recovers_synthetic_truth():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.3, 3.0, 400)
    h, a = 0.2093, 0.004
    rows = 120.0 + FY * h / (d - a) + rng.normal(0, 0.1, d.size)
    fit = cf.fit_height(d, rows, FY, cy=120.0)
    assert fit["h"] == pytest.approx(h, abs=0.001)
    assert fit["a"] == pytest.approx(a, abs=0.003)
    assert abs(fit["h"] - h) < fit["h_ci95"] * 3
    assert fit["rms_px"] == pytest.approx(0.1, abs=0.03)


def test_fit_height_exact_data_is_exact():
    d = np.linspace(0.3, 3.0, 30)
    rows = 120.0 + FY * 0.21 / (d - 0.0)
    fit = cf.fit_height(d, rows, FY, cy=120.0)
    assert fit["h"] == pytest.approx(0.21, abs=1e-9)
    assert fit["a"] == pytest.approx(0.0, abs=1e-9)


def test_fit_height_can_fit_the_principal_row():
    d = np.linspace(0.3, 3.0, 40)
    rows = 119.7 + FY * 0.21 / d
    fit = cf.fit_height(d, rows, FY, cy=None, offset=False)
    assert fit["cy"] == pytest.approx(119.7, abs=1e-6)
    assert fit["h"] == pytest.approx(0.21, abs=1e-6)


def synthetic_floor_image(row: float, floor=(0, 232, 0), wall=(40, 60, 90)) -> np.ndarray:
    """Floor below a sharp boundary at ``row`` (continuous), wall above, sky on top."""
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = wall
    img[:60] = 255
    for j in range(240):
        share = np.clip(j + 1 - row, 0.0, 1.0)  # floor share of pixel j
        img[j] = np.round(share * np.array(floor) + (1 - share) * np.array(wall))
        if j < 60:
            img[j] = 255
    return img


def test_boundary_rows_subpixel():
    from stage0_camera import green_floor

    for row in (150.0, 150.25, 150.5, 150.75, 171.4):
        b = cf.boundary_rows(synthetic_floor_image(row), [100, 160], green_floor)
        assert b == pytest.approx([row, row], abs=0.03)


def test_boundary_rows_invalid_when_floor_reaches_sky():
    img = np.full((240, 320, 3), 255, np.uint8)
    img[130:] = (0, 232, 0)  # floor straight up to the white background: no wall
    from stage0_camera import green_floor

    assert np.isnan(cf.boundary_rows(img, [160], green_floor)).all()


def test_scene_walls_are_thin_planes_through_cell_centres():
    occ = np.zeros((5, 7), bool)
    occ[:, 0] = occ[:, 6] = True  # walls at x = 0 and x = 1.2
    scene = cf.Scene(occ)
    z = scene.cast((0.4, 0.4), np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert z == pytest.approx([0.8, 0.4], abs=3e-3)  # plane through the wall cell centre


def test_scene_segments_follow_neighbours():
    occ = np.zeros((3, 3), bool)
    occ[1, 1] = occ[1, 2] = True  # two horizontal neighbours: a joining segment, free ends extend
    segs = {
        tuple(sorted([tuple(np.round(s[:2], 3)), tuple(np.round(s[2:], 3))]))
        for s in cf.Scene(occ).segments
    }
    assert segs == {
        ((0.2, 0.2), (0.4, 0.2)),
        ((0.1, 0.2), (0.2, 0.2)),
        ((0.4, 0.2), (0.5, 0.2)),
    }
    solo = cf.Scene(np.pad(np.ones((1, 1), bool), 1)).segments
    assert len(solo) == 1  # an isolated wall cell: one x segment


def test_raycast_depth_is_optical_axis_depth():
    occ = np.zeros((3, 9), bool)
    occ[:, 8] = True  # wall plane at x = 1.6
    scene = cf.Scene(occ)
    cols = np.array([150, 160, 170])
    z = cf.raycast_depths(scene, (0.0, 0.2), 0.0, cols, 92.0, 160.0)
    assert z == pytest.approx([1.6, 1.6, 1.6], abs=3e-3)  # a flat wall has constant depth


def test_floor_boundary_geometry_roundtrip():
    h = 0.209
    z = 1.7
    assert cf.predicted_rows(np.array([z]), FY, 120.0, h)[0] == pytest.approx(120 + FY * h / z)


def test_horizon_angle_and_clusters():
    f, cx, cy = 92.0, 160.0, 120.0
    beta = np.radians(20.0)  # a ground line heading 20 deg off the optical axis
    xh = cx - f * np.tan(beta)
    # two image segments of the same ground line, far from the horizon crossing
    segs = [(xh + 60, cy + 40, xh + 30, cy + 20), (xh + 120, cy + 80, xh + 90, cy + 60)]
    ang = [T.segment_horizon_angle(s) for s in segs]
    assert ang == pytest.approx([-20.0, -20.0], abs=1e-6)
    peaks = T.vp_clusters(np.array(ang), np.array([60.0, 60.0]))
    assert peaks[0][0] == pytest.approx(-20.0, abs=1.0)
    assert T.segment_horizon_angle((0, 150, 100, 150)) == 90.0  # parallel to the horizon


def test_vp_clusters_merge_infinity_and_reject_short():
    peaks = T.vp_clusters(np.array([89.6, -89.7]), np.array([30.0, 30.0]))
    assert len(peaks) == 1 and peaks[0][1] == pytest.approx(60.0)
    assert T.vp_clusters(np.array([10.0]), np.array([20.0])) == []


def test_expected_vp_angles_and_wrap():
    a, b = T.expected_vp_angles(np.radians(10.0))
    assert (a, b) == pytest.approx((10.0, -80.0))
    a, b = T.expected_vp_angles(np.radians(100.0))  # 100 deg == 10 deg modulo 90
    assert a == pytest.approx(10.0)
    assert T.angle_diff_deg(89.0, -89.0) == pytest.approx(2.0)


def test_real_floor_mask_accepts_both_textures_with_fog():
    img = np.zeros((2, 4, 3), np.uint8)
    img[0] = [(218, 181, 157), (232, 232, 232), (221, 195, 178), (50, 60, 70)]
    m = T.real_floor_mask(img)
    assert m[0].tolist() == [True, True, True, False]


def test_blind_flags_and_runs():
    inl = np.array([20, 5, 5, 5, 30])
    val = np.array([0.5, 0.1, 0.1, 0.5, 0.5])
    vp = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    f = T.blind_flags(inl, val, vp)
    assert f.tolist() == [False, True, True, False, False]
    assert T.longest_run(f) == 2
