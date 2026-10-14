"""Stage 2: back-projection and segmentation on synthetic images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import scan as sc  # noqa: E402
import segment as sg  # noqa: E402

CAM = sc.Camera(92.0, 92.0, 160.0, 120.0, 0.2099, 5.0, 320, 240)
BLUE, GRAY = (218, 181, 157), (232, 232, 232)


def test_backproject_recovers_projected_points_to_1e9():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 200)
    z = rng.uniform(0.2, 5.0, 200)
    u, v = sc.project(x, z, CAM)
    x2, z2 = sc.backproject(u, v, CAM)
    assert np.allclose(x2, x, atol=1e-9) and np.allclose(z2, z, atol=1e-9)


def test_far_plane_row_matches_stage0():
    _, v = sc.project(0.0, 5.0, CAM)
    assert v == pytest.approx(123.86, abs=0.01)


def synthetic(row: float, wall=(40, 60, 90), floor=BLUE, width=320):
    """Floor below a sharp boundary at ``row`` (continuous), wall above it, sky on top."""
    img = np.zeros((240, width, 3), np.uint8)
    for j in range(240):
        share = np.clip(j + 1 - row, 0.0, 1.0)
        img[j] = np.round(share * np.array(floor) + (1 - share) * np.array(wall))
    img[:40] = 255
    return img


@pytest.mark.parametrize("row", [150.0, 150.3, 150.5, 171.8, 200.1])
def test_subpixel_boundary(row):
    s = sg.segment(synthetic(row))
    assert np.all(s.status == sg.WALL_RETURN)
    assert s.rows == pytest.approx(np.full(320, row), abs=0.05)


def test_scan_depth_of_a_flat_wall():
    z_true = 1.0
    _, v = sc.project(0.0, z_true, CAM)
    scan = sc.scan_from_image(synthetic(v), r_max=2.5, cam=CAM)
    assert np.nanmedian(scan.z) == pytest.approx(z_true, abs=0.01)
    assert scan.hit.all()
    # a flat wall: lateral position grows with the column, depth does not
    assert np.nanstd(scan.z) < 0.01
    assert scan.x[0] < 0 < scan.x[-1]


def test_far_plane_is_no_return_not_a_wall():
    img = np.full((240, 320, 3), 255, np.uint8)
    img[124:] = GRAY  # floor reaches the far-plane edge, white above
    img[123] = 249
    s = sg.segment(img)
    assert np.all(s.status == sg.FAR)
    scan = sc.scan_from_image(img, r_max=2.5, cam=CAM)
    assert not scan.hit.any()
    assert np.all(scan.free_range == 2.5)


def test_wall_beyond_r_max_is_free_up_to_r_max_not_a_hit():
    _, v = sc.project(0.0, 3.5, CAM)
    scan = sc.scan_from_image(synthetic(v), r_max=2.5, cam=CAM)
    assert not scan.hit.any()
    assert np.all(scan.free_range == 2.5)


def test_wall_at_the_bottom_edge_is_invalid():
    img = np.full((240, 320, 3), (40, 60, 90), np.uint8)
    img[:40] = 255
    assert np.all(sg.segment(img).status == sg.INVALID)


def test_floor_coloured_wall_texture_above_the_floor_is_wall():
    img = synthetic(180.0)
    img[100:120, 50:80] = GRAY  # a grey patch on the wall, not connected to the floor
    s = sg.segment(img)
    assert not s.floor[100:120, 50:80].any()
    assert s.rows[60] == pytest.approx(180.0, abs=0.05)


def test_texture_seam_inside_the_floor_is_not_a_wall():
    img = synthetic(150.0)
    img[200:202, :] = (225, 215, 205)  # a blend between the two floor textures
    s = sg.segment(img)
    assert s.rows == pytest.approx(np.full(320, 150.0), abs=0.05)


def test_grey_and_blue_floor_are_both_floor():
    for f in (BLUE, GRAY, (222, 195, 178)):
        s = sg.segment(synthetic(160.0, floor=f))
        assert s.floor[200:, :].all()


def test_scan_points_are_in_the_robot_frame():
    _, v = sc.project(0.0, 1.0, CAM)
    scan = sc.scan_from_image(synthetic(v), r_max=2.5, cam=CAM)
    p = scan.points_robot()
    assert np.allclose(p[:, 0], 1.0, atol=0.01)  # forward
    assert p[0, 1] > 0 > p[-1, 1]  # left column is on the left
