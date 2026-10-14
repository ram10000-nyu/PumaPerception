"""Stage 4: the relative-motion estimators on synthetic geometry, and on cached dev frames."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
import scan as sc  # noqa: E402
import se2  # noqa: E402
import vo  # noqa: E402

CAM = sc.camera_config()
RNG = np.random.default_rng(5)


def wall_scan(segments, n_cols=320):
    """ScanPoints of walls (a list of ((x0, y0), (x1, y1)) in the robot frame): ray-cast one
    point per image column at the camera's own column angles, exactly (no noise)."""
    pts, cols = [], []
    for u in range(n_cols):
        d = np.array([1.0, -(u + 0.5 - CAM.cx) / CAM.fx])  # forward, left
        best = None
        for a, b in segments:
            a, b = np.array(a, float), np.array(b, float)
            m = np.array([d, a - b]).T
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            t, s = np.linalg.solve(m, a)
            if t > 0.05 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best):
                best = t
        if best is not None and best < 2.4:
            pts.append(best * d)
            cols.append(u)
    xy = np.array(pts)
    sig = np.full(len(xy), 0.004)
    normal, offset = vo._normals(xy, np.ones(len(xy), bool))
    return vo.ScanPoints(xy, sig, np.array(cols), normal, offset)


def to_frame(segments, pose):
    """The same walls seen from a robot at ``pose`` (x, y, theta) of the first frame."""
    inv = se2.inverse(np.asarray(pose))
    out = []
    for a, b in segments:
        pa = se2.compose(inv, [*a, 0.0])[:2]
        pb = se2.compose(inv, [*b, 0.0])[:2]
        out.append((pa, pb))
    return out


L_WALLS = [((1.2, -2.0), (1.2, 2.0)), ((-1.0, 0.6), (3.0, 0.6)), ((-1.0, -0.6), (3.0, -0.6))]
CORNER = [((1.0, -1.5), (1.0, 1.5)), ((-1.0, 0.8), (1.0, 0.8)), ((-1.0, -0.9), (1.0, -0.9))]


def test_rigid_fit_recovers_the_motion():
    b = RNG.uniform(-1, 1, (30, 2))
    t, th = np.array([0.12, -0.03]), 0.31
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = b @ R.T + t
    t2, th2 = vo._rigid(a, b)
    assert th2 == pytest.approx(th, abs=1e-12) and np.allclose(t2, t, atol=1e-12)


@pytest.mark.parametrize(
    "pose", [(0.15, 0.0, 0.0), (0.06, -0.02, 0.25), (0.0, 0.0, -0.3), (0.1, 0.02, 0.1)]
)
def test_icp_recovers_motion_on_a_corner(pose):
    target = wall_scan(CORNER)
    source = wall_scan(to_frame(CORNER, pose))
    m = vo.icp(target, source, np.array(pose) + [0.01, -0.01, 0.03])
    assert m is not None
    err = m.mean - np.array(pose)
    assert np.hypot(err[0], err[1]) < 0.012 and abs(err[2]) < np.radians(0.8)


def test_icp_covariance_shows_the_corridor_degeneracy():
    corridor = [((-2.0, 0.2), (3.0, 0.2)), ((-2.0, -0.2), (3.0, -0.2))]  # no end wall in view
    target = wall_scan(corridor)
    source = wall_scan(to_frame(corridor, (0.1, 0.0, 0.0)))
    m = vo.icp(target, source, np.array([0.1, 0.0, 0.0]))
    assert m is not None
    sd = np.sqrt(np.diag(m.cov))
    assert sd[0] > 0.2  # along the corridor: nothing constrains it
    assert sd[1] < 0.02  # across: tightly


def test_icp_covariance_is_tight_with_an_end_wall():
    m = vo.icp(
        wall_scan(CORNER), wall_scan(to_frame(CORNER, (0.1, 0.0, 0.0))), np.array([0.1, 0, 0])
    )
    assert np.sqrt(m.cov[0, 0]) < 0.05


def test_global_search_finds_a_large_rotation():
    pose = (0.04, -0.02, np.radians(-41.0))
    target = wall_scan(CORNER)
    source = wall_scan(to_frame(CORNER, pose))
    field = vo.likelihood_field(target)
    thetas = np.radians(np.arange(-100, 100.1, 1.5))
    scores, trans = vo.correlative_scores(field, source.xy, thetas)
    near = np.abs(np.degrees(thetas) - (-41.0)) < 2.0
    # a corner of parallel walls is nearly symmetric under 90 degrees, so the true rotation may
    # not be the single best peak: it must be a strong one (the hypotheses are scored by ICP)
    assert scores[near].max() >= 0.85 * scores.max()


def test_straight_pieces_ignore_a_depth_step():
    two = [((1.0, -1.0), (1.0, 0.0)), ((1.6, 0.0), (1.6, 1.0))]
    pts = wall_scan(two)
    has = np.isfinite(pts.normal[:, 0])
    assert has.sum() > 100
    assert np.allclose(np.abs(pts.normal[has][:, 0]), 1.0, atol=0.02)  # normals along x


def project(X, Y, Z):
    """Pixel of a point (forward X, left Y, height Z above the camera) for the course camera."""
    return CAM.cx + CAM.fx * (-Y) / X, CAM.cy - CAM.fy * Z / X


def wall_matches(pose, wall_x=1.0, n=40):
    """Matches of texture points on the plane x = wall_x between frame 0 and a robot at ``pose``."""
    ys = RNG.uniform(-0.6, 0.6, n)
    zs = RNG.uniform(-0.18, 0.08, n)
    zs = np.where(np.abs(zs) < 0.02, 0.05, zs)
    out = []
    inv = se2.inverse(np.asarray(pose))
    for y, z in zip(ys, zs, strict=True):
        u0, v0 = project(wall_x, y, z)
        p = se2.compose(inv, [wall_x, y, 0.0])
        u1, v1 = project(p[0], p[1], z)
        out.append((u0, v0, u1, v1))
    return np.array(out)


@pytest.mark.parametrize("pose", [(0.15, 0.0, 0.0), (0.1, 0.01, 0.05), (0.08, -0.02, -0.12)])
def test_wall_features_recover_the_motion_exactly(pose):
    prev = wall_scan([((1.0, -2.0), (1.0, 2.0))])
    m = vo.wall_motion(wall_matches(pose), prev)
    assert m is not None
    assert np.allclose(m.mean, pose, atol=3e-3)


def test_wall_features_need_enough_inliers():
    prev = wall_scan([((1.0, -2.0), (1.0, 2.0))])
    assert vo.wall_motion(wall_matches((0.1, 0, 0), n=4), prev) is None
    assert vo.wall_motion(np.zeros((0, 4)), prev) is None


def test_feature_depth_ratio_uses_the_row_ratio():
    prev = wall_scan([((1.0, -2.0), (1.0, 2.0))])
    a, b, _ = vo.feature_points(wall_matches((0.15, 0, 0)), prev)
    assert np.allclose(a[:, 0], 1.0, atol=2e-3)  # first frame: the scan depth
    assert np.allclose(b[:, 0], 0.85, atol=5e-3)  # second: Z1 * d0 / d1


def test_match_rotation_is_signed_and_ignores_outliers():
    th = np.radians(20.0)
    u0 = RNG.uniform(40, 280, 30)
    b1 = np.arctan((u0 - CAM.cx) / CAM.fx) + th
    u1 = CAM.cx + CAM.fx * np.tan(b1)
    m = np.column_stack([u0, np.full(30, 100.0), u1, np.full(30, 100.0)])
    m[:6, 2] += RNG.uniform(-60, 60, 6)  # outliers
    rot, n = vo.match_rotation(m)
    assert rot == pytest.approx(20.0, abs=0.2) and n >= 20
    m2 = m.copy()
    m2[:, 2] = CAM.cx + CAM.fx * np.tan(np.arctan((u0 - CAM.cx) / CAM.fx) - th)
    assert vo.match_rotation(m2)[0] == pytest.approx(-20.0, abs=0.2)
    assert np.isnan(vo.match_rotation(m[:2])[0])


def test_vp_rotation_unwraps_to_the_nearest_branch():
    d, sig = vo.vp_rotation(10.0, -44.0, 0.3, 0.3, predicted=np.radians(-0.9 + 90 / 90 * 0.0))
    # headings 10 -> -44 differ by -54 degrees = +36 modulo 90; near 0 the nearest branch is +36
    assert np.degrees(d) == pytest.approx(36.0, abs=1e-9)
    d2, _ = vo.vp_rotation(10.0, -44.0, 0.3, 0.3, predicted=np.radians(-50.0))
    assert np.degrees(d2) == pytest.approx(-54.0, abs=1e-9)
    assert sig == pytest.approx(np.radians(np.hypot(0.3, 0.3)))
