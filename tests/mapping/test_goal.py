"""Stage 8: goal camera height, heading modulo 90, geometric placement on a synthetic map, the
holdout gate of the final evaluation, and the rule 10 leakage test for goal.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal  # noqa: E402
import grid  # noqa: E402
import record_gt  # noqa: E402
import segment as sg  # noqa: E402
import vo  # noqa: E402
from test_grid import CAM, frames, maze, render  # noqa: E402


def test_camera_height_from_wall_bottom_and_top():
    h = 0.16
    z = np.linspace(0.3, 1.4, 320)
    seg = SimpleNamespace(rows=CAM.cy + CAM.fy * h / z, top_rows=CAM.cy - CAM.fy * (CAM.wall_height - h) / z,
                          status=np.full(320, sg.WALL_RETURN))
    assert abs(goal.camera_height([seg], CAM) - h) < 1e-3


def test_axis_heading_modulo_90():
    for theta in np.radians([3.0, 37.0, -20.0, 130.0]):
        phi = np.r_[np.zeros(50), np.full(50, np.pi / 2), np.full(50, np.pi)] - theta  # world normals on the axes
        gs = SimpleNamespace(normal=np.c_[np.cos(phi), np.sin(phi)], sigma=np.full(150, 0.005))
        d = (goal.axis_heading(gs) - theta + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(d) < 1e-6


@pytest.fixture(scope="module")
def synthetic_map(tmp_path_factory):
    G = maze()
    scans, poses = frames(G, n=500, seed=1)
    m = grid.build(scans, poses, poses[0][0, :2], spec=grid.GridSpec(np.array([-0.225, -0.225]), (50, 50)))
    m.signatures = grid.Signatures()
    f = tmp_path_factory.mktemp("map") / "map.npz"
    grid.save(m, f)
    return G, goal.load_map(f)


def goal_scan_at(G, pose):
    """A goal scan of the synthetic maze from ``pose``: four views at theta + k * 90 deg."""
    segs = grid.wall_segments(grid.CellGrid(0, 0, G, np.ones_like(G), np.zeros(G.shape)))
    XY, N, S, V = [], [], [], []
    for k in range(4):
        p = np.array([pose[0], pose[1], pose[2] + k * np.pi / 2])
        rows, top, status = render(segs, p)
        s = vo.scan_points(rows, top, status, CAM)
        a = k * np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        XY.append(s.xy @ R.T)
        N.append(s.normal @ R.T)
        S.append(s.sigma)
        V.append(np.full(len(s.xy), k))
    return goal.GoalScan(np.concatenate(XY), np.concatenate(S), np.concatenate(V), np.concatenate(N), CAM, 0.0, [])


def _close(p, q):
    return np.hypot(*(np.asarray(p[:2]) - q[:2])) < 0.02 and abs((p[2] - q[2] + np.pi) % (2 * np.pi) - np.pi) < np.radians(1.0)


@pytest.mark.parametrize("pose", [(0.6, 1.4, 0.3), (1.6, 0.6, 2.0), (1.0, 1.0, -1.2)])
def test_geometric_placement_on_a_synthetic_map(synthetic_map, pose):
    """Geometry alone: the search finds the true pose among its distinct candidates, and its
    best has a score at least the truth's (the small maze has a genuine 180-degree alias for
    the second pose; real maps need the texture term for such cases, stage 8 dev seed 10)."""
    G, mf = synthetic_map
    gs = goal_scan_at(G, pose)
    est = goal.place(mf, gs, use_texture=False)
    assert any(_close(t[0], np.array(pose)) for t in est.top)
    w = goal._weights(gs)
    assert goal._score(mf, gs.xy, w, est.pose[None])[0] >= goal._score(mf, gs.xy, w, np.array(pose)[None])[0] - 1e-3
    if pose != (1.6, 0.6, 2.0):
        assert _close(est.pose, np.array(pose))


def test_final_evaluation_refuses_holdout_seeds_without_the_flag():
    import final_eval

    holdout = record_gt.holdout_seeds()
    if not holdout:
        pytest.skip("no holdout seeds listed")
    assert not record_gt._FINAL
    with pytest.raises(record_gt.HoldoutError):
        final_eval.data_dir(holdout[0])
    record_gt.allow_final()
    try:
        assert final_eval.data_dir(holdout[0]).name.endswith(str(holdout[0]))
    finally:
        record_gt._FINAL = False


def test_goal_module_does_not_touch_ground_truth():
    forbidden = re.compile(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene|\"target\"|truth")
    body = (SRC / "goal.py").read_text().split('"""', 2)[-1]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert not forbidden.search(code)
