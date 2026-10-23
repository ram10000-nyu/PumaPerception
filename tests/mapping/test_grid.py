"""Stage 7: lattice and scale, snapping (including the layouts' special cases), collision
geometry, signatures, and the rule 10 leakage tests for grid.py, slam.py and build_map.py."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
import grid  # noqa: E402
import scan as sc  # noqa: E402
import segment as sg  # noqa: E402

CAM = sc.camera_config()


def maze() -> np.ndarray:
    """An 11 x 11 wall-cell grid with the special cases the snapping must get right."""
    G = np.zeros((11, 11), bool)
    G[0, :] = G[-1, :] = G[:, 0] = G[:, -1] = True
    G[4, 0:6] = True  # a wall from the outer wall to edge (4, 5) ...
    G[4, 7:11] = True  # ... a missing post at (4, 6) between collinear walls, then on to the outer wall
    G[7, 4] = True  # an isolated vertical edge cell (renders as a horizontal 0.2 m piece)
    G[1, 8] = True  # a stub off the bottom wall, free end at y = 0.3
    return G


def cells(G) -> grid.CellGrid:
    return grid.CellGrid(0, 0, G, np.ones_like(G), np.zeros(G.shape))


def render(segs, pose):
    """(rows, top, status) of one frame by ray casting the rendered walls."""
    x, y, yaw = pose
    u = np.arange(320) + 0.5
    b = np.arctan2(u - CAM.cx, CAM.fx)
    th = yaw - b
    d = np.stack([np.cos(th), np.sin(th)], 1)
    a, e = segs[:, :2], segs[:, 2:] - segs[:, :2]
    p = np.array([x, y])
    t_best = np.full(320, np.inf)
    for k in range(len(segs)):
        den = d[:, 0] * (-e[k, 1]) - d[:, 1] * (-e[k, 0])
        ok = np.abs(den) > 1e-12
        rhs = a[k] - p
        t = np.where(ok, (rhs[0] * (-e[k, 1]) - rhs[1] * (-e[k, 0])) / np.where(ok, den, 1), np.inf)
        s = np.where(ok, (d[:, 0] * rhs[1] - d[:, 1] * rhs[0]) / np.where(ok, den, 1), -1)
        hit = ok & (t > 0.02) & (s >= 0) & (s <= 1)
        t_best = np.where(hit & (t < t_best), t, t_best)
    z = t_best * np.cos(b)
    rows = np.full(320, np.nan)
    top = np.full(320, np.nan)
    status = np.full(320, sg.FAR, np.int8)
    hit = np.isfinite(z) & (z < CAM.far)
    r = CAM.cy + CAM.fy * CAM.h / np.where(hit, z, 1)
    floor = hit & (r < CAM.height - 0.5)
    rows[floor] = r[floor]
    status[floor] = sg.WALL_RETURN
    near = hit & ~floor
    top[hit] = CAM.cy - CAM.fy * (CAM.wall_height - CAM.h) / z[hit]
    status[near] = sg.TOP_RETURN
    return rows, top, status


def frames(G, n=500, seed=0):
    rng = np.random.default_rng(seed)
    col = grid.collision_segments(cells(G))
    segs = grid.wall_segments(cells(G))
    poses = []
    while len(poses) < n:
        p = np.r_[rng.uniform(0.1, 1.9, 2), rng.uniform(-np.pi, np.pi)]
        if grid.point_segment_distance(p[None, :2], col)[0] > 0.13:
            poses.append(p)
    poses = np.array(poses)
    return [[render(segs, p) for p in poses]], [poses]


@pytest.fixture(scope="module")
def synthetic():
    G = maze()
    scans, poses = frames(G)
    return G, scans, poses


def test_lattice_recovers_scale_and_offset(synthetic):
    G, scans, poses = synthetic
    anchor = poses[0][0, :2]
    s, shift = 0.99, np.array([0.012, -0.017])
    bad = poses[0].copy()
    bad[:, :2] = anchor + s * (bad[:, :2] - anchor) + shift
    c, ax, w, pc = grid.wall_points(scans, [bad])
    lat = grid.fit_lattice(c, ax, w, anchor, pc)
    assert abs(lat.scale - s) < 5e-4
    fixed = lat.apply(bad)
    assert np.abs(fixed[:, :2] - poses[0][:, :2]).max() < 0.004


def test_snap_recovers_the_layout_including_gaps_isolated_cells_and_stubs(synthetic):
    G, scans, poses = synthetic
    spec = grid.GridSpec(np.array([-0.225, -0.225]), (50, 50))
    m = grid.build(scans, poses, poses[0][0, :2], spec=spec)
    W = np.zeros_like(G)
    for r in range(G.shape[0]):
        for c in range(G.shape[1]):
            R, C = r - m.cells.r0, c - m.cells.c0
            W[r, c] = m.cells.walls[R, C]
    assert np.array_equal(W, G), np.argwhere(W != G)
    assert not W[4, 6]  # the gap between collinear walls
    assert W[7, 4] and W[1, 8]  # the isolated cell and the stub
    # every pose the robot used is in the configuration space
    iy, ix = spec.index(poses[0][:, 0], poses[0][:, 1])
    assert m.cspace[iy, ix].mean() > 0.99


def test_collision_geometry_has_no_free_end_extension():
    G = maze()
    col = grid.collision_segments(cells(G))
    ren = grid.wall_segments(cells(G))
    # the isolated cell is a point in collision, a 0.2 m piece when rendered
    assert any(np.allclose(s, [0.8, 1.4, 0.8, 1.4]) for s in col)
    assert any(np.allclose(s, [0.7, 1.4, 0.9, 1.4]) for s in ren)
    # the stub's free end: rendered 0.1 m past the cell centre, collision stops at it
    assert np.isclose(max(max(s[1], s[3]) for s in col if np.isclose(s[0], 1.6) and np.isclose(s[2], 1.6)), 0.2)
    assert np.isclose(max(max(s[1], s[3]) for s in ren if np.isclose(s[0], 1.6) and np.isclose(s[2], 1.6) and max(s[1], s[3]) < 1), 0.3)


def test_hellinger_distance():
    a = np.sqrt(np.r_[0.5, 0.5, 0, 0])
    b = np.sqrt(np.r_[0, 0, 0.5, 0.5])
    assert np.isclose(grid.hellinger(a, a), 0) and np.isclose(grid.hellinger(a, b), 1)


# ------------------------------------------------------------------------------- leakage


def test_stage7_pipeline_modules_do_not_touch_ground_truth():
    forbidden = re.compile(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene|true_|truth_grid")
    for name in ("grid.py", "slam.py", "build_map.py"):
        body = (SRC / name).read_text().split('"""', 2)[-1]
        code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
        assert not forbidden.search(code), name


def test_leakage_build_map_identical_without_ground_truth_files(tmp_path):
    """rule 10: raw images -> map on a copy without poses.json / gt.json is identical."""
    root = SRC.parents[1]
    src = root / "data" / "gt4-7"
    if not src.exists():
        pytest.skip("needs data/gt4-7")
    n = 40
    copies = {}
    for name, keep_truth in (("with", True), ("without", False)):
        dst = tmp_path / name / "gt4-7"
        for d in range(3):
            s, t = src / f"traj_{d}", dst / f"traj_{d}"
            t.mkdir(parents=True)
            info = json.loads((s / "data_info.json").read_text())[:n]
            (t / "data_info.json").write_text(json.dumps(info))
            for r in info:
                shutil.copy(s / r["image"], t / r["image"])
            if keep_truth:
                poses = json.loads((s / "poses.json").read_text())[:n]
                (t / "poses.json").write_text(json.dumps(poses))
        if keep_truth:
            shutil.copy(src / "gt.json", dst / "gt.json")
        out = tmp_path / name / "out"
        # the 40-frame copies are too short for the frame-rate detector (it raises rather than
        # guess), and this test is about leakage: the rate is given
        cmd = [sys.executable, str(SRC / "build_map.py"), "--data", str(dst), "--out", str(out), "--ticks", "5"]
        subprocess.run(cmd, check=True, capture_output=True)
        copies[name] = np.load(out / "map.npz", allow_pickle=True)
    a, b = copies["with"], copies["without"]
    for k in ("logodds", "wall_cells", "poses", "sig_hist"):
        assert np.array_equal(a[k], b[k]), k
