"""Stage 5: loop-closure candidates, verification and relative pose."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import loops  # noqa: E402
import se2  # noqa: E402
from test_vo import project, to_frame, wall_scan  # noqa: E402

RNG = np.random.default_rng(5)
WALL = [((1.0, -2.0), (1.0, 2.0))]


def wall_correspondences(pose, n=60, wall_x=1.0):
    """Pixels of points on the plane x = wall_x seen from the origin and from ``pose``."""
    ys = RNG.uniform(-0.5, 0.5, n)
    zs = RNG.uniform(-0.18, 0.08, n)
    inv = se2.inverse(np.asarray(pose, float))
    out = []
    for y, z in zip(ys, zs, strict=True):
        ua, va = project(wall_x, y, z)
        p = se2.compose(inv, [wall_x, y, 0.0])
        ub, vb = project(p[0], p[1], z)
        out.append((ua, va, ub, vb))
    return np.array(out)


@pytest.mark.parametrize("pose", [(0.3, 0.1, 0.0), (0.2, -0.15, 0.3), (-0.1, 0.2, -0.4)])
def test_homography_decomposition_recovers_the_pose(pose):
    m = wall_correspondences(pose)
    H, _ = cv2.findHomography(m[:, :2], m[:, 2:], 0)
    got = loops.pose_from_homography(H, m, wall_scan(WALL))
    assert got is not None
    assert np.hypot(*(got[:2] - pose[:2])) < 0.02
    assert abs(np.degrees(se2.wrap(got[2] - pose[2]))) < 0.5


@pytest.mark.parametrize("pose", [(0.3, 0.1, 0.0), (0.2, -0.15, 0.3)])
def test_feature_pose_recovers_the_pose(pose):
    m = wall_correspondences(pose)
    got = loops.pose_from_features(m, wall_scan(WALL), wall_scan(to_frame(WALL, pose)))
    assert got is not None and np.allclose(got.mean, pose, atol=0.01)


def test_candidates_exclude_own_drive_neighbours_and_keep_the_best():
    g = RNG.normal(size=(40, 8))
    g[25] = g[3] + 1e-3  # frame 3 (drive 0) and frame 25 (drive 1) look alike
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    drive = np.array([0] * 20 + [1] * 20)
    index = np.r_[np.arange(20), np.arange(20)]
    pairs = loops.candidates(g, drive, index, top_k=2, exclude=10)
    assert (3, 25) in {(i, j) for i, j, _ in pairs}
    for i, j, _ in pairs:
        assert i < j
        assert not (drive[i] == drive[j] and abs(index[i] - index[j]) <= 10)


def test_codebook_predicts_the_nearest_centre():
    c = RNG.normal(size=(5, 3))
    x = c[[2, 4, 0]] + 1e-3
    assert list(loops._Codebook(c).predict(x)) == [2, 4, 0]


def test_vocabulary_encodes_unit_vectors():
    feats = [loops.Features(RNG.uniform(0, 300, (50, 2)).astype(np.float32),
                            RNG.integers(0, 60, (50, 128)).astype(np.uint8)) for _ in range(30)]
    voc = loops.fit_vocabulary(feats, k=4, dim=8)
    g = voc.encode(feats[:5])
    assert g.shape == (5, 8) and np.allclose(np.linalg.norm(g, axis=1), 1.0)


def test_agree():
    assert loops.agree(np.array([0.1, 0, 0.0]), np.array([0.15, 0.02, np.radians(3)]))
    assert not loops.agree(np.array([0.1, 0, 0.0]), np.array([0.3, 0, 0.0]))


def test_envelope_and_gate():
    assert loops.in_envelope(np.array([0.3, 0.2, np.radians(20)]))
    assert not loops.in_envelope(np.array([0.4, 0.3, 0.0]))
    assert not loops.in_envelope(np.array([0.1, 0.0, np.radians(28)]))
    pa, pb = np.array([1.0, 1.0, 0.5]), np.array([1.2, 1.1, 0.6])
    rel = se2.relative(pa, pb)
    assert loops.odometry_gate(rel, pa, pb)
    assert not loops.odometry_gate(rel + [0.6, 0, 0], pa, pb)


def test_measure_on_a_real_revisit_pair():
    """Frames 40 and 41 of a dev drive: a verified pair with a tight pose (the pipeline path)."""
    import record_gt
    import scan as sc_
    import segment as sg
    import vo

    path = record_gt.gt_dir(7, prefix="gt4") / "traj_0"
    if not path.exists():
        pytest.skip("run record_gt.py --msaa 4 first")
    sift = cv2.SIFT_create(nfeatures=loops.N_FEATURES)
    out = []
    for k in (40, 41):
        img = cv2.imread(str(path / f"{k}.jpg"))
        seg = sg.segment(img)
        scan = sc_.scan_from_segmentation(seg)
        out.append((loops.frame_features(img, seg, sift),
                    vo.scan_points(seg.rows, seg.top_rows, scan.status)))
    c = loops.measure(out[0][0], out[1][0], out[0][1], out[1][1])
    assert c is not None and c.inliers >= loops.MIN_INLIERS
    assert np.hypot(*c.mean[:2]) < 0.3
