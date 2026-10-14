"""Stage 2: wall-segment rule, evaluation helpers, and the pipeline's no-ground-truth rule."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
import camera_fit as cf  # noqa: E402
import stage2 as s2  # noqa: E402


def segs(occ):
    """Segments as sets of normalised (x0, y0, x1, y1), the smaller endpoint first."""
    out = set()
    for s in cf.wall_segments(np.array(occ, bool)):
        a, b = (s[0], s[1]), (s[2], s[3])
        a, b = sorted([a, b])
        out.add((round(a[0], 3), round(a[1], 3), round(b[0], 3), round(b[1], 3)))
    return out


def test_isolated_cell_is_a_x_segment_of_0_2():
    assert segs([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == {(0.1, 0.2, 0.3, 0.2)}


def test_straight_chain_extends_only_at_free_ends():
    # three cells in a row: centres 0.2, 0.4, 0.6; free ends extend 0.1
    got = segs([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    assert (0.1, 0.2, 0.2, 0.2) in got and (0.6, 0.2, 0.7, 0.2) in got
    assert (0.2, 0.2, 0.4, 0.2) in got and (0.4, 0.2, 0.6, 0.2) in got


def test_corner_ends_at_the_cell_centre_not_past_it():
    # an L: cells (1,1), (1,2), (2,1); the corner (1,1) must not extend beyond its centre
    got = segs([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    xs = [s for s in got if s[1] == s[3]]
    ys = [s for s in got if s[0] == s[2]]
    assert max(max(s[0], s[2]) for s in ys) == pytest.approx(0.2)
    # the corner at (x, y) = (0.2, 0.2): nothing to its left on the x line, nothing above on y
    assert all(min(s[0], s[2]) >= 0.2 - 1e-9 for s in xs)
    assert all(min(s[1], s[3]) >= 0.2 - 1e-9 for s in ys)


def test_scene_hits_a_wall_plane_at_the_cell_centre():
    occ = np.zeros((3, 5), bool)
    occ[:, 4] = True  # vertical wall at x = 0.8
    z = cf.Scene(occ, half=0.0022).cast((0.2, 0.2), np.array([[1.0, 0.0]]))
    assert z[0] == pytest.approx(0.6 - 0.0022, abs=1e-6)


def test_point_segment_distance():
    segs_ = np.array([[0, 0, 1, 0], [0, 1, 1, 1]], float)
    p = np.array([[0.5, 0.2], [2.0, 0.0], [0.5, 0.9]])
    assert s2.point_segment_distance(p, segs_) == pytest.approx([0.2, 1.0, 0.1])


def test_iou_with_tolerance():
    a = np.zeros((20, 20), bool)
    a[10, 3:17] = True
    b = np.roll(a, 1, 0)  # one cell off
    assert s2.iou(a, b, dilate=0) == 0.0
    assert s2.iou(a, b, dilate=1) >= 0.5


def test_choose_r_max_stops_at_the_first_failing_bin():
    good = {"n": 100, "dist_median_m": 0.01, "dist_p95_m": 0.05, "rel_bias_median": 0.0}
    bad = {**good, "dist_p95_m": 0.2}
    bins = [
        {"lo": 0, "hi": 0.5, **good},
        {"lo": 0.5, "hi": 1.0, **good},
        {"lo": 1.0, "hi": 1.5, **bad},
    ]
    assert s2.choose_r_max(bins) == 1.0
    bins[1] = {"lo": 0.5, "hi": 1.0, **good, "rel_bias_median": 0.02}
    assert s2.choose_r_max(bins) == 0.5


def test_oracle_grid_puts_wall_planes_on_cell_centres():
    c = (0.2 - s2.ORIGIN[0]) / s2.RES  # a wall plane at x = 0.2 m
    assert c % 1 == pytest.approx(0.5)


def test_pipeline_modules_do_not_touch_ground_truth():
    """rule 10: segment.py and scan.py use only measured constants, never gt/poses/Scene."""
    forbidden = re.compile(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene|occupancy")
    for name in ("segment.py", "scan.py"):
        src = (SRC / name).read_text()
        code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert not forbidden.search(code.split('"""', 2)[-1]), name
