"""Stage 4: the filter, the collision treatment and the rule 10 leakage test."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
import motion as M  # noqa: E402
import odometry as od  # noqa: E402
import pipeline  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402
import vo  # noqa: E402
from test_vo import CAM, to_frame, wall_scan  # noqa: E402

# an asymmetric scene (no 90-degree look-alike): an end wall, a long left wall, a short right one
SCENE = [
    ((1.4, -1.0), (1.4, 1.6)),
    ((-1.0, 0.7), (1.4, 0.7)),
    ((0.2, -0.5), (1.4, -0.5)),
    ((0.6, 0.0), (0.6, 0.3)),
]


def test_vp_diff90_wraps():
    assert od.vp_diff90(44.0, -44.0) == pytest.approx(-2.0)
    assert od.vp_diff90(10.0, 100.0) == pytest.approx(0.0)


def test_collision_increment_picks_the_true_branch():
    pose = (0.05, -0.01, np.radians(-47.0))
    target = wall_scan(SCENE)
    source = wall_scan(to_frame(SCENE, pose))
    # headings 20 -> -27: a change of -47 degrees, which modulo 90 is +43; the branch
    # hypotheses are -47 (true), +43 and +133
    out = od.collision_increment(target, source, 20.0, -27.0, 0.3, 0.3, match_rot=-44.0, match_n=8)
    assert out is not None
    mean, cov, info = out
    assert np.degrees(mean[2]) == pytest.approx(-47.0, abs=1.5)
    assert np.hypot(*(mean[:2] - np.array(pose[:2]))) < 0.04
    assert info["chosen"] in ("match", "vp-1", "global")


def test_collision_increment_without_headings_uses_global_search_and_matches():
    pose = (0.05, 0.0, np.radians(38.0))
    target = wall_scan(SCENE)
    source = wall_scan(to_frame(SCENE, pose))
    mean, cov, info = od.collision_increment(
        target, source, np.nan, np.nan, np.nan, np.nan, match_rot=33.0, match_n=6
    )
    assert np.degrees(mean[2]) == pytest.approx(38.0, abs=2.0)


def test_a_wrong_branch_is_not_chosen_even_when_it_aligns_well():
    """Headings that fold to +43 offer the branch +43; the true rotation is -47. A trusted match
    rotation (-44) must keep the filter on the true branch."""
    pose = (0.05, 0.0, np.radians(-47.0))
    target, source = wall_scan(SCENE), wall_scan(to_frame(SCENE, pose))
    mean, _, info = od.collision_increment(target, source, 20.0, -27.0, 0.3, 0.3, -44.0, 8)
    assert abs(np.degrees(mean[2]) - (-47.0)) < 3.0


def test_collision_translation_is_regularised_by_the_prior():
    pose = (0.05, -0.01, np.radians(-47.0))
    prior = M.MotionPrior.load()
    target, source = wall_scan(SCENE), wall_scan(to_frame(SCENE, pose))
    mean, cov, _ = od.collision_increment(
        target, source, 20.0, -27.0, 0.3, 0.3, -44.0, 8, prior.by_label["FORWARD"].collision
    )
    assert np.hypot(*(mean[:2] - np.array([0.049, 0.0]))) < 0.12
    assert np.sqrt(cov[0, 0]) <= 0.036


# ------------------------------------------------------------------------ the filter itself


def cached_inputs(limit=150):
    try:
        import stage4 as s4
    except Exception:  # noqa: BLE001
        pytest.skip("stage 4 evaluation modules not importable")
    try:
        drives = s4.load_inputs([7], "gt4")
    except Exception:  # noqa: BLE001
        pytest.skip("run stage3_extract.py / stage4_extract.py for gt4 first")
    dr = drives[0]
    inp = dr["inp"]
    sl = slice(0, limit)
    cut = od.DriveInputs(
        inp.rows[sl],
        inp.top[sl],
        inp.status[sl],
        inp.psi[sl],
        inp.psi_sigma[sl],
        inp.labels[:limit],
        inp.flags[sl],
        inp.matches[:limit],
    )
    return cut, dr["true"][sl]


def test_filter_tracks_a_real_drive_start():
    inp, true = cached_inputs()
    res = od.run_drive(inp, M.MotionPrior.load(), "C")
    herr = np.degrees(se2.wrap(res.poses[:, 2] - true[:, 2]))
    perr = np.hypot(*(res.poses[:, :2] - true[:, :2]).T)
    assert np.sqrt(np.mean(herr**2)) < 2.0
    assert perr.max() < 0.4
    assert res.heading_accepted.sum() >= 0.95 * res.heading_valid.sum()


def test_vision_only_variant_does_not_use_the_labels():
    inp, true = cached_inputs(limit=80)
    prior = M.MotionPrior.load()
    a = od.run_drive(inp, prior, "B")
    shuffled = od.DriveInputs(
        inp.rows,
        inp.top,
        inp.status,
        inp.psi,
        inp.psi_sigma,
        list(np.roll(np.array(inp.labels), 7)),
        np.zeros(len(inp.labels), bool),
        inp.matches,
    )
    flags_off = od.DriveInputs(
        inp.rows,
        inp.top,
        inp.status,
        inp.psi,
        inp.psi_sigma,
        inp.labels,
        np.zeros(len(inp.labels), bool),
        inp.matches,
    )
    b = od.run_drive(shuffled, prior, "B")
    c = od.run_drive(flags_off, prior, "B")
    assert np.allclose(b.poses, c.poses)  # labels do not matter to variant B
    assert a.poses.shape == b.poses.shape


def test_a_confidently_wrong_heading_is_rejected_not_followed():
    inp, true = cached_inputs(limit=120)
    psi = inp.psi.copy()
    k = 60
    psi[k] = psi[k] + 30.0 if np.isfinite(psi[k]) else 30.0  # a wrong heading, isolated
    bad = od.DriveInputs(
        inp.rows, inp.top, inp.status, psi, inp.psi_sigma, inp.labels, inp.flags, inp.matches
    )
    res = od.run_drive(bad, M.MotionPrior.load(), "C")
    herr = np.degrees(se2.wrap(res.poses[:, 2] - true[:, 2]))
    assert not res.heading_accepted[k]
    assert abs(herr[k]) < 2.0


# ------------------------------------------------------------------------------- leakage


def test_pipeline_modules_do_not_touch_ground_truth():
    forbidden = re.compile(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene|occupancy")
    for name in ("vo.py", "odometry.py", "pipeline.py", "calibration.py", "loops.py"):
        body = (SRC / name).read_text().split('"""', 2)[-1]
        code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
        assert not forbidden.search(code), name


def test_leakage_trajectory_identical_without_ground_truth_files(tmp_path):
    """rule 10: images -> trajectory on a copy without poses.json/gt.json is identical."""
    src = record_gt.gt_dir(7, prefix="gt4") / "traj_0"
    if not src.exists():
        pytest.skip("run record_gt.py --msaa 4 first")
    n = 50
    info = json.loads((src / "data_info.json").read_text())[:n]
    dst = tmp_path / "traj_0"
    dst.mkdir()
    (dst / "data_info.json").write_text(json.dumps(info))
    for r in info:
        shutil.copy(src / r["image"], dst / r["image"])
    assert not (dst / "poses.json").exists()
    prior = M.MotionPrior.load()
    a = od.run_drive(pipeline.drive_inputs(src, n), prior, "C")
    b = od.run_drive(pipeline.drive_inputs(dst, n), prior, "C")
    assert np.array_equal(a.poses, b.poses)
    assert np.array_equal(a.inc_mean, b.inc_mean)


def test_image_pipeline_matches_the_cached_cue_path():
    """The images-to-trajectory path equals the evaluation's cached-cue path."""
    cut, _ = cached_inputs(limit=50)
    src = record_gt.gt_dir(7, prefix="gt4") / "traj_0"
    got = pipeline.drive_inputs(src, 50)
    assert np.allclose(got.rows, cut.rows, equal_nan=True)
    assert np.allclose(got.psi, cut.psi, equal_nan=True)
    assert np.array_equal(got.flags, cut.flags)
    assert all(np.allclose(a, b) for a, b in zip(got.matches, cut.matches, strict=True))
    prior = M.MotionPrior.load()
    assert np.allclose(od.run_drive(got, prior, "C").poses, od.run_drive(cut, prior, "C").poses)


def test_camera_constants_are_what_stage0_measured():
    assert CAM.h == pytest.approx(0.2099, abs=1e-3)
    assert vo.RANGE_CAP == 2.5
