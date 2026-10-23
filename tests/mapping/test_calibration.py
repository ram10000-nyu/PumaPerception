"""Stage 4b: scaled priors and the label-prior self-calibration."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source" / "mapping"
sys.path.insert(0, str(SRC))
import calibration as cal  # noqa: E402
import motion as M  # noqa: E402
import pipeline  # noqa: E402
import record_gt  # noqa: E402

RNG = np.random.default_rng(4)


@pytest.fixture(scope="module")
def prior():
    if not M.PRIOR_PATH.exists():
        pytest.skip("run stage1.py first")
    return M.MotionPrior.load()


def test_scaled_group_transforms_the_density(prior):
    g = prior.get("LEFT", "FORWARD")
    d = np.array([0.9, 0.9, 1.1])
    g2 = g.scaled(0.9, 1.1)
    x = g.mean + RNG.normal(size=(50, 3)) * np.sqrt(np.diag(g.cov))
    assert np.allclose(g2.logpdf(x * d), g.logpdf(x) - np.log(d).sum())
    assert np.array_equal(g2.contains(x * d), g.contains(x))
    assert np.allclose(g2.mean, g.mean * d)
    assert g2.collision is g.collision


def test_scaled_prior_scales_groups_by_their_current_label(prior):
    p2 = prior.scaled({"FORWARD": (0.8, 1.0)})
    fwd = prior.get("FORWARD", "LEFT").mean[0]
    assert np.allclose(p2.get("FORWARD", "LEFT").mean[0], 0.8 * fwd)
    assert np.allclose(p2.get("LEFT", "FORWARD").mean, prior.get("LEFT", "FORWARD").mean)
    assert p2.meta["scales"] == {"FORWARD": [0.8, 1.0]}


def test_ratio_with_se_is_the_ratio_of_sums():
    num, den = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])
    r, se = cal.ratio_with_se(num, den)
    assert r == pytest.approx(1.0) and se > 0


def test_consistency_test():
    assert cal.consistent(np.array([1.0, 1.01, 0.99]), np.array([0.01, 0.01, 0.01]))
    assert not cal.consistent(np.array([1.0, 1.3, 0.7]), np.array([0.01, 0.01, 0.01]))


def _rows(speed, turn, n=300):
    """Synthetic vision rows: prior means times the drive's scale plus noise."""
    out = []
    for _ in range(n):
        m = np.array([0.15, 0.0, 0.0])
        out.append(("FORWARD", m, m * [speed, 1, 1] + RNG.normal(0, [0.005, 0.002, 0.002])))
        m = np.array([0.03, 0.0, 0.25 * RNG.choice([-1, 1])])
        out.append(("LEFT", m, m * [speed, 1, turn] + RNG.normal(0, [0.005, 0.002, 0.003])))
    return out


def test_fit_drives_pools_agreeing_drives_and_separates_biased_ones(monkeypatch, prior):
    scales = {0: (1.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 1.0)}
    monkeypatch.setattr(cal, "vision_increments", lambda inp, p: _rows(*scales[inp]))
    out = cal.fit_drives([0, 1, 2], prior)
    assert all(d.speed.pooled and d.turn.pooled for d in out)
    assert out[0].speed.value == pytest.approx(1.0, abs=0.01)

    scales.update({1: (0.6, 1.2), 2: (1.3, 0.9)})
    out = cal.fit_drives([0, 1, 2], prior)
    assert not any(d.speed.pooled for d in out)
    got = [(d.speed.value, d.turn.value) for d in out]
    assert np.allclose(got, [scales[i] for i in range(3)], atol=0.02)


def test_fit_drives_keeps_scale_one_without_enough_pairs(monkeypatch, prior):
    monkeypatch.setattr(cal, "vision_increments", lambda inp, p: _rows(0.7, 1.2, n=10))
    out = cal.fit_drives([0], prior)
    assert out[0].speed.value == 1.0 and out[0].turn.value == 1.0


def test_calibration_leakage_identical_without_ground_truth_files(tmp_path, prior):
    """rule 10: the calibration's measurements on a copy without poses.json/gt.json match."""
    src = record_gt.gt_dir(7, prefix="gt4") / "traj_0"
    if not src.exists():
        pytest.skip("run record_gt.py --msaa 4 first")
    n = 60
    info = json.loads((src / "data_info.json").read_text())[:n]
    dst = tmp_path / "traj_0"
    dst.mkdir()
    (dst / "data_info.json").write_text(json.dumps(info))
    for r in info:
        shutil.copy(src / r["image"], dst / r["image"])
    a = cal.vision_increments(pipeline.drive_inputs(src, n), prior)
    b = cal.vision_increments(pipeline.drive_inputs(dst, n), prior)
    assert len(a) == len(b) > 10
    for (la, _, va), (lb, _, vb) in zip(a, b, strict=True):
        assert la == lb and np.array_equal(va, vb)
