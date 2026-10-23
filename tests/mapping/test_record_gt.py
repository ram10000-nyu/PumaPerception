"""Stage 0: the ground-truth dataset for one dev seed is student-equal and pose-aligned."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
record_gt = pytest.importorskip("record_gt")
sim = pytest.importorskip("vis_nav_sim")

SEED = json.loads(record_gt.SEEDS.read_text())["dev"][0]


@pytest.fixture(scope="module")
def dataset():
    path = record_gt.gt_dir(SEED)
    if not (path / "gt.json").exists():
        pytest.skip("run `uv run source/mapping/record_gt.py` first")
    return path


def test_seeds_are_disjoint_and_big_enough():
    cfg = json.loads(record_gt.SEEDS.read_text())
    assert len(cfg["dev"]) >= 3 and len(cfg["test"]) >= 3
    assert not set(cfg["dev"]) & set(cfg["test"])


def test_poses_align_with_records(dataset):
    poses = record_gt.load_poses(dataset)
    assert len(poses) == record_gt.ROUTES
    for i, p in enumerate(poses):
        info = json.loads((dataset / f"traj_{i}" / "data_info.json").read_text())
        assert p.shape == (len(info), 3)
        assert np.all(np.abs(p[:, 2]) < 100)


def test_pose_alignment_and_student_equality(dataset):
    textures = sim.Textures.find(download=False)
    assert record_gt.check_pose_alignment(SEED, dataset, textures)["max"] < 2.0
    r = record_gt.check_matches_student(SEED, dataset, textures)
    assert r["differ"] == []


def test_holdout_seeds_are_disjoint_from_dev_and_test():
    cfg = json.loads(record_gt.SEEDS.read_text())
    assert len(cfg["holdout"]) >= 5
    assert not set(cfg["holdout"]) & (set(cfg["dev"]) | set(cfg["test"]))


def test_holdout_is_refused_without_final(monkeypatch):
    monkeypatch.setattr(record_gt, "_FINAL", False)
    seed = record_gt.holdout_seeds()[0]
    for prefix in ("gt", "gt4"):
        with pytest.raises(record_gt.HoldoutError):
            record_gt.gt_dir(seed, prefix=prefix)
    # dev and test seeds are unaffected
    assert record_gt.gt_dir(SEED).name == f"gt-{SEED}"
    monkeypatch.setattr(record_gt, "_FINAL", True)
    assert record_gt.gt_dir(seed).name == f"gt-{seed}"
