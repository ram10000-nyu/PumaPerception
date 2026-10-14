"""Stage 3b: the collision component of the prior and the collision detector."""

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
import collision as col  # noqa: E402
import heading as hd  # noqa: E402
import motion as M  # noqa: E402
import record_gt  # noqa: E402

RNG = np.random.default_rng(3)


# --------------------------------------------------------------------------------- prior


def pairs_with_events(n=4000, n_events=20):
    lab = RNG.choice(["FORWARD", "IDLE", "LEFT"], n)
    inc = np.zeros((n, 3))
    f = lab == "FORWARD"
    inc[f, 0] = 0.14 + RNG.normal(0, 0.01, f.sum())
    inc[~f, 2] = RNG.normal(0.0, 0.05, (~f).sum())
    inc[:, 1] = RNG.normal(0, 0.001, n)
    idx = RNG.choice(np.where(f)[0], n_events, replace=False)
    inc[idx] = np.c_[
        RNG.uniform(0.03, 0.07, n_events),
        RNG.normal(0, 0.02, n_events),
        RNG.choice([-1, 1], n_events) * RNG.uniform(0.55, 0.9, n_events),
    ]
    pairs = {
        "inc": inc,
        "label": lab,
        "prev": np.array(["FORWARD"] * n),
        "next": lab,
        "seed": np.zeros(n, int),
        "drive": np.zeros(n, int),
        "k": np.arange(n),
    }
    return pairs, idx


def test_collision_component_contains_events_and_is_two_lobed():
    p, idx = pairs_with_events()
    prior = M.MotionPrior.fit(p)
    summary = prior.add_collision(p)
    assert summary["events"] == 20
    g = prior.get("FORWARD", "FORWARD")
    assert g.collision is not None and g.collision.means.shape == (2, 3)
    assert g.collision.means[0, 2] == pytest.approx(-g.collision.means[1, 2])
    assert g.contains_ext(p["inc"][idx], 0.99).all()
    assert prior.get("LEFT").collision is None  # only FORWARD and IDLE collide


def test_extended_region_is_a_union_not_a_wider_normal_region():
    p, _ = pairs_with_events()
    prior = M.MotionPrior.fit(p)
    prior.add_collision(p)
    g = prior.get("FORWARD", "FORWARD")
    assert g.contains_ext(np.array([[0.14, 0.0, 0.0]]), 0.95).all()
    # between the normal mass and the lobes: neither region claims it
    assert not g.contains_ext(np.array([[0.14, 0.0, 0.4]]), 0.95).any()


def test_extended_density_has_a_collision_floor():
    """Fit the mixture on normal pairs only, as a prior that has never seen a collision, then
    add the component: the plain mixture vetoes an event, the extended one does not."""
    normal, _ = pairs_with_events(n_events=0)
    with_events, idx = pairs_with_events()
    prior = M.MotionPrior.fit(normal)
    prior.add_collision(with_events)
    g = prior.get("FORWARD", "FORWARD")
    ev = with_events["inc"][idx]
    assert np.all(g.logpdf_ext(ev) > g.logpdf(ev))
    assert np.median(g.logpdf_ext(ev) - g.logpdf(ev)) > 5.0
    # and it costs a normal pair almost nothing: the factor (1 - weight)
    x = normal["inc"][normal["label"] == "FORWARD"][:50]
    assert np.allclose(g.logpdf_ext(x), g.logpdf(x), atol=0.05)


def test_collision_round_trip_keeps_old_fields(tmp_path):
    p, _ = pairs_with_events()
    prior = M.MotionPrior.fit(p)
    before = {k: v.to_json() for k, v in prior.by_label.items()}
    prior.add_collision(p)
    f = tmp_path / "p.json"
    prior.save(f)
    back = M.MotionPrior.load(f)
    g, h = prior.get("FORWARD", "FORWARD"), back.get("FORWARD", "FORWARD")
    assert h.collision is not None
    assert np.allclose(g.collision.means, h.collision.means)
    assert h.threshold99 == pytest.approx(g.threshold99)
    for key, old in before.items():  # the stage 1 fields are unchanged
        new = back.by_label[key].to_json()
        for field in ("weights", "means", "covs", "mean", "cov", "threshold", "n"):
            assert np.allclose(new[field], old[field])


# ------------------------------------------------------------------------------- detector


def test_flag_pair_uses_the_heading_change_modulo_90():
    assert col.flag_pair(10.0, -35.0, "FORWARD").flagged  # 45 degrees
    assert col.flag_pair(0.0, 30.0, "IDLE").flagged
    assert not col.flag_pair(0.0, 15.0, "FORWARD").flagged
    # across the +-45 cut: 44 -> -44 is a change of 2 degrees, not 88
    p = col.flag_pair(44.0, -44.0, "FORWARD")
    assert p.delta == pytest.approx(2.0) and not p.flagged


def test_turn_labels_are_never_flagged():
    assert not col.flag_pair(0.0, 40.0, "LEFT").flagged
    assert not col.flag_pair(0.0, 40.0, "RIGHT").flagged


def test_missing_heading_is_not_flagged():
    assert not col.flag_pair(np.nan, 40.0, "FORWARD").flagged
    assert not col.flag_pair(0.0, np.nan, "FORWARD").flagged


def test_flag_drive_skips_the_first_frame():
    psi = np.array([0.0, 2.0, 40.0, 41.0])
    flags, delta = col.flag_drive(psi, ["FORWARD"] * 4)
    assert flags.tolist() == [False, False, True, False]
    assert np.isnan(delta[0])


def test_best_psi_falls_back_to_the_scan_then_the_vp_cue():
    both = hd.HeadingEstimate(psi=5.0, valid=True, psi_scan=4.0, psi_vp=6.0)
    assert col.best_psi(both) == 5.0
    conflict = hd.HeadingEstimate(psi=np.nan, valid=False, psi_scan=4.0, psi_vp=30.0)
    assert col.best_psi(conflict) == 4.0
    only_vp = hd.HeadingEstimate(psi=np.nan, valid=False, psi_scan=np.nan, psi_vp=30.0)
    assert col.best_psi(only_vp) == 30.0


def test_detector_module_does_not_touch_ground_truth():
    src = (SRC / "collision.py").read_text().split('"""', 2)[-1]
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    forbidden = r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene|occupancy"
    assert not re.search(forbidden, code)


def test_leakage_same_flags_without_ground_truth_files(tmp_path):
    """rule 10: copy a drive without poses.json/gt.json; outputs must be identical."""
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
    assert not (dst / "poses.json").exists()
    a = col.detect_drive(src, limit=n)
    b = col.detect_drive(dst, limit=n)
    assert np.array_equal(a[0], b[0])
    assert np.allclose(a[1], b[1], equal_nan=True)
    assert np.allclose(a[2], b[2], equal_nan=True)


# ---------------------------------------------------------------------------- the rule


def test_rule_needs_both_measurements_to_agree():
    # heading says collision, matches show a small rotation: vetoed
    assert not col.decide(40.0, 5.0, 12, "FORWARD")
    # heading says collision, matches confirm
    assert col.decide(40.0, -38.0, 12, "FORWARD")
    # heading says collision, too few matches to judge: kept
    assert col.decide(40.0, np.nan, 1, "FORWARD")
    # heading says nothing, matches are strong: rescued
    assert col.decide(5.0, 36.0, 10, "IDLE")
    # ... but not when they are weak
    assert not col.decide(5.0, 36.0, 5, "IDLE")
    # a turn label is never a collision
    assert not col.decide(40.0, 38.0, 12, "LEFT")


def test_unresolved_pairs_are_flagged_conservatively():
    # no valid heading on a frame and too few matches: cannot rule a collision out
    assert col.decide(np.nan, np.nan, 0, "FORWARD", heading_ok=False)
    # with a valid heading the same pair is judged normally (no flag)
    assert not col.decide(3.0, np.nan, 0, "FORWARD", heading_ok=True)
    # matches that can judge override the unresolved rule
    assert not col.decide(np.nan, 4.0, 12, "FORWARD", heading_ok=False)


def test_decide_drive_uses_frame_validity():
    psi = np.array([0.0, 2.0, 3.0])
    rot = np.array([np.nan, np.nan, np.nan])
    n = np.zeros(3, int)
    valid = np.array([True, False, True])
    flags, _ = col.decide_drive(psi, rot, n, ["FORWARD"] * 3, valid)
    assert flags.tolist() == [False, True, True]  # both pairs touch the invalid frame
