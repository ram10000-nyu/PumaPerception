"""Stage 1: the motion prior on synthetic data, and against the fitted dev prior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import motion as M  # noqa: E402
import se2  # noqa: E402

RNG = np.random.default_rng(1)


def synthetic_pairs(n=3000):
    """Two labels with distinct, bimodal motions; the previous label shifts the mean."""
    labels = RNG.choice(["FORWARD", "LEFT"], n)
    prev = RNG.choice(["FORWARD", "LEFT"], n)
    inc = np.zeros((n, 3))
    f = labels == "FORWARD"
    inc[f, 0] = np.where(RNG.random(f.sum()) < 0.8, 0.15, 0.05) + RNG.normal(0, 0.005, f.sum())
    inc[~f, 2] = 0.2 + 0.05 * (prev[~f] == "LEFT") + RNG.normal(0, 0.03, (~f).sum())
    inc[:, 1] = RNG.normal(0, 0.001, n)
    return {
        "inc": inc,
        "label": labels,
        "prev": prev,
        "next": labels,
        "seed": np.zeros(n, int),
        "drive": np.zeros(n, int),
        "k": np.arange(n),
    }


def test_mixture_finds_both_modes():
    p = synthetic_pairs()
    g = M.fit_group(p["inc"][p["label"] == "FORWARD"], "label")
    assert len(g.weights) >= 2
    assert sorted(np.round(g.means[:, 0], 2))[-1] == pytest.approx(0.15, abs=0.01)
    assert g.mean[0] == pytest.approx(0.13, abs=0.01)


def test_hdr_covers_about_95_percent_of_held_out_data():
    a, b = synthetic_pairs(6000), synthetic_pairs(6000)
    g = M.fit_group(a["inc"][a["label"] == "FORWARD"], "label")
    cov = g.contains(b["inc"][b["label"] == "FORWARD"]).mean()
    assert 0.92 < cov < 0.98


def test_pair_prior_is_used_and_falls_back():
    p = synthetic_pairs()
    prior = M.MotionPrior.fit(p, min_group=100)
    assert prior.get("LEFT", "LEFT").source == "pair"
    assert prior.get("LEFT", "LEFT").mean[2] > prior.get("LEFT", "FORWARD").mean[2] + 0.03
    assert prior.get("LEFT", "START").source == "label"
    assert prior.get("LEFT").source == "label"


def test_save_load_roundtrip(tmp_path):
    prior = M.MotionPrior.fit(synthetic_pairs())
    f = tmp_path / "p.json"
    prior.save(f)
    back = M.MotionPrior.load(f)
    x = np.array([[0.15, 0.0, 0.0], [0.0, 0.0, 0.2]])
    for key in prior.by_pair:
        assert np.allclose(prior.by_pair[key].logpdf(x), back.by_pair[key].logpdf(x))
    assert json.loads(f.read_text())["meta"]["seeds"] == [0]


def test_dy_and_angle_are_never_degenerate():
    x = np.zeros((200, 3))
    x[:, 0] = RNG.normal(0.1, 0.01, 200)  # dy and dtheta exactly constant
    g = M.fit_group(x, "label")
    assert np.all(np.linalg.eigvalsh(g.cov) > 0)
    assert np.isfinite(g.logpdf(x)).all()


def test_mean_increments_use_previous_label():
    prior = M.MotionPrior.fit(synthetic_pairs())
    inc = prior.mean_increments(["LEFT", "LEFT", "FORWARD"])
    assert inc.shape == (3, 3)
    assert inc[1, 2] > inc[0, 2]  # LEFT after LEFT turns more than LEFT after START


def test_dead_reckoning_with_exact_increments_is_exact():
    inc = np.array([[0.1, 0, 0.2]] * 20)
    est = se2.integrate([0.2, 2.6, 0], inc)
    p = np.array([0.2, 2.6, 0])
    for i in inc:
        p = se2.compose(p, i)
    assert np.allclose(est[-1], p)


def test_fitted_prior_on_real_data_if_present():
    if not M.PRIOR_PATH.exists():
        pytest.skip("run uv run source/mapping/stage1.py first")
    prior = M.MotionPrior.load()
    assert set(prior.by_label) == set(M.LABELS)
    assert prior.get("FORWARD", "FORWARD").mean[0] == pytest.approx(0.149, abs=0.01)
    assert prior.get("LEFT", "FORWARD").mean[2] > 0
    assert prior.get("RIGHT", "FORWARD").mean[2] < 0
