"""Stage 1: SE(2) algebra."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import se2  # noqa: E402

RNG = np.random.default_rng(0)


def rand_pose():
    return np.array([*RNG.uniform(-3, 3, 2), RNG.uniform(-np.pi, np.pi)])


def same_pose(p, q, tol=1e-9):
    return np.allclose(p[:2], q[:2], atol=tol) and abs(se2.wrap(p[2] - q[2])) < tol


def test_inverse_composes_to_identity():
    for _ in range(50):
        a = rand_pose()
        assert same_pose(se2.compose(se2.inverse(a), a), np.zeros(3))
        assert same_pose(se2.compose(a, se2.inverse(a)), np.zeros(3))


def test_composition_is_associative():
    for _ in range(50):
        a, b, c = rand_pose(), rand_pose(), rand_pose()
        assert same_pose(se2.compose(se2.compose(a, b), c), se2.compose(a, se2.compose(b, c)))


def test_relative_recovers_the_increment():
    for _ in range(50):
        a, d = rand_pose(), rand_pose()
        assert same_pose(se2.relative(a, se2.compose(a, d)), d)


def test_wrap_at_pi():
    assert se2.wrap(np.pi) == pytest.approx(np.pi)
    assert se2.wrap(-np.pi) == pytest.approx(np.pi)  # (-pi, pi]
    assert se2.wrap(np.pi + 1e-9) == pytest.approx(-np.pi + 1e-9)
    assert se2.wrap(3 * np.pi) == pytest.approx(np.pi)
    assert se2.wrap(0.3 - 4 * np.pi) == pytest.approx(0.3)
    w = se2.wrap(np.linspace(-20, 20, 1001))
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)


def test_composition_wraps_heading():
    p = se2.compose([0, 0, 3.0], [0, 0, 0.5])
    assert p[2] == pytest.approx(3.5 - 2 * np.pi)


def test_batch_matches_single():
    a = np.array([rand_pose() for _ in range(5)])
    b = np.array([rand_pose() for _ in range(5)])
    batch = se2.compose(a, b)
    for i in range(5):
        assert same_pose(batch[i], se2.compose(a[i], b[i]))


def test_jacobians_match_numerical():
    for _ in range(30):
        a, b = rand_pose(), rand_pose()
        ja, jb = se2.compose_jacobians(a, b)
        assert np.allclose(
            ja, se2.numerical_jacobian(lambda x, b=b: se2.compose(x, b), a), atol=1e-6
        )
        assert np.allclose(
            jb, se2.numerical_jacobian(lambda x, a=a: se2.compose(a, x), b), atol=1e-6
        )
        ja, jb = se2.relative_jacobians(a, b)
        assert np.allclose(
            ja, se2.numerical_jacobian(lambda x, b=b: se2.relative(x, b), a), atol=1e-6
        )
        assert np.allclose(
            jb, se2.numerical_jacobian(lambda x, a=a: se2.relative(a, x), b), atol=1e-6
        )
        assert np.allclose(
            se2.inverse_jacobian(a), se2.numerical_jacobian(se2.inverse, a), atol=1e-6
        )


def test_jacobians_near_the_wrap_cut():
    a, b = np.array([1.0, 2.0, np.pi - 1e-7]), np.array([0.5, -0.2, 1e-3])
    ja, _ = se2.compose_jacobians(a, b)
    assert np.allclose(ja, se2.numerical_jacobian(lambda x, b=b: se2.compose(x, b), a), atol=1e-6)


def test_integrate_straight_and_turn():
    inc = [[0.1, 0, 0]] * 10
    assert same_pose(se2.integrate([0, 0, 0], inc)[-1], [1.0, 0, 0])
    inc = [[0.0, 0, np.pi / 2]] * 4
    assert same_pose(se2.integrate([1, 2, 0], inc)[-1], [1, 2, 0])
