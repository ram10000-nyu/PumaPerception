"""Stage 6: pose-graph toy loops (plan acceptance) and Jacobian check."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import posegraph as pg  # noqa: E402
import se2  # noqa: E402

SQUARE = np.array([[0, 0, 0], [1, 0, np.pi / 2], [1, 1, np.pi], [0, 1, -np.pi / 2]], float)
STEP = np.array([1.0, 0.0, np.pi / 2])  # each side of the square, in the robot frame


def _graph(drift_edge=None, drift=0.0):
    g = pg.Graph(4, anchor=SQUARE[0].copy())
    cov = np.diag([1e-4, 1e-4, 1e-4])
    g.add(-1, 0, np.zeros(3), np.diag([1e-8] * 3), "prior")
    for k in range(3):
        z = STEP.copy()
        if k == drift_edge:
            z[2] += drift
        g.add(k, k + 1, z, cov)
    g.add(3, 0, STEP, cov, "loop")  # closure: node 0 seen from node 3
    return g


def test_exact_square_from_perturbed_start():
    g = _graph()
    x0 = SQUARE + np.random.default_rng(0).normal(scale=[0.1, 0.1, 0.1], size=(4, 3))
    X, _ = g.optimise(x0, loss="linear")
    d = X - SQUARE
    d[:, 2] = se2.wrap(d[:, 2])
    assert np.abs(d).max() < 1e-6


def test_drift_is_spread_over_the_loop():
    g = _graph(drift_edge=1, drift=0.1)
    X, _ = g.optimise(SQUARE, loss="linear")
    r = g.residuals(X.ravel()).reshape(-1, 3)
    # equal weights: least squares spreads the 0.1 rad drift over all four edges, closure included
    assert abs(r[-1, 2] / 100.0) < 0.1 / 3
    rot = np.abs(r[1:, 2])  # odometry and closure rotation residuals
    assert rot.max() <= 0.5 * rot.sum() + 1e-9  # no single edge carries more than half


def test_jacobian_matches_numerical():
    g = _graph(drift_edge=0, drift=0.05)
    x = (SQUARE + 0.03).ravel()
    J = g.jacobian(x).toarray()
    eps = 1e-6
    num = np.stack([(g.residuals(x + eps * np.eye(12)[k]) - g.residuals(x - eps * np.eye(12)[k])) / (2 * eps) for k in range(12)], 1)
    assert np.allclose(J, num, atol=1e-4)
