"""SE(2) poses ``(x, y, theta)``: composition, inverse, relative pose, wrapping, Jacobians.

Metres and radians. Angles are wrapped to (-pi, pi]. Every function accepts a single pose
of shape (3,) or a batch of shape (n, 3).
"""

from __future__ import annotations

import numpy as np


def wrap(a):
    """Wrap an angle (or array) to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    w = np.pi - np.mod(np.pi - a, 2 * np.pi)  # in (-pi, pi]
    return w if w.ndim else float(w)


def compose(a, b):
    """``a (+) b``: pose ``b``, given in the frame of ``a``, expressed in ``a``'s parent frame."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack(
        [
            a[..., 0] + c * b[..., 0] - s * b[..., 1],
            a[..., 1] + s * b[..., 0] + c * b[..., 1],
            wrap(a[..., 2] + b[..., 2]),
        ],
        axis=-1,
    )


def inverse(a):
    """The pose ``a^-1`` with ``a (+) a^-1 = identity``."""
    a = np.asarray(a, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack(
        [
            -(c * a[..., 0] + s * a[..., 1]),
            s * a[..., 0] - c * a[..., 1],
            wrap(-a[..., 2]),
        ],
        axis=-1,
    )


def relative(a, b):
    """``a^-1 (+) b``: pose ``b`` as seen from ``a``."""
    return compose(inverse(a), b)


def compose_jacobians(a, b):
    """Jacobians of ``a (+) b`` with respect to ``a`` and ``b`` (3 x 3 each, single poses)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[2]), np.sin(a[2])
    ja = np.array([[1, 0, -s * b[0] - c * b[1]], [0, 1, c * b[0] - s * b[1]], [0, 0, 1]])
    jb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return ja, jb


def inverse_jacobian(a):
    a = np.asarray(a, float)
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        [
            [-c, -s, s * a[0] - c * a[1]],
            [s, -c, c * a[0] + s * a[1]],
            [0, 0, -1],
        ]
    )


def relative_jacobians(a, b):
    """Jacobians of ``a^-1 (+) b`` with respect to ``a`` and ``b``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    ja = np.array([[-c, -s, -s * dx + c * dy], [s, -c, -c * dx - s * dy], [0, 0, -1]])
    jb = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return ja, jb


def numerical_jacobian(f, x, eps: float = 1e-6):
    """Central-difference Jacobian of a pose-valued function; angle outputs are wrapped
    on the difference so the +-pi cut does not leak in."""
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        diff = np.asarray(f(x + d)) - np.asarray(f(x - d))
        diff[2] = wrap(diff[2])
        cols.append(diff / (2 * eps))
    return np.stack(cols, axis=1)


def integrate(start, increments):
    """Dead-reckon: compose ``start`` with each relative increment in turn. Returns the pose
    after every increment, shape (n, 3)."""
    out = np.empty((len(increments), 3))
    p = np.asarray(start, float)
    for k, inc in enumerate(increments):
        p = compose(p, inc)
        out[k] = p
    return out
