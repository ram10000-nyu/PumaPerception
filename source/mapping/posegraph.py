"""Pose-graph optimisation on SE(2) (stage 6). Pipeline code (rule 10): odometry and closures in,
optimised poses out; no ground truth.

Nodes are poses (x, y, theta). Edges are relative-pose measurements ``z`` of node j in node i's
frame with a covariance; a node may also be tied to a fixed pose (the spawn) by a prior edge.
The residual of an edge is ``wrap(relative(x_i, x_j) - z)`` whitened by the measurement's
square-root information; ``scipy.optimize.least_squares`` minimises the robust sum with an
analytic sparse Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import se2
from scipy.optimize import least_squares


@dataclass
class Edge:
    i: int  # -1 = the fixed anchor pose
    j: int
    z: np.ndarray  # relative pose of j in i's frame
    sqrt_info: np.ndarray  # 3x3, W such that W^T W = cov^-1
    kind: str = "odo"  # "odo", "prior", "loop"
    active: bool = True
    tag: object = None


def sqrt_info(cov: np.ndarray, floor=(1e-4, 1e-4, 1e-4)) -> np.ndarray:
    c = np.asarray(cov, float)
    c = np.where(np.isfinite(c), c, 0.0)
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    c = (v * np.maximum(w, 0.0)) @ v.T + np.diag(np.square(floor))
    w, v = np.linalg.eigh(c)
    return (v / np.sqrt(w)).T  # W with W^T W = c^-1


@dataclass
class Graph:
    n: int
    anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    edges: list[Edge] = field(default_factory=list)

    def add(self, i, j, z, cov, kind="odo", tag=None, floor=(1e-4, 1e-4, 1e-4)):
        self.edges.append(Edge(i, j, np.asarray(z, float), sqrt_info(cov, floor), kind, True, tag))

    def add_raw(self, i, j, z, W, kind, tag=None):
        self.edges.append(Edge(i, j, np.asarray(z, float), np.asarray(W, float), kind, True, tag))

    def _active(self):
        return [e for e in self.edges if e.active]

    @staticmethod
    def _pack(edges):
        I = np.array([e.i for e in edges])
        J = np.array([e.j for e in edges])
        Z = np.stack([e.z for e in edges])
        W = np.stack([e.sqrt_info for e in edges])
        return I, J, Z, W

    def _ab(self, X, I, J):
        A = np.where((I < 0)[:, None], self.anchor[None, :], X[np.maximum(I, 0)])
        return A, X[J]

    def residuals(self, x: np.ndarray, edges=None, packed=None) -> np.ndarray:
        X = np.asarray(x, float).reshape(-1, 3)
        edges = self._active() if edges is None else edges
        I, J, Z, W = packed or self._pack(edges)
        A, B = self._ab(X, I, J)
        d = se2.relative(A, B) - Z
        d[:, 2] = se2.wrap(d[:, 2])
        return np.einsum("kij,kj->ki", W, d).ravel()

    def jacobian(self, x: np.ndarray, edges=None, packed=None):
        X = np.asarray(x, float).reshape(-1, 3)
        edges = self._active() if edges is None else edges
        I, J, Z, W = packed or self._pack(edges)
        A, B = self._ab(X, I, J)
        c, s_ = np.cos(A[:, 2]), np.sin(A[:, 2])
        dx, dy = B[:, 0] - A[:, 0], B[:, 1] - A[:, 1]
        n = len(I)
        ja = np.zeros((n, 3, 3))
        ja[:, 0, 0], ja[:, 0, 1], ja[:, 0, 2] = -c, -s_, -s_ * dx + c * dy
        ja[:, 1, 0], ja[:, 1, 1], ja[:, 1, 2] = s_, -c, -c * dx - s_ * dy
        ja[:, 2, 2] = -1
        jb = np.zeros((n, 3, 3))
        jb[:, 0, 0], jb[:, 0, 1] = c, s_
        jb[:, 1, 0], jb[:, 1, 1] = -s_, c
        jb[:, 2, 2] = 1
        WA, WB = W @ ja, W @ jb
        rr, cc = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        k = np.arange(n)[:, None, None]
        rows = [(3 * k + rr).ravel()]
        cols = [(3 * J[:, None, None] + cc).ravel()]
        vals = [WB.ravel()]
        m = I >= 0
        rows.append((3 * k[m] + rr).ravel())
        cols.append((3 * I[m][:, None, None] + cc).ravel())
        vals.append(WA[m].ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * self.n))

    def optimise(self, x0: np.ndarray, loss: str = "cauchy", f_scale: float = 3.0, max_nfev: int = 50):
        edges = self._active()
        pk = self._pack(edges)
        res = least_squares(
            lambda x: self.residuals(x, edges, pk), np.asarray(x0, float).ravel(),
            jac=lambda x: self.jacobian(x, edges, pk), loss=loss, f_scale=f_scale,
            method="trf", tr_solver="lsmr", x_scale="jac", max_nfev=max_nfev,
        )
        X = res.x.reshape(-1, 3)
        X[:, 2] = se2.wrap(X[:, 2])
        return X, res

    def edge_chi2(self, X: np.ndarray, kinds=("loop",)) -> list[tuple[Edge, float]]:
        out = []
        for e in self.edges:
            if e.kind in kinds:
                r = self.residuals(X.ravel(), [e])
                out.append((e, float(r @ r)))
        return out
