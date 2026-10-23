"""Stages 5b and 6 as pipeline functions (rule 10: images and labels only): the union rule for
loop closures and the robust pose graph. ``stage6.py`` is the evaluation driver that chose
these parameters; this module repeats its steps on raw inputs for ``build_map.py``.

Closures (stage 5b union rule): a VLAD candidate pair with at least ``MIN_INL_BOUNDARY``
homography inliers whose method (c) relative pose is a revisit is a closure when
  - a texture boundary is seen in both frames at the same place (``texture.compare_boundaries``),
    or
  - it has at least ``loops.MIN_INLIERS`` inliers, passes the odometry gate (and the ICP
    agreement check) and no boundary contradicts it.

Pose graph (stage 6): odometry edges (covariance inflated ``ODO_INFL`` times), absolute heading
priors every ``YAW_EVERY`` frames, loop edges thinned to one per (drive, 5-frame window) pair,
Cauchy IRLS on loop edges only (weights first, at the odometry estimate), then a chi-square
re-check of every closure against the solution and a second solve.
"""

from __future__ import annotations

import numpy as np

import loops
import posegraph as pg
import se2
import texture as tx

SPAWN = np.array([0.2, 2.6, 0.0])  # the anchor (local recordings' spawn; any frame works)
P5B = dict(corr_min=0.8, corr_veto=0.2, pos_tol=0.05)  # frozen in stage 5b
MIN_INL_BOUNDARY = 12
MAX_REL_M, MAX_REL_DEG = 0.45, 27.0
THIN = 5
GATE_FLOOR = np.array([0.02, 0.02, np.radians(1.0)])
CHI2_99 = 11.34
CLOSURE_FLOOR = (0.005, 0.005, np.radians(0.3))
ODO_INFL = 16.0
IRLS_C = 3.0
IRLS_ROUNDS = 4
YAW_EVERY = 10
YAW_FLOOR = np.radians(1.0)


def _in_envelope(p) -> bool:
    return bool(np.hypot(p[0], p[1]) < MAX_REL_M and abs(np.degrees(p[2])) < MAX_REL_DEG)


def _gate_ok(pc, pb, rel_odo) -> bool:
    if pb is not None:
        d = pb - pc
        if np.hypot(d[0], d[1]) > loops.AGREE_M or abs(np.degrees(se2.wrap(d[2]))) > loops.AGREE_DEG:
            return False
    d = pc - rel_odo
    return bool(np.hypot(d[0], d[1]) < loops.GATE_M and abs(np.degrees(se2.wrap(d[2]))) < loops.GATE_DEG)


_W: dict = {}


def _init(features, boundaries, scans, poses):
    import cv2

    cv2.setNumThreads(1)
    _W.update(features=features, boundaries=boundaries, scans=scans, poses=poses, matcher=cv2.BFMatcher())


def _judge(pairs):
    """The union rule on candidate pairs ``((da, ka), (db, kb))`` (worker side)."""
    F, B, S, P, matcher = _W["features"], _W["boundaries"], _W["scans"], _W["poses"], _W["matcher"]
    out = []
    for (da, ka), (db, kb) in pairs:
        n, _H, m = loops.verify(F[da][ka], F[db][kb], matcher)
        if n < MIN_INL_BOUNDARY:
            continue
        sa, sb = S[da][ka], S[db][kb]
        pc = loops.pose_from_features(m, sa, sb)
        if pc is None or not _in_envelope(pc.mean):
            continue
        pb = loops.pose_from_icp(sa, sb, pc.mean)
        pb = None if pb is None else pb.mean
        match, contra = tx.compare_boundaries(B[da][ka], B[db][kb], pc.mean, **P5B)
        gate = n >= loops.MIN_INLIERS and _gate_ok(pc.mean, pb, se2.relative(P[da][ka], P[db][kb]))
        kind = "boundary" if match >= 1 else ("gate" if gate and contra == 0 else None)
        if kind is not None:
            out.append(dict(a=(da, ka), b=(db, kb), z=pc.mean, cov=pc.cov, inliers=n, kind=kind))
    return out


def closures(features, boundaries, scans, poses, vocab=None, workers: int = 1) -> list[dict]:
    """``features[d][k]`` (``loops.Features``), ``boundaries[d][k]`` (``texture.boundaries``),
    ``scans[d][k]`` (``vo.ScanPoints``) and the odometry ``poses[d]`` (one frame for all drives).
    Returns closures ``{a, b, z, cov, inliers, kind}`` with ``a``/``b`` = (drive, frame).
    ``workers`` > 1 splits the candidate pairs across processes (same result)."""
    vocab = vocab or loops.Vocabulary.load()
    flat = [f for d in features for f in d]
    drive = np.concatenate([np.full(len(d), i) for i, d in enumerate(features)])
    index = np.concatenate([np.arange(len(d)) for d in features])
    g = vocab.encode(flat)
    pairs = [((int(drive[i]), int(index[i])), (int(drive[j]), int(index[j]))) for i, j, _ in loops.candidates(g, drive, index)]
    if workers <= 1:
        _init(features, boundaries, scans, poses)
        return _judge(pairs)
    from concurrent.futures import ProcessPoolExecutor

    chunks = [pairs[w::workers * 4] for w in range(workers * 4)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(features, boundaries, scans, poses)) as ex:
        res = list(ex.map(_judge, chunks))
    out = [c for r in res for c in r]
    return sorted(out, key=lambda c: (c["a"], c["b"]))


def thin(cl):
    best = {}
    for c in cl:
        key = (c["a"][0], c["a"][1] // THIN, c["b"][0], c["b"][1] // THIN)
        if key not in best or c["inliers"] > best[key]["inliers"]:
            best[key] = c
    return list(best.values())


def build(odo, cl, anchor=SPAWN):
    """Graph of all drives: ``odo[d]`` is an ``odometry.DriveResult`` (poses, inc_cov, covs)."""
    lens = [len(v.poses) for v in odo]
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    g = pg.Graph(int(sum(lens)), anchor=np.asarray(anchor, float).copy())
    for i, v in enumerate(odo):
        P = v.poses
        g.add(-1, int(off[i]), se2.relative(anchor, P[0]), ODO_INFL * v.inc_cov[0], "prior")
        for k in range(1, len(P)):
            g.add(int(off[i] + k - 1), int(off[i] + k), se2.relative(P[k - 1], P[k]), ODO_INFL * v.inc_cov[k], "odo")
        if YAW_EVERY:
            for k in range(0, len(P), YAW_EVERY):
                sd = max(np.sqrt(max(v.covs[k][2, 2], 0.0)), YAW_FLOOR)
                g.add_raw(-1, int(off[i] + k), se2.relative(anchor, P[k]), np.diag([0.0, 0.0, 1.0 / sd]), "yaw")
    for c in cl:
        g.add(int(off[c["a"][0]] + c["a"][1]), int(off[c["b"][0]] + c["b"][1]), c["z"], c["cov"], "loop", tag=c, floor=CLOSURE_FLOOR)
    x0 = np.concatenate([v.poses for v in odo])
    return g, x0, off, lens


def irls(g, x0):
    edges = [e for e in g.edges if e.kind == "loop" and e.active]
    base = {id(e): e.sqrt_info.copy() for e in edges}
    X = np.asarray(x0, float)

    def reweight(X):
        for e in edges:
            e.sqrt_info = base[id(e)]
            r = g.residuals(X.ravel(), [e])
            w = 1.0 / (1.0 + (r @ r) / IRLS_C ** 2)
            e.sqrt_info = np.sqrt(w) * base[id(e)]

    reweight(X)  # weights first, at the starting estimate (stage 6, seed 22)
    for _ in range(IRLS_ROUNDS):
        X, _ = g.optimise(X, loss="linear")
        reweight(X)
    for e in edges:
        e.sqrt_info = base[id(e)]
    return X


def two_pass(g, x0):
    X1 = irls(g, x0)
    dropped = 0
    for e in g.edges:
        if e.kind != "loop":
            continue
        d = se2.relative(X1[e.i], X1[e.j]) - e.z
        d[2] = se2.wrap(d[2])
        cov = np.linalg.inv(e.sqrt_info.T @ e.sqrt_info) + np.diag(GATE_FLOOR ** 2)
        e.active = float(d @ np.linalg.solve(cov, d)) < CHI2_99
        dropped += not e.active
    return X1, irls(g, X1), dropped


def optimise(odo, cl, anchor=SPAWN):
    """Optimised poses per drive, and diagnostics."""
    cl = thin(cl)
    g, x0, off, lens = build(odo, cl, anchor)
    _, X, dropped = two_pass(g, x0)
    X = X.reshape(-1, 3)
    poses = [X[off[i]: off[i] + lens[i]] for i in range(len(odo))]
    return poses, {"closures": len(cl), "dropped": int(dropped), "by_kind": {k: sum(c["kind"] == k for c in cl) for k in ("boundary", "gate")}}
