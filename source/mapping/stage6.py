"""Stage 6 evaluation driver: pose-graph optimisation of all three drives of a seed with the
stage 5b union closures (reads ground truth only to score).

  python3 source/mapping/stage6.py run dev|test [--inject random|alias]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posegraph as pg  # noqa: E402
import se2  # noqa: E402
import stage5b as S  # noqa: E402
import texture as tx  # noqa: E402

ROOT = S.ROOT
OUT = ROOT / "cache" / "mapping" / "stage6"
SPAWN = np.array([0.2, 2.6, 0.0])
P5B = dict(corr_min=0.8, corr_veto=0.2, pos_tol=0.05, min_inl_boundary=12)  # frozen in stage 5b
THIN = 5  # frames per cluster window
GATE_FLOOR = np.array([0.02, 0.02, np.radians(1.0)])  # post-optimisation relative uncertainty
CHI2_99 = 11.34
CLOSURE_FLOOR = (0.005, 0.005, np.radians(0.3))
ODO_INFL = 16.0  # variance inflation of the odometry increments (chosen on dev)
IRLS_C = 3.0  # Cauchy scale on whitened closure residuals
IRLS_ROUNDS = 4
YAW_EVERY = 10  # absolute heading prior every N frames (0 = off)
YAW_FLOOR = np.radians(1.0)


def drives_odometry(prefix="gt4"):
    S.odometry(prefix)  # stubs the SDK modules
    name = "official" if prefix == "gt" else prefix
    d = pickle.loads((ROOT / "cache" / "mapping" / f"stage4b_results_{name}.pkl").read_bytes())
    return {(k[0], k[1]): v for k, v in d.items() if k[2] == "C"}


def true_poses(seed, prefix="gt4"):
    d = ROOT / "data" / f"{prefix}-{seed}"
    return [np.array(json.loads((d / f"traj_{i}" / "poses.json").read_text())) for i in range(3)]


def labels(seed, prefix="gt4"):
    d = ROOT / "data" / f"{prefix}-{seed}"
    return [[r["action"][0] for r in json.loads((d / f"traj_{i}" / "data_info.json").read_text())] for i in range(3)]


def closures(seed, prefix="gt4", P=None):
    """The stage 5b union rule (gate without contradiction, or boundary-verified)."""
    B = pickle.loads((S.OUT / f"boundaries_{prefix}_{seed}.pkl").read_bytes())
    rows, _ = S.rows_for(seed, prefix)
    out = []
    for r in rows:
        p = r["poses"].get("c") if r.get("poses") else None
        if p is None or not S.in_envelope(p) or r["inliers"] < P5B["min_inl_boundary"]:
            continue
        m, c = tx.compare_boundaries(B[r["a"][0]][r["a"][1]], B[r["b"][0]][r["b"][1]], p, P5B["corr_min"], P5B["corr_veto"], P5B["pos_tol"])
        gate = r["inliers"] >= S.MIN_INLIERS and S.gate_ok(r, P, seed)
        kind = "boundary" if m >= 1 else ("gate" if gate and c == 0 else None)
        if kind is None:
            continue
        ok = "c" in r["err"] and r["err"]["c"][0] < S.POSE_OK_M and r["err"]["c"][1] < S.POSE_OK_DEG
        out.append(dict(a=r["a"], b=r["b"], z=p, cov=r["poses"]["c_cov"], inliers=r["inliers"], kind=kind, ok=ok, gate_ok=gate))
    return out


def aliases(seed, prefix="gt4", P=None):
    """Stage 5 gated closures with a wrong pose (the 0.4 to 0.6 m aliases), for the robustness test."""
    rows, _ = S.rows_for(seed, prefix)
    out = []
    for r in rows:
        p = r["poses"].get("c") if r.get("poses") else None
        if p is None or r["inliers"] < S.MIN_INLIERS or not S.in_envelope(p) or not S.gate_ok(r, P, seed):
            continue
        if "c" in r["err"] and r["err"]["c"][0] < S.POSE_OK_M and r["err"]["c"][1] < S.POSE_OK_DEG:
            continue
        out.append(dict(a=r["a"], b=r["b"], z=p, cov=r["poses"]["c_cov"], inliers=r["inliers"], kind="alias", ok=False, gate_ok=True))
    return out


def thin(cl):
    best = {}
    for c in cl:
        key = (c["a"][0], c["a"][1] // THIN, c["b"][0], c["b"][1] // THIN)
        if key not in best or c["inliers"] > best[key]["inliers"]:
            best[key] = c
    return list(best.values())


def build(seed, odo, cl, prefix="gt4"):
    lens = [len(odo[(seed, i)].poses) for i in range(3)]
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    g = pg.Graph(int(sum(lens)), anchor=SPAWN.copy())
    for i in range(3):
        v = odo[(seed, i)]
        P = v.poses
        g.add(-1, int(off[i]), se2.relative(SPAWN, P[0]), ODO_INFL * v.inc_cov[0], "prior")
        for k in range(1, len(P)):
            g.add(int(off[i] + k - 1), int(off[i] + k), se2.relative(P[k - 1], P[k]), ODO_INFL * v.inc_cov[k], "odo")
        if YAW_EVERY:
            # absolute heading from the stage 4 filter (corrected by the stage 3 compass, which is
            # drift free modulo 90 deg): ties the map's orientation to the maze axes
            for k in range(0, len(P), YAW_EVERY):
                sd = max(np.sqrt(max(v.covs[k][2, 2], 0.0)), YAW_FLOOR)
                g.add_raw(-1, int(off[i] + k), se2.relative(SPAWN, P[k]), np.diag([0.0, 0.0, 1.0 / sd]), "yaw")
    for c in cl:
        g.add(int(off[c["a"][0]] + c["a"][1]), int(off[c["b"][0]] + c["b"][1]), c["z"], c["cov"], "loop", tag=c, floor=CLOSURE_FLOOR)
    x0 = np.concatenate([odo[(seed, i)].poses for i in range(3)])
    return g, x0, off, lens


def irls(g, x0):
    """Robust (Cauchy) weights on loop edges only, by iteratively reweighted least squares;
    odometry and prior edges stay Gaussian."""
    loops = [e for e in g.edges if e.kind == "loop" and e.active]
    base = {id(e): e.sqrt_info.copy() for e in loops}
    X = np.asarray(x0, float)

    def reweight(X):
        for e in loops:
            e.sqrt_info = base[id(e)]
            r = g.residuals(X.ravel(), [e])
            w = 1.0 / (1.0 + (r @ r) / IRLS_C ** 2)
            e.sqrt_info = np.sqrt(w) * base[id(e)]

    # weights first, at the starting estimate: solving with full weights first lets a cluster
    # of mutually consistent false closures pull the graph to itself, after which every later
    # round sees them as consistent (stage 6, seed 22)
    reweight(X)
    for _ in range(IRLS_ROUNDS):
        X, _ = g.optimise(X, loss="linear")
        reweight(X)
    for e in loops:
        e.sqrt_info = base[id(e)]
    return X


def two_pass(g, x0):
    X1 = irls(g, x0)
    dropped = 0
    for e in g.edges:
        if e.kind != "loop":
            continue
        a, b = X1[e.i], X1[e.j]
        d = se2.relative(a, b) - e.z
        d[2] = se2.wrap(d[2])
        cov = np.linalg.inv(e.sqrt_info.T @ e.sqrt_info) + np.diag(GATE_FLOOR ** 2)
        chi2 = float(d @ np.linalg.solve(cov, d))
        e.active = chi2 < CHI2_99
        dropped += not e.active
    X2 = irls(g, X1)
    return X1, X2, dropped


def umeyama_scale(est, true):
    mu_e, mu_t = est.mean(0), true.mean(0)
    E, T = est - mu_e, true - mu_t
    U, D, Vt = np.linalg.svd(T.T @ E / len(est))
    Sg = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        Sg[1, 1] = -1
    var_e = (E ** 2).sum() / len(est)
    return float(np.trace(np.diag(D) @ Sg) / var_e)


def straight_segments(true, est, lab, min_len=10):
    out = []
    k = 1
    while k < len(lab):
        if lab[k] != "FORWARD":
            k += 1
            continue
        j = k
        while j < len(lab) and lab[j] == "FORWARD":
            j += 1
        a, b = k - 1, j - 1
        if b - a >= min_len and abs(se2.wrap(true[b, 2] - true[a, 2])) < np.radians(2) and np.hypot(*(true[b, :2] - true[a, :2])) > 0.3:
            out.append(np.hypot(*(est[b, :2] - est[a, :2])) / np.hypot(*(true[b, :2] - true[a, :2])) - 1)
        k = j
    return out


def rigid_ate(est, true):
    """RMS position error after the best rigid (rotation + translation) alignment."""
    mu_e, mu_t = est.mean(0), true.mean(0)
    E, T = est - mu_e, true - mu_t
    U, _, Vt = np.linalg.svd(T.T @ E)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    aligned = (R @ E.T).T + mu_t
    return float(np.sqrt(((aligned - true) ** 2).sum(1).mean())), float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def metrics(X, true, off, lens, lab):
    per = []
    segs = []
    for i in range(3):
        e = X[off[i]: off[i] + lens[i]]
        t = true[i]
        d = e - t
        d[:, 2] = se2.wrap(d[:, 2])
        per.append(dict(ate=float(np.sqrt((d[:, :2] ** 2).sum(1).mean())), head_rms_deg=float(np.degrees(np.sqrt((d[:, 2] ** 2).mean()))),
                        scale=umeyama_scale(e[:, :2], t[:, :2])))
        segs += straight_segments(t, e, lab[i])
    allX, allT = X, np.concatenate(true)
    d = allX - allT
    d[:, 2] = se2.wrap(d[:, 2])
    ra, rot = rigid_ate(allX[:, :2], allT[:, :2])
    return dict(ate_rigid=ra, align_rot_deg=rot, ate_joint=float(np.sqrt((d[:, :2] ** 2).sum(1).mean())), head_rms_joint_deg=float(np.degrees(np.sqrt((d[:, 2] ** 2).mean()))), drives=per, segments=segs)


def run_seed(seed, prefix="gt4", inject=None, rng=None):
    odo = drives_odometry(prefix)
    P = {k: v.poses for k, v in odo.items()}
    cl = thin(closures(seed, prefix, P))
    if inject == "alias":
        cl += aliases(seed, prefix, P)
    if inject == "random":
        rng = rng or np.random.default_rng(seed)
        lens = [len(odo[(seed, i)].poses) for i in range(3)]
        for _ in range(5):
            da, db = rng.integers(0, 3, 2)
            cl.append(dict(a=(int(da), int(rng.integers(lens[da]))), b=(int(db), int(rng.integers(lens[db]))),
                           z=np.r_[rng.uniform(-0.4, 0.4, 2), rng.uniform(-0.4, 0.4)], cov=np.diag([1e-4, 1e-4, 1e-4]),
                           inliers=99, kind="random", ok=False, gate_ok=False))
    g, x0, off, lens = build(seed, odo, cl, prefix)
    t0 = time.time()
    X1, X2, dropped = two_pass(g, x0)
    dt = time.time() - t0
    true = true_poses(seed, prefix)
    lab = labels(seed, prefix)
    loops = [e for e in g.edges if e.kind == "loop"]
    res = dict(seed=seed, inject=inject, n_nodes=g.n, n_closures=len(loops), n_dropped=dropped,
               false_closures=sum(not e.tag["ok"] for e in loops), false_active=sum((not e.tag["ok"]) and e.active for e in loops),
               true_dropped=sum(e.tag["ok"] and not e.active for e in loops),
               kinds={k: sum(e.tag["kind"] == k for e in loops) for k in ("gate", "boundary", "alias", "random")},
               runtime_s=dt, before=metrics(x0, true, off, lens, lab), pass1=metrics(X1, true, off, lens, lab), after=metrics(X2, true, off, lens, lab))
    return res, X2, off, lens


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        for infl in (1.0, 4.0, 16.0, 64.0, 256.0):
            ODO_INFL = infl
            rs = [run_seed(s)[0] for s in S.DEV]
            print(f"infl {infl:6.0f}: ATE " + " ".join(f"{r['before']['ate_joint']:.3f}->{r['after']['ate_joint']:.3f}" for r in rs)
                  + f" | dropped {[r['n_dropped'] for r in rs]} true_dropped {[r['true_dropped'] for r in rs]} false_active {[r['false_active'] for r in rs]}", flush=True)
        sys.exit()
    split = sys.argv[2]
    inject = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--inject" else None
    seeds = S.DEV if split == "dev" else S.TEST
    OUT.mkdir(parents=True, exist_ok=True)
    allr = []
    for s in seeds:
        r, X, off, lens = run_seed(s, inject=inject)
        allr.append(r)
        np.save(OUT / f"poses_gt4_{s}{'_' + inject if inject else ''}.npy", X)
        b, a = r["before"], r["after"]
        print(f"seed {s}: closures {r['n_closures']} {r['kinds']} dropped {r['n_dropped']} false {r['false_closures']} false_active {r['false_active']} true_dropped {r['true_dropped']} | "
              f"ATE {b['ate_joint']:.3f} -> {r['pass1']['ate_joint']:.3f} -> {a['ate_joint']:.3f} m | head {b['head_rms_joint_deg']:.2f} -> {a['head_rms_joint_deg']:.2f} deg | "
              f"scale {[round(d['scale'] - 1, 4) for d in a['drives']]} | {r['runtime_s']:.1f} s", flush=True)
    (OUT / f"results_gt4_{split}{'_' + inject if inject else ''}.json").write_text(json.dumps(allr, indent=1, default=float))
