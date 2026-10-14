"""Visual odometry filter (stage 4): fuses the label prior, scan ICP and the heading cues.

Per consecutive frame pair ``(k-1, k)`` the relative motion is estimated in three steps:

1. a **prior** on the increment: variant C uses the stage 1 mixture for the (previous, current)
   label pair (moments for the fusion, the stage 3b extended density/region for gating);
   variant B uses a constant-velocity guess with a wide covariance;
2. a **measurement** from vision: scan-to-scan ICP (``vo.icp``) with its Hessian covariance,
   plus the rotation implied by the two frames' headings (modulo 90, unwrapped to the branch
   nearest the prior rotation). A measurement the label prior calls impossible (outside the
   99 % region of the mixture *including* the collision component) is rejected, never because of
   the normal component alone, so a FORWARD label cannot veto a large rotation that vision sees;
3. an information-weighted **fusion** of prior and measurement into one increment and covariance.

The increment moves an EKF on ``(x, y, theta)``; the frame's absolute heading (stage 3 fused
heading, unwrapped to the branch nearest the filter's heading) then updates ``theta`` through a
gate, which is what keeps a confidently wrong heading out.

Pipeline code: scans, headings, labels and the collision flags only (rule 10). The start pose is
the spawn constant of ``cache/mapping/spawn.json`` (a simulator constant).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import motion as M
import numpy as np
import se2
import vo

SPAWN_JSON = Path(__file__).resolve().parents[2] / "cache" / "mapping" / "spawn.json"
SPAWN = (
    tuple(json.loads(SPAWN_JSON.read_text())["pose"]) if SPAWN_JSON.exists() else (0.2, 2.6, 0.0)
)

B_PRIOR_SD = np.array([0.10, 0.05, 0.30])  # variant B: constant velocity, wide (m, m, rad)
B_GATE_TRANS = 0.30  # variant B: a measurement farther than this from the guess is rejected
B_GATE_ROT = np.radians(25.0)
MATCH_ROT_SD_DEG = 5.0  # its error: 3.6 degrees median, 4.8 at the 90th percentile on events
VP_SIGMA_DEG = 0.7  # heading measurement noise: median error 0.2 deg but a heavy tail (stage 3)
HEADING_GATE_CHI2 = 16.0  # innovation gate on theta (chi-square 1 dof, beyond 99.99 %)
REANCHOR_FRAMES = 3  # consecutive rejected headings with one constant offset: the filter is lost
REANCHOR_SPREAD_DEG = 3.0
REANCHOR_SD_DEG = 1.5
HEADING_GATE_MAX_DEG = 8.0  # ... and never more than this, however large the covariance
START_SD = np.array([0.001, 0.001, np.radians(0.1)])


@dataclass
class DriveInputs:
    """Everything the filter reads for one drive: no ground truth."""

    rows: np.ndarray  # (n, 320) floor-edge rows (stage 2)
    top: np.ndarray  # (n, 320) wall-top rows
    status: np.ndarray  # (n, 320)
    psi: np.ndarray  # (n,) fused heading modulo 90 (degrees), NaN where invalid
    psi_sigma: np.ndarray  # (n,) degrees
    labels: list[str]
    flags: np.ndarray  # (n,) stage 3b collision flags for the pair (k-1, k)
    matches: list | None = None  # per frame: SIFT matches (u0, v0, u1, v1) to the previous frame


@dataclass
class DriveResult:
    poses: np.ndarray  # (n, 3)
    covs: np.ndarray  # (n, 3, 3)
    inc_mean: np.ndarray  # (n, 3): fused relative motion of pair (k-1, k); row 0 is spawn -> 0
    inc_cov: np.ndarray  # (n, 3, 3)
    nis: np.ndarray  # (n,) NIS of the prior-vs-measurement innovation, NaN without a measurement
    used_icp: np.ndarray  # (n,) bool: the ICP measurement was accepted
    heading_accepted: np.ndarray  # (n,) bool: the absolute heading update was accepted
    wall_used: np.ndarray  # (n,) bool: the wall-feature measurement was accepted
    icp_sigma_along: np.ndarray  # (n,) m: ICP standard deviation along the corridor axis
    heading_valid: np.ndarray  # (n,) bool: a heading was offered
    heading_innovation_deg: np.ndarray  # (n,) innovation of the offered heading
    notes: dict = field(default_factory=dict)
    reanchored: np.ndarray | None = None  # (n,) bool: the heading was re-anchored at this frame


AGREE_CHI2 = 15.0  # two vision measurements agree if their difference is within this (3 dof)


def vp_diff90(a_deg: float, b_deg: float) -> float:
    """Difference of two angles (degrees) on the 90-degree circle, in (-45, 45]."""
    return float(45.0 - np.mod(45.0 - (a_deg - b_deg), 90.0))


def _agree(a: vo.Measurement, b: vo.Measurement) -> bool:
    d = a.mean - b.mean
    d[2] = se2.wrap(d[2])
    return bool(d @ np.linalg.solve(a.cov + b.cov, d) <= AGREE_CHI2)


def _rot_agree(a: vo.Measurement, b: vo.Measurement) -> bool:
    """Do two measurements agree on the rotation (within 3 sigma and a degree)?"""
    d = se2.wrap(a.mean[2] - b.mean[2])
    return bool(
        abs(d) <= 3.0 * np.hypot(np.sqrt(a.cov[2, 2]), np.sqrt(b.cov[2, 2])) + np.radians(1.0)
    )


def _info(cov: np.ndarray) -> np.ndarray:
    return np.linalg.inv(cov + 1e-12 * np.eye(3))


def run_drive(inp: DriveInputs, prior: M.MotionPrior, variant: str = "C") -> DriveResult:
    """Filter one drive. ``variant``: ``C`` (vision + action prior) or ``B`` (vision only).
    Pairs flagged by the stage 3b detector get the collision treatment
    (``collision_increment``) instead of the prior-weighted fusion."""
    n = len(inp.labels)
    poses = np.zeros((n, 3))
    covs = np.zeros((n, 3, 3))
    inc_m = np.zeros((n, 3))
    inc_c = np.zeros((n, 3, 3))
    nis = np.full(n, np.nan)
    used = np.zeros(n, bool)
    wall_used = np.zeros(n, bool)
    sig_along = np.full(n, np.nan)
    rejected: list = []
    reanchored = np.zeros(n, bool)
    h_acc = np.zeros(n, bool)
    h_val = np.zeros(n, bool)
    h_inn = np.full(n, np.nan)
    x = np.array(SPAWN, float)
    P = np.diag(START_SD**2)
    last_inc = np.zeros(3)
    pts_prev = None
    notes: dict = {}
    for k in range(n):
        label = inp.labels[k]
        prev = inp.labels[k - 1] if k > 0 else "START"
        group = prior.get(label, prev)
        pts = vo.scan_points(inp.rows[k], inp.top[k], inp.status[k])
        # ---- prior on the increment
        if variant == "C":
            mu_p, cov_p = group.mean.copy(), group.cov.copy()
        else:
            # constant velocity for the translation; rotation never exceeds about 17 degrees
            # between frames except at collisions, so its guess is zero (a rotation velocity
            # would put the heading-branch rounding on the edge after a collision)
            mu_p, cov_p = np.array([last_inc[0], last_inc[1], 0.0]), np.diag(B_PRIOR_SD**2)
        # ---- vision measurement of the increment (needs a previous frame)
        mu_m, info_m, icp_ok = None, np.zeros((3, 3)), False
        wall_ok = False
        hp_ok = k > 0 and np.isfinite(inp.psi[k]) and np.isfinite(inp.psi[k - 1])
        theta0 = mu_p[2]
        sig_rot = None
        sd_rot_prior = float(np.sqrt(cov_p[2, 2]))
        if hp_ok and not inp.flags[k]:
            theta_vp, sig_vp = vo.vp_rotation(
                inp.psi[k - 1], inp.psi[k], inp.psi_sigma[k - 1], inp.psi_sigma[k], mu_p[2]
            )
            sig_vp = max(sig_vp, np.radians(VP_SIGMA_DEG))
            # a heading-derived rotation is only as good as both headings (one may be wrong by
            # tens of degrees): ICP starts from it only if the prior finds it plausible, and it
            # is used as a measurement only after the other measurements confirm it (below)
            if abs(theta_vp - mu_p[2]) <= 3.0 * sd_rot_prior:
                theta0 = theta_vp
            sig_rot = sig_vp
        if k > 0 and pts_prev is not None:
            cands = []  # vision measurements of the increment: (name, Measurement)
            init = np.array([mu_p[0], mu_p[1], theta0])
            m = vo.icp(pts_prev, pts, init)
            if m is not None:
                cands.append(("icp", m))
                if np.isfinite(inp.psi[k]):
                    # corridor axis in the robot frame: the wall direction nearest forward
                    w = float(se2.wrap(-np.radians(inp.psi[k]) * 4.0) / 4.0)
                    u = np.array([np.cos(w), np.sin(w)])
                    sig_along[k] = float(np.sqrt(u @ m.cov[:2, :2] @ u))
            if inp.matches is not None:
                wm = vo.wall_motion(inp.matches[k], pts_prev)
                if wm is not None:
                    cands.append(("wall", wm))
            # independent rotation sources: the feature-match bearings, the headings
            mrot, mn = vo.match_rotation(inp.matches[k]) if inp.matches is not None else (np.nan, 0)
            sources = []
            if mn >= MATCH_MIN_N:
                sources.append((np.radians(mrot), np.radians(MATCH_ROT_SD_DEG)))
            if sig_rot is not None:
                sources.append((theta_vp, sig_rot))
            accepted: list[tuple[str, vo.Measurement]] = []
            for i, (name, c) in enumerate(cands):
                if variant == "C":
                    ok = bool(group.contains_ext(c.mean, 0.99)[0])
                else:
                    ok = bool(
                        np.hypot(*(c.mean[:2] - mu_p[:2])) < B_GATE_TRANS
                        and abs(se2.wrap(c.mean[2] - mu_p[2])) < B_GATE_ROT
                    )
                if not ok and name == "wall":
                    # the label prior is weak (a collision spin can run on into the next
                    # pair, against the label): accept when two independent rotation
                    # sources agree with the measurement's rotation
                    sd_c = float(np.sqrt(c.cov[2, 2]))
                    votes = [
                        abs(se2.wrap(c.mean[2] - r)) <= 3.0 * np.hypot(sd_c, sd) + np.radians(1.0)
                        for r, sd in sources
                    ]
                    votes += [
                        _rot_agree(c, o)
                        for j, (nm, o) in enumerate(cands)
                        if j != i and nm == "icp"
                    ]
                    ok = sum(votes) >= 2
                if ok:
                    accepted.append((name, c))
            # ICP outside the prior region is accepted only if it agrees with an accepted wall
            # measurement (a corridor can alias ICP to a confident wrong translation)
            for name, c in cands:
                if (
                    name == "icp"
                    and all(a is not c for _, a in accepted)
                    and any(_agree(c, a) for nm, a in accepted if nm == "wall")
                ):
                    accepted.append((name, c))
            for name, c in accepted:
                info_c = _info(c.cov)
                if mu_m is None:
                    mu_m, info_m = c.mean.copy(), info_c
                else:
                    tot_c = info_m + info_c
                    mu_m = np.linalg.solve(tot_c, info_m @ mu_m + info_c @ c.mean)
                    info_m = tot_c
                if name == "icp":
                    icp_ok = True
                else:
                    wall_ok = True
        if sig_rot is not None:
            # confirm the heading-derived rotation against the measurements accepted above
            rot_m = None if mu_m is None else float(mu_m[2])
            if rot_m is not None:
                sd_m = float(np.sqrt(np.linalg.inv(info_m + 1e-12 * np.eye(3))[2, 2]))
                ok_vp = abs(se2.wrap(theta_vp - rot_m)) <= 3.0 * np.hypot(sig_rot, sd_m)
            else:
                ok_vp = abs(theta_vp - mu_p[2]) <= 3.0 * sd_rot_prior
            if not ok_vp:
                sig_rot = None
        if sig_rot is not None:
            theta0 = theta_vp
            # the heading-derived rotation is a measurement of dtheta alone
            e = np.array([0.0, 0.0, 1.0])
            info_r = np.outer(e, e) / sig_rot**2
            if mu_m is None:
                mu_m, info_m = np.array([mu_p[0], mu_p[1], theta0]), info_r.copy()
            else:
                # combine ICP and the heading rotation in information form
                tot = info_m + info_r
                mu_m = np.linalg.solve(tot, info_m @ mu_m + info_r @ np.array([0.0, 0.0, theta0]))
                info_m = tot
        # ---- fusion of prior and measurement
        if mu_m is None:
            mu_f, cov_f = mu_p, cov_p
        else:
            info_p = _info(cov_p)
            tot = info_p + info_m
            cov_f = np.linalg.inv(tot)
            mu_f = cov_f @ (info_p @ mu_p + info_m @ mu_m)
            mu_f[2] = float(se2.wrap(mu_f[2]))
            if icp_ok:
                d = mu_m - mu_p
                d[2] = se2.wrap(d[2])
                S = cov_p + np.linalg.inv(info_m + 1e-9 * np.eye(3))
                nis[k] = float(d @ np.linalg.solve(S, d))
        # ---- a flagged pair: the ordinary fused estimate stands if vision verified a small
        # rotation (a false flag, or a 20 to 30 degree slide); otherwise it is a collision
        if inp.flags[k] and k > 0 and pts_prev is not None:
            normal_ok = (icp_ok or wall_ok) and abs(mu_f[2]) < np.radians(NORMAL_ROT_MAX_DEG)
            if normal_ok and np.isfinite(inp.psi[k]) and np.isfinite(inp.psi[k - 1]):
                # the rotation must agree with the heading difference modulo 90 degrees (ICP can
                # stick at a small wrong rotation when the true one is 30 to 55 degrees)
                dh = vp_diff90(np.degrees(mu_f[2]), inp.psi[k] - inp.psi[k - 1])
                normal_ok = bool(abs(dh) <= NORMAL_HEADING_TOL_DEG)
            if normal_ok:
                mrot, mn = (
                    vo.match_rotation(inp.matches[k]) if inp.matches is not None else (np.nan, 0)
                )
                if mn >= MATCH_MIN_N:
                    normal_ok = bool(abs(np.degrees(mu_f[2]) - mrot) <= NORMAL_MATCH_TOL_DEG)
            if not normal_ok:
                mrot, mn = (
                    vo.match_rotation(inp.matches[k]) if inp.matches is not None else (np.nan, 0)
                )
                hyp = collision_increment(
                    pts_prev,
                    pts,
                    inp.psi[k - 1],
                    inp.psi[k],
                    inp.psi_sigma[k - 1],
                    inp.psi_sigma[k],
                    mrot,
                    mn,
                    prior.by_label["FORWARD"].collision,
                )
                if hyp is not None:
                    mu_f, cov_f = hyp[0], hyp[1]
                    notes[k] = hyp[2]
                    icp_ok = wall_ok = False
        used[k] = icp_ok
        wall_used[k] = wall_ok
        inc_m[k], inc_c[k] = mu_f, cov_f
        # ---- EKF predict
        ja, jb = se2.compose_jacobians(x, mu_f)
        x = se2.compose(x, mu_f)
        P = ja @ P @ ja.T + jb @ cov_f @ jb.T
        # ---- absolute heading update, gated
        if np.isfinite(inp.psi[k]):
            h_val[k] = True
            z = np.radians(inp.psi[k])
            z = z + np.round((x[2] - z) / (np.pi / 2)) * np.pi / 2
            Rz = np.radians(max(inp.psi_sigma[k], VP_SIGMA_DEG)) ** 2
            y = float(se2.wrap(z - x[2]))
            S = P[2, 2] + Rz
            h_inn[k] = np.degrees(y)
            # never more than 8 degrees off a confident filter; an uncertain one (the first frame
            # of a vision-only drive, after a flagged pair) may be pulled further
            limit = max(HEADING_GATE_MAX_DEG, 4.0 * np.degrees(np.sqrt(P[2, 2])))
            if y * y / S <= HEADING_GATE_CHI2 and abs(np.degrees(y)) <= limit:
                # heavy-tailed heading noise: an innovation beyond one standard deviation is
                # downweighted (the measurement variance grows with its squared size), so a
                # heading that is wrong by a few degrees cannot drag the filter with it
                Rz = Rz * max(1.0, y * y / S)
                S = P[2, 2] + Rz
                K = P[:, 2] / S
                x = x + K * y
                x[2] = float(se2.wrap(x[2]))
                P = (np.eye(3) - np.outer(K, np.array([0, 0, 1.0]))) @ P
                P = 0.5 * (P + P.T)
                h_acc[k] = True
                rejected.clear()
            else:
                # a confidently wrong heading is an isolated frame; a *run* of headings that
                # disagree with the filter by one constant offset means the filter lost its
                # heading (a rotation it could not measure): re-anchor on the headings
                rejected.append((y, z))
                if len(rejected) >= REANCHOR_FRAMES:
                    ys = np.array([r[0] for r in rejected[-REANCHOR_FRAMES:]])
                    if np.degrees(ys.max() - ys.min()) <= REANCHOR_SPREAD_DEG:
                        x[2] = float(se2.wrap(rejected[-1][1]))
                        P[2, :] = 0.0
                        P[:, 2] = 0.0
                        P[2, 2] = np.radians(REANCHOR_SD_DEG) ** 2
                        reanchored[k] = True
                        h_acc[k] = True
                        rejected.clear()
        poses[k], covs[k] = x, P
        last_inc = mu_f.copy()
        pts_prev = pts
    return DriveResult(
        poses,
        covs,
        inc_m,
        inc_c,
        nis,
        used,
        h_acc,
        wall_used,
        sig_along,
        h_val,
        h_inn,
        notes,
        reanchored,
    )


# ---------------------------------------------------------------------- collision pairs

COLLISION_COV_INFLATION = 4.0
BRANCH_MARGIN = 0.15  # relative cost difference needed to call a branch hypothesis the winner
SEARCH_STEP_DEG = 1.5
SEARCH_RANGE_DEG = 100.0
PEAK_COUNT = 4  # distinct peaks of the global rotation search that are refined
PEAK_FRACTION = 0.6  # ... if at least this fraction of the best score
PEAK_SEPARATION_DEG = 8.0
MATCH_MIN_N = 4  # feature matches that must agree for the match rotation to be trusted
MATCH_PENALTY = 100.0  # cost of a hypothesis far from a trusted match rotation
NORMAL_HEADING_TOL_DEG = 5.0  # ... and agree with the heading difference modulo 90
NORMAL_MATCH_TOL_DEG = 15.0  # ... and with a trusted feature-match rotation
NORMAL_ROT_MAX_DEG = 25.0  # a flagged pair with a verified rotation below this is not a collision
RANGE_FREE_DEG = 60.0  # collisions rotate by at most about 55 degrees (stage 0)
RANGE_PENALTY = 100.0  # cost per 30 degrees beyond that: a branch 90 degrees away is not credible
MATCH_PENALTY_DEG = 25.0


def collision_increment(
    pts_prev,
    pts,
    psi_prev,
    psi_cur,
    sig_prev,
    sig_cur,
    match_rot=np.nan,
    match_n=0,
    collision=None,
):
    """Increment of a pair flagged as a collision: the rotation is found globally first, because
    the label prior is no guide and the nearest-branch rule of the heading is unsafe.

    Hypotheses (each refined by ICP and scored by its robust alignment cost): the rotations
    ``g + 90 k`` (``k`` in -1, 0, 1) implied by the two headings (``g`` their difference modulo
    90) when both are valid; the best peaks of a correlative scan-matching search over the whole
    rotation range; and the feature-match rotation (sign and rough size, no 90-degree
    ambiguity) when at least ``MATCH_MIN_N`` matches agree. A hypothesis far from a trusted
    match rotation pays a penalty. Returns ``(mean, cov, info)`` or ``None`` if nothing aligns."""
    if pts_prev is None or len(pts.xy) < 10 or len(pts_prev.xy) < 10:
        return None
    field = vo.likelihood_field(pts_prev)
    thetas = np.radians(np.arange(-SEARCH_RANGE_DEG, SEARCH_RANGE_DEG + 1e-9, SEARCH_STEP_DEG))
    scores, trans = vo.correlative_scores(field, pts.xy, thetas)
    starts = []
    # the best distinct peaks of the global search
    order = np.argsort(-scores)
    for i in order:
        if scores[i] < PEAK_FRACTION * scores[order[0]] or len(starts) >= PEAK_COUNT:
            break
        if all(abs(thetas[i] - t0) > np.radians(PEAK_SEPARATION_DEG) for _, t0, _ in starts):
            starts.append(("global", thetas[i], trans[i]))
    g = None
    if np.isfinite(psi_prev) and np.isfinite(psi_cur):
        g = float(np.radians(psi_cur - psi_prev))
        g = float(se2.wrap(g * 4.0) / 4.0)  # fold to (-45, 45] degrees
        for kb in (-1, 0, 1):
            th = g + kb * np.pi / 2
            if abs(th) > np.radians(SEARCH_RANGE_DEG):
                continue
            near = np.where(np.abs(thetas - th) <= np.radians(3.0))[0]
            if len(near) == 0:
                continue
            kbest = near[scores[near].argmax()]
            starts.append((f"vp{kb:+d}", thetas[kbest], trans[kbest]))
    trusted = bool(match_n >= MATCH_MIN_N and np.isfinite(match_rot))
    if trusted:
        near = np.where(np.abs(thetas - np.radians(match_rot)) <= np.radians(6.0))[0]
        if len(near):
            kbest = near[scores[near].argmax()]
            starts.append(("match", thetas[kbest], trans[kbest]))
    cands = []
    for name, th, t in starts:
        init = np.array([t[0], t[1], th])
        m = vo.icp(pts_prev, pts, init, iters=40, dmax_start=0.12, dmax_end=0.04)
        if m is None:
            continue
        cost, nmatch = vo.robust_cost(pts_prev, pts, m.mean)
        beyond = max(0.0, abs(np.degrees(m.mean[2])) - RANGE_FREE_DEG)
        cost += RANGE_PENALTY * beyond / 30.0
        if trusted:
            off = abs(np.degrees(se2.wrap(m.mean[2] - np.radians(match_rot))))
            cost += MATCH_PENALTY * min(1.0, (off / MATCH_PENALTY_DEG) ** 2)
        cands.append({"name": name, "mean": m.mean, "cov": m.cov, "cost": cost, "matched": nmatch})
    if not cands:
        return None
    cands.sort(key=lambda c: c["cost"])
    best = cands[0]
    # distinct hypotheses only (the global peak often coincides with a heading branch)
    others = [
        c for c in cands[1:] if abs(se2.wrap(c["mean"][2] - best["mean"][2])) > np.radians(10.0)
    ]
    margin = (others[0]["cost"] - best["cost"]) / max(abs(best["cost"]), 1.0) if others else np.inf
    mean = best["mean"].copy()
    cov = best["cov"] * COLLISION_COV_INFLATION
    # inside a branch the heading fixes the rotation precisely: use it for the fine angle
    if g is not None:
        branch = np.round((mean[2] - g) / (np.pi / 2))
        if abs(branch) <= 1:
            mean[2] = g + branch * np.pi / 2
            cov = cov.copy()
            cov[2, 2] = min(cov[2, 2], np.radians(1.5) ** 2)
    # a collision moves the robot only a few centimetres: the collision component's translation
    # prior regularises ICP, whose translation is weak against a wall and can alias along a
    # corridor to a confident wrong value (beyond 4 sigma of the prior it is dropped)
    if collision is not None:
        mu_c = collision.means[0][:2]
        sd_c = np.sqrt(np.diag(collision.cov))[:2]
        prior_info = np.diag([1 / sd_c[0] ** 2, 1 / sd_c[1] ** 2, 0.0])
        far = bool(np.hypot(*((mean[:2] - mu_c) / sd_c)) > 4.0)
        info_icp = np.linalg.inv(cov + 1e-12 * np.eye(3))
        if far:
            info_icp[:2, :] = 0.0
            info_icp[:, :2] = 0.0
        tot = info_icp + prior_info
        # rotation keeps its own precision; the translation is fused with the prior
        cov_new = np.linalg.inv(tot + np.diag([0.0, 0.0, 1e-9]))
        mean_new = cov_new @ (info_icp @ mean + prior_info @ np.array([mu_c[0], mu_c[1], 0.0]))
        mean_new[2] = mean[2]
        cov_new[2, 2] = cov[2, 2]
        mean, cov = mean_new, cov_new
    info = {
        "chosen": best["name"],
        "margin": float(margin),
        "hypotheses": [
            (c["name"], float(np.degrees(c["mean"][2])), float(c["cost"])) for c in cands
        ],
        "ambiguous": bool(margin < BRANCH_MARGIN),
    }
    return mean, cov, info
