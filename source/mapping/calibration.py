"""Self-calibration of the action-label prior against a drive's own vision (stage 4b).

The stage 1 prior was fitted on local recordings without a motion bias. With the simulator's
``motion_bias`` a run's linear and angular speeds are scaled by factors drawn once per run, and
``Simulator.record`` draws them **per route**, so each of a dataset's three drives can carry a
different bias (measured in stage 4b: 0.41, 0.92 and 1.27 times the forward speed on one
dataset). The vision measurements (scan ICP and wall features) do not depend on the prior, so on
pairs where both are tight and agree they measure the real motion; the ratio of measured to
prior-mean motion is the drive's scale.

Model: one **speed scale** (from FORWARD pairs: summed measured forward motion over summed
prior-mean forward motion) and one **turn scale** (from LEFT and RIGHT pairs: ``sum(v m) /
sum(m^2)`` on the rotation), applied to every label's prior. Each drive is fitted on its own;
when the drives of a dataset agree (chi-square test on the per-drive estimates), they share the
pooled estimate, which is three times as precise.

Pipeline code (rule 10): images and labels only.
"""

from __future__ import annotations

from dataclasses import dataclass

import motion as M
import numpy as np
import odometry as od
import vo
from scipy.stats import chi2

LABELS_ROT = ("LEFT", "RIGHT")
MIN_PAIRS = 60  # fewer well-measured pairs in a drive: that scale stays 1
SCALE_RANGE = (0.3, 2.0)
MAX_PAIRS_PER_DRIVE = 1500  # evenly spaced pairs per drive (the ICP and wall fits cost time)
SD_MAX = np.array([0.010, 0.020, np.radians(1.5)])  # fused standard deviation required
ROT_MAX = np.radians(20.0)  # larger rotations are collisions: not a motion scale
INIT_SIGMA = np.array([0.30, 0.30, np.radians(40.0)])  # ICP's own prior must not pull to the label
POOL_P = 0.01  # drives are pooled unless their estimates disagree at this level


@dataclass
class Scale:
    value: float
    se: float
    n: int
    pooled: bool = False


@dataclass
class DriveScales:
    speed: Scale
    turn: Scale

    def as_dict(self) -> dict[str, tuple[float, float]]:
        """``{label: (s_trans, s_rot)}`` for ``MotionPrior.scaled``."""
        return {lab: (self.speed.value, self.turn.value) for lab in M.LABELS}


def vision_increments(inp: od.DriveInputs, prior: M.MotionPrior, max_pairs=MAX_PAIRS_PER_DRIVE):
    """Per sampled pair with a tight, mutually consistent ICP and wall measurement: the label,
    the prior-mean increment of its group and the fused vision increment."""
    n = len(inp.labels)
    ks = np.arange(1, n)
    if len(ks) > max_pairs:
        ks = ks[np.linspace(0, len(ks) - 1, max_pairs).astype(int)]
    out = []
    for k in ks:
        if inp.flags[k] or inp.matches is None:
            continue
        group = prior.get(inp.labels[k], inp.labels[k - 1])
        p0 = vo.scan_points(inp.rows[k - 1], inp.top[k - 1], inp.status[k - 1])
        p1 = vo.scan_points(inp.rows[k], inp.top[k], inp.status[k])
        icp = vo.icp_symmetric(p0, p1, group.mean, init_sigma=INIT_SIGMA)
        wall = vo.wall_motion(inp.matches[k], p0, scan_cur=p1)
        if icp is None or wall is None or not od._agree(icp, wall):
            continue
        info = od._info(icp.cov) + od._info(wall.cov)
        cov = np.linalg.inv(info)
        mean = np.linalg.solve(info, od._info(icp.cov) @ icp.mean + od._info(wall.cov) @ wall.mean)
        if np.any(np.sqrt(np.diag(cov)) > SD_MAX) or abs(mean[2]) > ROT_MAX:
            continue
        out.append((inp.labels[k], group.mean.copy(), mean))
    return out


def ratio_with_se(num: np.ndarray, den: np.ndarray, n_boot: int = 200) -> tuple[float, float]:
    """``sum(num) / sum(den)`` and its bootstrap standard error."""
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(num), len(num)) for _ in range(n_boot)]
    return float(num.sum() / den.sum()), float(np.std([num[b].sum() / den[b].sum() for b in idx]))


def _terms(rows) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Numerator and denominator terms of the speed and turn ratios."""
    f = [(m, v) for lab, m, v in rows if lab == "FORWARD"]
    t = [(m, v) for lab, m, v in rows if lab in LABELS_ROT]
    mf = np.array([m for m, _ in f]).reshape(-1, 3)
    vf = np.array([v for _, v in f]).reshape(-1, 3)
    mt = np.array([m for m, _ in t]).reshape(-1, 3)
    vt = np.array([v for _, v in t]).reshape(-1, 3)
    return {
        "speed": (vf[:, 0], mf[:, 0]),
        "turn": (vt[:, 2] * mt[:, 2], mt[:, 2] ** 2),
    }


def _fit(num: np.ndarray, den: np.ndarray) -> Scale:
    if len(num) < MIN_PAIRS:
        return Scale(1.0, 0.0, len(num))
    v, se = ratio_with_se(num, den)
    return Scale(float(np.clip(v, *SCALE_RANGE)), se, len(num))


def consistent(values: np.ndarray, ses: np.ndarray, p: float = POOL_P) -> bool:
    """Do per-drive estimates agree with their weighted mean (chi-square, ``len - 1`` dof)?"""
    if len(values) < 2 or np.any(ses <= 0):
        return True
    w = 1.0 / ses**2
    mean = (w * values).sum() / w.sum()
    stat = float((w * (values - mean) ** 2).sum())
    return bool(stat <= chi2.ppf(1.0 - p, len(values) - 1))


def fit_drives(inputs: list[od.DriveInputs], prior: M.MotionPrior) -> list[DriveScales]:
    """Speed and turn scale for each drive of one dataset (pooled where the drives agree)."""
    terms = [_terms(vision_increments(inp, prior)) for inp in inputs]
    out: dict[str, list[Scale]] = {}
    for key in ("speed", "turn"):
        each = [_fit(*t[key]) for t in terms]
        fitted = [s for s in each if s.se > 0]
        if len(fitted) == len(each) and consistent(
            np.array([s.value for s in each]), np.array([s.se for s in each])
        ):
            num = np.concatenate([t[key][0] for t in terms])
            den = np.concatenate([t[key][1] for t in terms])
            pooled = _fit(num, den)
            pooled.pooled = True
            each = [Scale(pooled.value, pooled.se, pooled.n, True) for _ in each]
        out[key] = each
    return [DriveScales(s, t) for s, t in zip(out["speed"], out["turn"], strict=True)]


def calibrated_priors(inputs: list[od.DriveInputs], prior: M.MotionPrior):
    """One prior per drive, scaled to that drive, and the fitted scales."""
    scales = fit_drives(inputs, prior)
    return [prior.scaled(s.as_dict()) for s in scales], scales
