"""Action labels as a weak motion prior (stage 1).

A record's label is only the last tick of a 5-tick window (stage 0), so the motion between
two saved frames is a *distribution* given the label, not a constant. This module holds the
empirical distribution of the true relative motion ``(dx, dy, dtheta)`` (robot frame of the
first pose) given the label and, because it is far more informative, the previous record's
label. Each (previous, current) group is a Gaussian mixture of at most three components,
because collisions and window mixing make single Gaussians a poor fit.

    prior = MotionPrior.load()
    p = prior.get("LEFT", prev="FORWARD")
    p.mean, p.cov         # moments, inflated so the 95 % ellipse covers 95 % of dev samples
    p.logpdf(x), p.contains(x)   # mixture density and its 95 % highest-density region
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_gt  # noqa: E402
import se2  # noqa: E402

LABELS = ("FORWARD", "LEFT", "RIGHT", "IDLE")
PREV = (*LABELS, "START")
PRIOR_PATH = record_gt.ROOT / "cache" / "mapping" / "motion_prior.json"
SCALES = np.array([0.004, 0.001, 0.01])  # std floors of the mixture components: dx, dy, dth
MIN_GROUP = 100
CHI2_95_3DOF = 7.814727903251179
CHI2_99_3DOF = 11.344866730144373
COLLISION_LABELS = ("FORWARD", "IDLE")
COLLISION_ROT_DEG = 30.0  # a collision: true rotation above this under a FORWARD/IDLE label


# ------------------------------------------------------------------------------- the data


def load_pairs(seeds) -> dict[str, np.ndarray]:
    """True relative motion for every consecutive pair of saved frames (the first from the
    spawn), with the record's label, the previous and next label, and where it came from."""
    cols = {k: [] for k in ("inc", "label", "prev", "next", "seed", "drive", "k")}
    for s in seeds:
        gt = json.loads((record_gt.gt_dir(s) / "gt.json").read_text())
        for d, P in enumerate(record_gt.load_poses(record_gt.gt_dir(s))):
            info = json.loads((record_gt.gt_dir(s) / f"traj_{d}" / "data_info.json").read_text())
            lab = [r["action"][0] for r in info]
            q = np.vstack([gt["spawn"], P])
            cols["inc"].append(se2.relative(q[:-1], q[1:]))
            cols["label"] += lab
            cols["prev"] += ["START", *lab[:-1]]
            cols["next"] += [*lab[1:], "END"]
            cols["seed"] += [s] * len(lab)
            cols["drive"] += [d] * len(lab)
            cols["k"] += list(range(len(lab)))
    out = {k: np.array(v) for k, v in cols.items() if k != "inc"}
    out["inc"] = np.concatenate(cols["inc"])
    return out


# ------------------------------------------------------------------------------ the prior


def _logpdf_mixture(x, w, mu, cov):
    x = np.atleast_2d(x)
    lp = []
    for wi, mi, ci in zip(w, mu, cov, strict=True):
        d = x - mi
        sol = np.linalg.solve(ci, d.T).T
        _, logdet = np.linalg.slogdet(ci)
        lp.append(
            np.log(wi) - 0.5 * (np.einsum("ij,ij->i", d, sol) + logdet + 3 * np.log(2 * np.pi))
        )
    return np.logaddexp.reduce(np.stack(lp), axis=0)


@dataclass
class Collision:
    """Wall-collision component: the robot is stopped and spun by a wall, so it moves a few
    centimetres forward and rotates by about +-0.7 rad (the true events have |dtheta| between
    0.52 and 0.96 rad): two equally weighted lobes, symmetric in the rotation, sharing a
    diagonal covariance that is broad in translation (a collision can happen anywhere in the
    5-tick window)."""

    weight: float  # fraction of the label's pairs that are collisions (dev seeds)
    means: np.ndarray  # (2, 3): (dx, 0, +mu) and (dx, 0, -mu)
    cov: np.ndarray  # (3, 3), diagonal
    n_events: int

    def logpdf(self, x) -> np.ndarray:
        return _logpdf_mixture(x, np.array([0.5, 0.5]), self.means, np.stack([self.cov, self.cov]))

    def mahalanobis2(self, x) -> np.ndarray:
        """Squared Mahalanobis distance to the nearer lobe."""
        x = np.atleast_2d(x)
        out = []
        for m in self.means:
            d = x - m
            out.append(np.einsum("ij,ij->i", d, np.linalg.solve(self.cov, d.T).T))
        return np.minimum(*out)

    def to_json(self) -> dict:
        return {
            "weight": self.weight,
            "means": self.means.tolist(),
            "cov": self.cov.tolist(),
            "n_events": self.n_events,
        }

    @classmethod
    def from_json(cls, d: dict) -> Collision:
        return cls(d["weight"], np.array(d["means"]), np.array(d["cov"]), d["n_events"])


@dataclass
class GroupPrior:
    n: int
    weights: np.ndarray
    means: np.ndarray  # (k, 3)
    covs: np.ndarray  # (k, 3, 3)
    mean: np.ndarray  # mixture mean
    cov: np.ndarray  # mixture covariance, inflated (see ``inflation``)
    inflation: float
    threshold: float  # log-density of the 95 % highest-density region
    source: str  # "pair" (prev, label) or "label"
    collision: Collision | None = None  # stage 3b: set for FORWARD and IDLE
    threshold99: float | None = None  # 99 % highest-density threshold of the non-collision mixture

    def logpdf(self, x) -> np.ndarray:
        return _logpdf_mixture(x, self.weights, self.means, self.covs)

    def contains(self, x) -> np.ndarray:
        """In the 95 % highest-density region of the mixture."""
        return self.logpdf(x) >= self.threshold

    def mahalanobis2(self, x) -> np.ndarray:
        d = np.atleast_2d(x) - self.mean
        return np.einsum("ij,ij->i", d, np.linalg.solve(self.cov, d.T).T)

    def logpdf_ext(self, x) -> np.ndarray:
        """Density of the mixture extended with the collision component:
        ``(1 - w) p_normal + w p_collision`` (stage 4 gating uses this so that a label never
        vetoes a large rotation seen by vision). Without a collision component: ``logpdf``."""
        if self.collision is None:
            return self.logpdf(x)
        w = self.collision.weight
        return np.logaddexp(np.log1p(-w) + self.logpdf(x), np.log(w) + self.collision.logpdf(x))

    def contains_ext(self, x, level: float = 0.95) -> np.ndarray:
        """In the extended ``level`` (0.95 or 0.99) region: the normal mixture's highest-density
        region **or** the collision component's own chi-square ellipse. A union, because a broad
        low-density component can never reach the density level set of the sharp one."""
        thr, chi2 = (
            (self.threshold, CHI2_95_3DOF) if level == 0.95 else (self.threshold99, CHI2_99_3DOF)
        )
        inside = self.logpdf(x) >= thr
        if self.collision is not None:
            inside = inside | (self.collision.mahalanobis2(x) <= chi2)
        return inside

    def to_json(self) -> dict:
        extra = {}
        if self.collision is not None:
            extra["collision"] = self.collision.to_json()
        if self.threshold99 is not None:
            extra["threshold99"] = self.threshold99
        return {
            **extra,
            "n": self.n,
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covs": self.covs.tolist(),
            "mean": self.mean.tolist(),
            "cov": self.cov.tolist(),
            "inflation": self.inflation,
            "threshold": self.threshold,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, d: dict) -> GroupPrior:
        return cls(
            d["n"],
            np.array(d["weights"]),
            np.array(d["means"]),
            np.array(d["covs"]),
            np.array(d["mean"]),
            np.array(d["cov"]),
            d["inflation"],
            d["threshold"],
            d["source"],
            Collision.from_json(d["collision"]) if "collision" in d else None,
            d.get("threshold99"),
        )


def fit_group(x: np.ndarray, source: str, max_components: int = 3, seed: int = 0) -> GroupPrior:
    """Mixture of 1 to ``max_components`` Gaussians chosen by BIC on dx, dy, dtheta scaled by
    ``SCALES`` (with a variance floor of one scale unit, so dy and the angle are never
    degenerate), then the 95 % highest-density threshold and the inflated moment ellipse."""
    from sklearn.mixture import GaussianMixture

    z = x / SCALES
    best = None
    for k in range(1, max_components + 1):
        if len(z) < 30 * k:
            break
        gm = GaussianMixture(k, covariance_type="full", reg_covar=1.0, n_init=3, random_state=seed)
        gm.fit(z)
        if best is None or gm.bic(z) < best.bic(z):
            best = gm
    w = best.weights_
    mu = best.means_ * SCALES
    cov = best.covariances_ * np.outer(SCALES, SCALES)
    mean = (w[:, None] * mu).sum(0)
    d = mu - mean
    mcov = (w[:, None, None] * (cov + d[:, :, None] * d[:, None, :])).sum(0)
    thr = float(np.percentile(_logpdf_mixture(x, w, mu, cov), 5))
    m2 = np.einsum("ij,ij->i", x - mean, np.linalg.solve(mcov, (x - mean).T).T)
    infl = float(max(1.0, np.percentile(m2, 95) / CHI2_95_3DOF))
    return GroupPrior(len(x), w, mu, cov, mean, mcov * infl, infl, thr, source)


class MotionPrior:
    """The fitted priors: one per label, and one per (previous label, label) with enough data."""

    def __init__(self, by_label: dict, by_pair: dict, meta: dict | None = None):
        self.by_label, self.by_pair, self.meta = by_label, by_pair, meta or {}

    @classmethod
    def fit(cls, pairs: dict, min_group: int = MIN_GROUP, use_prev: bool = True) -> MotionPrior:
        by_label, by_pair = {}, {}
        for lab in LABELS:
            sel = pairs["label"] == lab
            if sel.sum() < 30:
                continue
            by_label[lab] = fit_group(pairs["inc"][sel], "label")
            if not use_prev:
                continue
            for pl in PREV:
                g = sel & (pairs["prev"] == pl)
                if g.sum() >= min_group:
                    by_pair[f"{pl}>{lab}"] = fit_group(pairs["inc"][g], "pair")
        return cls(by_label, by_pair, {"seeds": sorted({int(s) for s in pairs["seed"]})})

    def add_collision(self, pairs: dict) -> dict:
        """Add the collision component (stage 3b) fitted on ``pairs`` (dev seeds): events are
        pairs with a true rotation above ``COLLISION_ROT_DEG`` under FORWARD or IDLE. The weight
        is per label; the Gaussian is shared (translation near ``(mean dx, 0)``, rotation
        zero-mean and broad). Also stores each group's 99 % density threshold of the
        non-collision mixture. Returns a summary."""
        inc, lab = pairs["inc"], pairs["label"]
        rot = np.degrees(np.abs(inc[:, 2]))
        ev = np.isin(lab, COLLISION_LABELS) & (rot > COLLISION_ROT_DEG)
        x = inc[ev]
        mu = float(np.abs(x[:, 2]).mean())
        mean_dx = float(x[:, 0].mean())
        means = np.array([[mean_dx, 0.0, mu], [mean_dx, 0.0, -mu]])
        # a collision can happen anywhere in the 5-tick window, so its translation spans 0 to
        # 0.15 m: floors from that physical range (the few dev events span only 0.03 to 0.06 m
        # in dx); the rotation spread floor keeps the lobes wide enough for 0.5 to 1.0 rad
        sd = np.array(
            [max(x[:, 0].std(), 0.035), max(x[:, 1].std(), 0.035), max(np.abs(x[:, 2]).std(), 0.2)]
        )
        cov = np.diag(sd**2)
        summary = {
            "events": int(ev.sum()),
            "means": means.tolist(),
            "sd": sd.tolist(),
            "weights": {},
        }
        for label in (lab_ for lab_ in COLLISION_LABELS if lab_ in self.by_label):
            sel = lab == label
            w = float((ev & sel).sum() / max(sel.sum(), 1))
            summary["weights"][label] = w
            comp = Collision(w, means, cov, int((ev & sel).sum()))
            groups = [(self.by_label[label], sel)] + [
                (g, sel & (pairs["prev"] == key.split(">")[0]))
                for key, g in self.by_pair.items()
                if key.split(">")[1] == label
            ]
            for g, mask in groups:
                g.collision = comp
                normal = inc[mask & ~ev]
                g.threshold99 = float(np.percentile(g.logpdf(normal), 1))
        for label in (
            lab_ for lab_ in LABELS if lab_ not in COLLISION_LABELS and lab_ in self.by_label
        ):
            sel = lab == label
            groups = [(self.by_label[label], sel)] + [
                (g, sel & (pairs["prev"] == key.split(">")[0]))
                for key, g in self.by_pair.items()
                if key.split(">")[1] == label
            ]
            for g, mask in groups:
                g.threshold99 = float(np.percentile(g.logpdf(inc[mask]), 1))
        return summary

    def get(self, label: str, prev: str | None = None) -> GroupPrior:
        if prev is not None and f"{prev}>{label}" in self.by_pair:
            return self.by_pair[f"{prev}>{label}"]
        return self.by_label[label]

    def save(self, path: Path = PRIOR_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "meta": self.meta,
                    "labels": {k: v.to_json() for k, v in self.by_label.items()},
                    "pairs": {k: v.to_json() for k, v in self.by_pair.items()},
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: Path = PRIOR_PATH) -> MotionPrior:
        d = json.loads(path.read_text())
        return cls(
            {k: GroupPrior.from_json(v) for k, v in d["labels"].items()},
            {k: GroupPrior.from_json(v) for k, v in d["pairs"].items()},
            d["meta"],
        )

    def mean_increments(self, labels, use_prev: bool = True) -> np.ndarray:
        """Prior-mean relative motion for a sequence of labels (dead reckoning input)."""
        out = []
        for k, lab in enumerate(labels):
            prev = labels[k - 1] if k > 0 else "START"
            out.append(self.get(lab, prev if use_prev else None).mean)
        return np.array(out)
