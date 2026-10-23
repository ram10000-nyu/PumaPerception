"""Loop closures (stage 5): revisit pairs between any two frames of a dataset's drives, with a
measured relative pose and covariance.

1. **Candidates.** VLAD (the baseline's ``vlad.VLADExtractor`` encoding) of RootSIFT on the
   stage 2 **wall-only mask**, with a fixed vocabulary and a PCA-whitening projection
   (``cache/mapping/stage5_vocab.npz``, fitted once on dev seeds: a property of the texture
   pack, not of a maze). For each frame, the ``TOP_K`` most similar frames of any drive,
   excluding frames within ``EXCLUDE`` indices of it in its own drive.
2. **Verification.** Ratio-test matches and a RANSAC homography; accepted with at least
   ``min_inliers`` inliers.
3. **Relative pose** of frame b in frame a's robot frame (forward, left, yaw):
   (a) homography decomposition, scale from the scan depth of the wall;
   (b) scan ICP (``vo.icp_symmetric``) initialised from (c), else from (a);
   (c) the matched wall features as metric points, depth from each frame's own scan
       (``vo.wall_motion`` with two scans, stage 4b). The most accurate on dev (stage 5); it is
       the closure's measurement, and (b) is required to agree with it when (b) exists.
4. **Revisit envelope.** Only pairs whose measured relative pose is a revisit in the plan's
   sense (within ``MAX_REL_M`` and ``MAX_REL_DEG``, a margin inside 0.5 m and 30 deg) are
   closures; wider-baseline matches are accurate but are not claimed.
5. **Odometry gate.** The textures repeat across a maze, so a different place can match as
   well as the same place (a third of dev pairs with 30+ inliers are 1 to 5 m apart). Every
   drive's odometry starts at the spawn, so all drives share one frame; a closure must agree
   with the odometry's relative pose within ``GATE_M`` and ``GATE_DEG`` (drift on dev and test
   stays under 0.25 m; 99 % of correct dev closures are within 0.34 m of the odometry).

Pipeline code (rule 10): images and labels only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import scan as sc
import se2
import vo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vlad  # noqa: E402  (the baseline's VLAD encoding)

VOCAB_PATH = Path(__file__).resolve().parents[2] / "cache" / "mapping" / "stage5_vocab.npz"
N_FEATURES = 400  # SIFT keypoints per frame (as stage 4)
VLAD_K = 64  # visual words
PCA_DIM = 256
TOP_K = 20  # candidates per frame (plan: 5; 20 chosen on dev for recall, stage 5)
EXCLUDE = 10  # frames of the same drive closer than this in index are not candidates
RATIO = 0.8
RANSAC_PX = 3.0
ICP_INIT_SIGMA = np.array([0.3, 0.3, np.radians(20.0)])
MIN_INLIERS = 20  # homography inliers (chosen on dev, stage 5)
AGREE_M, AGREE_DEG = 0.10, 5.0  # (b) and (c) must agree when both exist (plan)
GATE_M, GATE_DEG = 0.5, 10.0  # closure vs odometry relative pose
MAX_REL_M, MAX_REL_DEG = 0.45, 27.0  # a closure is a revisit (plan: 0.5 m, 30 deg), with margin
COV_INFLATION = 1.0  # multiplies method (c)'s covariance (calibrated on dev, stage 5)


@dataclass
class Features:
    pts: np.ndarray  # (n, 2) float32, continuous pixel coordinates
    desc: np.ndarray  # (n, 128) uint8, raw SIFT


@dataclass
class Closure:
    a: tuple[int, int]  # (drive, frame)
    b: tuple[int, int]
    inliers: int
    score: float  # VLAD similarity
    mean: np.ndarray | None = None  # pose of b in a: (forward, left, yaw)
    cov: np.ndarray | None = None
    method: str = ""
    poses: dict = field(default_factory=dict)  # every method's estimate, for the report
    matches: np.ndarray | None = None  # homography inliers (ua, va, ub, vb)


def frame_features(img: np.ndarray, seg, sift) -> Features:
    pts, desc = vo.wall_features(img, seg, sift)
    if desc is None:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
    return Features(pts.astype(np.float32), np.clip(np.round(desc), 0, 255).astype(np.uint8))


def root_sift(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    d /= np.maximum(d.sum(1, keepdims=True), 1e-9)
    return np.sqrt(d)


# ------------------------------------------------------------------------------ candidates


class _Codebook:
    """The minimal k-means interface ``vlad.VLADExtractor`` needs (fixed centres)."""

    def __init__(self, centers: np.ndarray) -> None:
        self.cluster_centers_ = centers

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = (x**2).sum(1)[:, None] - 2 * x @ self.cluster_centers_.T
        return np.argmin(d + (self.cluster_centers_**2).sum(1)[None], axis=1)


@dataclass
class Vocabulary:
    centers: np.ndarray  # (K, 128) RootSIFT
    pca_mean: np.ndarray  # (K * 128,)
    pca_proj: np.ndarray  # (PCA_DIM, K * 128), whitening included

    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> Vocabulary:
        z = np.load(path)
        return cls(z["centers"], z["pca_mean"], z["pca_proj"])

    def save(self, path: Path = VOCAB_PATH, **provenance) -> None:
        np.savez(path, centers=self.centers, pca_mean=self.pca_mean, pca_proj=self.pca_proj,
                 **{k: np.array(str(v)) for k, v in provenance.items()})

    def raw_vlad(self, feats: Features) -> np.ndarray:
        ext = vlad.VLADExtractor(n_clusters=len(self.centers))
        ext.codebook = _Codebook(self.centers)
        return ext._vlad(root_sift(feats.desc) if len(feats.desc) else None)

    def encode(self, feats: list[Features]) -> np.ndarray:
        """(N, PCA_DIM) unit vectors."""
        v = np.array([self.raw_vlad(f) for f in feats])
        g = (v - self.pca_mean) @ self.pca_proj.T
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def fit_vocabulary(feats: list[Features], k: int = VLAD_K, dim: int = PCA_DIM, seed: int = 0):
    """Vocabulary (k-means on a sample of RootSIFT descriptors) and PCA whitening (on the raw
    VLAD vectors of the same frames)."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    desc = np.concatenate([root_sift(f.desc) for f in feats if len(f.desc)])
    sample = desc[rng.choice(len(desc), min(len(desc), 200_000), replace=False)]
    km = KMeans(n_clusters=k, n_init=1, max_iter=100, random_state=seed).fit(sample)
    voc = Vocabulary(km.cluster_centers_.astype(np.float32), np.zeros(1), np.zeros(1))
    v = np.array([voc.raw_vlad(f) for f in feats])
    mean = v.mean(0)
    _, s, vt = np.linalg.svd(v - mean, full_matrices=False)
    proj = vt[:dim] / (s[:dim, None] / np.sqrt(len(v)) + 1e-6)
    return Vocabulary(voc.centers, mean, proj)


def candidates(g: np.ndarray, drive: np.ndarray, index: np.ndarray, top_k: int = TOP_K,
               exclude: int = EXCLUDE) -> list[tuple[int, int, float]]:
    """Unordered pairs ``(i, j, similarity)`` (i < j, rows of ``g``): each frame's ``top_k``
    most similar frames, excluding its own drive within ``exclude`` indices."""
    out = {}
    for start in range(0, len(g), 1024):
        sim = g[start : start + 1024] @ g.T
        rows = np.arange(start, min(start + 1024, len(g)))
        same = (drive[rows, None] == drive[None]) & (
            np.abs(index[rows, None] - index[None]) <= exclude
        )
        sim[same] = -np.inf
        top = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        for r, cols in zip(rows, top, strict=True):
            for c in cols:
                i, j = (r, c) if r < c else (c, r)
                out[(i, j)] = max(out.get((i, j), -np.inf), float(sim[r - start, c]))
    return [(i, j, s) for (i, j), s in sorted(out.items())]


# ---------------------------------------------------------------------------- verification


def verify(fa: Features, fb: Features, matcher=None):
    """Ratio-test matches and a RANSAC homography: ``(inliers, H, inlier matches)``."""
    matcher = matcher or cv2.BFMatcher()
    if len(fa.desc) < 4 or len(fb.desc) < 4:
        return 0, None, np.zeros((0, 4))
    knn = matcher.knnMatch(root_sift(fa.desc), root_sift(fb.desc), k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 4:
        return 0, None, np.zeros((0, 4))
    pa = fa.pts[[m.queryIdx for m in good]]
    pb = fb.pts[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return 0, None, np.zeros((0, 4))
    m = mask.ravel().astype(bool)
    return int(m.sum()), H, np.hstack([pa[m], pb[m]]).astype(np.float64)


# ------------------------------------------------------------------------------ relative pose


def pose_from_homography(H: np.ndarray, matches: np.ndarray, scan_a: vo.ScanPoints,
                         cam: sc.Camera | None = None):
    """Method (a): decompose ``H`` (image a -> image b), keep the solution whose plane is
    vertical, in front of camera a and parallel to the scan's wall at the inlier features (the
    decomposition has two physically valid solutions; the scan's wall normal separates them),
    and set the scale from the scan depth of that wall. Returns the pose of b in a's robot
    frame, or ``None``."""
    cam = cam or sc.camera_config()
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
    _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    zcol = np.full(cam.width, np.nan)
    zcol[scan_a.col] = scan_a.xy[:, 0]
    ncol = np.full((cam.width, 2), np.nan)
    ncol[scan_a.col] = scan_a.normal
    u, v = matches[:, 0], matches[:, 1]
    cols = np.clip(np.floor(u).astype(int), 0, cam.width - 1)
    z = zcol[cols]
    ok = np.isfinite(z)
    if ok.sum() < 3:
        return None
    X = np.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], 1)[ok]
    wn = ncol[cols][ok]
    wn = wn[np.isfinite(wn[:, 0])]
    if len(wn) == 0:
        return None
    wn = wn * np.sign(wn @ wn[0])[:, None]  # one orientation, then the mean direction
    wall_n = wn.mean(0) / np.linalg.norm(wn.mean(0))
    best = None
    for R, t, n in zip(Rs, ts, ns, strict=True):
        n = n.ravel()
        if abs(n[1]) > 0.2 or R[1, 1] < np.cos(np.radians(5.0)):
            continue  # a wall's normal is horizontal; the rotation is about the vertical axis
        d = X @ n
        if np.median(d) <= 0:
            continue
        n_robot = np.array([n[2], -n[0]]) / np.hypot(n[2], n[0])
        fit = abs(float(n_robot @ wall_n))
        if best is None or fit > best[0]:
            best = (fit, R, t.ravel() * float(np.median(d)))
    if best is None or best[0] < np.cos(np.radians(15.0)):
        return None
    _, R, tm = best
    c = -R.T @ tm  # camera b's centre in camera a
    yaw = float(np.arctan2(-R[2, 0], R[2, 2]))
    out = np.array([c[2], -c[0], yaw])
    return out if np.all(np.isfinite(out)) else None


def pose_from_features(matches: np.ndarray, scan_a, scan_b):
    """Method (c): metric wall-feature correspondences, depth from each frame's own scan."""
    return vo.wall_motion(matches, scan_a, scan_cur=scan_b)


def pose_from_icp(scan_a, scan_b, init):
    """Method (b): symmetric scan ICP from ``init``."""
    if not np.all(np.isfinite(init)):
        return None
    return vo.icp_symmetric(scan_a, scan_b, init, init_sigma=ICP_INIT_SIGMA)


def agree(p: np.ndarray, q: np.ndarray, trans: float = 0.10, rot_deg: float = 5.0) -> bool:
    d = p - q
    return bool(np.hypot(d[0], d[1]) <= trans and abs(np.degrees(se2.wrap(d[2]))) <= rot_deg)


def in_envelope(mean: np.ndarray) -> bool:
    rot = abs(np.degrees(se2.wrap(mean[2])))
    return bool(np.hypot(mean[0], mean[1]) <= MAX_REL_M and rot <= MAX_REL_DEG)


def odometry_gate(mean: np.ndarray, pose_a: np.ndarray, pose_b: np.ndarray,
                  max_m: float = GATE_M, max_deg: float = GATE_DEG) -> bool:
    """Does a closure's relative pose agree with the odometry's (both drives in the spawn
    frame)?"""
    return agree(mean, se2.relative(pose_a, pose_b), max_m, max_deg)


def measure(fa: Features, fb: Features, scan_a, scan_b, matcher=None) -> Closure | None:
    """Verification and relative pose of one candidate pair (no gate). ``None`` when the
    homography has too few inliers, method (c) fails, or (b) disagrees with (c)."""
    n, H, m = verify(fa, fb, matcher)
    if n < MIN_INLIERS:
        return None
    pc = pose_from_features(m, scan_a, scan_b)
    if pc is None or not in_envelope(pc.mean):
        return None
    pb = pose_from_icp(scan_a, scan_b, pc.mean)
    if pb is not None and not agree(pb.mean, pc.mean, AGREE_M, AGREE_DEG):
        return None
    pa = pose_from_homography(H, m, scan_a)
    return Closure(
        (0, 0), (0, 0), n, 0.0, pc.mean, pc.cov * COV_INFLATION, "c",
        {"a": pa, "b": None if pb is None else pb.mean, "c": pc.mean}, m,
    )


def find_closures(features: list[list[Features]], scans: list[list], poses: list[np.ndarray],
                  vocab: Vocabulary | None = None) -> list[Closure]:
    """All accepted closures of one dataset: ``features`` and ``scans`` per drive and frame,
    ``poses`` the odometry trajectories (spawn frame) per drive."""
    vocab = vocab or Vocabulary.load()
    flat = [f for d in features for f in d]
    drive = np.concatenate([np.full(len(d), i) for i, d in enumerate(features)])
    index = np.concatenate([np.arange(len(d)) for d in features])
    g = vocab.encode(flat)
    matcher = cv2.BFMatcher()
    out = []
    for i, j, sim in candidates(g, drive, index):
        (da, ka), (db, kb) = (int(drive[i]), int(index[i])), (int(drive[j]), int(index[j]))
        c = measure(flat[i], flat[j], scans[da][ka], scans[db][kb], matcher)
        if c is None or not odometry_gate(c.mean, poses[da][ka], poses[db][kb]):
            continue
        c.a, c.b, c.score = (da, ka), (db, kb), sim
        out.append(c)
    return out
