"""Wall-collision detector (stage 3b): flags a consecutive frame pair as a likely collision.

A collision spins the robot by 30 to 55 degrees (up to 0.96 rad) under a FORWARD or IDLE label
with almost no translation. Normal motion between saved frames never rotates by more than about
17 degrees (a 5-tick window at 0.06 rad per tick, plus noise). Two independent measurements of
that rotation are combined:

1. **Heading change.** Stage 3 gives every frame its heading modulo 90 degrees, and 30 to 55
   degrees is exactly the range that wraps to a large change modulo 90. A change of at least
   ``HEADING_T`` degrees under a FORWARD or IDLE label is a candidate. The best available heading
   is used: the fused one when valid, else the scan cue, else the vanishing-point cue.
2. **Rotation from feature matches.** SIFT matches between the two frames (on the stage 2 wall
   mask, ratio test 0.8); a collision moves every distant feature by the same bearing change, so
   the dominant cluster of bearing shifts (within 3 degrees) estimates the rotation directly,
   with no 90-degree ambiguity, from as few as 3 consistent matches.

**Rule.** Flag a FORWARD/IDLE pair when the heading change is a candidate **and** the matches do
not contradict it (their rotation is at least ``CONFIRM_DEG``, or there are too few matches to
judge), **or** when the matches alone are strong (``RESCUE_N`` or more consistent matches with a
rotation of ``RESCUE_DEG`` or more), **or** when the pair cannot be judged at all (a
FORWARD/IDLE pair where either frame has no valid fused heading and the matches are too few):
a missed event is the costly error, so an unresolved pair is flagged. The second measurement
removes most false flags (a wrong heading on a frame whose fused heading is invalid); the
rescue keeps recall when the heading is wrong at the collision itself. Constants were tuned on
the dev seeds of the anti-aliased recordings. A false flag only costs extra work in stage 4;
a missed event can leave a permanent 90-degree heading error, which is why recall matters more
than precision.

Pipeline code: images and labels only (rule 10). It never reads ``poses.json`` or ``gt.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import heading as hd
import numpy as np
import scan as sc
import segment as sg
import vp

HEADING_T = 22.0  # degrees of heading change modulo 90
THRESHOLD = HEADING_T  # the heading-only helpers below
CONFIRM_DEG = 15.0  # the matches must show at least this much rotation to confirm a candidate
CONFIRM_MIN_N = 3  # fewer consistent matches than this cannot judge
RESCUE_N = 8
RESCUE_DEG = 30.0
CONSENSUS_DEG = 3.0
LABELS = ("FORWARD", "IDLE")  # collisions only happen under these (stage 0)


def best_psi(h: hd.HeadingEstimate) -> float:
    """The best available yaw mod 90: fused if valid, else the scan cue, else the VP cue."""
    if h.valid:
        return h.psi
    if np.isfinite(h.psi_scan):
        return h.psi_scan
    return h.psi_vp


@dataclass
class PairFlag:
    flagged: bool
    delta: float  # degrees: heading change modulo 90 (NaN if a heading is missing)


def flag_pair(
    psi_prev: float, psi_cur: float, label: str, threshold: float = THRESHOLD
) -> PairFlag:
    """Is the pair ``(previous frame, this frame)`` a collision? ``label`` is this record's."""
    if not (np.isfinite(psi_prev) and np.isfinite(psi_cur)):
        return PairFlag(False, np.nan)
    delta = float(abs(vp.diff90(psi_cur, psi_prev)))
    return PairFlag(bool(label in LABELS and delta >= threshold), delta)


def flag_drive(psi: np.ndarray, labels: list[str], threshold: float = THRESHOLD):
    """Flags and heading changes for every pair of a drive; the first frame has no previous one."""
    n = len(psi)
    flags = np.zeros(n, bool)
    delta = np.full(n, np.nan)
    for k in range(1, n):
        f = flag_pair(psi[k - 1], psi[k], labels[k], threshold)
        flags[k], delta[k] = f.flagged, f.delta
    return flags, delta


def sift_features(img: np.ndarray, seg: sg.Segmentation, sift):
    """SIFT keypoints (as bearings from the principal point) and descriptors on the wall mask."""
    mask = cv2.erode(seg.wall.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    kp, desc = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    cam = sc.camera_config()
    pts = np.array([k.pt[0] for k in kp], np.float64)
    return np.degrees(np.arctan((pts - cam.cx) / cam.fx)), desc


def rotation_from_matches(f0, f1, matcher) -> tuple[float, int, int]:
    """Rotation between two frames from the bearing shifts of matched features: the dominant
    cluster (within ``CONSENSUS_DEG``) of ``bearing1 - bearing0``. Returns
    ``(rotation_deg, consistent_matches, matches)``; NaN with too few matches."""
    (b0, d0), (b1, d1) = f0, f1
    if d0 is None or d1 is None or len(b0) < 4 or len(b1) < 4:
        return np.nan, 0, 0
    good = [
        m
        for m, n in (x for x in matcher.knnMatch(d0, d1, k=2) if len(x) == 2)
        if m.distance < 0.8 * n.distance
    ]
    if len(good) < 3:
        return np.nan, 0, len(good)
    shift = np.array([b1[m.trainIdx] - b0[m.queryIdx] for m in good])
    best_n, best_c = 0, 0.0
    for c in shift:
        n = int((np.abs(shift - c) < CONSENSUS_DEG).sum())
        if n > best_n or (n == best_n and abs(c) > abs(best_c)):
            best_n, best_c = n, float(c)
    inliers = np.abs(shift - best_c) < CONSENSUS_DEG
    return float(np.median(shift[inliers])), int(inliers.sum()), len(good)


def decide(
    delta: float,
    rot: float,
    n: int,
    label: str,
    heading_t: float = HEADING_T,
    heading_ok: bool = True,
) -> bool:
    """The rule: ``delta`` the heading change (NaN if unknown), ``rot`` and ``n`` the match
    rotation (signed, degrees) and its consistent matches, ``heading_ok`` whether both frames
    have a valid fused heading."""
    if label not in LABELS:
        return False
    judged = n >= CONFIRM_MIN_N and np.isfinite(rot)
    confirmed = (not judged) or abs(rot) >= CONFIRM_DEG
    candidate = bool(np.isfinite(delta) and delta >= heading_t)
    rescue = bool(judged and n >= RESCUE_N and abs(rot) >= RESCUE_DEG)
    unresolved = bool(not judged and not heading_ok)
    return bool((candidate and confirmed) or rescue or unresolved)


def drive_cues(path: str | Path, limit: int | None = None):
    """Per-frame cues for one drive from its images and labels only: the best heading and, for
    FORWARD/IDLE records, the match rotation to the previous frame. Returns
    ``(psi, rot, n, labels, valid)``."""
    path = Path(path)
    info = json.loads((path / "data_info.json").read_text())[:limit]
    lsd = vp.make_lsd()
    sift = cv2.SIFT_create(nfeatures=400)
    matcher = cv2.BFMatcher()
    psi = np.full(len(info), np.nan)
    valid = np.zeros(len(info), bool)
    rot = np.full(len(info), np.nan)
    n = np.zeros(len(info), int)
    labels = [r["action"][0] for r in info]
    prev = None
    for k, r in enumerate(info):
        img = cv2.imread(str(path / r["image"]))
        seg = sg.segment(img)
        h = hd.heading_from_segmentation(seg, lsd)
        psi[k], valid[k] = best_psi(h), h.valid
        feats = sift_features(img, seg, sift)
        if prev is not None and labels[k] in LABELS:
            rot[k], n[k], _ = rotation_from_matches(prev, feats, matcher)
        prev = feats
    return psi, rot, n, labels, valid


def decide_drive(psi, rot, n, labels, valid, heading_t: float = HEADING_T):
    """Flags and heading changes for every pair of a drive from its cues (``valid``: whether
    each frame's fused heading is valid)."""
    flags = np.zeros(len(psi), bool)
    delta = np.full(len(psi), np.nan)
    for k in range(1, len(psi)):
        f = flag_pair(psi[k - 1], psi[k], labels[k], 0.0)
        delta[k] = f.delta
        ok = bool(valid[k] and valid[k - 1])
        flags[k] = decide(f.delta, rot[k], int(n[k]), labels[k], heading_t, ok)
    return flags, delta


def detect_drive(path: str | Path, limit: int | None = None):
    """Run the whole detector on one drive directory (``data_info.json`` and the images only).
    Returns ``(flags, heading_change, psi)``."""
    psi, rot, n, labels, valid = drive_cues(path, limit)
    flags, delta = decide_drive(psi, rot, n, labels, valid)
    return flags, delta, psi
