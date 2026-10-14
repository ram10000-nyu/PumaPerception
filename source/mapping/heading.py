"""Heading from one frame: the vanishing-point cue and the scan cue, fused (stage 3).

``yaw mod 90`` in degrees, in (-45, 45], relative to the maze axes (the world x axis; stage 0
found the offset is zero), with a validity flag and a confidence.

Fusion rule (tuned on the dev seeds, see reports/stage_3.md): the two cues are independent
measurements (one from the floor-seam lines in the image, one from the wall directions in the
virtual scan). When they **agree within ``TAU`` degrees** the result is their inverse-variance
circular mean and it is valid. When they disagree the frame is flagged ``conflict`` and is not
valid: on the dev seeds neither cue is right more than about 75 % of the time on those frames
(the per-cue validity tests below do not separate them), so no choice between them reaches the
95 % precision the validity flag has to keep. A frame with only one cue is valid only if the
vanishing-point cue passes its own validity test.

Pipeline code: images and camera constants only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scan as sc
import scan_heading as sh
import segment as sg
import vp

TAU = 3.0  # degrees: the cues must agree within this
VP_TWO_MIN_LEN = 25.0  # px per family for the Manhattan test
VP_F_TOL = 0.05  # the implied focal length must be within 5 % of fx
VP_CENTRE_ALPHA = 20.0  # degrees: "near-centre" peak
VP_SIDE_LEN = 40.0  # px of inlier segments on each side of the image for a centre peak
VP_MIN_SUPPORT = 80.0  # px of inlier segments in total for a centre peak


@dataclass
class HeadingEstimate:
    psi: float = np.nan  # degrees, yaw mod 90 in (-45, 45]
    sigma: float = np.nan  # degrees, of the fused estimate
    valid: bool = False
    confidence: float = 0.0  # 0..1: 1 - disagreement / TAU for agreeing cues
    source: str = "none"  # both | conflict | vp | scan | none
    psi_vp: float = np.nan
    psi_scan: float = np.nan
    delta: float = np.nan  # degrees between the cues
    vp_valid: bool = False


def vp_valid(r: vp.VPResult, fx: float = 92.0) -> bool:
    """The plan's validity test for the vanishing-point cue: either two peaks consistent with
    the Manhattan constraint and the known focal length (within 5 %), each supported by
    enough segment length, or one strong peak near the image centre supported by segments from
    both sides of the image."""
    if not np.isfinite(r.psi):
        return False
    two = (
        min(r.family_len) >= VP_TWO_MIN_LEN
        and np.isfinite(r.f_implied)
        and abs(r.f_implied / fx - 1.0) < VP_F_TOL
    )
    centre = (
        abs(r.alpha_a) <= VP_CENTRE_ALPHA
        and r.left_len >= VP_SIDE_LEN
        and r.right_len >= VP_SIDE_LEN
        and r.support >= VP_MIN_SUPPORT
    )
    return bool(two or centre)


def fuse(
    r_vp: vp.VPResult, r_sc: sh.ScanHeading, fx: float = 92.0, tau: float = TAU
) -> HeadingEstimate:
    out = HeadingEstimate(psi_vp=r_vp.psi, psi_scan=r_sc.psi, vp_valid=vp_valid(r_vp, fx))
    has_vp, has_sc = np.isfinite(r_vp.psi), np.isfinite(r_sc.psi)
    if has_vp and has_sc:
        out.delta = float(abs(vp.diff90(r_vp.psi, r_sc.psi)))
        if out.delta < tau:
            w1 = 1.0 / max(r_vp.sigma, 0.2) ** 2
            w2 = 1.0 / max(r_sc.sigma, 0.2) ** 2
            out.psi = vp.circ_mean90(np.array([r_vp.psi, r_sc.psi]), np.array([w1, w2]))
            out.sigma = float(1.0 / np.sqrt(w1 + w2))
            out.valid = True
            out.confidence = float(max(0.0, 1.0 - out.delta / tau))
            out.source = "both"
        else:
            out.source = "conflict"
        return out
    if has_vp and out.vp_valid:
        out.psi, out.sigma, out.valid, out.source, out.confidence = (
            r_vp.psi,
            r_vp.sigma,
            True,
            "vp",
            0.5,
        )
    elif has_vp or has_sc:
        out.source = "vp" if has_vp else "scan"
    return out


def heading_from_segmentation(seg: sg.Segmentation, lsd=None) -> HeadingEstimate:
    """Both cues and their fusion from a frame that has already been segmented."""
    cam = sc.camera_config()
    scan = sc.scan_from_segmentation(seg, cam)
    return fuse(vp.estimate(seg, "lsd", lsd, cam), sh.estimate(scan, cam), cam.fx)


def estimate_heading(img: np.ndarray, lsd=None) -> HeadingEstimate:
    """The whole pipeline for one frame: segmentation, scan, both cues, fusion."""
    return heading_from_segmentation(sg.segment(img), lsd)
