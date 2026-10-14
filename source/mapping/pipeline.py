"""Images to trajectory: the stage 4 pipeline as one function chain (rule 10: images and labels
only; no ground truth, no ``poses.json``).

    inputs = drive_inputs(drive_dir)            # segmentation, scans, headings, matches, flags
    result = odometry.run_drive(inputs, MotionPrior.load(), "C")
"""

from __future__ import annotations

import json
from pathlib import Path

import collision as col
import cv2
import heading as hd
import numpy as np
import odometry as od
import scan as sc
import segment as sg
import vo
import vp


def drive_inputs(path: str | Path, limit: int | None = None) -> od.DriveInputs:
    """Everything the filter reads for one drive directory (``data_info.json`` and the images)."""
    path = Path(path)
    info = json.loads((path / "data_info.json").read_text())[:limit]
    n = len(info)
    lsd = vp.make_lsd()
    sift = cv2.SIFT_create(nfeatures=400)
    bf = cv2.BFMatcher()
    rows = np.full((n, 320), np.nan, np.float32)
    top = np.full((n, 320), np.nan, np.float32)
    status = np.zeros((n, 320), np.int8)
    psi = np.full(n, np.nan)
    psi_sigma = np.full(n, np.nan)
    psi_best = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    rot = np.full(n, np.nan)
    rot_n = np.zeros(n, int)
    labels = [r["action"][0] for r in info]
    matches = [np.zeros((0, 4))]
    prev_feat = prev_det = None
    for k, r in enumerate(info):
        img = cv2.imread(str(path / r["image"]))
        seg = sg.segment(img)
        scan = sc.scan_from_segmentation(seg)
        rows[k], top[k], status[k] = seg.rows, seg.top_rows, scan.status
        h = hd.heading_from_segmentation(seg, lsd)
        if h.valid:
            psi[k], psi_sigma[k] = h.psi, h.sigma
        psi_best[k], valid[k] = col.best_psi(h), h.valid
        feat = vo.wall_features(img, seg, sift)
        det = col.sift_features(img, seg, sift)
        if prev_feat is not None:
            matches.append(vo.match_features(prev_feat, feat, bf))
            if labels[k] in col.LABELS:
                rot[k], rot_n[k], _ = col.rotation_from_matches(prev_det, det, bf)
        prev_feat, prev_det = feat, det
    flags, _ = col.decide_drive(psi_best, rot, rot_n, labels, valid)
    return od.DriveInputs(rows, top, status, psi, psi_sigma, labels, flags, matches)
