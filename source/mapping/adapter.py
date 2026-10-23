"""Frame-rate adapter (stage 9): per-tick challenge data to the 5-tick spacing every stage was
tuned on (pipeline code, rule 10: images and action labels only).

Every frame-counted setting (the motion prior, collision thresholds, closure exclusion window,
thinning window, heading priors, odometry inflation) assumes frames 5 ticks apart. A challenge
download can be one frame per tick (stage 4b: ``ab7efaeb1abf``). Keeping every k-th frame makes
such data identical in kind to the local recordings, and the ticks in between give each kept
frame its exact action composition.

* ``ticks_per_frame(drive_dir)`` detects k from the data alone: how far the robot moves and
  turns between consecutive frames, in units of one tick, with the label-repeat rate as a
  consistency check. It raises rather than guesses.
* ``subsample(drive_dir, k, out_dir)`` keeps every k-th frame. Its ``action`` is the **last**
  tick's label (the stage 1 convention for a window); ``composition`` counts every action over
  the window's k ticks.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import odometry as od
import scan as sc
import segment as sg
import vo
import vp

TICK_M = 0.03  # FORWARD, per tick (simulator constant, stage 0)
TICK_DEG = float(np.degrees(0.06))  # LEFT/RIGHT, per tick
N_PAIRS = 240  # consecutive-frame pairs measured per drive
REPEAT_PER_TICK_MIN = 0.65  # label repeat rate: per-tick data 0.80, 5-tick data 0.45
MAX_K = 10


class AdapterError(RuntimeError):
    """The frame rate could not be determined (or the evidence is inconsistent)."""


@dataclass
class Evidence:
    k: int
    repeat_rate: float  # (a)
    forward_ticks: float  # (b) median forward motion per FORWARD-after-FORWARD frame / 0.03 m
    turn_ticks: float  # (c) turn per LEFT/RIGHT-after-same frame / 3.44 deg (see TURN_QUANTILE)
    n_forward: int
    n_turn: int

    def as_dict(self) -> dict:
        return asdict(self)


TURN_QUANTILE = 0.5  # of the per-frame turn; chosen on dev (stage 9)


def drive_dirs(data: Path) -> list[Path]:
    data = Path(data)
    if (data / "data_info.json").exists():
        return [data]
    return sorted(data.glob("traj_*"), key=lambda p: int(p.name.split("_")[1]))


def labels_of(drive_dir: Path) -> list[str]:
    info = json.loads((Path(drive_dir) / "data_info.json").read_text())
    return [r["action"][0] if r["action"] else "IDLE" for r in info]


def repeat_rate(labels: list[str]) -> float:
    """Fraction of frames whose label equals the previous frame's."""
    return float(np.mean([a == b for a, b in zip(labels[1:], labels[:-1], strict=True)]))


def _eligible(labels: list[str], kinds: tuple[str, ...]) -> list[int]:
    return [i for i in range(1, len(labels)) if labels[i] in kinds and labels[i - 1] == labels[i]]


def _spread(idx: list[int], n: int) -> list[int]:
    if len(idx) <= n:
        return idx
    return [idx[j] for j in np.linspace(0, len(idx) - 1, n).astype(int)]


def window_evidence(drive_dir: Path, n_pairs: int = N_PAIRS):
    """Per-pair motion between consecutive frames, in ticks: ``(forward, turn)`` arrays.
    Forward: scan ICP translation on FORWARD-after-FORWARD pairs. Turn: the change of the
    stage 3 compass (modulo 90 deg) on LEFT/RIGHT-after-the-same pairs."""
    drive_dir = Path(drive_dir)
    info = json.loads((drive_dir / "data_info.json").read_text())
    labels = labels_of(drive_dir)
    lsd = vp.make_lsd()
    cache: dict[int, tuple] = {}

    def frame(i):
        if i not in cache:
            img = cv2.imread(str(drive_dir / info[i]["image"]))
            seg = sg.segment(img)
            scan = sc.scan_from_segmentation(seg)
            pts = vo.scan_points(seg.rows, seg.top_rows, scan.status)
            import heading as hd

            h = hd.heading_from_segmentation(seg, lsd)
            cache[i] = (pts, (h.psi, h.sigma) if h.valid else None)
        return cache[i]

    fwd, turn = [], []
    for i in _spread(_eligible(labels, ("FORWARD",)), n_pairs):
        a, b = frame(i - 1)[0], frame(i)[0]
        init = np.array([0.5 * TICK_M, 0.0, 0.0])
        m = vo.icp_symmetric(a, b, init, init_sigma=np.array([0.5, 0.5, 0.3]))
        if m is not None and np.sqrt(m.cov[0, 0]) < 0.02:  # the corridor axis is measured
            fwd.append(abs(m.mean[0]) / TICK_M)
    for i in _spread(_eligible(labels, ("LEFT", "RIGHT")), n_pairs):
        pa, pb = frame(i - 1)[1], frame(i)[1]
        if pa is not None and pb is not None:
            turn.append(abs(od.vp_diff90(pb[0], pa[0])) / TICK_DEG)
    return np.array(fwd), np.array(turn)


def ticks_per_frame(data: Path, n_pairs: int = N_PAIRS) -> tuple[int, Evidence]:
    """Ticks between consecutive frames, from the data alone: ``(k, evidence)``. ``data`` is a
    drive directory or a download holding ``traj_<i>`` directories (evidence is pooled)."""
    dirs = drive_dirs(data)
    labels = [labels_of(d) for d in dirs]
    same = [[a == b for a, b in zip(lab[1:], lab[:-1], strict=True)] for lab in labels]
    rate = float(np.mean(np.concatenate(same)))
    fwd, turn = [], []
    for d in dirs:
        f, t = window_evidence(d, n_pairs)
        fwd.append(f)
        turn.append(t)
    fwd, turn = np.concatenate(fwd), np.concatenate(turn)
    if len(fwd) < 20 or len(turn) < 20:
        raise AdapterError(f"too little evidence ({len(fwd)} forward, {len(turn)} turn pairs)")
    kb, kc = float(np.median(fwd)), float(np.quantile(turn, TURN_QUANTILE))
    return decide_k(rate, kb, kc, len(fwd), len(turn))


def decide_k(
    rate: float, forward_ticks: float, turn_ticks: float, n_forward: int = 0, n_turn: int = 0
):
    """The frame rate from the three pieces of evidence: ``k`` is the rounded median of (b) and
    (c) (their mean), and (a) must agree. Raises ``AdapterError`` rather than guess."""
    k = int(round(float(np.median([forward_ticks, turn_ticks]))))
    ev = Evidence(k, rate, forward_ticks, turn_ticks, n_forward, n_turn)
    if not 1 <= k <= MAX_K:
        raise AdapterError(f"implausible frame rate k={k}: {ev}")
    if int(round(forward_ticks)) != int(round(turn_ticks)):
        raise AdapterError(f"forward and turn evidence disagree on the frame rate: {ev}")
    if (k == 1) != (rate >= REPEAT_PER_TICK_MIN):
        raise AdapterError(f"the label-repeat rate ({rate:.2f}) disagrees with k={k}: {ev}")
    return k, ev


# ----------------------------------------------------------------------------- subsampling


def window_labels(labels: list[list[str]], k: int, offset: int) -> list[dict]:
    """For each kept frame ``offset + j k`` the action composition over its window of ``k``
    ticks (frames ``offset + j k - k + 1`` ... ``offset + j k``; a window that starts before
    frame 0 is cut there), and its last tick's label."""
    out = []
    for j, i in enumerate(range(offset, len(labels), k)):
        window = labels[max(0, i - k + 1) : i + 1]
        comp: dict[str, int] = {}
        for a in window:
            comp[a] = comp.get(a, 0) + 1
        out.append({"frame": i, "action": labels[i], "composition": comp, "index": j})
    return out


def step_to_five_ticks(k: int) -> int:
    """Frames to skip so that the kept frames are 5 ticks apart, for data ``k`` ticks per frame."""
    if k < 1 or 5 % k:
        raise AdapterError(f"data {k} ticks per frame cannot be subsampled to 5 ticks per frame")
    return 5 // k


def subsample(drive_dir: Path, k: int, out_dir: Path, offset: int | None = None) -> list[dict]:
    """Keep every ``k``-th frame, ``offset, offset + k, ...``, of a drive whose frames are one
    tick apart (``k = 5`` gives the local 5-tick spacing; for other data use
    ``step_to_five_ticks``), written to ``out_dir`` in the local format (``data_info.json`` and
    ``<j>.jpg``). ``offset`` defaults to ``k - 1``: the first saved frame is ``k`` ticks after
    the spawn, as locally (stage 0), i.e. the first per-tick frame is one tick after it."""
    drive_dir, out_dir = Path(drive_dir), Path(out_dir)
    offset = k - 1 if offset is None else offset
    info = json.loads((drive_dir / "data_info.json").read_text())
    labels = labels_of(drive_dir)
    kept = window_labels(labels, k, offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for w in kept:
        src = drive_dir / info[w["frame"]]["image"]
        name = f"{w['index']}.jpg"
        dst = out_dir / name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        records.append(
            {
                "step": w["index"],
                "image": name,
                "action": [w["action"]],
                "composition": w["composition"],
            }
        )
    (out_dir / "data_info.json").write_text(json.dumps(records))
    return records


def subsample_dataset(data: Path, k: int, out: Path, offset: int | None = None) -> Path:
    """``subsample`` every ``traj_<i>`` of a download; copies ``target.jpg`` along."""
    data, out = Path(data), Path(out)
    for d in drive_dirs(data):
        subsample(d, k, out / d.name, offset)
    if (data / "target.jpg").exists():
        shutil.copyfile(data / "target.jpg", out / "target.jpg")
    meta = {"k": k, "offset": k - 1 if offset is None else offset, "source": str(data)}
    (out / "adapter.json").write_text(json.dumps(meta))
    return out
