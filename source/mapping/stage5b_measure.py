"""Stage 5b task 0: texture-adjacency go/no-go measurement (EVALUATION ONLY, rule 10).

Reads true poses (`poses.json`) and the true grid (`gt.json`), so it must never be imported
by pipeline code. For every recorded frame, each image column is ray-cast at the true pose
against the rendered wall segments; the wall pixels of that column are sampled at fixed wall
heights and accumulated per wall *line* (orientation, grid line, viewing side) and 1 cm
position bin. That rebuilds every visible wall face as a rectified strip. From the strips:

1. where along a wall the texture changes (phase of the change relative to the 0.2 m grid);
2. texture units, their identity (clustered by appearance) and ordered neighbour pairs;
3. how unique pairs and triples are within a maze and across mazes.

Usage: python3 source/mapping/stage5b_measure.py build 7 8 9 10 21 22 23 24 25
       python3 source/mapping/stage5b_measure.py analyse 7 8 9 10 21 22 23 24 25
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_fit import Scene, column_dirs  # noqa: E402

HOLDOUT = set(json.loads((Path(__file__).resolve().parent / "seeds.json").read_text()).get("holdout", []))


def data_dir(seed: int) -> Path:
    """Refuses holdout seeds (rule 4); record_gt.gt_dir does the same but imports the simulator."""
    if seed in HOLDOUT:
        raise SystemExit(f"seed {seed} is a holdout seed")
    return ROOT / "data" / f"{PREFIX}-{seed}"

OUT = ROOT / "cache" / "mapping" / "stage5b"
FX = FY = 92.0
CX, CY = 160.0, 120.0
H_CAM, H_WALL = 0.2099, 0.2999
NZ = 16            # wall-height samples
BIN = 0.01         # 1 cm along the wall
NB = 700           # bins per wall line (7 m)
T_MIN, T_MAX = 0.18, 1.4
PREFIX = "gt4"


def cast_index(scene: Scene, origin, dirs):
    """Like Scene.cast, but also returns the index of the box hit first."""
    o = np.asarray(origin, float)
    d = np.asarray(dirs, float)[:, None, :]
    b = scene.boxes
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0 = (b[None, :, 0] - o[0]) / d[..., 0]
        tx1 = (b[None, :, 1] - o[0]) / d[..., 0]
        ty0 = (b[None, :, 2] - o[1]) / d[..., 1]
        ty1 = (b[None, :, 3] - o[1]) / d[..., 1]
    inx = (o[0] >= b[:, 0]) & (o[0] <= b[:, 1])
    iny = (o[1] >= b[:, 2]) & (o[1] <= b[:, 3])
    tx0 = np.where(d[..., 0] == 0, np.where(inx, -np.inf, np.inf), tx0)
    tx1 = np.where(d[..., 0] == 0, np.where(inx, np.inf, -np.inf), tx1)
    ty0 = np.where(d[..., 1] == 0, np.where(iny, -np.inf, np.inf), ty0)
    ty1 = np.where(d[..., 1] == 0, np.where(iny, np.inf, -np.inf), ty1)
    tn = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
    tf = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
    hit = (tn <= tf) & (tf > 1e-9)
    tn = np.where(hit, np.maximum(tn, 0.0), np.inf)
    idx = tn.argmin(axis=1)
    t = tn[np.arange(len(idx)), idx]
    return np.where(np.isfinite(t), t, np.nan), idx


def build(seed: int, step: int = 2) -> Path:
    d = data_dir(seed)
    gt = json.loads((d / "gt.json").read_text())
    occ = np.array([[c == "#" for c in row] for row in gt["grid"]])
    scene = Scene(occ)
    seg = scene.segments
    horiz = np.isclose(seg[:, 1], seg[:, 3])
    cols = np.arange(320)
    zk = (np.arange(NZ) + 0.5) * H_WALL / NZ
    acc: dict[tuple, list] = {}
    for tdir in sorted(d.glob("traj_*")):
        poses = json.loads((tdir / "poses.json").read_text())
        for k in range(0, len(poses), step):
            x, y, yaw = poses[k]
            img = cv2.imread(str(tdir / f"{k}.jpg"))
            if img is None:
                continue
            dirs = column_dirs(yaw, cols, FX, CX)
            t, idx = cast_index(scene, (x, y), dirs)
            ok = np.isfinite(t) & (t > T_MIN) & (t < T_MAX)
            for u in np.nonzero(ok)[0]:
                i = idx[u]
                p = np.array([x, y]) + t[u] * dirs[u]
                if horiz[i]:
                    key = ("h", int(round(seg[i, 1] / 0.2)), int(np.sign(y - seg[i, 1])))
                    pos = p[0]
                else:
                    key = ("v", int(round(seg[i, 0] / 0.2)), int(np.sign(x - seg[i, 0])))
                    pos = p[1]
                if key[2] == 0:
                    continue
                b = int(round((pos + 0.3) / BIN))
                if not 0 <= b < NB:
                    continue
                v = np.round(CY + FY * (H_CAM - zk) / t[u] - 0.5).astype(int)
                if v.min() < 0 or v.max() > 239:
                    continue
                if key not in acc:
                    acc[key] = [np.zeros((NB, NZ, 3)), np.zeros(NB)]
                acc[key][0][b] += img[v, u].astype(float)
                acc[key][1][b] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sorted(acc)
    path = OUT / f"strips_{PREFIX}_{seed}.npz"
    np.savez_compressed(
        path,
        keys=np.array([f"{a}|{b}|{c}" for a, b, c in keys]),
        sums=np.stack([acc[k][0] for k in keys]).astype(np.float32),
        counts=np.stack([acc[k][1] for k in keys]).astype(np.float32),
    )
    return path


if __name__ == "__main__":
    cmd, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    if cmd == "build":
        for s in seeds:
            print(s, build(s), flush=True)
