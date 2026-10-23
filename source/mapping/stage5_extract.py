"""Stage 5: wall-mask SIFT keypoints and descriptors for every frame, cached (pipeline path:
images only). Loop closures need any frame's features, not just the previous frame's, so they
are stored per frame: keypoints (float32, continuous pixel coordinates) and raw SIFT
descriptors (uint8; RootSIFT is applied when they are used).

    uv run python source/mapping/stage5_extract.py [gt4|official] ...
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loops  # noqa: E402
import record_gt  # noqa: E402
import segment as sg  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping" / "stage5"
CFG = json.loads(record_gt.SEEDS.read_text())


def code_hash() -> str:
    h = hashlib.sha1()
    h.update(inspect.getsource(loops.frame_features).encode())
    h.update(str(loops.N_FEATURES).encode())
    h.update((Path(__file__).with_name("segment.py")).read_bytes())
    return h.hexdigest()[:8]


def drive_features(path: Path) -> list[loops.Features]:
    """Features of every frame of one drive directory."""
    info = json.loads((path / "data_info.json").read_text())
    sift = cv2.SIFT_create(nfeatures=loops.N_FEATURES)
    out = []
    for r in info:
        img = cv2.imread(str(path / r["image"]))
        out.append(loops.frame_features(img, sg.segment(img), sift))
    return out


def _drive(args):
    seed, drive, variant = args
    prefix = "gt4" if variant == "gt4" else "gt"
    return drive_features(record_gt.gt_dir(seed, prefix=prefix) / f"traj_{drive}")


def save(path: Path, drives: list[list[loops.Features]]) -> None:
    arr = {}
    for d, feats in enumerate(drives):
        arr[f"n{d}"] = np.array([len(f.pts) for f in feats])
        arr[f"pts{d}"] = np.concatenate([f.pts for f in feats]).astype(np.float32)
        arr[f"desc{d}"] = np.concatenate([f.desc for f in feats]).astype(np.uint8)
    np.savez(path, **arr)


def load(path: Path) -> list[list[loops.Features]]:
    z = np.load(path)
    out = []
    d = 0
    while f"n{d}" in z:
        n, pts, desc = z[f"n{d}"], z[f"pts{d}"], z[f"desc{d}"]
        ends = np.cumsum(n)
        starts = ends - n
        out.append([loops.Features(pts[a:b], desc[a:b]) for a, b in zip(starts, ends, strict=True)])
        d += 1
    return out


def run_seed(seed: int, variant: str, force: bool = False) -> list[list[loops.Features]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"features_{variant}_{seed}_{code_hash()}.npz"
    if f.exists() and not force:
        return load(f)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_drive, [(seed, d, variant) for d in range(3)]))
    save(f, res)
    print("features", variant, seed, flush=True)
    return res


def main() -> None:
    for variant in sys.argv[1:] or ["gt4", "official"]:
        for s in CFG["dev"] + CFG["test"]:
            run_seed(s, variant)


if __name__ == "__main__":
    main()
