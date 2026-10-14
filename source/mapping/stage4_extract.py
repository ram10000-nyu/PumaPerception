"""Stage 4: SIFT matches between consecutive frames, cached (pipeline path: images only).

uv run source/mapping/stage4_extract.py [gt4|official] ...
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
import record_gt  # noqa: E402
import segment as sg  # noqa: E402
import vo  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
CFG = json.loads(record_gt.SEEDS.read_text())


def code_hash() -> str:
    h = hashlib.sha1()
    for fn in (vo.wall_features, vo.match_features):
        h.update(inspect.getsource(fn).encode())
    h.update((Path(__file__).with_name("segment.py")).read_bytes())
    return h.hexdigest()[:8]


def _drive(args):
    seed, drive, variant = args
    prefix = "gt4" if variant == "gt4" else "gt"
    path = record_gt.gt_dir(seed, prefix=prefix) / f"traj_{drive}"
    info = json.loads((path / "data_info.json").read_text())
    sift = cv2.SIFT_create(nfeatures=400)
    bf = cv2.BFMatcher()
    out = [np.zeros((0, 4))]
    prev = None
    for r in info[0:]:
        img = cv2.imread(str(path / r["image"]))
        f = vo.wall_features(img, sg.segment(img), sift)
        if prev is not None:
            out.append(vo.match_features(prev, f, bf))
        prev = f
    return out


def run_seed(seed: int, variant: str, force: bool = False):
    f = CACHE / f"stage4_matches_{variant}_{seed}_{code_hash()}.npz"
    if f.exists() and not force:
        z = np.load(f, allow_pickle=True)
        return [list(z[f"d{d}"]) for d in range(3)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_drive, [(seed, d, variant) for d in range(3)]))
    arr = {}
    for d in range(3):
        o = np.empty(len(res[d]), dtype=object)
        for i, m in enumerate(res[d]):
            o[i] = m
        arr[f"d{d}"] = o
    np.savez(f, **arr)
    print("matched", variant, seed, flush=True)
    return res


def main() -> None:
    for variant in sys.argv[1:] or ["gt4", "official"]:
        for s in CFG["dev"] + CFG["test"]:
            run_seed(s, variant)


if __name__ == "__main__":
    main()
