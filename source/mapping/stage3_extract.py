"""Stage 3: run the pipeline cues on every frame and cache the results (pipeline code path:
images and camera constants only; ground truth is read only by ``stage3.py``).

    uv run source/mapping/stage3_extract.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_gt  # noqa: E402
import scan as sc  # noqa: E402
import scan_heading as sh  # noqa: E402
import segment as sg  # noqa: E402
import vp  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
CFG = json.loads(record_gt.SEEDS.read_text())
VP_FIELDS = (
    "psi",
    "sigma",
    "support",
    "n_inl",
    "fam_a",
    "fam_b",
    "left",
    "right",
    "f_impl",
    "nseg",
)
SH_FIELDS = ("psi", "sigma", "support", "n_pieces", "n_inl", "fam_x", "fam_y")


def code_hash() -> str:
    h = hashlib.sha1()
    here = Path(__file__).parent
    for f in ("segment.py", "scan.py", "vp.py", "scan_heading.py"):
        h.update((here / f).read_bytes())
    h.update((CACHE / "camera.json").read_bytes())
    return h.hexdigest()[:8]


def vp_row(r: vp.VPResult) -> list:
    return [
        r.psi,
        r.sigma,
        r.support,
        r.n_inliers,
        r.family_len[0],
        r.family_len[1],
        r.left_len,
        r.right_len,
        r.f_implied,
        r.n_segments,
    ]


def sh_row(r: sh.ScanHeading) -> list:
    return [
        r.psi,
        r.sigma,
        r.support,
        r.n_pieces,
        r.n_inliers,
        r.family_support[0],
        r.family_support[1],
    ]


def msaa4_frames(seed: int, drive: int):
    """EVALUATION ONLY: frames as the server renders them (msaa 4, anti-aliased edges), made
    by rendering at the true poses of the recorded frames and encoding with the recorder's JPEG
    settings. The official local recordings use the simulator default msaa 1."""
    import vis_nav_sim as sim

    world = sim.Simulator(sim.Textures.find(), seed, msaa=4)
    poses = record_gt.load_poses(record_gt.gt_dir(seed))[drive]
    for x, y, yaw in poses:
        frame = world.render(float(x), float(y), float(yaw))
        jpg = world.jpeg(frame, 90, "444")
        yield cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def process_drive(args) -> dict:
    seed, drive, variant = args
    prefix = "gt4" if variant == "gt4" else "gt"
    path = record_gt.gt_dir(seed, prefix=prefix) / f"traj_{drive}"
    info = json.loads((path / "data_info.json").read_text())
    n = len(info)
    images = (
        msaa4_frames(seed, drive)
        if variant == "msaa4"
        else (cv2.imread(str(path / r["image"])) for r in info)
    )
    lsd = vp.make_lsd()
    rows = np.full((n, 320), np.nan, np.float32)
    top = np.full((n, 320), np.nan, np.float32)
    status = np.zeros((n, 320), np.int8)
    v_lsd = np.full((n, len(VP_FIELDS)), np.nan)
    v_poly = np.full((n, len(VP_FIELDS)), np.nan)
    s_h = np.full((n, len(SH_FIELDS)), np.nan)
    for k, img in enumerate(images):
        seg = sg.segment(img)
        scan = sc.scan_from_segmentation(seg)
        rows[k], top[k], status[k] = seg.rows, seg.top_rows, scan.status
        v_lsd[k] = vp_row(vp.estimate(seg, "lsd", lsd))
        v_poly[k] = vp_row(vp.estimate(seg, "polyline"))
        s_h[k] = sh_row(sh.estimate(scan))
    return {
        "rows": rows,
        "top": top,
        "status": status,
        "vp_lsd": v_lsd,
        "vp_poly": v_poly,
        "sh": s_h,
    }


def run_seed(seed: int, force: bool = False, variant: str = "official") -> list[dict]:
    f = CACHE / f"stage3_cues_{variant}_{seed}_{code_hash()}.npz"
    if f.exists() and not force:
        z = np.load(f)
        return [
            {k: z[f"{k}{d}"] for k in ("rows", "top", "status", "vp_lsd", "vp_poly", "sh")}
            for d in range(3)
        ]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(process_drive, [(seed, d, variant) for d in range(3)]))
    np.savez(f, **{f"{k}{d}": res[d][k] for d in range(3) for k in res[d]})
    print("extracted", seed, [len(r["rows"]) for r in res], flush=True)
    return res


def main() -> None:
    variants = sys.argv[1:] or ["official", "msaa4", "gt4"]
    for variant in variants:
        for s in CFG["dev"] + CFG["test"]:
            run_seed(s, variant=variant)


if __name__ == "__main__":
    main()
