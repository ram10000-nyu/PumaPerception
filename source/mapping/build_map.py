"""Raw exploration data to the top-down map, end to end (pipeline code, rule 10: images and
action labels only).

    python3 source/mapping/build_map.py --data data/gt4-21              # every step
    python3 source/mapping/build_map.py --data <dir> --steps inputs     # one step (resumable)

``<dir>`` holds ``traj_0`` ... with ``data_info.json`` and the images. Steps, each cached in
``--out`` (default ``cache/mapping/build/<dir name>``) so a run can be split across calls:

1. ``inputs``    per drive, in parallel: segmentation, scan, heading cues, frame-to-frame SIFT
                 matches and collision flags (``pipeline.drive_inputs``, stages 2 to 4), the
                 loop-closure features (stage 5) and texture boundaries (stage 5b);
2. ``odometry``  per-drive self-calibrated priors and the stage 4 filter (variant C, stage 4b);
3. ``closures``  stage 5b union-rule loop closures (``slam.closures``);
4. ``graph``     stage 6 pose graph (``slam.optimise``);
5. ``map``       stage 7 map, snapped walls, configuration space, wall-face signatures
                 (``grid``), written to ``map.npz``, ``map.png``, ``map_snapped.png``.

``--ticks auto|N`` (default ``auto``, stage 9): the pipeline is tuned on frames 5 ticks apart.
``auto`` detects the data's frame rate (``adapter.ticks_per_frame``); if it is not 5 the drives
are subsampled to every 5th frame into ``<out>/adapted`` first, and the map is built from that.
The detected rate and its evidence are written to ``<out>/adapter.json``.

``timing.json`` records each step's wall time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapter  # noqa: E402
import calibration as cal  # noqa: E402
import grid  # noqa: E402
import loops  # noqa: E402
import motion as M  # noqa: E402
import odometry as od  # noqa: E402
import pipeline  # noqa: E402
import scan as sc  # noqa: E402
import segment as sg  # noqa: E402
import slam  # noqa: E402
import texture as tx  # noqa: E402
import vo  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STEPS = ("inputs", "odometry", "closures", "graph", "map")


def drives(data: Path) -> list[Path]:
    return sorted(data.glob("traj_*"), key=lambda p: int(p.name.split("_")[1]))


def _drive_inputs(path: Path):
    cv2.setNumThreads(1)
    inp = pipeline.drive_inputs(path)
    info = json.loads((path / "data_info.json").read_text())
    sift = cv2.SIFT_create(nfeatures=loops.N_FEATURES)
    cam = sc.camera_config()
    feats, bounds = [], []
    for r in info:
        img = cv2.imread(str(path / r["image"]))
        feats.append(loops.frame_features(img, sg.segment(img), sift))
        bounds.append(tx.boundaries(img, cam=cam))
    return inp, feats, bounds


def step_inputs(data, out, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_drive_inputs, drives(data)))
    (out / "inputs.pkl").write_bytes(pickle.dumps(res))


def _run(args):
    inp, prior = args
    return od.run_drive(inp, prior, "C")


def step_odometry(data, out, workers):
    res = pickle.loads((out / "inputs.pkl").read_bytes())
    inputs = [r[0] for r in res]
    priors, scales = cal.calibrated_priors(inputs, M.MotionPrior.load())
    with ProcessPoolExecutor(workers) as ex:
        odo = list(ex.map(_run, [(i, p) for i, p in zip(inputs, priors, strict=True)]))
    (out / "odometry.pkl").write_bytes(pickle.dumps(odo))
    (out / "calibration.json").write_text(json.dumps([s.as_dict() for s in scales], indent=1, default=float))


def step_closures(data, out, workers):
    res = pickle.loads((out / "inputs.pkl").read_bytes())
    odo = pickle.loads((out / "odometry.pkl").read_bytes())
    scans = [[vo.scan_points(inp.rows[k], inp.top[k], inp.status[k]) for k in range(len(inp.labels))] for inp, _, _ in res]
    cl = slam.closures([r[1] for r in res], [r[2] for r in res], scans, [v.poses for v in odo], workers=workers)
    (out / "closures.pkl").write_bytes(pickle.dumps(cl))


def step_graph(data, out, workers):
    odo = pickle.loads((out / "odometry.pkl").read_bytes())
    cl = pickle.loads((out / "closures.pkl").read_bytes())
    poses, diag = slam.optimise(odo, cl)
    (out / "poses_optimized.json").write_text(json.dumps({"drives": [np.round(p, 6).tolist() for p in poses], "graph": diag}))


def step_map(data, out, workers):
    res = pickle.loads((out / "inputs.pkl").read_bytes())
    poses = [np.array(p) for p in json.loads((out / "poses_optimized.json").read_text())["drives"]]
    scans = [[(inp.rows[k], inp.top[k], inp.status[k]) for k in range(len(inp.labels))] for inp, _, _ in res]
    m = grid.build(scans, poses, slam.SPAWN[:2])

    def frames():
        for d, path in enumerate(drives(data)):
            info = json.loads((path / "data_info.json").read_text())
            inp = res[d][0]
            for k, r in enumerate(info):
                yield d, k, cv2.imread(str(path / r["image"])), inp.rows[k], inp.top[k], inp.status[k]

    grid.add_signatures(m, frames())
    grid.save(m, out / "map.npz")
    lat = m.lattice
    title = f"{data.name}: cell size {lat.period / 2:.4f} m, scale corrected by {100 * (1 / lat.scale - 1):+.2f} %"
    grid.render(m, out / "map.png", title)
    grid.render(m, out / "map_snapped.png", title + "; snapped walls, robot-centre free space", snapped=True)


def adapt(data: Path, out: Path, ticks: str) -> Path:
    """The directory to build from: ``data`` itself at 5 ticks per frame, else its subsampled
    copy in ``<out>/adapted``. Resumable: an existing ``adapter.json`` is reused."""
    f = out / "adapter.json"
    if f.exists() and ticks == "auto":
        info = json.loads(f.read_text())
        return Path(info["built_from"])
    if ticks == "auto":
        k, ev = adapter.ticks_per_frame(data)
        evidence = ev.as_dict()
    else:
        k, evidence = int(ticks), None
    info = {"k": k, "evidence": evidence, "source": str(data), "built_from": str(data)}
    if k != 5:
        t0 = time.perf_counter()
        adapted = out / "adapted"
        step = adapter.step_to_five_ticks(k)  # frames to skip: 5 ticks apart after subsampling
        adapter.subsample_dataset(data, step, adapted)
        info.update(built_from=str(adapted), subsample_step=step, subsample_s=round(time.perf_counter() - t0, 2))
    f.write_text(json.dumps(info, indent=1))
    return Path(info["built_from"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out")
    p.add_argument("--steps", default="all", help="comma-separated subset of " + ",".join(STEPS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ticks", default="auto", help="ticks per frame: auto (detect) or N; not 5 means subsample")
    a = p.parse_args()
    data = Path(a.data).resolve()
    out = Path(a.out) if a.out else ROOT / "cache" / "mapping" / "build" / data.name
    out.mkdir(parents=True, exist_ok=True)
    steps = STEPS if a.steps == "all" else tuple(a.steps.split(","))
    tfile = out / "timing.json"
    timing = json.loads(tfile.read_text()) if tfile.exists() else {}
    t0 = time.perf_counter()
    built = adapt(data, out, a.ticks)
    timing["adapter"] = round(time.perf_counter() - t0, 2)
    k = json.loads((out / "adapter.json").read_text())["k"]
    print(f"adapter: {k} tick(s) per frame{'' if built == data else f', subsampled into {built}'} ({timing['adapter']:.1f} s)", flush=True)
    for s in steps:
        t0 = time.perf_counter()
        globals()[f"step_{s}"](built, out, a.workers)
        timing[s] = round(time.perf_counter() - t0, 2)
        tfile.write_text(json.dumps(timing, indent=1))
        print(f"{s}: {timing[s]:.1f} s", flush=True)
    total = sum(timing.get(s, 0) for s in ("adapter", *STEPS))
    print(f"total so far: {total:.1f} s ({', '.join(s for s in STEPS if s not in timing) or 'all steps done'})")


if __name__ == "__main__":
    main()
