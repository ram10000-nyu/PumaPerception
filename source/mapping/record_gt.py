"""Ground-truth exploration datasets for the top-down mapping work (local simulator only).

For each seed, ``data/gt-<seed>/`` holds exactly what ``cli.record_exploration_data`` writes
(``traj_<i>/<k>.jpg``, ``data_info.json``, ``target.jpg``) plus ``traj_<i>/poses.json``, the
true ``[x, y, yaw]`` of every saved frame by index, and ``gt.json``: spawn, goal pose, the
maze grid and the camera. ``record(student_files=False)`` is what writes the poses; the
driving is the same code, so the frames are the ones a challenge would hand out.

    uv run source/mapping/record_gt.py                  # every seed in seeds.json
    uv run source/mapping/record_gt.py --seeds 7 8      # chosen seeds
    uv run source/mapping/record_gt.py --verify         # also run the stage 0 checks
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli  # noqa: E402
import vis_nav_sim as sim  # noqa: E402

ROOT = cli.ROOT
SEEDS = Path(__file__).with_name("seeds.json")
CAPTURE_EVERY = 5
ROUTES = 3


class HoldoutError(RuntimeError):
    """A holdout seed was requested outside the final evaluation."""


_FINAL = False


def holdout_seeds() -> list[int]:
    return list(json.loads(SEEDS.read_text()).get("holdout", []))


def allow_final() -> None:
    """Called by an evaluation script given ``--final`` (the end of stage 8); until then
    ``gt_dir`` refuses the holdout seeds (plan rule 4)."""
    global _FINAL
    _FINAL = True


def _path(seed: int, dest: str | Path, prefix: str) -> Path:
    return Path(dest) / f"{prefix}-{seed}"


def gt_dir(seed: int, dest: str | Path = ROOT / "data", prefix: str = "gt") -> Path:
    """``data/gt-<seed>`` (the official ``msaa=1`` recording) or ``data/gt4-<seed>`` (same
    seed, routes and noise, rendered with the server's ``msaa=4``). Holdout seeds are refused
    unless the script was started with ``--final`` (``allow_final``)."""
    if int(seed) in holdout_seeds() and not _FINAL:
        raise HoldoutError(
            f"seed {seed} is a holdout seed: it is opened once, by the final evaluation "
            "(pass --final); see rule 4 of the plan"
        )
    return _path(seed, dest, prefix)


def make_world(seed: int, textures: sim.Textures | None = None, msaa: int = 1) -> sim.Simulator:
    """The simulator configured as ``cli.record_exploration_data`` configures it (``msaa=1``,
    the simulator default); ``msaa=4`` is what the server renders with."""
    textures = textures or sim.Textures.find()
    return sim.Simulator(
        textures, seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed, msaa=msaa
    )


def record(seed: int, dest: str | Path = ROOT / "data", textures=None, msaa: int = 1) -> Path:
    path = _path(seed, dest, "gt4" if msaa == 4 else "gt")
    world = make_world(seed, textures, msaa)
    summary = world.record(
        path, routes=ROUTES, seed=seed, capture_every=CAPTURE_EVERY, student_files=False
    )
    meta = {
        "seed": seed,
        "spawn": world.spawn,
        "target": world.target,
        "cell_size": world.cell_size,
        "grid": world.grid,
        "camera": world.camera,
        "maze_sha256": world.maze_sha256,
        "summary": summary,
    }
    (path / "gt.json").write_text(json.dumps(meta, indent=1))
    return path


def record_ticks(
    seed: int,
    capture_every: int,
    dest: str | Path = ROOT / "data",
    textures=None,
    msaa: int = 4,
) -> Path:
    """Stage 9: the same maze, routes and noise as ``record`` but one frame every
    ``capture_every`` ticks, in ``data/tick<N>-<seed>``. With ``capture_every=1`` this is the
    format of a challenge download. The driving does not depend on ``capture_every``
    (stage 0), so frame ``N * j + c`` of a per-tick recording is frame ``j`` of the 5-tick one."""
    if seed in holdout_seeds():
        raise HoldoutError(f"seed {seed} is a holdout seed")
    path = _path(seed, dest, f"tick{capture_every}")
    world = make_world(seed, textures, msaa)
    summary = world.record(
        path, routes=ROUTES, seed=seed, capture_every=capture_every, student_files=False
    )
    meta = {
        "seed": seed,
        "spawn": world.spawn,
        "target": world.target,
        "cell_size": world.cell_size,
        "grid": world.grid,
        "camera": world.camera,
        "maze_sha256": world.maze_sha256,
        "summary": summary,
        "capture_every": capture_every,
    }
    (path / "gt.json").write_text(json.dumps(meta, indent=1))
    return path


def record_biased(
    seed: int,
    motion_bias: tuple[float, float, float],
    motion_seed: int,
    dest: str | Path = ROOT / "data",
    textures=None,
    msaa: int = 4,
) -> Path:
    """Stage 4b: the same maze and routes as ``record`` but a run with a systematic motion bias
    (``Simulator(motion_bias=...)``, drawn once from ``motion_seed``), in ``data/bias-<seed>``.
    Holdout seeds are refused. ``gt.json`` also stores the drawn bias (evaluation only)."""
    if seed in holdout_seeds():
        raise HoldoutError(f"seed {seed} is a holdout seed")
    textures = textures or sim.Textures.find()
    world = sim.Simulator(
        textures,
        seed,
        motion_noise=sim.DEFAULT_NOISE,
        motion_bias=motion_bias,
        motion_seed=motion_seed,
        msaa=msaa,
    )
    path = _path(seed, dest, "bias")
    summary = world.record(
        path, routes=ROUTES, seed=seed, capture_every=CAPTURE_EVERY, student_files=False
    )
    meta = {
        "seed": seed,
        "spawn": world.spawn,
        "target": world.target,
        "cell_size": world.cell_size,
        "grid": world.grid,
        "camera": world.camera,
        "maze_sha256": world.maze_sha256,
        "summary": summary,
        "motion_bias_sigmas": list(motion_bias),
        "motion_seed": motion_seed,
        "drawn_bias": list(world.drawn_bias()),
    }
    (path / "gt.json").write_text(json.dumps(meta, indent=1))
    return path


def load_poses(path: str | Path) -> list[np.ndarray]:
    """Per drive, an (n, 3) array of true ``x, y, yaw`` aligned with ``data_info.json``."""
    path = Path(path)
    drives = sorted(path.glob("traj_*"), key=lambda p: int(p.name.split("_")[1]))
    return [np.array(json.loads((d / "poses.json").read_text())) for d in drives]


def check_matches_student(seed: int, gt: Path, textures) -> dict:
    """Frame-for-frame equality with ``cli.record_exploration_data`` for the same seed."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = cli.record_exploration_data(seed, tmp)
        files = sorted(p.relative_to(ref) for p in ref.rglob("*") if p.is_file())
        differ = [str(f) for f in files if not filecmp.cmp(ref / f, gt / f, shallow=False)]
        extra = sorted(
            str(p.relative_to(gt))
            for p in gt.rglob("*")
            if p.is_file() and p.relative_to(gt) not in set(files)
        )
    return {"files": len(files), "differ": differ, "extra_in_gt": extra}


def check_pose_alignment(seed: int, gt: Path, textures, n: int = 20) -> dict:
    """Re-render the simulator at the logged pose of random frames; compare to the jpg."""
    world = make_world(seed, textures)
    rng = np.random.default_rng(seed)
    poses = load_poses(gt)
    errs = []
    for _ in range(n):
        d = int(rng.integers(len(poses)))
        k = int(rng.integers(len(poses[d])))
        x, y, yaw = poses[d][k]
        img = cv2.imread(str(gt / f"traj_{d}" / f"{k}.jpg"))
        # the saved frame went through jpeg; push the render through the same encoder
        ref = world.render(float(x), float(y), float(yaw))
        ref = cv2.imdecode(np.frombuffer(world.jpeg(ref), np.uint8), cv2.IMREAD_COLOR)
        errs.append(float(np.abs(ref.astype(int) - img.astype(int)).mean()))
    return {"mean": float(np.mean(errs)), "max": float(np.max(errs)), "n": n}


def check_gt4_matches_gt(seed: int, dest: str | Path = ROOT / "data") -> dict:
    """Byte-level comparison only (no frame or pose is decoded or inspected). The ``msaa=4``
    recording must differ from the ``msaa=1`` one only in rendering: identical ``poses.json``,
    ``data_info.json`` and ``gt.json`` (minus the image encoding), same frames."""
    a, b = _path(seed, dest, "gt"), _path(seed, dest, "gt4")
    out = {"same_poses": True, "same_data_info": True, "same_frame_count": True}
    for i in range(ROUTES):
        out["same_poses"] &= (a / f"traj_{i}" / "poses.json").read_bytes() == (
            b / f"traj_{i}" / "poses.json"
        ).read_bytes()
        out["same_data_info"] &= (a / f"traj_{i}" / "data_info.json").read_bytes() == (
            b / f"traj_{i}" / "data_info.json"
        ).read_bytes()
        out["same_frame_count"] &= len(list((a / f"traj_{i}").glob("*.jpg"))) == len(
            list((b / f"traj_{i}").glob("*.jpg"))
        )
    ga, gb = json.loads((a / "gt.json").read_text()), json.loads((b / "gt.json").read_text())
    out["same_layout_spawn_target"] = all(ga[k] == gb[k] for k in ("grid", "spawn", "target"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="*", help="default: dev + test from seeds.json")
    p.add_argument("--verify", action="store_true", help="run the stage 0 equivalence checks")
    p.add_argument("--dest", default=str(ROOT / "data"))
    p.add_argument("--msaa", type=int, default=1, choices=[1, 4], help="4: the server's rendering")
    p.add_argument("--motion-bias", type=float, nargs=3, help="record data/bias-<seed> (4b)")
    p.add_argument("--motion-seed", type=int, default=0)
    p.add_argument(
        "--capture-every",
        type=int,
        default=CAPTURE_EVERY,
        help="ticks per saved frame (default 5: the existing recordings); other values write "
        "data/tick<N>-<seed> (stage 9: 1 is the format of a challenge download)",
    )
    args = p.parse_args()
    if args.capture_every != CAPTURE_EVERY:
        textures = sim.Textures.find()
        for seed in args.seeds or []:
            path = record_ticks(seed, args.capture_every, args.dest, textures, msaa=args.msaa)
            frames = json.loads((path / "gt.json").read_text())["summary"]["frames"]
            print(f"seed {seed}: {frames} frames, one per {args.capture_every} tick(s) -> {path}")
        return
    if args.motion_bias:
        for seed in args.seeds or []:
            bias = tuple(args.motion_bias)
            path = record_biased(seed, bias, args.motion_seed, args.dest, msaa=args.msaa)
            drawn = json.loads((path / "gt.json").read_text())["drawn_bias"]
            print(f"seed {seed}: biased run {drawn} -> {path}")
        return
    seeds = args.seeds
    if not seeds:
        cfg = json.loads(SEEDS.read_text())
        seeds = cfg["dev"] + cfg["test"]
    textures = sim.Textures.find()
    for seed in seeds:
        path = record(seed, args.dest, textures, args.msaa)
        meta = json.loads((path / "gt.json").read_text())
        print(f"seed {seed}: {meta['summary']['frames']} frames -> {path}")
        held = seed in holdout_seeds()
        if args.verify and args.msaa == 4:
            print("  gt4 vs gt:", check_gt4_matches_gt(seed, args.dest))
        elif args.verify:
            print("  student-equal:", check_matches_student(seed, path, textures))
            if held:  # decodes frames: not before the final evaluation
                print("  pose alignment: skipped (holdout)")
            else:
                print("  pose alignment:", check_pose_alignment(seed, path, textures))


if __name__ == "__main__":
    main()
