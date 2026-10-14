"""Stage 0 artefacts: ``cache/mapping/camera.json`` and the true-trajectory figures.

uv run source/mapping/stage0_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_gt  # noqa: E402

ROOT = record_gt.ROOT
COLORS = ["tab:red", "tab:green", "tab:blue"]


def write_camera(seed: int) -> None:
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    cam = gt["camera"]
    out = {
        "K": [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]],
        "width": cam["width"],
        "height": cam["height"],
        "camera": cam,
        "source": f"vis_nav_sim Simulator(seed={seed}).camera (course camera)",
    }
    dest = ROOT / "cache" / "mapping" / "camera.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))


def plot_seed(seed: int) -> Path:
    path = record_gt.gt_dir(seed)
    gt = json.loads((path / "gt.json").read_text())
    cs = gt["cell_size"]
    grid = gt["grid"]
    poses = record_gt.load_poses(path)
    fig, ax = plt.subplots(figsize=(8, 8))
    walls = [
        (c * cs, r * cs) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "#"
    ]
    for x, y in walls:
        ax.add_patch(plt.Rectangle((x - cs / 2, y - cs / 2), cs, cs, color="0.8", lw=0))
    for i, p in enumerate(poses):
        ax.plot(p[:, 0], p[:, 1], color=COLORS[i], lw=0.8, label=f"drive {i} ({len(p)} frames)")
        ax.plot(*p[0, :2], "o", color=COLORS[i], ms=7)
        ax.plot(*p[-1, :2], "s", color=COLORS[i], ms=7)
    tx, ty, _ = gt["target"]
    ax.plot(tx, ty, "*", color="gold", mec="k", ms=16, label="goal")
    ax.set_aspect("equal")
    ax.set_xlim(-cs, len(grid[0]) * cs)
    ax.set_ylim(-cs, len(grid) * cs)
    ax.set_title(f"seed {seed}: true drives (o start, s end)")
    ax.legend(loc="upper right", fontsize=8)
    out = ROOT / "reports" / "figures" / f"stage0_true_trajectories_{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    cfg = json.loads(record_gt.SEEDS.read_text())
    seeds = cfg["dev"] + cfg["test"]
    write_camera(seeds[0])
    for s in seeds:
        print(plot_seed(s))


if __name__ == "__main__":
    main()
