"""Stage 10 evaluation: ping sweeps recorded at random free positions with their true poses, and
how well ``ping.locate`` places the robot. EVALUATION CODE (reads ground truth and the true
layout); the pipeline is ``ping.py``.

    uv run python source/mapping/stage10.py record 7 8 9 10 [--n 30]     # data/ping-<seed>/
    uv run python source/mapping/stage10.py run 7 8 9 10 [--tag name]    # dev
    uv run python source/mapping/stage10.py report 7 8 9 10 [--tag dev_final]   # markdown tables
    uv run python source/mapping/stage10.py figures 7 --tag dev_final
    uv run python source/mapping/stage10.py run 21 22 23 24 25 --allow-test   # ask the user first

Each sweep: a random position at least 0.15 m from the walls, a random heading, then LEFT ticks
(0.06 rad, 10 % per-tick noise as the challenge default) until the compass has turned 360
degrees, one rendered frame (``msaa=4``, the server's) per tick. Frames are stored as JPEG, the
format of every recording; ``--raw`` evaluates the same sweeps from unencoded renders.
Test seeds are refused without ``--allow-test`` (plan seed discipline), holdout seeds always.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_fit as cf  # noqa: E402
import goal  # noqa: E402
import grid  # noqa: E402
import ping  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping" / "stage10"
FIG = ROOT / "reports" / "figures"
CFG = json.loads(record_gt.SEEDS.read_text())
MIN_WALL_M = 0.15
TURN_NOISE = 0.1  # per-tick sigma of the angular speed (sim.DEFAULT_NOISE)
ACCEPT_M, ACCEPT_DEG = 0.10, 10.0


def check_seed(seed: int, allow_test: bool) -> None:
    if seed in CFG["test"] and not allow_test:
        raise SystemExit(f"seed {seed} is a test seed: ask the user, then pass --allow-test")
    record_gt.gt_dir(seed)  # raises HoldoutError for a holdout seed


def sweep_dir(seed: int, j: int) -> Path:
    return ROOT / "data" / f"ping-{seed}" / f"sweep_{j:03d}"


def map_path(seed: int) -> Path:
    return ROOT / "cache" / "mapping" / "build" / f"gt4-{seed}" / "map.npz"


# ---------------------------------------------------------------------------------- recording


def free_positions(seed: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random free positions at least ``MIN_WALL_M`` from every wall (the true layout)."""
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    occ = np.array([[c == "#" for c in row] for row in gt["grid"]])
    segs = cf.wall_segments(occ)
    out = []
    hi = 0.2 * (occ.shape[0] - 1)
    while len(out) < n:
        p = rng.uniform(0.0, hi, 2)
        r, c = int(round(p[1] / 0.2)), int(round(p[0] / 0.2))
        if occ[r, c] or grid.point_segment_distance(p[None], segs)[0] < MIN_WALL_M:
            continue
        out.append(p)
    return np.array(out)


def simulate_sweep(world, pos, yaw0: float, rng: np.random.Generator):
    """Yield ``(frame, true yaw)``: before the spin, then after every LEFT tick. The caller
    stops iterating when its sweep is done."""
    yaw = yaw0
    for _ in range(ping.MAX_TICKS + 2):
        yield world.render(float(pos[0]), float(pos[1]), float(yaw)), yaw
        yaw += 0.06 * (1.0 + TURN_NOISE * rng.normal())


def record_seed(args) -> int:
    seed, n, allow_test = args
    import vis_nav_sim as sim

    check_seed(seed, allow_test)
    world = record_gt.make_world(seed, sim.Textures.find(), msaa=4)
    rng = np.random.default_rng(1000 + seed)
    pos = free_positions(seed, n, rng)
    for j, p in enumerate(pos):
        d = sweep_dir(seed, j)
        d.mkdir(parents=True, exist_ok=True)
        yaw0 = float(rng.uniform(-np.pi, np.pi))
        sweep = ping.Sweep()
        sweep.start()
        yaws = []
        for i, (frame, yaw) in enumerate(simulate_sweep(world, p, yaw0, rng)):
            cv2.imwrite(str(d / f"{i}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yaws.append(yaw)
            sweep.add(cv2.imread(str(d / f"{i}.jpg")))
            if sweep.done:
                break
        (d / "truth.json").write_text(json.dumps({"seed": seed, "pos": p.tolist(), "yaw0": yaw0, "yaws": yaws}))
    return seed


# ---------------------------------------------------------------------------------- evaluation

_MAPS: dict = {}


def _map(seed: int):
    if seed not in _MAPS:
        _MAPS[seed] = goal.load_map(map_path(seed))
    return _MAPS[seed]


def load_sweep(seed: int, j: int, raw: bool = False):
    d = sweep_dir(seed, j)
    truth = json.loads((d / "truth.json").read_text())
    n = len(truth["yaws"])
    if raw:
        import vis_nav_sim as sim

        world = record_gt.make_world(seed, sim.Textures.find(), msaa=4)
        frames = [world.render(truth["pos"][0], truth["pos"][1], float(y)) for y in truth["yaws"]]
    else:
        frames = [cv2.imread(str(d / f"{i}.jpg")) for i in range(n)]
    return frames, truth


def build_sweep(frames, until: int | None = None) -> ping.Sweep:
    s = ping.Sweep()
    s.start()
    for f in frames[: until or len(frames)]:
        s.add(f)
    return s


def errors(res: ping.PingResult, truth) -> tuple[float, float]:
    end = np.array([truth["pos"][0], truth["pos"][1], truth["yaws"][-1]])
    d = np.hypot(*(res.pose[:2] - end[:2]))
    h = abs(np.degrees(se2.wrap(res.pose[2] - end[2])))
    return float(d), float(h)


VARIANTS = {
    "full": {},
    "no_texture": {"use_texture": False},
    "half_spin": {"max_deg": 180.0},
    "single_frame": {"frames": [0], "slice_deg": 120.0},
}


def _eval_one(args):
    seed, j, raw, variants, overrides = args
    frames, truth = load_sweep(seed, j, raw)
    sweep = build_sweep(frames)
    mf = _map(seed)
    out = {"seed": seed, "j": j, "ticks": sweep.ticks, "turned_deg": sweep.turned_deg, "true_turn_deg": float(np.degrees(truth["yaws"][-1] - truth["yaws"][0])), "mismatch_deg": sweep.mismatch_deg(), "problems": sweep.problems()}
    for name in variants:
        kw = {**VARIANTS[name], **overrides}
        t0 = time.perf_counter()
        try:
            res = ping.locate(mf, sweep, **kw)
        except ValueError as exc:
            out[name] = {"error": str(exc)}
            continue
        d, h = errors(res, truth)
        out[name] = {"pos_err_m": d, "head_err_deg": h, "confidence": res.confidence, "seconds": res.seconds, "wall_s": time.perf_counter() - t0, "warnings": res.warnings, "n_points": res.n_points, "n_frames": res.n_frames, "ok": bool(d < ACCEPT_M and h < ACCEPT_DEG)}
    return out


def run(seeds, tag, raw=False, variants=("full",), overrides=None, limit=None, workers=6, allow_test=False):
    for s in seeds:
        check_seed(s, allow_test)
    jobs = []
    for s in seeds:
        n = len(list((ROOT / "data" / f"ping-{s}").glob("sweep_*")))
        jobs += [(s, j, raw, variants, overrides or {}) for j in range(n if limit is None else min(n, limit))]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_eval_one, jobs))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def summary(res, variant="full", conf_min=None):
    rows = [r[variant] for r in res if variant in r and "error" not in r[variant]]
    if not rows:
        return {}
    d = np.array([r["pos_err_m"] for r in rows])
    h = np.array([r["head_err_deg"] for r in rows])
    ok = np.array([r["ok"] for r in rows])
    c = np.array([r["confidence"] for r in rows])
    out = {"n": len(rows), "within": float(ok.mean()), "pos_median_m": float(np.median(d)), "pos_p90_m": float(np.quantile(d, 0.9)),
           "head_median_deg": float(np.median(h)), "head_p90_deg": float(np.quantile(h, 0.9)),
           "seconds_median": float(np.median([r["seconds"] for r in rows])), "seconds_max": float(max(r["seconds"] for r in rows))}
    if conf_min is not None:
        hi = c >= conf_min
        out.update(conf_min=conf_min, reliable=int(hi.sum()), high_conf_failures=int((hi & ~ok).sum()), failures=int((~ok).sum()), failures_flagged=int((~ok & ~hi).sum()))
    return out


def tables(res, seeds, conf_min) -> str:
    rows = ["| seed | sweeps | within 0.10 m and 10 deg | median error | p90 error | reliable | failures (all flagged?) | locate time median / max |", "|---|---|---|---|---|---|---|---|"]
    for s in [*seeds, None]:
        sub = [r for r in res if s is None or r["seed"] == s]
        sm = summary(sub, "full", conf_min)
        rows.append(
            f"| {'all' if s is None else s} | {sm['n']} | {100 * sm['within']:.1f} % | {1000 * sm['pos_median_m']:.1f} mm, {sm['head_median_deg']:.2f} deg | "
            f"{1000 * sm['pos_p90_m']:.1f} mm, {sm['head_p90_deg']:.2f} deg | {sm['reliable']} | {sm['failures']} ({'yes' if sm['failures_flagged'] == sm['failures'] else 'NO'}) | "
            f"{sm['seconds_median']:.2f} s / {sm['seconds_max']:.2f} s |"
        )
    rows += ["", "| variant | within 0.10 m and 10 deg | median error | reliable at the threshold | high-confidence failures |", "|---|---|---|---|---|"]
    for v in ("full", "no_texture", "half_spin", "single_frame"):
        sm = summary(res, v, conf_min)
        if sm:
            rows.append(f"| {v} | {100 * sm['within']:.1f} % | {1000 * sm['pos_median_m']:.1f} mm, {sm['head_median_deg']:.2f} deg | {sm['reliable']} / {sm['n']} | {sm['high_conf_failures']} |")
    return "\n".join(rows)


def figures(res, tag: str, conf_min: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for seed in sorted({r["seed"] for r in res}):
        sub = [r["full"] for r in res if r["seed"] == seed]
        ax[0].scatter([x["confidence"] for x in sub], [max(x["pos_err_m"], 1e-3) for x in sub], s=14, label=f"seed {seed}")
    ax[0].axvline(conf_min, color="r", lw=0.8, ls="--")
    ax[0].axhline(ACCEPT_M, color="0.5", lw=0.8, ls=":")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("ping confidence (score margin)")
    ax[0].set_ylabel("position error [m]")
    ax[0].set_title(f"full spin ({tag}); dashed: PING_CONF_MIN = {conf_min}")
    ax[0].legend(fontsize=7)
    names = ["full", "no_texture", "half_spin", "single_frame"]
    vals = [100 * summary(res, v)["within"] for v in names]
    ax[1].bar(names, vals, color=["C2", "C1", "C1", "C1"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 1, f"{v:.0f} %", ha="center", fontsize=8)
    ax[1].axhline(90, color="r", lw=0.8, ls="--")
    ax[1].set_ylabel("sweeps within 0.10 m and 10 deg [%]")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("what each part buys")
    fig.tight_layout()
    fig.savefig(FIG / f"stage10_{tag}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["record", "run", "report", "figures"])
    p.add_argument("seeds", type=int, nargs="+")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--tag", default="dev")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--variants", default="full")
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-test", action="store_true")
    a = p.parse_args()
    if a.cmd == "record":
        with ProcessPoolExecutor(min(len(a.seeds), 4)) as ex:
            for s in ex.map(record_seed, [(s, a.n, a.allow_test) for s in a.seeds]):
                print("recorded", s, flush=True)
    elif a.cmd == "run":
        res = run(a.seeds, a.tag, a.raw, tuple(a.variants.split(",")), limit=a.limit, allow_test=a.allow_test)
        for v in a.variants.split(","):
            print(v, summary(res, v, ping.PING_CONF_MIN))
    elif a.cmd == "figures":
        figures(json.loads((CACHE / f"{a.tag}.json").read_text()), a.tag, ping.PING_CONF_MIN)
    else:
        res = json.loads((CACHE / f"{a.tag}.json").read_text())
        print(tables(res, a.seeds, ping.PING_CONF_MIN))


if __name__ == "__main__":
    main()
