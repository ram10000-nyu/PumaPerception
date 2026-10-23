"""Stage 10: sweep bookkeeping on synthetic rotations, angle unwrapping, the 4-view case
reproducing ``goal.place``, merged-scan geometry, the agent's key handling, and rule 10 leakage."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source"
sys.path.insert(0, str(SRC / "mapping"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal  # noqa: E402
import grid  # noqa: E402
import ping  # noqa: E402
import record_gt  # noqa: E402
from test_grid import cells, maze, render  # noqa: E402

BLANK = np.zeros((240, 320, 3), np.uint8)


def synthetic_sweep(yaw0, n, step_deg=ping.TICK_DEG, drop=(), segs=None, pos=(1.0, 1.0)):
    """A sweep built without images: the compass reads the true yaw modulo 90 degrees."""
    s = ping.Sweep()
    s.start()
    for i in range(n):
        yaw = np.radians(yaw0 + i * step_deg)
        psi = np.nan if i in drop else float(np.degrees(yaw) % 90.0)
        if segs is not None:
            rows, top, status = render(segs, (pos[0], pos[1], yaw))
        else:
            rows, top, status = (np.full(320, np.nan), np.full(320, np.nan), np.zeros(320, np.int8))
        s.frames.append(ping.SweepFrame(BLANK, rows, top, status, psi))
    return s


def test_relative_angles_recover_the_rotation_across_the_90_degree_wrap():
    for yaw0 in (0.0, 37.0, 80.0, -50.0, 170.0):
        s = synthetic_sweep(yaw0, 110)
        rel = s.relative_angles()
        assert np.allclose(rel, np.arange(110) * ping.TICK_DEG, atol=1e-6)
        assert s.turned_deg == pytest.approx(109 * ping.TICK_DEG)


def test_relative_angles_with_invalid_frames_interpolated_and_extrapolated():
    drop = {0, 1, 40, 41, 42, 108, 109}  # leading, middle and trailing gaps
    s = synthetic_sweep(12.0, 110, drop=drop)
    rel = s.relative_angles()
    assert np.allclose(rel, np.arange(110) * ping.TICK_DEG, atol=0.05)


def test_a_long_gap_in_the_compass_is_unwrapped_with_the_ticks():
    # 30 invalid frames in a row hide 103 degrees: the compass alone would lose a quarter turn
    for start in (10, 40, 70):
        s = synthetic_sweep(12.0, 106, drop=set(range(start, start + 30)))
        rel = s.relative_angles()
        assert np.allclose(rel, np.arange(106) * ping.TICK_DEG, atol=0.1)
    # a turn 20 % slower than nominal across such a gap still resolves to the right branch
    s = synthetic_sweep(12.0, 106, step_deg=0.8 * ping.TICK_DEG, drop=set(range(40, 70)))
    assert s.turned_deg == pytest.approx(105 * 0.8 * ping.TICK_DEG, abs=0.1)


def test_valid_but_wrong_compass_readings_are_refused():
    # the compass is sometimes valid and wrong by about 45 degrees: the angles must not follow it
    rng = np.random.default_rng(3)
    for yaw0 in (3.0, 61.0):
        s = synthetic_sweep(yaw0, 110)
        bad = rng.choice(np.arange(5, 105), 14, replace=False)
        for i in bad:
            s.frames[i].psi = float((s.frames[i].psi + 45.0 * rng.choice([-1, 1])) % 90.0)
        assert np.allclose(s.relative_angles(), np.arange(110) * ping.TICK_DEG, atol=0.1)
    # two outliers in a row, and one right after a gap
    s = synthetic_sweep(20.0, 110, drop=set(range(30, 36)))
    for i in (36, 37, 70, 71, 72):
        s.frames[i].psi = float((s.frames[i].psi + 44.0) % 90.0)
    assert np.allclose(s.relative_angles(), np.arange(110) * ping.TICK_DEG, atol=0.1)


def test_a_turn_that_slows_down_is_still_followed():
    # a robot rubbing a wall: the rate halves for 25 ticks, then recovers
    t = ping.TICK_DEG
    steps = np.r_[np.full(40, t), np.full(25, 0.5 * t), np.full(45, t)]
    yaw = np.r_[0.0, np.cumsum(steps)] + 7.0
    s = ping.Sweep()
    s.start()
    for y in yaw:
        nan = np.full(320, np.nan)
        s.frames.append(ping.SweepFrame(BLANK, nan, nan, np.zeros(320, np.int8), float(y % 90.0)))
    assert np.allclose(s.relative_angles(), yaw - yaw[0], atol=0.2)


def test_angles_follow_the_compass_not_the_tick_count():
    # a turn 5 % slower than the nominal tick: the compass total is the truth, and the small
    # disagreement with the ticks is a note for the HUD, not a failure
    s = synthetic_sweep(5.0, 112, step_deg=0.95 * ping.TICK_DEG)
    assert s.turned_deg == pytest.approx(111 * 0.95 * ping.TICK_DEG, abs=1e-6)
    assert abs(s.mismatch_deg()) == pytest.approx(111 * 0.05 * ping.TICK_DEG)
    assert ping.MISMATCH_WARN_DEG < abs(s.mismatch_deg()) < ping.MISMATCH_FAIL_DEG
    assert not any("ticks" in p for p in s.problems())
    # 30 % slower (a robot rubbing a wall): too far from the ticks to trust, and it says so
    slow = synthetic_sweep(5.0, 120, step_deg=0.7 * ping.TICK_DEG)
    assert slow.turned_deg == pytest.approx(119 * 0.7 * ping.TICK_DEG, abs=1e-6)
    assert any("compass" in p and "ticks" in p for p in slow.problems())


def test_sweep_ends_at_360_degrees_or_at_the_tick_limit():
    s = ping.Sweep(max_ticks=30)
    s.frames = synthetic_sweep(0.0, 106).frames
    assert s.done and s.turned_deg >= 360
    short = ping.Sweep(max_ticks=30)
    short.frames = synthetic_sweep(0.0, 50, step_deg=1.0).frames  # barely turning
    assert short.done  # gave up at the tick limit ...
    assert any("turned only" in p for p in short.problems())  # ... and says why


def test_sweep_problems_are_reported():
    ok = synthetic_sweep(0.0, 108)
    assert ok.problems() == []
    no_compass = synthetic_sweep(0.0, 108, drop=set(range(108)))
    assert any("compass valid" in p for p in no_compass.problems())
    stuck = synthetic_sweep(0.0, 140, step_deg=2.5)  # compass 347 deg after 139 ticks (478 nominal)
    assert any("compass" in p and "ticks" in p for p in stuck.problems())


def test_merged_scan_puts_the_walls_where_they_are():
    G = maze()
    segs = grid.wall_segments(cells(G))
    pos, yaw0 = (1.0, 1.0), 0.8
    s = synthetic_sweep(np.degrees(yaw0), 106, segs=segs, pos=pos)
    gs = ping.merged_scan(s)
    assert len(gs.xy) > 500 and len(gs.views) >= 5
    # in the first frame's frame: world = pos + R(yaw0) xy
    c, sn = np.cos(yaw0), np.sin(yaw0)
    world = np.array(pos) + gs.xy @ np.array([[c, -sn], [sn, c]]).T
    assert np.median(grid.point_segment_distance(world, segs)) < 0.01
    assert np.quantile(grid.point_segment_distance(world, segs), 0.95) < 0.05
    # every direction is covered: the points surround the robot
    ang = np.degrees(np.arctan2(gs.xy[:, 1], gs.xy[:, 0]))
    assert np.histogram(ang, bins=8, range=(-180, 180))[0].min() > 20


def test_single_frame_and_half_spin_restrict_the_scan():
    segs = grid.wall_segments(cells(maze()))
    s = synthetic_sweep(0.0, 106, segs=segs)
    full = ping.merged_scan(s)
    half = ping.merged_scan(s, max_deg=180.0)
    one = ping.merged_scan(s, frames=[0], slice_deg=120.0)
    assert len(one.xy) < len(half.xy) < len(full.xy)
    assert np.degrees(np.arctan2(one.xy[:, 1], one.xy[:, 0])).max() < 61


# ----------------------------------------------- the 4-view case reproduces goal.place exactly


def test_four_views_with_explicit_angles_reproduce_goal_place_exactly():
    mp = record_gt.ROOT / "cache" / "mapping" / "7" / "map.npz"
    tj = record_gt.ROOT / "data" / "gt4-7" / "target.jpg"
    if not (mp.exists() and tj.exists()):
        pytest.skip("needs the dev seed 7 map and target.jpg")
    mf = goal.load_map(mp)
    target = cv2.imread(str(tj))
    gs = goal.goal_scan(target)
    angled = goal.GoalScan(gs.xy, gs.sigma, gs.view, gs.normal, gs.cam, gs.overlap_residual,
                           [(v[0], *v[1:5], v[0] * np.pi / 2) for v in gs.views])
    a, b = goal.place(mf, gs), goal.place(mf, angled)
    assert a.pose.tobytes() == b.pose.tobytes() and a.confidence == b.confidence
    assert [t[0].tobytes() for t in a.top] == [t[0].tobytes() for t in b.top]
    assert goal.texture_score(mf, gs, a.pose) == goal.texture_score(mf, angled, a.pose)


# --------------------------------------------------------------------------- the agent keys


class FakeViewer:
    def __init__(self):
        self.calls = 0
        self.script: list[set] = []

    def keys(self):
        self.calls += 1
        names = self.script.pop(0) if self.script else set()
        return SimpleNamespace(__contains__=lambda k: k in names)


class Keys(set):
    pass


def make_agent():
    from my_agent import MyAgent

    agent = MyAgent(Path("data/none"))
    agent.viewer = SimpleNamespace(calls=0, names=set())

    def keys():
        agent.viewer.calls += 1
        return Keys(agent.viewer.names)

    agent.viewer.keys = keys
    return agent


def test_agent_reads_the_keys_once_per_step_and_handles_p_and_escape():
    from vis_nav_sdk import WAIT, Action

    agent = make_agent()
    obs = SimpleNamespace(image=BLANK)
    agent.viewer.names = {"up"}
    act = agent.act(obs)
    assert act == (Action.FORWARD, 1) and agent.viewer.calls == 1
    agent.viewer.names = {"escape"}
    assert agent.act(obs) == Action.QUIT and agent.viewer.calls == 2
    agent.viewer.names = {"p"}  # no map: a ping is not possible, nothing happens
    assert agent.act(obs) == WAIT and not agent.sweep.active


def test_agent_ignores_the_arrows_during_a_sweep_and_spins_left(monkeypatch):
    from vis_nav_sdk import Action

    agent = make_agent()
    agent.map = object()  # a map is loaded
    def add(self, image):  # a compass-only frame: 3.44 degrees more than the last one
        psi = (len(self.frames) * ping.TICK_DEG) % 90.0
        nan = np.full(320, np.nan)
        self.frames.append(ping.SweepFrame(image, nan, nan, np.zeros(320, np.int8), psi))
        if self.done:
            self.active = False

    monkeypatch.setattr(ping.Sweep, "add", add)
    obs = SimpleNamespace(image=BLANK)
    agent.viewer.names = {"p"}
    assert agent.act(obs) == (Action.LEFT, 1) and agent.sweep.active
    agent.viewer.names = {"up", "left", "space"}
    for _ in range(3):
        assert agent.act(obs) == (Action.LEFT, 1)  # arrows and space are ignored
    assert agent.viewer.calls == 4
    agent.viewer.names = {"escape"}
    assert agent.act(obs) == Action.QUIT  # escape still quits


# --------------------------------------------------------------------------------- leakage


def _code(path: Path) -> str:
    body = path.read_text().split('"""', 2)[-1]
    return "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))


def test_ping_and_the_agents_ping_path_do_not_touch_ground_truth():
    forbidden = re.compile(
        r"gt\.json|poses\.json|truth|camera_fit|record_gt|vis_nav_sim|Scene\(|occupancy\("
    )
    assert not forbidden.search(_code(SRC / "mapping" / "ping.py"))
    agent = _code(SRC / "my_agent.py")
    assert not re.search(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim", agent)
    assert ".simulator" not in agent and ".pose()" not in agent  # no peeking at the true pose


def test_locate_is_identical_without_the_truth_files(tmp_path):
    src = record_gt.ROOT / "data" / "ping-7" / "sweep_000"
    mp = record_gt.ROOT / "cache" / "mapping" / "build" / "gt4-7" / "map.npz"
    if not (src.exists() and mp.exists()):
        pytest.skip("needs the dev ping recordings (stage10.py record 7) and the seed 7 map")
    copy = tmp_path / "copy"
    copy.mkdir()
    n = len(json.loads((src / "truth.json").read_text())["yaws"])
    for i in range(n):
        shutil.copy(src / f"{i}.jpg", copy / f"{i}.jpg")
    assert not (copy / "truth.json").exists()
    mf = goal.load_map(mp)
    out = []
    for d in (src, copy):
        s = ping.Sweep()
        s.start()
        for i in range(n):
            s.add(cv2.imread(str(d / f"{i}.jpg")))
        out.append(ping.locate(mf, s))
    assert out[0].pose.tobytes() == out[1].pose.tobytes()
    assert out[0].confidence == out[1].confidence
