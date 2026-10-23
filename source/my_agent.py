"""
Your agent. Fill in the methods below, then:

    uv run source/my_agent.py --challenge <id>      # on the server, one attempt
    uv run source/my_agent.py --local 7             # on your machine, maze 7, free

Everything your agent may use is on this list; there is nothing else.

    exploration data   images and action labels from earlier drives through this maze
                       (cli.exploration_data downloads them into data/<challenge>/)
    info.targets       four views from the goal, facing front, left, back, right
    info.camera        image size and intrinsic matrix
    info.limits        step budget and attempt count
    obs.image          the camera frame, (240, 320, 3) uint8, BGR
    obs.step / obs.steps_left

The robot's pose, the map and the goal position live on the server and are never sent.

Current state (top-down map plan, stages 0 to 9): the agent builds the metric map from the
exploration data in ``__init__`` (cached in ``cache/mapping/build/<data dir>/``; about 3 minutes
the first time, before anything touches the server), places the goal on it from the four goal
views in ``setup``, and shows both in the viewer's panel. There is no live localization or
control yet: you drive with the keyboard (arrows; space checks in; escape quits).

Press P to locate the robot on the map (stage 10): it spins in place through 360 degrees (about
105 ticks, the arrows are ignored meanwhile), then draws a green circle at where it thinks it
is, with a tick for its heading. A hollow circle means the ping is not reliable. The last five
pings stay as fading circles. This is on demand only: nothing tracks the robot between pings.

The pipeline is tuned on one frame per five ticks. Challenge downloads can be one frame per
tick; ``build_map.py`` detects the frame rate from the data itself and subsamples to every fifth
frame when it has to (``mapping/adapter.py``), so any exploration data works.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cli
import cv2
import numpy as np
from keyboard_agent import KEYMAP, KeyboardAgent
from vis_nav_sdk import WAIT, Action, Observation, Result, SessionInfo, run
from vis_nav_sdk import viewer as sdk_viewer

MAPPING = Path(__file__).resolve().parent / "mapping"
sys.path.insert(0, str(MAPPING))
import goal  # noqa: E402
import ping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGE_W = 320  # camera frame width; the viewer scales it by ``viewer.scale``
GOAL_VIEWS_W = sdk_viewer.COLUMN_W - 2 * sdk_viewer.PAD  # width of the goal views above
MAP_SCALE = 0.6  # of that width
PANEL_PX = round(MAP_SCALE * GOAL_VIEWS_W)
ARROW_PX = 16  # length of the goal heading arrow on the map, whatever the map's scale
CARD_BGR = (36, 29, 26)  # the viewer's card colour, so the strip around the map blends in
DRIVE_BGR = ((40, 40, 214), (44, 160, 44), (180, 119, 31))  # red, green, blue
PING_BGR = (0, 190, 0)  # the ping marker
PING_RUNNER_UP_BGR = (150, 150, 150)
PINGS_KEPT = 5


class MyAgent(KeyboardAgent):
    def __init__(self, data_dir: Path) -> None:
        # Runs before anything touches the server: a crash here costs no attempt.
        super().__init__()
        self.data_dir = Path(data_dir)
        self.map = None
        self.map_npz = None
        self.goal = None
        self.status = self.map_status = ""
        self._panel = None
        self._map_img = None
        self.sweep = ping.Sweep()
        self.pings: list = []  # the last PINGS_KEPT results, oldest first
        self.ping_lines: list[str] = []
        out = ROOT / "cache" / "mapping" / "build" / self.data_dir.name
        if not list(self.data_dir.glob("traj_*/data_info.json")):
            self.status = self.map_status = f"map: no exploration data in {self.data_dir}"
            return
        if not (out / "map.npz").exists():
            print(f"building the map from {self.data_dir} into {out} (about 3 minutes, once) ...", flush=True)
            subprocess.run([sys.executable, str(MAPPING / "build_map.py"), "--data", str(self.data_dir), "--out", str(out)], check=True)
        self.map_npz = np.load(out / "map.npz", allow_pickle=True)
        self.map = goal.load_map(self.map_npz)
        lat = float(self.map_npz["lattice_scale"])
        rate = out / "adapter.json"
        ticks = json.loads(rate.read_text())["k"] if rate.exists() else 5
        self.status = self.map_status = (
            f"map: {out.relative_to(ROOT)} ({ticks} tick(s) per frame, "
            f"scale corrected {100 * (1 / lat - 1):+.2f} %)"
        )

    def setup(self, info: SessionInfo) -> None:
        # Once per session (twice per run: the first call is a pre-flight check with random
        # frames), with the goal views in info.targets; the session clock is running.
        self.targets = info.targets
        self.goal = None
        self.status = self.map_status  # a pre-flight failure message must not linger
        self._panel = self._map_img = None
        self.sweep = ping.Sweep()
        self.pings, self.ping_lines = [], []
        if self.map is not None:
            try:
                self.goal = goal.place(self.map, list(info.targets))
                x, y, th = self.goal.pose
                print(
                    f"goal on the map: x {x:.2f} y {y:.2f} m, heading {np.degrees(th):.0f} deg, "
                    f"confidence {self.goal.confidence:.2f} "
                    f"({'reliable' if self.goal.reliable else 'NOT reliable'})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - a display aid must not end the run
                self.status = f"goal placement failed: {exc}"
                print(self.status, flush=True)

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        if self.viewer is None:
            raise RuntimeError("MyAgent needs the viewer window; run with viewer=True")
        # The keys are read once per step: ``Viewer.keys()`` clears the tap record, so a second
        # call (for instance through ``KeyboardAgent.act``) would lose the taps.
        keys = self.viewer.keys()
        if "escape" in keys:
            return Action.QUIT
        if self.sweep.active:  # a ping is running: the arrows are ignored
            self.sweep.add(obs.image)
            if self.sweep.active:
                turned = min(self.sweep.turned_deg, 360)
                self.ping_lines = [f"ping: spinning {turned:.0f} / 360 deg"]
                self._panel = None
                return (Action.LEFT, 1)
            self._locate()
            return WAIT
        if "space" in keys:
            return Action.CHECKIN
        if "p" in keys and self.map is not None:
            self.sweep.start()
            self.sweep.add(obs.image)
            self.ping_lines = ["ping: spinning 0 / 360 deg"]
            self._panel = None
            return (Action.LEFT, 1)
        action = Action.IDLE
        for name, bit in KEYMAP.items():
            if name in keys:
                action |= bit
        return WAIT if action == Action.IDLE else (action, 1)

    def _locate(self) -> None:
        """The spin is over: place the robot, redraw the panel in the same step."""
        try:
            res = ping.locate(self.map, self.sweep)
        except Exception as exc:  # noqa: BLE001 - a display aid must not end the run
            self.ping_lines = [f"ping failed: {exc}"]
            print(self.ping_lines[0], flush=True)
        else:
            x, y, th = res.pose
            self.pings = (self.pings + [res])[-PINGS_KEPT:]
            tag = "located" if res.reliable else "NOT reliable"
            self.ping_lines = [
                f"ping {tag}: x {x:.2f} y {y:.2f} m, heading {np.degrees(th):.0f} deg",
                f"confidence {res.confidence:.2f}, {res.seconds:.1f} s, {res.ticks} ticks",
                *[f"  {w}" for w in res.warnings],
            ]
            print(" | ".join(self.ping_lines), flush=True)
        self._panel = self._map_img = None

    def finish(self, result: Result | None) -> None:
        pass

    def hud(self) -> list[str]:
        return []  # no "agent" card: only the map is added to the window

    def panel(self) -> np.ndarray | list | None:
        if self.map_npz is None:
            return None
        if self._panel is None:
            if self._map_img is None:
                self._map_img = render_map(self.map_npz, self.goal, self.pings)
            scale = self.viewer.scale if self.viewer is not None else 2
            self._panel = under_goal_views(self._map_img, scale, self.ping_lines)
        return self._panel


def under_goal_views(
    img: np.ndarray, scale: int, text_lines: list[str] | None = None
) -> np.ndarray:
    """The viewer stretches the panel to the width of camera plus right column. Draw the map
    on a strip of exactly that width, centred under the goal views at its own size instead of
    being blown up to the full window width, with the legend image just left of it."""
    pad, col_w = sdk_viewer.PAD, sdk_viewer.COLUMN_W
    view_w = IMAGE_W * scale
    legend = render_legend()
    strip = np.full((img.shape[0], view_w + pad + col_w, 3), CARD_BGR, np.uint8)
    x0 = view_w + 2 * pad + (GOAL_VIEWS_W - img.shape[1]) // 2  # centred under the goal views
    strip[:, x0 : x0 + img.shape[1]] = img
    lx = x0 - pad - legend.shape[1]
    strip[: legend.shape[0], lx : lx + legend.shape[1]] = legend
    for i, line in enumerate(text_lines or []):  # the ping state, left of the legend
        cv2.putText(strip, line, (pad + 4, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (225, 225, 230), 1, cv2.LINE_AA)
    return strip


def render_legend() -> np.ndarray:
    """A small separate image: what the marks on the map mean."""
    w, h, row = 190, 96, 22
    img = np.full((h, w, 3), 255, np.uint8)  # light, like the map, so the marks look the same
    text = (60, 60, 60)
    items = (
        ("spawn", lambda c: cv2.circle(img, c, 5, (0, 0, 0), -1, cv2.LINE_AA)),
        ("goal estimate", lambda c: cv2.arrowedLine(img, (c[0] - 8, c[1]), (c[0] + 8, c[1]),
                                                    (0, 0, 220), 3, cv2.LINE_AA, tipLength=0.5)),
        ("runner-up goal places", lambda c: cv2.circle(img, c, 7, (0, 140, 255), 2, cv2.LINE_AA)),
        ("ping: where the robot is", lambda c: cv2.circle(img, c, 6, PING_BGR, -1, cv2.LINE_AA)),
    )
    for i, (label, mark) in enumerate(items):
        y = 12 + row * i
        mark((16, y))
        cv2.putText(img, label, (34, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text, 1, cv2.LINE_AA)
    return img


def render_map(z, est=None, pings=()) -> np.ndarray:
    """BGR image of the map: robot free space (green), walls (black), the exploration drives,
    and the goal estimate (red arrow; runner-up places as orange circles)."""
    segs = z["wall_segments"]
    lo = np.minimum(segs[:, :2].min(0), segs[:, 2:].min(0)) - 0.15
    hi = np.maximum(segs[:, :2].max(0), segs[:, 2:].max(0)) + 0.15
    s = (PANEL_PX - 1) / float((hi - lo).max())
    img = np.full((PANEL_PX, PANEL_PX, 3), 255, np.uint8)

    def px(x, y):
        return int(round((x - lo[0]) * s)), int(round(PANEL_PX - 1 - (y - lo[1]) * s))

    # free space for the robot centre
    res = float(z["resolution"])
    org = z["origin"]
    iy, ix = np.nonzero(z["cspace"])
    for x, y in zip(org[0] + (ix + 0.5) * res, org[1] + (iy + 0.5) * res, strict=True):
        a, b = px(x - res / 2, y + res / 2), px(x + res / 2, y - res / 2)
        cv2.rectangle(img, a, b, (205, 235, 205), -1)
    # exploration drives (faint)
    P = z["poses"]
    for d in range(int(P[:, 0].max()) + 1):
        pts = np.array([px(x, y) for x, y in P[P[:, 0] == d][:, 2:4]], np.int32)
        cv2.polylines(img, [pts], False, tuple(int(0.4 * c + 150) for c in DRIVE_BGR[d % 3]), 1, cv2.LINE_AA)
    for sg in segs:
        cv2.line(img, px(*sg[:2]), px(*sg[2:]), (0, 0, 0), 2, cv2.LINE_AA)
    sp = P[P[:, 1] == 0][0, 2:4]
    cv2.circle(img, px(*sp), 5, (0, 0, 0), -1, cv2.LINE_AA)
    if est is not None:
        for t in est.top[1:]:
            cv2.circle(img, px(*t[0][:2]), 7, (0, 140, 255), 2, cv2.LINE_AA)
        x, y, th = est.pose
        col = (0, 0, 220) if est.reliable else (0, 140, 255)
        a = px(x, y)
        tip = (int(round(a[0] + ARROW_PX * np.cos(th))), int(round(a[1] - ARROW_PX * np.sin(th))))
        cv2.arrowedLine(img, a, tip, col, 2, cv2.LINE_AA, tipLength=0.4)
        cv2.circle(img, a, 3, col, -1, cv2.LINE_AA)
    draw_pings(img, px, pings)
    return img


def draw_pings(img: np.ndarray, px, pings) -> None:
    """The last pings, oldest first: the newest is a green circle (filled when reliable, hollow
    when not, with its runner-up places in grey) and a tick for the heading; older ones fade."""
    n = len(pings)
    for i, res in enumerate(pings):
        age = n - 1 - i
        fade = max(0.0, 1.0 - 0.22 * age)
        col = tuple(int(255 - (255 - c) * fade) for c in PING_BGR)
        x, y, th = res.pose
        c = px(x, y)
        if age == 0 and not res.reliable:
            for q in res.top[1:]:
                cv2.circle(img, px(*q[0][:2]), 3, PING_RUNNER_UP_BGR, 1, cv2.LINE_AA)
        cv2.circle(img, c, 5 if age == 0 else 4, col, -1 if (res.reliable or age) else 2, cv2.LINE_AA)
        if age == 0:
            tip = (int(round(c[0] + 12 * np.cos(th))), int(round(c[1] - 12 * np.sin(th))))
            cv2.line(img, c, tip, col, 2, cv2.LINE_AA)


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    run(MyAgent(data), cli.challenge(args), **cli.run_options(args))
