"""Stage 9: the frame-rate detector's decision, the subsampling bookkeeping, and the rule 10
leakage checks for the adapter, the build and the agent's map path."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "source"
sys.path.insert(0, str(SRC / "mapping"))
sys.path.insert(0, str(SRC))
import adapter as ad  # noqa: E402
import record_gt  # noqa: E402


def test_decide_k_on_five_tick_and_per_tick_evidence():
    k, ev = ad.decide_k(0.45, 4.98, 4.87, 300, 600)
    assert k == 5 and ev.k == 5
    assert ad.decide_k(0.80, 1.01, 1.02)[0] == 1


def test_decide_k_refuses_to_guess():
    with pytest.raises(ad.AdapterError, match="disagree"):
        ad.decide_k(0.45, 5.0, 3.1)  # forward says 5, turn says 3
    with pytest.raises(ad.AdapterError, match="label-repeat"):
        ad.decide_k(0.80, 5.0, 5.0)  # a 5-tick motion with a per-tick repeat rate
    with pytest.raises(ad.AdapterError, match="label-repeat"):
        ad.decide_k(0.45, 1.0, 1.0)
    with pytest.raises(ad.AdapterError, match="implausible"):
        ad.decide_k(0.45, 40.0, 40.0)
    with pytest.raises(ad.AdapterError, match="implausible"):
        ad.decide_k(0.80, 0.0, 0.0)


def test_repeat_rate():
    assert ad.repeat_rate(["A", "A", "B", "B", "B"]) == pytest.approx(0.75)
    assert ad.repeat_rate(["A", "B", "A", "B"]) == 0.0


def test_window_labels_bookkeeping():
    # per-tick labels: frame i is the tick-i action
    labels = list("RRRRRRRRRRIIFFFFFLL") + ["L"]
    out = ad.window_labels(labels, 5, 4)
    assert [w["frame"] for w in out] == [4, 9, 14, 19]
    assert [w["action"] for w in out] == ["R", "R", "F", "L"]  # the last tick of each window
    assert out[0]["composition"] == {"R": 5}
    assert out[1]["composition"] == {"R": 5}
    assert out[2]["composition"] == {"I": 2, "F": 3}
    assert all(sum(w["composition"].values()) == 5 for w in out)
    assert [w["index"] for w in out] == [0, 1, 2, 3]


def test_window_labels_other_offsets_and_rates():
    labels = list("ABCDEFGHIJ")
    out = ad.window_labels(labels, 3, 2)
    assert [w["frame"] for w in out] == [2, 5, 8]
    assert out[0]["composition"] == {"A": 1, "B": 1, "C": 1}
    assert ad.window_labels(labels, 1, 0)[7]["composition"] == {"H": 1}
    short = ad.window_labels(list("AB"), 5, 4)  # no complete window
    assert short == []


def _tiny_drive(path: Path, labels: list[str]) -> None:
    path.mkdir(parents=True)
    rng = np.random.default_rng(0)
    info = []
    for i, a in enumerate(labels):
        cv2.imwrite(str(path / f"{i}.jpg"), rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
        info.append({"step": i, "image": f"{i}.jpg", "action": [a]})
    (path / "data_info.json").write_text(json.dumps(info))


def test_subsample_writes_the_local_format(tmp_path):
    labels = list("RRRRRFFFFFFFLLLLLIII")
    _tiny_drive(tmp_path / "src", labels)
    recs = ad.subsample(tmp_path / "src", 5, tmp_path / "out")
    info = json.loads((tmp_path / "out" / "data_info.json").read_text())
    assert info == recs
    assert [r["action"] for r in info] == [["R"], ["F"], ["L"], ["I"]]
    assert [r["image"] for r in info] == ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]
    assert [r["step"] for r in info] == [0, 1, 2, 3]
    assert info[1]["composition"] == {"F": 5}
    # the kept image of window j is per-tick frame 5 j + 4
    for j, r in enumerate(info):
        kept = (tmp_path / "out" / r["image"]).read_bytes()
        assert kept == (tmp_path / "src" / f"{5 * j + 4}.jpg").read_bytes()


def test_subsample_dataset_copies_the_target(tmp_path):
    for d in range(2):
        _tiny_drive(tmp_path / "data" / f"traj_{d}", list("RRRRRLLLLL"))
    (tmp_path / "data" / "target.jpg").write_bytes(b"x")
    out = ad.subsample_dataset(tmp_path / "data", 5, tmp_path / "out")
    assert (out / "target.jpg").read_bytes() == b"x"
    assert json.loads((out / "adapter.json").read_text())["k"] == 5
    assert len(ad.drive_dirs(out)) == 2


# ------------------------------------------------------------------------------- leakage

FORBIDDEN = re.compile(r"gt\.json|poses\.json|camera_fit|record_gt|vis_nav_sim|Scene\(|occupancy\(")


def _code(path: Path) -> str:
    body = path.read_text().split('"""', 2)[-1]
    return "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))


def test_pipeline_modules_do_not_touch_ground_truth():
    for p in (SRC / "mapping" / "adapter.py", SRC / "mapping" / "build_map.py"):
        assert not FORBIDDEN.search(_code(p)), p.name


def test_the_agents_map_path_does_not_touch_ground_truth():
    code = _code(SRC / "my_agent.py")
    assert not re.search(r"gt\.json|poses\.json|camera_fit|record_gt|Scene\(", code)
    assert "five_tick_data" not in code  # the name check is gone: the detector decides


def test_adapter_output_is_identical_without_the_truth_files(tmp_path):
    src = record_gt.gt_dir(7, prefix="tick1") / "traj_0"
    if not src.exists():
        pytest.skip("record_gt.py --msaa 4 --capture-every 1 --seeds 7 first")
    n = 160
    info = json.loads((src / "data_info.json").read_text())[:n]
    copy = tmp_path / "copy" / "traj_0"
    copy.mkdir(parents=True)
    (copy / "data_info.json").write_text(json.dumps(info))
    for r in info:
        shutil.copy(src / r["image"], copy / r["image"])
    assert not (copy.parent / "gt.json").exists() and not (copy / "poses.json").exists()
    # the same drive with the truth files next to it: identical outputs
    with_truth = tmp_path / "with_truth"
    shutil.copytree(copy.parent, with_truth)
    (with_truth / "gt.json").write_text("{}")
    (with_truth / "traj_0" / "poses.json").write_text("[]")
    a = ad.subsample(copy, 5, tmp_path / "a")
    b = ad.subsample(with_truth / "traj_0", 5, tmp_path / "b")
    assert a == b
    fa, ta = ad.window_evidence(copy, 12)
    fb, tb = ad.window_evidence(with_truth / "traj_0", 12)
    assert fa.size and np.array_equal(fa, fb) and np.array_equal(ta, tb)


def test_step_to_five_ticks():
    assert ad.step_to_five_ticks(1) == 5  # per-tick data: keep every fifth frame
    assert ad.step_to_five_ticks(5) == 1
    for k in (2, 3, 4, 10):
        with pytest.raises(ad.AdapterError, match="cannot be subsampled"):
            ad.step_to_five_ticks(k)


def test_build_map_adapts_per_tick_data_to_every_fifth_frame(tmp_path):
    import build_map

    data = tmp_path / "data"
    for d in range(2):
        _tiny_drive(data / f"traj_{d}", list("RRRRRRRRRRLLLLLLLLLL"))
    (data / "target.jpg").write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    built = build_map.adapt(data, out, "1")  # the rate given instead of detected
    info = json.loads((out / "adapter.json").read_text())
    assert info["k"] == 1 and info["subsample_step"] == 5 and Path(info["built_from"]) == built
    lengths = [len(json.loads((d / "data_info.json").read_text())) for d in ad.drive_dirs(built)]
    assert lengths == [4, 4]
    assert build_map.adapt(data, out, "auto") == built  # resumable: reuses adapter.json
    five = tmp_path / "five"
    for d in range(2):
        _tiny_drive(five / f"traj_{d}", list("RRRRRLLLLL"))
    out5 = tmp_path / "out5"
    out5.mkdir()
    assert build_map.adapt(five, out5, "5") == five  # 5-tick data is used as is
