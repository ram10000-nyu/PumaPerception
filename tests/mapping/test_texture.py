"""Stage 5b: texture boundaries (pipeline module texture.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "mapping"))
import texture as tx  # noqa: E402


def _desc(seed):
    return tx._desc(np.random.default_rng(seed).normal(size=(3, tx.NZ, tx.NW)))


def _b(l, r, pos):
    return tx.Boundary(_desc(l), _desc(r), np.array(pos, float), float(pos[0]))


def test_transform_identity_and_rotation():
    pts = np.array([[1.0, 0.0], [0.5, 0.2]])
    assert np.allclose(tx.transform(np.zeros(3), pts), pts)
    out = tx.transform(np.array([0.0, 0.0, np.pi / 2]), np.array([[1.0, 0.0]]))
    assert np.allclose(out, [[0.0, 1.0]])


def test_match_requires_both_sides_and_position():
    a = [_b(1, 2, (1.0, 0.0))]
    rel = np.array([0.1, 0.0, 0.0])  # frame b is 0.1 m ahead of frame a
    same = [_b(1, 2, (0.9, 0.0))]  # the same boundary seen from b
    assert tx.compare_boundaries(a, same, rel) == (1, 0)
    # same left texture, different right texture at the same spot: a contradiction
    assert tx.compare_boundaries(a, [_b(1, 7, (0.9, 0.0))], rel) == (0, 1)
    # the right pair but 0.6 m away (the stage 5 alias shift): neither match nor contradiction
    assert tx.compare_boundaries(a, [_b(1, 2, (0.3, 0.0))], rel) == (0, 0)
    assert tx.compare_boundaries([], same, rel) == (0, 0)


def test_texture_module_never_names_ground_truth():
    src = (Path(tx.__file__)).read_text()
    for word in ("poses.json", "gt.json", "vis_nav_sim", "Scene", "true_"):
        assert word not in src, word
