"""Stage 3b evaluation: the collision component of the prior and the collision detector.
EVALUATION CODE (reads ground truth); the detector itself is ``collision.py``.

    uv run source/mapping/stage3b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hashlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import collision as col  # noqa: E402
import motion as M  # noqa: E402
import record_gt  # noqa: E402
import stage3 as s3  # noqa: E402
import vp  # noqa: E402

ROOT = record_gt.ROOT
CACHE = ROOT / "cache" / "mapping"
FIG = ROOT / "reports" / "figures"
DEV, TEST = s3.DEV, s3.TEST
RESULTS = CACHE / "stage3b_results.json"
VARIANTS = ("gt4", "official")  # gt4: server-like frames (tuning); official: the gt recordings


def prior_checks(dev, test) -> dict:
    prior = M.MotionPrior.fit(dev)
    summary = prior.add_collision(dev)
    prior.save()
    out = {"fit": summary}
    inc, lab, prev = test["inc"], test["label"], test["prev"]
    rot = np.degrees(np.abs(inc[:, 2]))
    ev = np.isin(lab, M.COLLISION_LABELS) & (rot > M.COLLISION_ROT_DEG)
    ev45 = ev & (rot > 45)
    for level in (0.95, 0.99):
        inside = np.array(
            [
                prior.get(la, pr).contains_ext(x, level)[0]
                for x, la, pr in zip(inc, lab, prev, strict=True)
            ]
        )
        groups = {}
        for la in M.LABELS:
            for pl in M.PREV:
                m = (lab == la) & (prev == pl)
                if m.sum() >= 30 and f"{pl}>{la}" in prior.by_pair:
                    groups[f"{pl}>{la}"] = {"n": int(m.sum()), "coverage": float(inside[m].mean())}
        out[f"region_{int(level * 100)}"] = {
            "by_label": {la: float(inside[lab == la].mean()) for la in M.LABELS},
            "groups_min": min(g["coverage"] for g in groups.values()),
            "groups_max": max(g["coverage"] for g in groups.values()),
            "groups": groups,
            "events_inside": f"{int(inside[ev].sum())}/{int(ev.sum())}",
            "events_over_45_inside": f"{int(inside[ev45].sum())}/{int(ev45.sum())}",
        }
    # what the collision component does to the density at the events
    plain = np.array(
        [
            prior.get(la, pr).logpdf(x)[0]
            for x, la, pr in zip(inc[ev], lab[ev], prev[ev], strict=True)
        ]
    )
    ext = np.array(
        [
            prior.get(la, pr).logpdf_ext(x)[0]
            for x, la, pr in zip(inc[ev], lab[ev], prev[ev], strict=True)
        ]
    )
    out["event_log_density"] = {
        "plain_median": float(np.median(plain)),
        "extended_median": float(np.median(ext)),
        "events_under_1e-3_plain": int((plain < np.log(1e-3)).sum()),
        "events_under_1e-3_extended": int((ext < np.log(1e-3)).sum()),
        "events": int(ev.sum()),
    }
    return out


def _hash() -> str:
    """Cache key of the match-rotation cues: the code they depend on (not the decision rule)."""
    import inspect

    h = hashlib.sha1()
    for f in ("segment.py", "scan.py"):
        h.update((Path(__file__).with_name(f)).read_bytes())
    for fn in (col.sift_features, col.rotation_from_matches):
        h.update(inspect.getsource(fn).encode())
    h.update(str((col.CONSENSUS_DEG,)).encode())
    h.update((CACHE / "camera.json").read_bytes())
    return h.hexdigest()[:8]


def _sift_drive(args):
    """Match rotation to the previous frame for every FORWARD/IDLE record of one drive."""
    seed, drive, variant = args
    if variant == "gt4":
        path = record_gt.gt_dir(seed, prefix="gt4") / f"traj_{drive}"
        info = json.loads((path / "data_info.json").read_text())
        images = (cv2.imread(str(path / r["image"])) for r in info)
    elif variant == "official":
        path = record_gt.gt_dir(seed) / f"traj_{drive}"
        info = json.loads((path / "data_info.json").read_text())
        images = (cv2.imread(str(path / r["image"])) for r in info)
    else:
        raise ValueError(variant)
    sift = cv2.SIFT_create(nfeatures=400)
    matcher = cv2.BFMatcher()
    rot = np.full(len(info), np.nan)
    n = np.zeros(len(info), int)
    prev = None
    for k, img in enumerate(images):
        feats = col.sift_features(img, col.sg.segment(img), sift)
        if prev is not None and info[k]["action"][0] in col.LABELS:
            rot[k], n[k], _ = col.rotation_from_matches(prev, feats, matcher)
        prev = feats
    return rot, n


def sift_cues(seeds, variant):
    rots, ns = [], []
    for s in seeds:
        f = CACHE / f"stage3b_sift_{variant}_{s}_{_hash()}.npz"
        if f.exists():
            z = np.load(f)
            res = [(z[f"rot{d}"], z[f"n{d}"]) for d in range(3)]
        else:
            with ProcessPoolExecutor() as ex:
                res = list(ex.map(_sift_drive, [(s, d, variant) for d in range(3)]))
            np.savez(
                f,
                **{f"rot{d}": res[d][0] for d in range(3)},
                **{f"n{d}": res[d][1] for d in range(3)},
            )
        rots += [r for r, _ in res]
        ns += [n for _, n in res]
    return np.concatenate(rots), np.concatenate(ns)


def drive_arrays(seeds, variant):
    D = s3.load(seeds, variant)
    F = s3.fused(D)
    pairs = M.load_pairs(seeds)
    assert np.all(pairs["k"] == D["k"])
    psi = np.where(
        F["valid"], F["psi"], np.where(np.isfinite(D["sh"][:, 0]), D["sh"][:, 0], D["lsd"][:, 0])
    )
    D["rot"], D["n"] = sift_cues(seeds, variant)
    D["valid"] = F["valid"]
    return D, F, pairs, psi


def flags_for(D, pairs, psi, threshold=col.HEADING_T, mode="final"):
    """Detector output for every pair of the concatenated drives (first frame of a drive: none).
    ``mode``: ``final`` (the rule), ``heading`` (heading change only), ``matches`` (matches
    only)."""
    flag = np.zeros(len(psi), bool)
    delta = np.full(len(psi), np.nan)
    for s in np.unique(D["seed"]):
        for dr in range(3):
            idx = np.where((D["seed"] == s) & (D["drive"] == dr))[0]
            labels = list(pairs["label"][idx])
            if mode == "heading":
                f, d = col.flag_drive(psi[idx], labels, threshold)
            else:
                f, d = col.decide_drive(
                    psi[idx], D["rot"][idx], D["n"][idx], labels, D["valid"][idx], threshold
                )
                if mode == "matches":
                    f = np.array(
                        [
                            labels[j] in col.LABELS
                            and D["n"][idx][j] >= col.RESCUE_N
                            and abs(D["rot"][idx][j]) >= col.RESCUE_DEG
                            for j in range(len(idx))
                        ]
                    )
            flag[idx], delta[idx] = f, d
    return flag, delta


def metrics(pairs, flag, delta) -> dict:
    rot = np.degrees(np.abs(pairs["inc"][:, 2]))
    fi = np.isin(pairs["label"], col.LABELS)
    ev = fi & (rot > 30)
    ev45 = ev & (rot > 45)
    n_pairs = len(rot)
    return {
        "pairs": int(n_pairs),
        "events_over_30": int(ev.sum()),
        "events_over_45": int(ev45.sum()),
        "recall_over_30": float((flag & ev).sum() / ev.sum()),
        "recall_over_45": float((flag & ev45).sum() / ev45.sum()),
        "missed_over_30": int((ev & ~flag).sum()),
        "missed_over_45": int((ev45 & ~flag).sum()),
        "flagged_fraction": float(flag.sum() / n_pairs),
        "flagged": int(flag.sum()),
        "precision_over_30": float((flag & ev).sum() / max(flag.sum(), 1)),
        "flags_that_are_not_over_30": int((flag & ~ev).sum()),
    }


def false_flag_causes(D, pairs, flag) -> dict:
    """The flagged pairs that are not events: what is wrong with them?"""
    rot = np.degrees(np.abs(pairs["inc"][:, 2]))
    fi = np.isin(pairs["label"], col.LABELS)
    ev = fi & (rot > 30)
    fp = np.where(flag & ~ev)[0]
    mid = fp[rot[fp] > 20]  # real but smaller rotations
    return {
        "false_flags": int(len(fp)),
        "with_true_rotation_20_to_30": int(len(mid)),
        "with_true_rotation_under_20": int(len(fp) - len(mid)),
    }


def consistency_check(seed=21, drive=2, limit=150) -> dict:
    """The pipeline path from images (collision.detect_drive) against the cached-cue path."""
    path = record_gt.gt_dir(seed, prefix="gt4") / f"traj_{drive}"
    flags, delta, psi = col.detect_drive(path, limit=limit)
    D, F, pairs, psi_c = drive_arrays([seed], "gt4")
    idx = np.where((D["seed"] == seed) & (D["drive"] == drive))[0][:limit]
    sub = {
        "seed": D["seed"][idx],
        "drive": D["drive"][idx],
        "rot": D["rot"][idx],
        "n": D["n"][idx],
        "valid": D["valid"][idx],
    }
    fc, _ = flags_for(sub, {"label": pairs["label"][idx]}, psi_c[idx])
    return {
        "frames": int(limit),
        "flags_equal": bool(np.array_equal(flags, fc)),
        "max_abs_psi_difference": float(np.nanmax(np.abs(vp.diff90(psi, psi_c[idx])))),
        "flags_image_path": int(flags.sum()),
        "flags_cached_path": int(fc.sum()),
    }


def plot_trajectories(seed, variant_flags, pairs, D):
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    cs = gt["cell_size"]
    poses = record_gt.load_poses(record_gt.gt_dir(seed))
    fig, ax = plt.subplots(figsize=(9, 9))
    for r, row in enumerate(gt["grid"]):
        for c, ch in enumerate(row):
            if ch == "#":
                ax.add_patch(
                    plt.Rectangle((c * cs - cs / 2, r * cs - cs / 2), cs, cs, color="0.9", lw=0)
                )
    colors = ["tab:red", "tab:green", "tab:blue"]
    sel = D["seed"] == seed
    rot = np.degrees(np.abs(pairs["inc"][:, 2]))
    fi = np.isin(pairs["label"], col.LABELS)
    for d in range(3):
        ax.plot(poses[d][:, 0], poses[d][:, 1], color=colors[d], lw=0.5, alpha=0.6)
        idx = np.where(sel & (D["drive"] == d))[0]
        k = D["k"][idx]
        ev = fi[idx] & (rot[idx] > 30)
        for kk, e, f, rr in zip(k, ev, variant_flags[idx], rot[idx], strict=True):
            if e:
                ax.plot(
                    *poses[d][kk, :2],
                    "o" if f else "x",
                    ms=9 if rr > 45 else 6,
                    color="k" if f else "m",
                    mfc="none" if rr <= 45 else None,
                    mew=1.5,
                )
    ax.set_aspect("equal")
    ax.set_title(
        f"seed {seed}: collision events (>30 deg). o detected, magenta x missed, big = >45 deg"
    )
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"stage3b_collisions_{seed}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def plot_event_frames(D, F, pairs, psi, flag, delta, variant="gt4", n=6):
    rot = np.degrees(np.abs(pairs["inc"][:, 2]))
    fi = np.isin(pairs["label"], col.LABELS)
    ev = np.where(fi & (rot > 30) & np.isin(D["seed"], TEST))[0]
    rng = np.random.default_rng(0)
    big = [i for i in ev if rot[i] > 45]
    pick = list(rng.choice(big, min(4, len(big)), replace=False)) + list(
        rng.choice([i for i in ev if i not in big], 2, replace=False)
    )
    prefix = "gt4" if variant == "gt4" else "gt"
    fig, ax = plt.subplots(2, n, figsize=(4 * n, 6.5))
    for j, i in enumerate(pick):
        s, d, k = int(D["seed"][i]), int(D["drive"][i]), int(D["k"][i])
        base = record_gt.gt_dir(s, prefix=prefix) / f"traj_{d}"
        for r, kk in enumerate((k - 1, k)):
            ax[r, j].imshow(cv2.imread(str(base / f"{kk}.jpg"))[..., ::-1])
            ax[r, j].axis("off")
        ax[0, j].set_title(
            f"seed {s} drive {d} #{k - 1} -> #{k}\n"
            f"{pairs['label'][i]}, true rotation {rot[i]:.0f} deg",
            fontsize=8,
        )
        ax[1, j].set_title(
            f"psi {psi[i - 1]:+.1f} -> {psi[i]:+.1f}: change {delta[i]:.1f} deg\n"
            f"{'FLAGGED' if flag[i] else 'missed'}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIG / "stage3b_event_frames.png", dpi=80)
    plt.close(fig)


def main() -> None:
    res: dict = {"threshold_deg": col.THRESHOLD}
    dev, test = M.load_pairs(DEV), M.load_pairs(TEST)
    res["prior"] = prior_checks(dev, test)
    print(
        json.dumps({k: v for k, v in res["prior"].items() if not k.startswith("region")}, indent=1)[
            :1500
        ]
    )
    for p in ("region_95", "region_99"):
        r = res["prior"][p]
        print(
            p,
            r["by_label"],
            r["groups_min"],
            r["groups_max"],
            r["events_inside"],
            r["events_over_45_inside"],
        )
    sweep = {}
    for variant in VARIANTS:
        Dd, Fd, Pd, psid = drive_arrays(DEV, variant)
        Dt, Ft, Pt, psit = drive_arrays(TEST, variant)
        sweep[variant] = {
            str(t): {
                "dev": metrics(Pd, *flags_for(Dd, Pd, psid, t)),
                "test": metrics(Pt, *flags_for(Dt, Pt, psit, t)),
            }
            for t in (15, 18, 20, 22, 25, 28)
        }
        res[variant] = {}
        for mode in ("heading", "matches", "final"):
            res[variant][mode] = {
                "dev": metrics(Pd, *flags_for(Dd, Pd, psid, mode=mode)),
                "test": metrics(Pt, *flags_for(Dt, Pt, psit, mode=mode)),
            }
        flag_t, delta_t = flags_for(Dt, Pt, psit)
        flag_d, delta_d = flags_for(Dd, Pd, psid)
        res[variant]["false_flags_test"] = false_flag_causes(Dt, Pt, flag_t)
        rot = np.degrees(np.abs(Pt["inc"][:, 2]))
        ev = np.isin(Pt["label"], col.LABELS) & (rot > 30)
        res[variant]["missed_test"] = [
            {
                "seed": int(Dt["seed"][i]),
                "drive": int(Dt["drive"][i]),
                "k": int(Dt["k"][i]),
                "rotation_deg": float(rot[i]),
                "label": str(Pt["label"][i]),
                "heading_change_deg": float(delta_t[i]) if np.isfinite(delta_t[i]) else None,
                "match_rotation_deg": float(Dt["rot"][i]) if np.isfinite(Dt["rot"][i]) else None,
                "match_n": int(Dt["n"][i]),
                "valid_heading_before_after": [bool(Ft["valid"][i - 1]), bool(Ft["valid"][i])],
            }
            for i in np.where(ev & ~flag_t)[0]
        ]
        res[variant]["per_seed_test"] = {
            str(s): metrics(
                {k: v[Dt["seed"] == s] for k, v in Pt.items()},
                flag_t[Dt["seed"] == s],
                delta_t[Dt["seed"] == s],
            )
            for s in TEST
        }
        if variant == "gt4":
            for s in DEV + TEST:
                D_, P_ = (Dd, Pd) if s in DEV else (Dt, Pt)
                plot_trajectories(s, flag_d if s in DEV else flag_t, P_, D_)
            plot_event_frames(Dt, Ft, Pt, psit, flag_t, delta_t)
        for mode in ("heading", "matches", "final"):
            m = res[variant][mode]
            print(
                variant,
                mode,
                "dev",
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in m["dev"].items()
                    if k in ("recall_over_30", "recall_over_45", "flagged_fraction", "flagged")
                },
                "| test",
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in m["test"].items()
                    if k
                    in (
                        "recall_over_30",
                        "recall_over_45",
                        "flagged_fraction",
                        "flagged",
                        "missed_over_45",
                    )
                },
                flush=True,
            )
        print("   missed (final, test):", res[variant]["missed_test"], flush=True)
    res["threshold_sweep"] = sweep
    res["consistency"] = consistency_check()
    print(res["consistency"])
    RESULTS.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
