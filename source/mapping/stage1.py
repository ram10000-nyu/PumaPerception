"""Stage 1: fit the action-label motion prior on dev seeds, check it on test seeds, and
dead-reckon from the spawn for the reference numbers.

    uv run source/mapping/stage1.py

Writes ``cache/mapping/motion_prior.json``, ``cache/mapping/stage1_results.json`` and the
figures ``reports/figures/stage1_*.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion as M  # noqa: E402
import record_gt  # noqa: E402
import se2  # noqa: E402

ROOT = record_gt.ROOT
FIG = ROOT / "reports" / "figures"


def det_ratio(pairs, keys_fn, label_sel, min_n=30) -> tuple[float, int]:
    """Weighted geometric mean of group covariance determinants over the unconditional one,
    for the groups of ``keys_fn`` with at least ``min_n`` samples. Also returns how many
    samples fell in such groups."""
    x = pairs["inc"][label_sel]
    d0 = np.linalg.det(np.cov(x.T))
    keys = keys_fn(label_sel)
    ld = n = 0.0
    for k in set(keys):
        g = x[keys == k]
        if len(g) >= min_n:
            ld += len(g) * np.log(np.linalg.det(np.cov(g.T)))
            n += len(g)
    return float(np.exp(ld / n) / d0), int(n)


def conditioning(dev) -> dict:
    out = {}
    for lab in M.LABELS:
        sel = dev["label"] == lab
        pr = lambda s: dev["prev"][s]  # noqa: E731
        pn = lambda s: np.char.add(np.char.add(dev["prev"][s], ">"), dev["next"][s])  # noqa: E731
        base_prev, n1 = det_ratio(dev, pr, sel)
        with_next, n2 = det_ratio(dev, pn, sel)
        out[lab] = {
            "det_ratio_prev": base_prev,
            "samples_prev": n1,
            "det_ratio_prev_next": with_next,
            "samples_prev_next": n2,
            "shrink_prev_pct": 100 * (1 - base_prev),
            "keep_prev": bool(base_prev < 0.7),
        }
    return out


def calibration(prior: M.MotionPrior, pairs: dict, use_prev=True) -> dict:
    out = {}
    for lab in M.LABELS:
        sel = pairs["label"] == lab
        inside = np.zeros(sel.sum(), bool)
        ell = np.zeros(sel.sum(), bool)
        x = pairs["inc"][sel]
        prevs = pairs["prev"][sel]
        for i in range(len(x)):
            g = prior.get(lab, prevs[i] if use_prev else None)
            inside[i] = bool(g.contains(x[i])[0])
            ell[i] = bool(g.mahalanobis2(x[i])[0] <= M.CHI2_95_3DOF)
        out[lab] = {
            "n": int(sel.sum()),
            "hdr_coverage": float(inside.mean()),
            "ellipse_coverage": float(ell.mean()),
        }
        # per (prev) group
        grp = {}
        for pl in M.PREV:
            g = prevs == pl
            if g.sum() >= 30:
                grp[pl] = {"n": int(g.sum()), "hdr_coverage": float(inside[g].mean())}
        out[lab]["by_prev"] = grp
    return out


def mean_agreement(dev, test) -> dict:
    """Dev vs test per-label means in standard errors. ``naive`` treats samples as
    independent; ``clustered`` uses the spread of per-drive means (samples within a drive
    are correlated)."""
    out = {}
    for lab in M.LABELS:
        r = {}
        for name, d in (("dev", dev), ("test", test)):
            sel = d["label"] == lab
            x = d["inc"][sel]
            keys = d["seed"][sel] * 10 + d["drive"][sel]
            means = np.array([x[keys == k].mean(0) for k in np.unique(keys)])
            r[name] = {
                "mean": x.mean(0),
                "se_naive": x.std(0, ddof=1) / np.sqrt(len(x)),
                "se_clustered": means.std(0, ddof=1) / np.sqrt(len(means)),
                "n": len(x),
                "drives": len(means),
            }
        diff = r["test"]["mean"] - r["dev"]["mean"]
        z_naive = diff / np.hypot(r["dev"]["se_naive"], r["test"]["se_naive"])
        z_clu = diff / np.hypot(r["dev"]["se_clustered"], r["test"]["se_clustered"])
        out[lab] = {
            "dev_mean": r["dev"]["mean"].tolist(),
            "test_mean": r["test"]["mean"].tolist(),
            "z_naive": z_naive.tolist(),
            "z_clustered": z_clu.tolist(),
            "pass_naive": bool(np.all(np.abs(z_naive) <= 2)),
            "pass_clustered": bool(np.all(np.abs(z_clu) <= 2)),
        }
    return out


def dead_reckon(seed: int, prior: M.MotionPrior, use_prev: bool) -> list[dict]:
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    res = []
    for d, P in enumerate(record_gt.load_poses(record_gt.gt_dir(seed))):
        info = json.loads((record_gt.gt_dir(seed) / f"traj_{d}" / "data_info.json").read_text())
        labels = [r["action"][0] for r in info]
        est = se2.integrate(gt["spawn"], prior.mean_increments(labels, use_prev))
        perr = np.hypot(*(est[:, :2] - P[:, :2]).T)
        herr = se2.wrap(est[:, 2] - P[:, 2])
        res.append(
            {
                "est": est,
                "true": P,
                "pos_err": perr,
                "head_err": herr,
                "ate_rms": float(np.sqrt(np.mean(perr**2))),
                "final_err": float(perr[-1]),
                "heading_rms_deg": float(np.degrees(np.sqrt(np.mean(herr**2)))),
                "n": len(P),
            }
        )
    return res


# ------------------------------------------------------------------------------- figures


def plot_label_scatter(prior, dev, test):
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for j, lab in enumerate(M.LABELS):
        for i, (name, d) in enumerate((("dev", dev), ("test", test))):
            sel = d["label"] == lab
            x = d["inc"][sel]
            g = prior.by_label[lab]
            ins = g.contains(x)
            a = ax[i, j]
            a.plot(
                x[ins, 0], x[ins, 2], ".", ms=2, color="0.6", label=f"inside HDR {ins.mean():.1%}"
            )
            a.plot(x[~ins, 0], x[~ins, 2], ".", ms=4, color="tab:red", label="outside")
            a.plot(g.means[:, 0], g.means[:, 2], "kx", ms=8)
            a.plot(*g.mean[[0, 2]], "b+", ms=12)
            a.set_title(f"{lab}: {name} seeds (label only prior)")
            a.set_xlabel("dx [m]")
            a.set_ylabel("dtheta [rad]")
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_label_scatter.png", dpi=90)
    plt.close(fig)


def plot_pair_scatter(prior, dev, test):
    keys = [k for k in prior.by_pair]
    fig, ax = plt.subplots(4, 5, figsize=(22, 14))
    for a in ax.ravel():
        a.axis("off")
    for a, key in zip(ax.ravel(), keys, strict=False):
        pl, lab = key.split(">")
        g = prior.by_pair[key]
        sel = (test["label"] == lab) & (test["prev"] == pl)
        x = test["inc"][sel]
        ins = g.contains(x)
        a.axis("on")
        a.plot(x[ins, 0], x[ins, 2], ".", ms=2, color="0.6")
        a.plot(x[~ins, 0], x[~ins, 2], ".", ms=4, color="tab:red")
        a.plot(g.means[:, 0], g.means[:, 2], "kx", ms=8)
        a.set_title(f"prev {pl} -> {lab}: test, HDR {ins.mean():.1%} (n={sel.sum()})", fontsize=8)
        a.set_xlabel("dx [m]", fontsize=7)
        a.set_ylabel("dtheta [rad]", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_pair_scatter.png", dpi=80)
    plt.close(fig)


def plot_dr(seed, res, tag):
    fig, ax = plt.subplots(3, 3, figsize=(15, 12))
    gt = json.loads((record_gt.gt_dir(seed) / "gt.json").read_text())
    cs = gt["cell_size"]
    for i, r in enumerate(res):
        a = ax[0, i]
        for rr, c in enumerate(gt["grid"]):
            for cc, ch in enumerate(c):
                if ch == "#":
                    a.add_patch(
                        plt.Rectangle(
                            (cc * cs - cs / 2, rr * cs - cs / 2), cs, cs, color="0.88", lw=0
                        )
                    )
        a.plot(r["true"][:, 0], r["true"][:, 1], "k", lw=0.8, label="true")
        a.plot(r["est"][:, 0], r["est"][:, 1], "tab:red", lw=0.8, label="dead reckoned")
        a.plot(*gt["spawn"][:2], "go")
        a.set_aspect("equal")
        a.set_title(
            f"seed {seed} drive {i}: ATE {r['ate_rms']:.2f} m, final {r['final_err']:.2f} m"
        )
        a.legend(fontsize=7)
        ax[1, i].plot(r["pos_err"], lw=0.6)
        ax[1, i].set_ylabel("position error [m]")
        ax[2, i].plot(np.degrees(r["head_err"]), lw=0.6)
        ax[2, i].set_ylabel("heading error [deg]")
        ax[2, i].set_xlabel("frame")
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(FIG / f"stage1_dead_reckoning_{seed}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    cfg = json.loads(record_gt.SEEDS.read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    dev = M.load_pairs(cfg["dev"])
    test = M.load_pairs(cfg["test"])
    prior = M.MotionPrior.fit(dev)
    prior.save()
    label_only = M.MotionPrior(prior.by_label, {}, prior.meta)
    results = {
        "n_dev": int(len(dev["inc"])),
        "n_test": int(len(test["inc"])),
        "conditioning": conditioning(dev),
        "calibration_test": calibration(prior, test),
        "calibration_test_label_only": calibration(label_only, test, use_prev=False),
        "calibration_dev": calibration(prior, dev),
        "mean_agreement": mean_agreement(dev, test),
        "groups": {
            k: {
                "n": g.n,
                "components": len(g.weights),
                "weights": g.weights.tolist(),
                "means": g.means.tolist(),
                "inflation": g.inflation,
            }
            for k, g in {**prior.by_label, **prior.by_pair}.items()
        },
        "dead_reckoning": {},
    }
    plot_label_scatter(prior, dev, test)
    plot_pair_scatter(prior, dev, test)
    for s in cfg["dev"] + cfg["test"]:
        row = {}
        for name, up in (("pair_prior", True), ("label_prior", False)):
            r = dead_reckon(s, prior, up)
            row[name] = [
                {
                    k: v
                    for k, v in d.items()
                    if k in ("ate_rms", "final_err", "heading_rms_deg", "n")
                }
                for d in r
            ]
            if up:
                plot_dr(
                    s,
                    r,
                    f"seed {s} ({'dev, in-sample prior' if s in cfg['dev'] else 'test'}): "
                    "dead reckoning with (previous, current) prior means",
                )
        results["dead_reckoning"][str(s)] = row
    (ROOT / "cache" / "mapping" / "stage1_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["conditioning"], indent=0)[:1500])
    for lab in M.LABELS:
        c = results["calibration_test"][lab]
        print(
            lab,
            "test HDR",
            round(c["hdr_coverage"], 3),
            "ellipse",
            round(c["ellipse_coverage"], 3),
            "| label-only HDR",
            round(results["calibration_test_label_only"][lab]["hdr_coverage"], 3),
            "| mean z naive",
            np.round(results["mean_agreement"][lab]["z_naive"], 1),
            "clustered",
            np.round(results["mean_agreement"][lab]["z_clustered"], 1),
        )
    for s, row in results["dead_reckoning"].items():
        print(
            s,
            [
                (round(d["ate_rms"], 2), round(d["final_err"], 2), round(d["heading_rms_deg"], 1))
                for d in row["pair_prior"]
            ],
            [(round(d["ate_rms"], 2), round(d["heading_rms_deg"], 1)) for d in row["label_prior"]],
        )


if __name__ == "__main__":
    main()
