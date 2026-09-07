"""fig_attribution_agreement -- two independent attributions, cross-checked.

Round-2 referee Finding 2: the t-SNE panel "still does not earn its place... it
shows only that the representation separates sleep stages -- which any accurate
stager would show... Neither statement is falsifiable." The suggested replacement
is an attribution that could have come out otherwise: does per-feature
permutation importance implicate the channels the retrain-ablation implicates?

This figure answers that directly. Each point is one modality. The x axis is the
performance lost when the model is RETRAINED without it; the y axis is the
performance lost when it is SHUFFLED at inference in the trained model. The two
methods share no machinery, so agreement is a real check rather than a
restatement -- and disagreement would have been visible as a cloud with no trend.

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/fig_attribution_agreement.py
"""
import json
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
from mmnet_repro import REV, RUNS          # noqa: E402

FIG = os.path.join(REV, "figures")
FD = os.path.join(REV, "figdata")
os.makedirs(FIG, exist_ok=True)
os.makedirs(FD, exist_ok=True)
SEEDS = [42, 1, 7]

# ablation condition -> permutation-importance modality name
PAIRS = [("-EEG", "EEG"), ("-EOG", "EOG"), ("-EMG", "EMG"), ("-SpO2", "SpO2"),
         ("-pulse/HRV", "pulse/HRV"), ("-ECG", "ECG"), ("-airflow", "airflow"),
         ("-effort", "effort")]


def pooled(src, cond, metric):
    v = [src["%s|%d" % (cond, s)][metric] for s in SEEDS if "%s|%d" % (cond, s) in src]
    return np.array(v).ravel() if v else None


def collect():
    ms = json.load(open(os.path.join(RUNS, "multiseed_ablation.json")))
    extra = {}
    for f in glob.glob(os.path.join(RUNS, "foundation", "ablation_shards", "*.json")):
        extra.update(json.load(open(f)))
    perm = json.load(open(os.path.join(RUNS, "permutation_importance.json")))["importance"]
    base_acc, base_auc = pooled(ms, "full", "acc"), pooled(ms, "full", "auc")
    rows = []
    for cond, mod in PAIRS:
        src = ms if "%s|42" % cond in ms else extra
        a, u = pooled(src, cond, "acc"), pooled(src, cond, "auc")
        if a is None or mod not in perm:
            print("  skipped (missing):", cond)
            continue
        rows.append(dict(modality=mod,
                         abl_acc=float(base_acc.mean() - a.mean()),
                         perm_acc=float(perm[mod]["acc_drop"]),
                         abl_auc=float(base_auc.mean() - u.mean()),
                         perm_auc=float(perm[mod]["auc_drop"])))
    return rows


# Candidate label offsets in points, tried in order of preference. Several of
# these modalities sit almost on top of one another (SpO2/airflow differ by
# 0.0002 in staging), so a fixed offset guarantees overlapping text.
_OFFSETS = [(7, 5), (7, -12), (-11, 5), (-11, -12), (7, 14), (-11, 14),
            (7, -22), (-11, -22), (16, 0), (-20, 0)]


def _place_labels(ax, labels, x, y, fontsize, colour="0.25"):
    """Greedy declutter: give each label the first offset that hits nothing.

    Points are placed largest-effect first, so the modalities that carry the
    result get the cleanest positions and the near-zero cluster takes whatever
    is left. A leader line is drawn whenever a label had to be pushed far enough
    that its owner would otherwise be ambiguous.
    """
    fig = ax.figure
    fig.canvas.draw()                      # renderer needed for text extents
    rend = fig.canvas.get_renderer()
    placed = []
    order = sorted(range(len(labels)), key=lambda i: -(abs(x[i]) + abs(y[i])))
    for i in order:
        px, py = ax.transData.transform((x[i], y[i]))
        chosen = None
        for dx, dy in _OFFSETS:
            t = ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize, color=colour, zorder=4)
            bb = t.get_window_extent(renderer=rend).expanded(1.06, 1.16)
            if not any(bb.overlaps(b) for b in placed):
                placed.append(bb); chosen = (dx, dy, bb); break
            t.remove()
        if chosen is None:                 # everything collided: take the last slot
            dx, dy = _OFFSETS[-1]
            t = ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize, color=colour, zorder=4)
            placed.append(t.get_window_extent(renderer=rend))
            chosen = (dx, dy, placed[-1])
        dx, dy, _ = chosen
        if abs(dx) > 12 or abs(dy) > 15:   # far enough to need a pointer
            ax.annotate("", (x[i], y[i]), textcoords="offset points", xytext=(dx, dy),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.7",
                                        shrinkA=0, shrinkB=3), zorder=2)


def _scatter(ax, rows, x, y, fontsize=8, label_thresh=None):
    """Draw the points with decluttered labels."""
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter(x, y, s=46, c="#2b6cb0", edgecolor="white", linewidth=0.8, zorder=3)
    keep = [i for i in range(len(rows))
            if label_thresh is None or (abs(x[i]) + abs(y[i])) >= label_thresh]
    if keep:
        xk = np.asarray(x)[keep]; yk = np.asarray(y)[keep]
        _place_labels(ax, [rows[i]["modality"] for i in keep], xk, yk, fontsize)


def panel(ax, rows, xk, yk, title, inset_below=None):
    x = np.array([r[xk] for r in rows])
    y = np.array([r[yk] for r in rows])
    rho, p = spearmanr(x, y)
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls=":", c="0.6", lw=1.0,
            zorder=1, label="identity")
    _scatter(ax, rows, x, y, label_thresh=inset_below)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("retrain ablation  (loss when removed)")
    ax.set_ylabel("permutation  (loss when shuffled)")
    star = "*" if p < 0.05 else ""
    ax.set_title("%s\nSpearman $\\rho$ = %.3f, p = %.4f%s  (n = %d)"
                 % (title, rho, p, star, len(rows)), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    if inset_below is not None:
        # For this head only two modalities matter; the rest sit at the origin in
        # BOTH methods, which is itself the result. The inset shows that cluster
        # is genuinely near zero rather than merely unresolved at this scale.
        sel = [i for i in range(len(rows)) if abs(x[i]) + abs(y[i]) < inset_below]
        if len(sel) >= 2:
            axin = ax.inset_axes([0.50, 0.09, 0.47, 0.44])
            xs, ys = x[sel], y[sel]
            sub = [rows[i] for i in sel]
            m = max(float(np.abs(np.r_[xs, ys]).max()), 1e-4) * 2.1
            axin.plot([-m, m], [-m, m], ls=":", c="0.7", lw=0.8, zorder=1)
            _scatter(axin, sub, xs, ys, fontsize=6.5)
            axin.set_xlim(-m, m); axin.set_ylim(-m, m)
            axin.tick_params(labelsize=6)
            axin.set_title("near zero in both methods", fontsize=7, color="0.35", pad=3)
            axin.grid(alpha=0.2, lw=0.4)
            for sp in axin.spines.values():
                sp.set_color("0.7")
    return float(rho), float(p)


def main():
    rows = collect()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    r_acc = panel(axes[0], rows, "abl_acc", "perm_acc", "Staging accuracy", inset_below=0.02)
    r_auc = panel(axes[1], rows, "abl_auc", "perm_auc", "Respiratory AUC")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig_attribution_agreement.%s" % ext),
                    dpi=200, bbox_inches="tight")
    json.dump({"rows": rows,
               "staging": {"rho": r_acc[0], "p": r_acc[1]},
               "respiratory": {"rho": r_auc[0], "p": r_auc[1]}},
              open(os.path.join(FD, "attribution_agreement.json"), "w"), indent=1)
    print("staging     rho=%.3f p=%.4f" % r_acc)
    print("respiratory rho=%.3f p=%.4f" % r_auc)
    print("wrote", os.path.join(FIG, "fig_attribution_agreement.pdf"))


if __name__ == "__main__":
    main()
