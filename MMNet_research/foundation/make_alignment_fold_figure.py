"""Per-fold effect of embedding alignment on downstream staging (manuscript figure).

For each of the ten patient-independent folds, the three-seed staging accuracy of MM-Net with
a sleep-adapted encoder whose training embeddings are cross-fitted but not aligned, and with the
same encoder cross-fitted and ridge-aligned, each minus the frozen-encoder reference
(base-nocw-cross). Source: results/revision/runs/lesion/mmnet_ftl.json (test folds, seed
ensemble), the rows of Table tab:components.

    python make_alignment_fold_figure.py
Output: MMNet_research/submission/bspc_cas/manuscript/figures/revision/fig_alignment_folds.pdf
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "revision", "runs", "lesion", "mmnet_ftl.json")
OUT = os.path.join(HERE, "..", "submission", "bspc_cas", "manuscript", "figures", "revision",
                   "fig_alignment_folds.pdf")

d = json.load(open(RES))
acc = {k: np.array(d[k]["te"]["ensemble"]["acc"]) for k in
       ("base-nocw-cross", "base-nocw-cross-ftl", "base-nocw-cross-ftla")}
ref = acc["base-nocw-cross"]
series = [("Adapted, cross-fitted, not aligned", acc["base-nocw-cross-ftl"] - ref, "#eb6834", "s"),
          ("Adapted, cross-fitted, aligned", acc["base-nocw-cross-ftla"] - ref, "#2a78d6", "s")]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.0, 2.5))
x = np.arange(1, 11)
w = 0.36
for i, (name, delta, color, marker) in enumerate(series):
    pos = x + (i - 0.5) * (w + 0.04)
    ax.bar(pos, 100 * delta, width=w, color=color, edgecolor="white", linewidth=0.8, zorder=2)
    up = int((delta > 0).sum())
    ax.plot([], [], marker=marker, linestyle="none", color=color, markersize=6,
            label=f"{name}: mean {100 * delta.mean():+.1f} pp, {up}/10 folds above")
ax.axhline(0, color="#52514e", linewidth=0.8, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in x])
ax.set_xlabel("Patient-independent fold", color="#52514e")
ax.set_ylabel("Accuracy vs frozen\nencoder (pp)", color="#52514e")
ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.5, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1, frameon=False, fontsize=7.5,
          handletextpad=0.4)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
for name, delta, _, _ in series:
    print(name, np.round(100 * delta, 2), "mean %+.2f" % (100 * delta.mean()))
