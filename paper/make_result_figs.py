"""Result figures for the Experiments section: staging benchmark bar chart and an
accuracy-vs-size efficiency plot. Built from the same real numbers as the tables."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

# ---- Fig: staging-accuracy benchmark (horizontal bars, grouped by family) ----
groups = [
    ("Published",  ["LSTM", "Transformer", "CNN-ResNet18"],
                   [0.747, 0.674, 0.617], "#8C8CB0"),
    ("Deep",       ["DeepSleepNet", "AttnSleep", "CNN+BiLSTM", "Sleep-EDF transfer",
                    "Seq CNN-BiLSTM (4ch)", "Seq CNN-BiLSTM (7ch)", "Raw multimodal CNN"],
                   [0.615, 0.686, 0.613, 0.623, 0.645, 0.654, 0.655], "#E0A458"),
    ("Feature",    ["Gradient boosting", "Feature-seq BiLSTM", "Learnable refiner",
                    "Graph attention + SSM", "Stacking ensemble", "GB ensemble"],
                   [0.655, 0.676, 0.704, 0.730, 0.735, 0.746], "#55A868"),
    ("Proposed",   ["MM-Net (proposed)"], [0.721], "#C44E52"),
]
labels, vals, colors = [], [], []
for _, names, accs, col in groups:
    for n, a in zip(names, accs):
        labels.append(n); vals.append(a); colors.append(col)
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(3.4, 4.4))
bars = ax.barh(y, vals, color=colors, edgecolor="white", height=0.72)
# highlight proposed
for i, lab in enumerate(labels):
    if "proposed" in lab:
        bars[i].set_edgecolor("#7a1d20"); bars[i].set_linewidth(1.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2)
ax.set_xlim(0.58, 0.785); ax.set_xlabel("Staging accuracy")
for yi, v in zip(y, vals):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=6.4)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, _, c in groups]
ax.legend(handles, [g[0] for g in groups], fontsize=6.6, loc="center right",
          bbox_to_anchor=(1.0, 0.42), framealpha=0.95)
ax.set_title("Staging accuracy by architecture", fontsize=8.6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_benchmark.pdf"), bbox_inches="tight"); plt.close(fig)
print("wrote fig_benchmark.pdf")

# ---- Fig: accuracy vs model size (efficiency) ----
pts = [("CNN-ResNet18", 11, 0.617, "#8C8CB0"), ("DeepSleepNet", 21, 0.615, "#E0A458"),
       ("AttnSleep", 0.5, 0.686, "#E0A458"), ("CNN+BiLSTM", 0.6, 0.613, "#E0A458"),
       ("Sleep-EDF transfer", 0.6, 0.623, "#E0A458"), ("Seq (7ch)", 0.8, 0.654, "#E0A458"),
       ("Raw multimodal CNN", 0.95, 0.655, "#E0A458"), ("Feature-seq BiLSTM", 1.2, 0.676, "#55A868"),
       ("Learnable refiner", 0.5, 0.704, "#55A868"), ("Graph attention + SSM", 0.49, 0.730, "#55A868"),
       ("MM-Net (proposed)", 0.86, 0.721, "#C44E52")]
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for name, p, a, c in pts:
    big = "proposed" in name
    ax.scatter(p, a, s=90 if big else 45, color=c, edgecolor="#333" if big else "white",
               linewidth=1.4 if big else 0.6, zorder=3)
    if big or name in ("Graph attention + SSM", "GB ensemble", "DeepSleepNet"):
        ax.annotate(name, (p, a), fontsize=6.4, xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("Staging accuracy"); ax.set_ylim(0.60, 0.76)
ax.set_title("Accuracy versus model size", fontsize=8.6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_efficiency.pdf"), bbox_inches="tight"); plt.close(fig)
print("wrote fig_efficiency.pdf")
print("done")
