"""Two clean figures for the deck (graph only, no titles/labels/callouts): the cohort
sleep-disordered-breathing burden, and the staging benchmark across architectures."""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "revision", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif", "figure.dpi": 140})
def save(fig, n): fig.savefig(f"{FIG}/{n}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{n}.png", bbox_inches="tight"); plt.close(fig)

# cohort SDB burden: per-patient fraction of epochs with a scored event
prev = []
for f in sorted(glob.glob(os.path.join(ROOT, "data", "mm_features", "SN*.npz"))):
    if int(re.search(r"SN(\d+)", f).group(1)) == 28: continue
    prev.append(100 * np.load(f)["apnea"].mean())
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.hist(prev, bins=20, color="#55A868", edgecolor="white")
ax.set_xlabel("% of epochs with a scored respiratory event"); ax.set_ylabel("number of patients")
save(fig, "fig_sdb_burden")

# staging accuracy across architectures (grouped by family), clean horizontal bars
groups = [("Published", ["LSTM", "Transformer", "CNN-ResNet18"], [0.747, 0.674, 0.617], "#8C8CB0"),
          ("Deep (healthy-built)", ["DeepSleepNet", "AttnSleep", "Sleep-EDF transfer", "Raw multimodal CNN"],
           [0.615, 0.686, 0.623, 0.655], "#E0A458"),
          ("Feature / ensemble", ["Feature-seq BiLSTM", "Graph + SSM", "Gradient boosting"],
           [0.676, 0.730, 0.746], "#55A868"),
          ("Proposed", ["MM-Net"], [0.722], "#C44E52")]
labels, vals, colors = [], [], []
for _, ns, vs, c in groups:
    for n, v in zip(ns, vs): labels.append(n); vals.append(v); colors.append(c)
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.barh(y, vals, color=colors, edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.set_xlim(0.58, 0.77)
ax.set_xlabel("staging accuracy")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, _, c in groups]
ax.legend(handles, [g[0] for g in groups], fontsize=8, loc="lower right")
save(fig, "fig_benchmark_clean")
print("wrote fig_sdb_burden, fig_benchmark_clean")
