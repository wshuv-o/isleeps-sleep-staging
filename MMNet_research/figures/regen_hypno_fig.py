"""Regenerate fig_hypnogram from the arrays the notebook already saved (no retraining).
Same layout as the notebook's figure cell, with the corrected tag placement (left-aligned
titles above each hypnogram instead of text overlapping the trace)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "results", "revision", "figures")
d = np.load(os.path.join(FIGDIR, "hypnogram_data.npz"), allow_pickle=True)
ref, pred, sp = d["ref"], d["pred"], d["stage_probs"]
Sdb, spec_f, spec_t = d["spec_db"], d["spec_f"], d["spec_t"]
acc = float(d["acc"]); sid = str(d["subject"])
n_ep = len(ref); T = n_ep * 30 / 3600.0
te_hr = (np.arange(n_ep) + 0.5) * 30 / 3600.0

LADDER = np.array([4, 2, 1, 0, 3]); YT = [0, 1, 2, 3, 4]; YL = ["N3", "N2", "N1", "R", "W"]
COL = {"N1": "#A6CEE3", "N2": "#3B8BC2", "N3": "#2C5378", "R": "#8E4A9E", "W": "#F2C230"}
plt.rcParams.update({"font.size": 11, "font.family": "serif"})
fig = plt.figure(figsize=(9, 8))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1.7, 1.7], hspace=0.42)

def hypno(ax, stages, tag, color):
    ax.step(te_hr, LADDER[stages], where="post", color="k", lw=0.8)
    rem = stages == 4
    if rem.any():
        ax.step(te_hr, np.where(rem, LADDER[stages].astype(float), np.nan), where="post", color="#C0392B", lw=1.8)
    ax.set_yticks(YT); ax.set_yticklabels(YL, fontsize=8); ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, T); ax.set_ylabel("Stage", fontsize=9)
    ax.set_title(tag, loc="left", fontsize=11, fontweight="bold", color=color, pad=3)
    ax.tick_params(labelbottom=False)

axA1 = fig.add_subplot(gs[0]); hypno(axA1, pred, "Predicted", "#25507B")
axA2 = fig.add_subplot(gs[1]); hypno(axA2, ref, "Reference", "#C77A17")
axB = fig.add_subplot(gs[2])
vmin, vmax = np.percentile(Sdb, [5, 97])
axB.pcolormesh(spec_t, spec_f, Sdb, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
axB.set_ylim(0, 25); axB.set_xlim(0, T); axB.set_ylabel("Frequency [Hz]", fontsize=9); axB.tick_params(labelbottom=False)
axC = fig.add_subplot(gs[3])
order = ["N1", "N2", "N3", "R", "W"]; idx = {"N1": 1, "N2": 2, "N3": 3, "R": 4, "W": 0}
axC.stackplot(te_hr, *[sp[:, idx[k]] for k in order], colors=[COL[k] for k in order], labels=order)
axC.plot(te_hr, sp.max(1), color="k", lw=0.9)
axC.set_xlim(0, T); axC.set_ylim(0, 1); axC.set_ylabel("Probability", fontsize=9); axC.set_xlabel("Time [hrs]", fontsize=10)
axC.legend(loc="lower left", ncol=1, fontsize=7, framealpha=0.9)
for ax, lab, yy in [(axA1, "A", 1.10), (axB, "B", 1.02), (axC, "C", 1.02)]:
    ax.text(-0.085, yy, lab, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")

fig.savefig(os.path.join(FIGDIR, "fig_hypnogram.pdf"), bbox_inches="tight", dpi=200)
fig.savefig(os.path.join(FIGDIR, "fig_hypnogram.png"), bbox_inches="tight", dpi=200)
print(f"regenerated fig_hypnogram ({sid}, acc={acc:.3f}) with corrected tag placement")
