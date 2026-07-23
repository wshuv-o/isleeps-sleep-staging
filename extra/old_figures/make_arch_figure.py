"""
make_arch_figure.py -- detailed architecture figure, black-and-white base with colour
reserved for the novel contribution (event/physiological features + clinical readout).
Solid fills only (no transparency). Rendered + verified.
Output: paper/figures/architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams["font.family"] = "DejaVu Sans"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "architecture.png")

WHITE, BLACK = "#FFFFFF", "#000000"
ORNG_F, ORNG_E, ORNG_T = "#F6C877", "#B26A00", "#5E3A00"   # solid orange = contribution

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 160); ax.set_ylim(0, 92); ax.axis("off")


def group(x, y, w, h, ec, title, tfs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                linewidth=1.6, edgecolor=ec, facecolor=WHITE))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=tfs,
            fontweight="bold", color=ec)


def cell(x, y, w, h, fc, ec, tc, head, body="", hfs=9.4, bfs=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    if body:
        ax.text(x + w / 2, y + h - 2.2, head, ha="center", va="top", fontsize=hfs,
                fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=bfs,
                color=tc, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, head, ha="center", va="center", fontsize=hfs,
                fontweight="bold", color=tc)


def arrow(x1, y1, x2, y2, color=BLACK, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0))


def dim(x, y, s, c=BLACK):
    ax.text(x, y, s, ha="center", va="center", fontsize=10, color=c, style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc=WHITE, ec="none"))


# ================= ROW 1 : input -> features -> context =================
cell(3, 62, 22, 20, WHITE, BLACK, BLACK, "Input (PSG epoch)",
     "7 ch × 3000\n30 s @ 100 Hz\nEEG ×4, EOG ×2, EMG")

# ---- Feature Extraction group with two lanes ----
group(30, 40, 62, 46, BLACK, "Feature Extraction")
# lane A: spectral + temporal  (black & white)
ax.text(46, 82.2, "Spectral + temporal   (23 / ch × 7 = 161)", ha="center", va="top",
        fontsize=9.2, fontweight="bold", color=BLACK)
cell(33, 71.5, 26, 6.4, WHITE, BLACK, BLACK, "band power δ/θ/α/σ/β (abs+rel), 10/ch", "", hfs=7.6)
cell(33, 64.3, 26, 6.4, WHITE, BLACK, BLACK, "spectral entropy + edge, 2/ch", "", hfs=7.8)
cell(33, 57.1, 26, 6.4, WHITE, BLACK, BLACK, "Hjorth: activity/mobility/complexity, 3/ch", "", hfs=7.4)
cell(33, 49.9, 26, 6.4, WHITE, BLACK, BLACK, "time-domain: var, ZCR, percentiles, 8/ch", "", hfs=7.4)
# lane B: event / physiological  (COLOURED = contribution)
ax.text(76, 82.2, "Event / physiological   (27)", ha="center", va="top",
        fontsize=9.2, fontweight="bold", color=ORNG_E)
cell(63, 71.5, 26, 6.4, ORNG_F, ORNG_E, ORNG_T, "spindle density / amp / var  (×4 EEG)", "", hfs=7.6)
cell(63, 64.3, 26, 6.4, ORNG_F, ORNG_E, ORNG_T, "slow-wave p2p / amplitude  (×4 EEG)", "", hfs=7.6)
cell(63, 57.1, 26, 6.4, ORNG_F, ORNG_E, ORNG_T, "ocular movement / p95  (×2 EOG)", "", hfs=7.6)
cell(63, 49.9, 26, 6.4, ORNG_F, ORNG_E, ORNG_T, "EMG log-RMS / p90 / diff-var", "", hfs=7.6)
ax.text(61, 45.0, r"per-epoch feature vector  $\varphi \in \mathbb{R}^{188}$", ha="center",
        va="center", fontsize=9.6, color=BLACK)

cell(97, 62, 24, 20, WHITE, BLACK, BLACK, "Temporal Context",
     "stack ±3 epochs\n7-epoch window\nedge-padded")

# ================= ROW 2 : ensemble -> HMM -> output =================
group(96, 8, 26, 34, BLACK, "Boosting Ensemble")
cell(99, 33.5, 20, 4.8, WHITE, BLACK, BLACK, "XGBoost  (500, depth 6, GPU)", "", hfs=7.4)
cell(99, 28.1, 20, 4.8, WHITE, BLACK, BLACK, "LightGBM  (600, 63 leaves)", "", hfs=7.4)
cell(99, 22.7, 20, 4.8, WHITE, BLACK, BLACK, "HistGB  (400 iter)", "", hfs=7.4)
cell(99, 17.3, 20, 4.8, WHITE, BLACK, BLACK, "CatBoost  (700, GPU)", "", hfs=7.4)
ax.text(109, 13.4, "class-balanced soft-vote", ha="center", va="center", fontsize=8.2, color=BLACK)
ax.text(109, 10.6, r"$\bar{P}(y \mid \Phi) \in \mathbb{R}^{5}$", ha="center", va="center",
        fontsize=9.6, color=BLACK)

cell(60, 14, 28, 22, WHITE, BLACK, BLACK, "HMM Decoding",
     "transition matrix A (5×5),\nprior π from train hypnograms\nViterbi:\n"
     r"$\hat{s}_{1:T}=\arg\max_s P(s\mid x)$", bfs=8.6)

cell(26, 14, 26, 22, WHITE, BLACK, BLACK, "Hypnogram",
     "5-class sequence\nW  N1  N2  N3  R\n10-fold,\nsubject-independent", bfs=8.6)

# ---- clinical readout branch  (COLOURED = contribution) ----
cell(30, 1.0, 64, 7.5, ORNG_F, ORNG_E, ORNG_T, "Clinical readout  (event features reused)",
     r"spindle asymmetry  AI $=\dfrac{P_{ipsi}-P_{contra}}{P_{ipsi}+P_{contra}}$   vs   stroke severity (NIHSS)",
     hfs=9.4, bfs=8.6)

# ================= arrows + dimension annotations =================
arrow(25, 72, 30, 72); dim(27.5, 76, r"$\mathbb{R}^{7\times3000}$")
arrow(92, 72, 97, 72); dim(94.5, 76, r"$\mathbb{R}^{188}$")
arrow(109, 62, 109, 42); dim(115.5, 52, r"$\mathbb{R}^{1316}$")
arrow(96, 25, 88, 25); dim(92, 29, r"$\bar{P}$")
arrow(60, 25, 52, 25); dim(56, 29, r"$\hat{y}_{1:T}$")
# branch: event lane -> clinical readout (dashed orange = contribution path)
ax.plot([89, 92.5], [53.1, 53.1], color=ORNG_E, ls="--", lw=1.5)
arrow(92.5, 53.1, 92.5, 8.7, color=ORNG_E, ls="--")
dim(92.5, 44, "event\nfeatures\nreused", c=ORNG_E)

# ================= legend =================
ax.add_patch(FancyBboxPatch((3, 40), 3.0, 2.2, boxstyle="square,pad=0",
                            linewidth=1.1, edgecolor=BLACK, facecolor=WHITE))
ax.text(7.2, 41.1, "standard staging pipeline", ha="left", va="center", fontsize=8.8, color=BLACK)
ax.add_patch(FancyBboxPatch((3, 35.5), 3.0, 2.2, boxstyle="square,pad=0",
                            linewidth=1.1, edgecolor=ORNG_E, facecolor=ORNG_F))
ax.text(7.2, 36.6, "physiological features (our contribution)", ha="left", va="center",
        fontsize=8.8, color=ORNG_T)

plt.tight_layout()
plt.savefig(OUT, dpi=185, bbox_inches="tight", facecolor="white")
print("saved", OUT)
