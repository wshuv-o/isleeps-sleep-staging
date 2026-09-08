"""fig_architecture -- the dual-prior multimodal network.

Draws the architecture the ablations selected, and labels each stream with what
it contributed, so the figure carries the evidence rather than only the shape.

The claim the diagram makes: on a small pathological cohort, representation must
be IMPORTED rather than learned, and there are two independent import routes --
expert physiological features and self-supervised pretraining -- which encode
different information and therefore fuse profitably.

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/fig_architecture.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
from mmnet_repro import REV                                  # noqa: E402

FIG = os.path.join(REV, "figures")
os.makedirs(FIG, exist_ok=True)

# palette: inputs grey, imported-prior streams blue, learned blocks slate,
# outputs green, the discarded component red
C_IN = "#e8eaed"
C_PRIOR_A = "#bcd4ee"        # expert physiology
C_PRIOR_B = "#a8c8e8"        # self-supervised
C_CARD = "#d6e8d5"
C_BLOCK = "#dfe3e8"
C_OUT = "#bfe0c4"
C_DROP = "#f2d0d0"
EDGE = "#5a6570"


def box(ax, x, y, w, h, text, fc, fontsize=8.2, weight="normal", ec=EDGE, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.0, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#1b2430", zorder=3,
            linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, style="-|>", lw=1.1, color=EDGE, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1.5, shrinkB=2.5, zorder=1,
                                 connectionstyle="arc3,rad=%.2f" % rad))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def main():
    fig, ax = plt.subplots(figsize=(14.6, 6.6))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 6.6); ax.axis("off")

    ax.text(0.05, 6.36, "Dual-prior multimodal network", fontsize=13.5,
            fontweight="bold", color="#1b2430")
    ax.text(0.05, 6.06,
            "one 30 s polysomnogram epoch $\\rightarrow$ sleep stage + respiratory event",
            fontsize=9.2, color="#55606c")

    # ---- inputs -------------------------------------------------------------
    inp = box(ax, 0.05, 4.30, 1.62, 0.95,
              "PSG epoch\n7 ch @ 100 Hz\n30 s = 3000", C_IN, 8.0)
    card_in = box(ax, 0.05, 1.05, 1.62, 0.95,
                  "cardiorespiratory\nECG, flow, thorax,\nabdomen, SpO$_2$, pulse", C_IN, 7.6)

    # ---- the two imported priors -------------------------------------------
    ax.text(2.05, 5.62, "IMPORTED REPRESENTATIONS", fontsize=8.4,
            fontweight="bold", color="#31465e")
    ax.text(2.05, 5.40, "the only two components that improved the model",
            fontsize=7.4, color="#6b7784", style="italic")

    feat = box(ax, 2.05, 4.52, 2.62, 0.80,
               "expert physiology\n188 engineered features", C_PRIOR_A, 8.2, "bold")
    ax.text(2.05, 4.34, "spectral, spindle, slow-wave, EOG, EMG",
            fontsize=6.9, color="#6b7784")

    found = box(ax, 2.05, 3.16, 2.62, 0.80,
                "self-supervised encoder\nCBraMod 1400-d  /  LaBraM 200-d", C_PRIOR_B, 8.2, "bold")
    ax.text(2.05, 2.98, "frozen; pretrained on healthy sleep",
            fontsize=6.9, color="#6b7784")

    arrow(ax, right(inp), left(feat), rad=0.10)
    arrow(ax, right(inp), left(found), rad=-0.08)

    # ---- fusion of the EEG side --------------------------------------------
    cat = box(ax, 5.10, 3.84, 1.15, 1.30, "concat\n1588-d\n/ 388-d", C_BLOCK, 8.0)
    arrow(ax, right(feat), (5.10, 4.72), rad=0.0)
    arrow(ax, right(found), (5.10, 4.26), rad=0.0)

    eeg_enc = box(ax, 6.62, 4.10, 1.30, 0.78, "MLP\n$\\rightarrow$ 128-d", C_BLOCK, 8.2)
    arrow(ax, right(cat), left(eeg_enc))

    card_enc = box(ax, 6.62, 1.14, 1.30, 0.78, "MLP\n$\\rightarrow$ 64-d", C_CARD, 8.2)
    arrow(ax, right(card_in), left(card_enc), rad=0.0)

    fuse = box(ax, 8.30, 2.70, 1.22, 0.86, "fusion\n128-d", C_BLOCK, 8.2)
    arrow(ax, right(eeg_enc), (8.30, 3.34), rad=-0.10)
    arrow(ax, right(card_enc), (8.30, 2.92), rad=0.10)

    # ---- the component the ablation removed, drawn ON the path it occupied ---
    # Placed inline rather than off to one side: the point is that the temporal
    # decoder sat between fusion and the heads, and that deleting it IMPROVED
    # staging, which is why the published architecture is not the best one.
    drop = box(ax, 9.86, 3.92, 1.42, 0.62, "BiLSTM", C_DROP, 8.2, "bold",
               ec="#b46a6a", ls="--")
    ax.plot([9.96, 11.18], [4.44, 4.02], color="#b46a6a", lw=1.3, zorder=4)
    ax.plot([9.96, 11.18], [4.02, 4.44], color="#b46a6a", lw=1.3, zorder=4)
    arrow(ax, (8.91, 3.56), (10.20, 3.92), color="#c79a9a", ls=(0, (3, 2)), lw=1.0, rad=-0.12)
    ax.text(10.57, 3.76, "removed: $+0.015$ accuracy,\nbut $-0.022$ respiratory AUC",
            fontsize=6.9, color="#a05a5a", style="italic", linespacing=1.3,
            ha="center", va="top")

    # ---- heads --------------------------------------------------------------
    stage = box(ax, 11.62, 3.28, 1.42, 0.72, "staging head\n5 classes", C_OUT, 8.2)
    hmm = box(ax, 11.62, 2.36, 1.42, 0.62, "HMM\nViterbi", C_BLOCK, 8.0)
    apnea = box(ax, 11.62, 1.14, 1.42, 0.72, "respiratory head\nevent probability", C_OUT, 8.0)

    arrow(ax, right(fuse), left(stage), rad=-0.10)
    arrow(ax, right(fuse), left(apnea), rad=0.10)
    arrow(ax, (12.33, 3.28), (12.33, 2.98))

    # validated cardiorespiratory bypass
    arrow(ax, (7.92, 1.32), (11.62, 1.38), color="#4f7d52", ls=(0, (4, 2)), lw=1.2, rad=-0.08)
    ax.text(8.30, 0.80, "validated bypass: SpO$_2$ cues reach the head undiminished",
            fontsize=7.0, color="#4f7d52", style="italic")

    # ---- outputs ------------------------------------------------------------
    ax.text(13.12, 2.62, "hypnogram", fontsize=8.6, fontweight="bold", color="#2f5a34")
    ax.text(13.12, 1.44, "AHI burden", fontsize=8.6, fontweight="bold", color="#2f5a34")

    # ---- results strip ------------------------------------------------------
    ax.plot([0.05, 14.4], [0.52, 0.52], color="#c9cfd6", lw=0.9)
    res = ("published  0.7275 acc / 0.6536 mF1        "
           "+ CBraMod  0.7450 / 0.6730        "
           "+ LaBraM  0.7485 / 0.6620   ($+0.021$ acc, $p<0.0001$)")
    ax.text(0.05, 0.24, res, fontsize=8.0, color="#3a4552", family="DejaVu Sans")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig_architecture.%s" % ext), dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("wrote", os.path.join(FIG, "fig_architecture.pdf"))


if __name__ == "__main__":
    main()
