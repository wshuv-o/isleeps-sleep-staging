"""Graphical abstract for the BSPC submission (Elsevier: >= 1328 x 531 px, readable at 13 x 5 cm).

Every number is a headline value of the manuscript (main.tex): iSLEEPS three-seed MM-Net,
ISRUC-I ten-fold cross-validation, and the iSLEEPS-trained model applied without retraining.

    python make_graphical_abstract.py   ->  submission/bspc_cas/graphical_abstract.{pdf,png,tif}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "submission", "bspc_cas", "graphical_abstract")

INK, MUTED = "#1f2933", "#52606d"
NEURAL, CARDIO, FUSE, OUTC = "#dbe7f6", "#f7e1d2", "#e4dcf2", "#dcefe0"
NEURAL_E, CARDIO_E, FUSE_E, OUT_E = "#3b6fb6", "#c0652b", "#6b4fa3", "#2e7d4f"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(13.28, 5.31))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 133); ax.set_ylim(0, 53); ax.axis("off")


def box(x, y, w, h, fc, ec, title, lines=(), tsize=12.5, lsize=10.5, gap=3.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.4 - i * gap, l, ha="center", va="top", fontsize=lsize,
                color=MUTED)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=INK, shrinkA=0, shrinkB=0))


ax.text(66.5, 51.2, "MM-Net: joint sleep staging and respiratory-event detection after ischemic stroke",
        ha="center", va="top", fontsize=15, fontweight="bold", color=INK)

# inputs
box(2, 26, 25, 20, NEURAL, NEURAL_E, "Neural PSG", ["4 EEG + 2 EOG + chin EMG", "30-s epochs"])
box(2, 5, 25, 18, CARDIO, CARDIO_E, "Cardiorespiratory PSG",
    ["airflow, nasal pressure,", "effort, SpO₂, pulse, ECG", "8 raw channels"])

# streams
box(33, 26, 31, 20, NEURAL, NEURAL_E, "Sleep-adapted EEG encoder",
    ["LaBraM fine-tuned per fold", "cross-fitted (K = 3) +", "ridge-aligned embeddings",
     "+ 188 physiological descriptors"], lsize=10, gap=3.3)
box(33, 5, 31, 18, CARDIO, CARDIO_E, "Learned cardiorespiratory",
    ["convolutional encoder", "over raw waveforms", "(CardioCNN)"])
arrow(27.6, 36, 32.4, 36); arrow(27.6, 14, 32.4, 14)

# fusion + temporal
box(70, 12, 24, 26, FUSE, FUSE_E, "Shared temporal model",
    ["cross-attention fusion", "BiLSTM over", "20 epochs (10 min)", "two task heads"])
arrow(64.6, 36, 69.4, 31); arrow(64.6, 14, 69.4, 19)

# outputs
box(100, 33.5, 31, 11.5, OUTC, OUT_E, "Sleep stages", ["76.3% accuracy, κ = 0.656"])
box(100, 19.5, 31, 11.5, OUTC, OUT_E, "Respiratory events", ["AUC 0.825"])
box(100, 5.5, 31, 11.5, OUTC, OUT_E, "Apnea–hypopnea index", ["Spearman ρ = 0.765"])
arrow(94.6, 31, 99.4, 39); arrow(94.6, 25, 99.4, 25); arrow(94.6, 19, 99.4, 11)

ax.text(66.5, 1.4, "iSLEEPS, 99 patients with subacute ischemic stroke, ten-fold patient-independent "
        "evaluation  •  ISRUC-I: 80.3% accuracy, AUC 0.889  •  without retraining: "
        "AUC 0.833 (ISRUC-I), 0.841 (UCD)", ha="center", va="bottom", fontsize=9.6, color=MUTED)

for ext, kw in (("pdf", {}), ("png", {"dpi": 200}), ("tif", {"dpi": 300})):
    fig.savefig("%s.%s" % (OUT, ext), facecolor="white", **kw)
print("wrote", os.path.normpath(OUT) + ".{pdf,png,tif}")
