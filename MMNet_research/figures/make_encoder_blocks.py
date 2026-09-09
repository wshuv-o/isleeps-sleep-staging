"""Isometric layer-block diagrams of the two encoders, in the PlotNeuralNet idiom.

Why not the earlier make_mlp_3d.py: it used a real matplotlib 3-D axis, which
spends most of the canvas on empty projection volume, gives no control over where
the tick-free labels land, and rendered the blocks small with the layer names
overlapping each other. PlotNeuralNet and visualkeras are not 3-D either --- they
draw a 2-D isometric projection of a cuboid --- and doing the same here makes the
geometry exact, the canvas tight, and label collisions impossible to get by
accident.

Each block is three polygons: the front face, the top face sheared right, and the
right face sheared up. What the two axes mean differs by encoder and is stated in
each call:

    FeatMLP     front height  ~ log(units)         depth fixed
    CardioCNN   front height  ~ log(temporal len)  depth ~ log(channels)

The CNN case is the informative one: it shows the trade the stack actually makes,
spending time resolution to buy channels (750 -> 23 samples while 7 -> 96
channels). Both are log-scaled because those two ranges do not share a linear
axis without one collapsing to a line.

Outputs, transparent PNG, cropped tight:
    eeg_encoder_iso.png      cardio_encoder_iso.png

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/make_encoder_blocks.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
SKEW_X, SKEW_Y = 0.42, 0.30          # isometric offset per unit of depth


def _shade(hexc, f):
    """f > 0 lightens toward white, f < 0 darkens."""
    r, g, b = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
    if f >= 0:
        rgb = [c + (255 - c) * f for c in (r, g, b)]
    else:
        rgb = [c * (1 + f) for c in (r, g, b)]
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def cuboid(ax, x, h, d, face, edge="#33383d", lw=0.9):
    """Isometric block: front face at x with height h, depth d receding up-right."""
    w = 0.34                                     # front-face width, constant
    dx, dy = d * SKEW_X, d * SKEW_Y
    front = [(x, 0), (x + w, 0), (x + w, h), (x, h)]
    top = [(x, h), (x + w, h), (x + w + dx, h + dy), (x + dx, h + dy)]
    side = [(x + w, 0), (x + w + dx, dy), (x + w + dx, h + dy), (x + w, h)]
    for pts, col in ((top, _shade(face, 0.28)), (side, _shade(face, -0.18)),
                     (front, face)):
        ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor=edge,
                             linewidth=lw, joinstyle="round"))
    return w, dx, dy


def draw(stages, fname, base, pitch=2.05, hmax=3.0, dmax=1.5):
    """stages: (label, shape_text, height_frac, depth_frac, kind).

    pitch has to exceed the widest label or the names below collide; keep the
    labels to one short line each and put the detail in the surrounding figure.
    """
    colmap = {"in": "#c6cad0", "main": base, "light": _shade(base, 0.42),
              "out": _shade(base, -0.22)}
    fig, ax = plt.subplots(figsize=(0.78 * pitch * len(stages) + 0.8, 2.0))
    geom = []
    for k, (lab, shape, hf, df, kind) in enumerate(stages):
        x = k * pitch
        h = 0.55 + hmax * hf
        d = 0.45 + dmax * df
        w, dx, dy = cuboid(ax, x, h, d, colmap[kind])
        geom.append((x, w, dx, h, dy))
        ax.text(x + w / 2 + dx / 2, h + dy + 0.14, shape, ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color="#15181b")
        ax.text(x + w / 2 + dx / 2, -0.22, lab, ha="center", va="top", fontsize=7.4,
                color="#3a4149", linespacing=1.25)
    # connectors: from the right-most extent of block k to the left face of k+1
    for k in range(len(stages) - 1):
        x0, w0, dx0, _, _ = geom[k]
        x1 = geom[k + 1][0]
        y = 0.34
        ax.annotate("", xy=(x1 - 0.04, y), xytext=(x0 + w0 + dx0 + 0.04, y),
                    arrowprops=dict(arrowstyle="-|>", color="#5a6570", lw=1.0,
                                    shrinkA=0, shrinkB=0))
    x_end = geom[-1][0] + geom[-1][1] + geom[-1][2]
    ax.set_xlim(-0.30, x_end + 0.35)
    ax.set_ylim(-1.15, hmax + dmax * SKEW_Y + 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(HERE, fname), dpi=220, bbox_inches="tight",
                pad_inches=0.02, transparent=True)
    plt.close(fig)
    print("wrote", fname)


L750, L192 = np.log10(750.0), np.log10(192.0)

# --- EEG encoder: FeatMLP over [f_eeg ; z_fnd], 388 -> 128 -----------------
u = lambda n: np.log10(n) / np.log10(388.0)
draw([("input", "388", u(388), 0.30, "in"),
      ("Linear W$_1$", "128", u(128), 0.30, "main"),
      ("LN+GELU", "128", u(128), 0.30, "light"),
      ("Linear W$_2$", "128", u(128), 0.30, "main"),
      ("LN+GELU", "128", u(128), 0.30, "light"),
      ("e", "128", u(128), 0.30, "out")],
     "eeg_encoder_iso.png", "#4c9a63")

# --- Cardio encoder: CardioCNN over the raw 7 x 750 tensor -> 64 ----------
# lengths follow the real module: 750 -(stride 2)-> 375 -(pool 4)-> 93 -(pool 4)-> 23
t = lambda n: np.log10(n) / L750
c = lambda n: np.log10(n) / L192
draw([("raw", "7×750", t(750), c(7), "in"),
      ("Conv 25 /2", "48×93", t(93), c(48), "main"),
      ("Conv 15", "96×23", t(23), c(96), "main"),
      ("Conv 7", "96×23", t(23), c(96), "main"),
      ("mean⊕max", "192", t(1), c(192), "light"),
      ("Linear", "64", t(1), c(64), "out")],
     "cardio_encoder_iso.png", "#d08a3e")
print("done")
