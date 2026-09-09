"""Encoder diagrams in the feature-map-sheet idiom used across the CNN literature.

The visual grammar, copied from the papers this is meant to sit beside rather
than invented here:

  * a layer is a THIN TALL SHEET seen at a slight angle, not a chunky cube --
    it stands for a feature map, so its face is the map and its width is
    nothing. An earlier attempt drew isometric cubes and read as a stack of
    boxes rather than as a network.
  * a convolution is a BANK of such sheets, drawn as several offset planes with
    the leading edges tinted, which is what gives those figures their
    characteristic rainbow spine. One sheet would understate that a conv layer
    produces many maps.
  * operation type is carried by COLOUR with a legend, not by text, so the eye
    can follow conv -> norm -> pool without reading.
  * shape sits above each block, operation name below. Never both on one line.

Height encodes the temporal length and the number of stacked planes encodes
channels (both log-compressed), so the convolutional stack shows the trade it
makes: time resolution spent to buy channels, 750 -> 23 samples against
7 -> 96 channels.

Outputs, transparent PNG cropped tight:
    eeg_encoder_iso.png      cardio_encoder_iso.png

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/make_encoder_blocks.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

DX, DY = 0.30, 0.20              # depth offset of the sheet, per unit depth
EDGE = "#2b2f33"

# operation palette, legend-able
C_IN = "#c8ccd2"
C_CONV = "#2f4fd0"
C_NORM = "#eceff2"
C_POOL = "#e34ee0"
C_FC = "#3fae5a"
C_OUT = "#f0a03c"
SPINE = ["#d92b2b", "#f0821e", "#f5d020", "#3fae5a"]   # tinted leading edges


def sheet(ax, x, h, w, face, depth=0.55, lw=0.9, edge=EDGE):
    """One sheet: front face, sheared top, sheared right side."""
    dx, dy = depth * DX, depth * DY
    front = [(x, 0), (x + w, 0), (x + w, h), (x, h)]
    top = [(x, h), (x + w, h), (x + w + dx, h + dy), (x + dx, h + dy)]
    side = [(x + w, 0), (x + w + dx, dy), (x + w + dx, h + dy), (x + w, h)]
    for pts, col in ((top, _sh(face, 0.30)), (side, _sh(face, -0.22)), (front, face)):
        ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor=edge,
                             linewidth=lw, joinstyle="round", zorder=3))
    return dx, dy


def bank(ax, x, h, face, n_planes=4, w=0.15, step=0.085, depth=0.55):
    """A convolution: several offset sheets with tinted leading edges."""
    total = 0.0
    for k in range(n_planes):
        col = SPINE[k % len(SPINE)] if k < len(SPINE) and n_planes > 1 else face
        sheet(ax, x + k * step, h, w, col, depth=depth, lw=0.7)
        total = k * step
    dx, dy = sheet(ax, x + total + step, h, w * 2.6, face, depth=depth)
    return total + step + w * 2.6 + dx, dy


def _sh(hexc, f):
    r, g, b = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
    rgb = [c + (255 - c) * f if f >= 0 else c * (1 + f) for c in (r, g, b)]
    return "#%02x%02x%02x" % tuple(int(round(max(0, min(255, c)))) for c in rgb)


def draw(stages, fname, groups=(), pitch=2.25, hmax=3.1, figh=2.45):
    """stages: (kind, shape_label, op_label, height_frac, n_planes)."""
    colmap = {"in": C_IN, "conv": C_CONV, "norm": C_NORM, "pool": C_POOL,
              "fc": C_FC, "out": C_OUT}
    fig, ax = plt.subplots(figsize=(pitch * len(stages) + 1.1, figh))
    xs = []
    for k, (kind, shape, op, hf, npl) in enumerate(stages):
        x = k * pitch
        h = 0.6 + hmax * hf
        if kind == "conv":
            wtot, dy = bank(ax, x, h, colmap[kind], n_planes=npl)
        else:
            w = 0.58 if kind in ("norm", "pool") else 0.42
            dx, dy = sheet(ax, x, h, w, colmap[kind])
            wtot = w + dx
        cx = x + wtot / 2
        xs.append((x, wtot, h + dy))
        ax.text(cx, h + dy + 0.16, shape, ha="center", va="bottom", fontsize=8.0,
                fontweight="bold", color="#15181b", zorder=6)
        ax.text(cx, -0.22, op, ha="center", va="top", fontsize=7.4,
                color="#3a4149", linespacing=1.3, zorder=6)
    # arrows between blocks, at a low common height
    for k in range(len(stages) - 1):
        x0, w0, _ = xs[k]
        x1 = xs[k + 1][0]
        if x1 - (x0 + w0) > 0.06:
            ax.annotate("", xy=(x1 - 0.02, 0.30), xytext=(x0 + w0 + 0.02, 0.30),
                        arrowprops=dict(arrowstyle="-|>", color="#5a6570", lw=1.0,
                                        shrinkA=0, shrinkB=0), zorder=2)
    # dashed stage groupings, as in the reference figures
    for lo, hi, name in groups:
        x0 = xs[lo][0] - 0.16
        x1 = xs[hi][0] + xs[hi][1] + 0.16
        top = max(xs[k][2] for k in range(lo, hi + 1)) + 0.46
        ax.add_patch(FancyBboxPatch((x0, -1.16), x1 - x0, top + 1.16,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    linestyle=(0, (5, 3)), linewidth=0.9,
                                    edgecolor="#8a9099", facecolor="none", zorder=1))
        ax.text((x0 + x1) / 2, top + 0.10, name, ha="center", va="bottom",
                fontsize=8.2, fontweight="bold", color="#4a5158", zorder=6)
    x_end = xs[-1][0] + xs[-1][1]
    ax.set_xlim(-0.34, x_end + 0.34)
    ax.set_ylim(-1.55, hmax + 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(HERE, fname), dpi=230, bbox_inches="tight",
                pad_inches=0.03, transparent=True)
    plt.close(fig)
    print("wrote", fname)


# --- EEG encoder: FeatMLP over [f_eeg ; z_fnd], 388 -> 128 -----------------
u = lambda n: np.log10(n) / np.log10(388.0)
draw([("in", "388", "input\n[f$^{eeg}$; z$^{fnd}$]", u(388), 1),
      ("fc", "128", "Linear W$_1$", u(128), 1),
      ("norm", "128", "LN + GELU", u(128), 1),
      ("fc", "128", "Linear W$_2$", u(128), 1),
      ("norm", "128", "LN + GELU", u(128), 1),
      ("out", "128", "e", u(128), 1)],
     "eeg_encoder_iso.png", groups=[(1, 4, "FeatMLP")])

# --- Cardio encoder: CardioCNN over the raw 7 x 750 tensor -> 64 ----------
# lengths are the real ones: 750 -(stride 2)-> 375 -(pool 4)-> 93 -(pool 4)-> 23
L = np.log10(750.0)
t = lambda n: np.log10(n) / L
draw([("in", "7×750", "raw", t(750), 1),
      ("conv", "48×375", "Conv 25, s2\nBN+GELU", t(375), 4),
      ("pool", "48×93", "Pool 4", t(93), 1),
      ("conv", "96×93", "Conv 15\nBN+GELU", t(93), 5),
      ("pool", "96×23", "Pool 4", t(23), 1),
      ("conv", "96×23", "Conv 7\nBN+GELU", t(23), 5),
      ("pool", "192", "mean⊕max", t(3), 1),
      ("out", "64", "Linear", t(2), 1)],
     "cardio_encoder_iso.png",
     groups=[(1, 2, "stage 1"), (3, 4, "stage 2"), (5, 5, "stage 3")])
print("done")
