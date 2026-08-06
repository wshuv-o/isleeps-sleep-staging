"""3D block visualizations (visualkeras style) of the two FeatMLP encoders in MM-Net,
derived from the real layer structure in model/mm_feature_net.py:
  FeatMLP = [Linear(fin,d), LayerNorm(d), GELU, Dropout, Linear(d,d), LayerNorm(d), GELU, Dropout]
EEG encoder: 188 -> 128   ·   Cardio encoder: 14 -> 64.
Output: eeg_mlp_3d.png, cardio_mlp_3d.png (transparent)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "DejaVu Sans"})

def cuboid(ax, x0, dx, dy, dz, face, edge="#3a3a3a", alpha=0.96):
    """draw an axis-aligned box spanning x0..x0+dx, y -dy/2..dy/2, z 0..dz."""
    x1, y0, y1, z0, z1 = x0 + dx, -dy / 2, dy / 2, 0, dz
    V = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    pc = Poly3DCollection([V[f] for f in F], facecolors=face, edgecolors=edge, linewidths=0.7, alpha=alpha)
    ax.add_collection3d(pc)

def draw_mlp(layers, title, fname, base):
    """layers: list of (label, units, kind). kind in {in,linear,act,out}."""
    fig = plt.figure(figsize=(9.5, 2.9)); ax = fig.add_subplot(111, projection="3d")
    colmap = {"in": "#c9ccd1", "linear": base, "act": _lighten(base, 0.5), "out": _darken(base, 0.15)}
    gap, x = 1.9, 0.0
    maxu = max(u for _, u, _ in layers)
    for lab, units, kind in layers:
        dz = 0.8 + 2.8 * (units / maxu)          # height encodes number of units
        dx = 0.55 if kind in ("linear", "out", "in") else 0.34
        cuboid(ax, x, dx, 1.6, dz, colmap[kind])
        ax.text(x + dx / 2, 0, dz + 0.30, str(units), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#111")           # units on top
        ax.text(x + dx / 2, 0, -0.55, lab, ha="center", va="top",
                fontsize=8.5, color="#333")                              # layer type below
        x += dx + gap
    ax.set_title(title, fontsize=13, fontweight="bold", color=_darken(base, 0.2), pad=0, y=0.98)
    ax.set_xlim(0, x); ax.set_ylim(-2.2, 2.2); ax.set_zlim(-1.2, 4.0)
    ax.set_box_aspect((x, 2.4, 2.6)); ax.view_init(elev=16, azim=-74)
    ax.set_axis_off()
    fig.savefig(os.path.join(HERE, fname), dpi=200, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig); print("wrote", fname)

def _lighten(hexc, f):
    r, g, b = [int(hexc[i:i+2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))
def _darken(hexc, f):
    r, g, b = [int(hexc[i:i+2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(c * (1 - f)) for c in (r, g, b))

# EEG encoder  phi_eeg : 188 -> 128
draw_mlp([("input", 188, "in"), ("Linear", 128, "linear"), ("LN+GELU", 128, "act"),
          ("Linear", 128, "linear"), ("LN+GELU", 128, "act"), ("z_e", 128, "out")],
         "EEG encoder  (phi_eeg, FeatMLP):  188 -> 128", "eeg_mlp_3d.png", "#4c9a63")

# Cardio encoder  phi_car : 14 -> 64
draw_mlp([("input", 14, "in"), ("Linear", 64, "linear"), ("LN+GELU", 64, "act"),
          ("Linear", 64, "linear"), ("LN+GELU", 64, "act"), ("z_c", 64, "out")],
         "Cardio encoder  (phi_car, FeatMLP):  14 -> 64", "cardio_mlp_3d.png", "#d08a3e")
print("done")
