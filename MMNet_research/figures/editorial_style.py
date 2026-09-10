"""Editorial plotting style for the manuscript's figures.

Matplotlib's defaults are built for screens: sans-serif type that clashes with a
LaTeX body, a saturated colour cycle, a full box of spines, and a solid grid that
competes with the data. This makes the figures look like they belong in the paper
they sit in.

  type      serif throughout, Latin Modern Roman first so plot text matches the
            body face, falling back to DejaVu Serif and then Times. Mathtext uses
            Computer Modern for the same reason.
  colour    a six-colour editorial cycle in place of the default. Dark blue leads
            because most figures here have one series that matters.
  frame     top and right spines removed, 0.8 pt axis lines, ticks pointing out.
            Ink is #222222 rather than pure black, which reads softer in print.
  legend    frameless at 9 pt. A box around a legend is a border around nothing.
  grid      dotted, 0.6 pt, alpha 0.7. Present enough to read a value against,
            recessive enough to stay behind the data.

  from editorial_style import apply
  apply()
"""
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

INK = "#222222"

# dark blue, editorial red, green, amber, slate, purple
PALETTE = ["#1f4e79", "#a6212d", "#2e7d32", "#c46e0d", "#5b6770", "#6a3d9a"]

BLUE, RED, GREEN, AMBER, SLATE, PURPLE = PALETTE


def apply():
    matplotlib.rcParams.update({
        # ---- type -------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif", "Times New Roman",
                       "Times"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,

        # ---- colour -----------------------------------------------------
        "axes.prop_cycle": cycler(color=PALETTE),

        # ---- frame ------------------------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # ---- legend -----------------------------------------------------
        "legend.frameon": False,
        "legend.fontsize": 9,

        # ---- grid -------------------------------------------------------
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "grid.color": INK,
        "axes.axisbelow": True,

        # ---- output -----------------------------------------------------
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
    return plt
