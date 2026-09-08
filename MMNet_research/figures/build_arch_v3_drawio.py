"""Architecture figure v3 -- the dual-prior network, as an editable draw.io file.

Updates the paper's architecture diagram for what the ablations actually
selected. Two things changed since v2 and both are visible here:

  * The EEG branch now carries TWO representations, not one. Expert physiology
    (188 engineered features) and a frozen self-supervised encoder are
    concatenated before the encoder MLP. Each beats the other's absence
    significantly, and their union beats both.
  * The BiLSTM is drawn dashed, because removing it raises staging by 0.0148 and
    costs 0.0215 respiratory AUC. It is a documented trade-off rather than a
    fixed part of the design, and the figure says so instead of hiding it.

One layout model emits both artefacts, so the .drawio and the preview can never
drift apart. draw.io cannot be rendered here, so the preview is how the layout is
checked before the file is opened.

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/build_arch_v3_drawio.py

Outputs: mm_architecture_v3.drawio  +  mm_architecture_v3_preview.png
"""
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# palette matched to the existing figure so v3 sits beside v2 without reworking
GREEN, ORANGE, BLUE, YELLOW, PURPLE, PINK, GREY = (
    "#e6f4ea", "#fdecdd", "#e7effb", "#fdf6da", "#efe7fb", "#fde7e7", "#f4f5f7")
S_GREEN, S_ORANGE, S_BLUE, S_YELLOW, S_PURPLE, S_PINK, S_GREY = (
    "#5a8f66", "#c98a52", "#4a72b0", "#b39a3e", "#7a5fa8", "#b8607a", "#8a9099")

BOX, TXT, ARR = "box", "txt", "arr"
E = []
byid = {}


def box(bid, x, y, w, h, title, detail="", fill="#ffffff", stroke="#333333",
        fs=12, dashed=False):
    e = dict(t=BOX, id=bid, x=x, y=y, w=w, h=h, title=title, detail=detail,
             fill=fill, stroke=stroke, fs=fs, dashed=dashed)
    E.append(e); byid[bid] = e
    return bid


def txt(x, y, w, h, text, fs=11, bold=False, color="#333333", italic=False):
    E.append(dict(t=TXT, x=x, y=y, w=w, h=h, text=text, fs=fs, bold=bold,
                  color=color, italic=italic))


def arr(src, dst, color="#333333", dashed=False, lab="", exit_=None, entry=None):
    E.append(dict(t=ARR, src=src, dst=dst, color=color, dashed=dashed, lab=lab,
                  exit=exit_, entry=entry))


# ---------------------------------------------------------------- main column
txt(40, 18, 900, 30, "MM-Net v3: dual-prior multimodal network", fs=17, bold=True)
txt(40, 46, 900, 24,
    "one 30 s polysomnogram epoch -> sleep stage + respiratory event",
    fs=12, color="#666f78")

box("psg", 60, 92, 300, 66, "PSG epoch",
    "7 ch @ 100 Hz  ·  30 s = 3000 samples", GREY, S_GREY, 12)
box("cardsig", 500, 92, 270, 66, "Cardiorespiratory signals",
    "ECG, flow, thorax, abdomen, SpO2, pulse", ORANGE, S_ORANGE, 12)

# Kept narrow and low: the PSG -> SSL-encoder arrow crosses this band diagonally,
# and a wide caption here is struck through by it. The rationale that used to sit
# under this heading now lives in the "Why two priors" panel instead of being said
# twice.
txt(60, 196, 130, 18, "IMPORTED PRIORS", fs=10, bold=True, color=S_BLUE)

box("feat", 60, 216, 200, 92, "Expert physiology",
    "f_eeg in R^188\nband power, spindle,\nHjorth", GREEN, S_GREEN, 12)
box("fnd", 278, 216, 200, 92, "Frozen SSL encoder",
    "z_fnd in R^200 (LaBraM)\nor R^1400 (CBraMod)", BLUE, S_BLUE, 12)
box("card", 500, 216, 270, 92, "Cardio features",
    "f_car in R^14\nSpO2, effort,\nHRV, airflow", ORANGE, S_ORANGE, 12)

box("cat", 165, 344, 210, 54, "concat  (+)",
    "388-d  /  1588-d", GREY, S_GREY, 12)
box("eegenc", 155, 428, 230, 78, "EEG encoder  phi_eeg",
    "FeatMLP -> e in R^128\n49,792 p  (388-d input)", GREEN, S_GREEN, 12)
box("cardenc", 505, 428, 260, 78, "Cardio encoder  phi_car",
    "FeatMLP -> c in R^64\n5,376 p", ORANGE, S_ORANGE, 12)

box("fuse", 260, 534, 300, 70, "Fusion  (concat)",
    "[e ; c] -> z in R^128  ·  24,704 p", BLUE, S_BLUE, 12)

box("lstm", 240, 640, 340, 74, "BiLSTM  (optional)",
    "2 layers · context L = 20 · h in R^256\n+0.0148 acc when REMOVED, -0.0215 AUC",
    YELLOW, S_YELLOW, 12, dashed=True)

box("stage", 120, 762, 250, 82, "Staging head",
    "softmax(W_s h_t) -> 5 classes\n+ HMM Viterbi decode", PURPLE, S_PURPLE, 12)
box("resp", 430, 762, 280, 82, "Respiratory head",
    "sigma(W_2 [h_t ; c_t])\ndirect cardio bypass", PINK, S_PINK, 12)

txt(60, 872, 700, 24,
    "Joint objective:  L = CE_sqrt-w(y_stg) + lambda · BCE_pw(y_apn),  lambda = 1",
    fs=12)
txt(60, 900, 700, 22,
    "published 0.7275 acc / 0.6536 mF1   ->   v3 0.7485 / 0.6620   (+0.021 acc, p < 0.0001)",
    fs=11, color="#3a6b43", bold=True)

arr("psg", "feat"); arr("psg", "fnd")
arr("feat", "cat"); arr("fnd", "cat")
arr("cat", "eegenc")
arr("cardsig", "card"); arr("card", "cardenc")
arr("eegenc", "fuse"); arr("cardenc", "fuse")
arr("fuse", "lstm")
arr("lstm", "stage"); arr("lstm", "resp")
arr("cardenc", "resp", color=S_ORANGE, dashed=True, lab="direct c_t")

# ------------------------------------------------------------- detail panels
box("p1", 830, 92, 380, 300, "Feature encoder (FeatMLP)",
    "e = GELU(LN(W2 · GELU(LN(W1 f))))\n\n"
    "Linear W1  ->  LayerNorm  ->  GELU\n"
    "Dropout(0.3)\n"
    "Linear W2  ->  LayerNorm  ->  GELU\n\n"
    "EEG: 388 -> 128    Cardio: 14 -> 64", GREY, S_GREY, 12)

box("p2", 830, 416, 380, 320, "Frozen self-supervised encoder",
    "raw 30 s epoch, 100 Hz\n"
    "  resample -> 200 Hz\n"
    "  patch into 1 s tokens\n"
    "  frozen transformer (4.9-5.8 M p)\n"
    "  mean-pool over patches\n\n"
    "CBraMod  7 ch -> R^1400\n"
    "LaBraM   4 ch -> R^200   (10-20 names)\n\n"
    "no gradient; embeddings cached once", BLUE, S_BLUE, 12)

box("p3", 830, 760, 380, 162, "Why two priors",
    "99 patients is too few to LEARN a\n"
    "representation, enough to USE one.\n\n"
    "+ SSL vs features alone: +0.0081 acc (p=0.003)\n"
    "+ SSL vs random init:   +0.0203 acc (p=0.0001)\n"
    "a third SSL stream adds nothing (n.s.)", GREEN, S_GREEN, 12)

# ------------------------------------------------------------------- drawio
def esc(s):
    return html.escape(s).replace("\n", "&#10;")


cells = []
for e in E:
    if e["t"] == BOX:
        label = e["title"] + ("\n" + e["detail"] if e["detail"] else "")
        st = ("rounded=1;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;fontSize=%d;"
              "align=center;verticalAlign=top;spacingTop=6;arcSize=8;%s"
              % (e["fill"], e["stroke"], e["fs"], "dashed=1;" if e["dashed"] else ""))
        cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">'
                     '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
                     % (e["id"], esc(label), st, e["x"], e["y"], e["w"], e["h"]))
    elif e["t"] == TXT:
        st = ("text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
              "fontSize=%d;fontColor=%s;%s%s"
              % (e["fs"], e["color"], "fontStyle=1;" if e["bold"] else "",
                 "fontStyle=2;" if e.get("italic") else ""))
        cells.append('<mxCell id="t%d" value="%s" style="%s" vertex="1" parent="1">'
                     '<mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
                     % (len(cells), esc(e["text"]), st, e["x"], e["y"], e["w"], e["h"]))
for i, e in enumerate(E):
    if e["t"] == ARR:
        st = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=%s;%s"
              "endArrow=block;endFill=1;fontSize=10;"
              % (e["color"], "dashed=1;" if e["dashed"] else ""))
        cells.append('<mxCell id="e%d" value="%s" style="%s" edge="1" parent="1" '
                     'source="%s" target="%s"><mxGeometry relative="1" as="geometry"/></mxCell>'
                     % (i, esc(e["lab"]), st, e["src"], e["dst"]))

xml = ('<mxfile host="app.diagrams.net"><diagram name="MM-Net v3 architecture" id="mmnetv3">'
       '<mxGraphModel dx="1500" dy="1100" grid="1" gridSize="10" guides="1" page="1" '
       'pageWidth="1280" pageHeight="960" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
dst = os.path.join(HERE, "mm_architecture_v3.drawio")
open(dst, "w", encoding="utf-8").write(xml)

# ------------------------------------------------------- preview, same coords
W, H = 1260, 950
fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis(); ax.axis("off")


def edge_point(b, other):
    """Anchor on the box border facing the other box, so arrows do not cross text."""
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    ox, oy = other["x"] + other["w"] / 2, other["y"] + other["h"] / 2
    if abs(oy - cy) >= abs(ox - cx):
        return (cx, b["y"] + b["h"] if oy > cy else b["y"])
    return (b["x"] + b["w"] if ox > cx else b["x"], cy)


for e in E:
    if e["t"] == BOX:
        ax.add_patch(mpatches.FancyBboxPatch(
            (e["x"], e["y"]), e["w"], e["h"],
            boxstyle="round,pad=0,rounding_size=8", linewidth=1.2,
            edgecolor=e["stroke"], facecolor=e["fill"],
            linestyle="--" if e["dashed"] else "-", zorder=2))
        ax.text(e["x"] + e["w"] / 2, e["y"] + 9, e["title"], ha="center", va="top",
                fontsize=e["fs"] * 0.78, fontweight="bold", color=e["stroke"], zorder=3)
        if e["detail"]:
            ax.text(e["x"] + e["w"] / 2, e["y"] + 9 + e["fs"] * 1.35, e["detail"],
                    ha="center", va="top", fontsize=e["fs"] * 0.66, color="#3a4149",
                    zorder=3, linespacing=1.5)
    elif e["t"] == TXT:
        ax.text(e["x"], e["y"] + e["h"] / 2, e["text"], ha="left", va="center",
                fontsize=e["fs"] * 0.82, fontweight="bold" if e["bold"] else "normal",
                style="italic" if e.get("italic") else "normal",
                color=e["color"], zorder=3, linespacing=1.4)

for e in E:
    if e["t"] == ARR:
        a, b = byid[e["src"]], byid[e["dst"]]
        p0, p1 = edge_point(a, b), edge_point(b, a)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                     color=e["color"], lw=1.2, zorder=1,
                                     linestyle="--" if e["dashed"] else "-",
                                     shrinkA=1, shrinkB=3,
                                     connectionstyle="arc3,rad=0.0"))
        if e["lab"]:
            ax.text((p0[0] + p1[0]) / 2 + 8, (p0[1] + p1[1]) / 2, e["lab"],
                    fontsize=8, color=e["color"], style="italic", zorder=4)

prev = os.path.join(HERE, "mm_architecture_v3_preview.png")
fig.savefig(prev, dpi=120, bbox_inches="tight", facecolor="white")
print("wrote %s" % dst)
print("wrote %s  (%d boxes, %d arrows)"
      % (prev, sum(1 for e in E if e["t"] == BOX), sum(1 for e in E if e["t"] == ARR)))
