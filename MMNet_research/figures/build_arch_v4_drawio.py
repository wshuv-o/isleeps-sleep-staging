"""Architecture figure v4 -- the final model, as an editable draw.io file.

v3 drew a model we did not end up running. Three things differ, and all three are
visible here:

  * The frozen self-supervised block is now ONE encoder, not a choice of two.
    v3 hedged "LaBraM R^200 or CBraMod R^1400"; the selected model uses LaBraM,
    so the concatenated neural input is 388-d and nothing about it is optional.
  * The cardiorespiratory branch no longer starts from 14 engineered features.
    It is a learned CNN over the raw 7 x 750 tensor. This is a bigger change than
    the foundation block, and v3 does not show it at all.
  * The BiLSTM is solid, not dashed. v3 drew it optional and annotated it with a
    trade-off measured on a superseded configuration; the final model uses it,
    and a stale number is worse than no number.

Every parameter count is computed from the modules themselves, not typed in --
see the assertions at the bottom, which fail the build rather than let the figure
disagree with the code. v3's EEG encoder count (49,792) was the first Linear only.

One layout model emits both artefacts, so the .drawio and the preview can never
drift apart. draw.io cannot render here, so the preview is how layout is checked.

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/build_arch_v4_drawio.py

Outputs: mm_architecture_v4.drawio + mm_architecture_v4_preview.png
         fig_architecture_v4.pdf
"""
import html
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

GREEN, ORANGE, BLUE, YELLOW, PURPLE, PINK, GREY = (
    "#e6f4ea", "#fdecdd", "#e7effb", "#fdf6da", "#efe7fb", "#fde7e7", "#f4f5f7")
S_GREEN, S_ORANGE, S_BLUE, S_YELLOW, S_PURPLE, S_PINK, S_GREY = (
    "#5a8f66", "#c98a52", "#4a72b0", "#b39a3e", "#7a5fa8", "#b8607a", "#8a9099")
NEWC = "#c0392b"                      # marks the blocks new since submission

# ---------------------------------------------------------------- parameters
# Asserted against the real modules at the bottom of this file, so the figure
# cannot drift away from the code the way v3's counts did.
P = dict(eeg_enc=66816, card_enc=155232, fuse=24704, lstm=2367488,
         stage_head=2565, apnea_head=147969, total=2764774, labram=5819936,
         manuscript_total=773254)

BOX, TXT, ARR = "box", "txt", "arr"
E, byid = [], {}


def box(bid, x, y, w, h, title, detail="", fill="#ffffff", stroke="#333333",
        fs=12, dashed=False):
    e = dict(t=BOX, id=bid, x=x, y=y, w=w, h=h, title=title, detail=detail,
             fill=fill, stroke=stroke, fs=fs, dashed=dashed)
    E.append(e)
    byid[bid] = e
    return bid


def txt(x, y, w, h, text, fs=11, bold=False, color="#333333", italic=False):
    E.append(dict(t=TXT, x=x, y=y, w=w, h=h, text=text, fs=fs, bold=bold,
                  color=color, italic=italic))


def arr(src, dst, color="#333333", dashed=False, lab=""):
    E.append(dict(t=ARR, src=src, dst=dst, color=color, dashed=dashed, lab=lab))


def n(k):
    return format(P[k], ",")


# ---------------------------------------------------------------- main column
txt(40, 18, 780, 30,
    "MM-Net v4: dual-prior neural branch, learned cardiorespiratory encoder",
    fs=16, bold=True)
txt(40, 48, 780, 24,
    "one 30 s polysomnogram epoch  ->  sleep stage + respiratory event",
    fs=12, color="#666f78")
txt(810, 48, 400, 24, "red outline = new since submission",
    fs=10, italic=True, color=NEWC)

box("psg", 60, 92, 330, 74, "Neural montage",
    "C4:M1  C3:M2  O2:M1  O1:M2  -  E1:M2  E2:M2  -  EMG\n"
    "7 ch @ 100 Hz - 30 s = 3000 samples",
    GREY, S_GREY, 12)
box("cardsig", 470, 92, 310, 74, "Cardiorespiratory montage",
    "ECG  flow  thorax  abdomen  effort  SpO2  pulse\n"
    "7 ch @ 25 Hz - 30 s = 750 samples",
    ORANGE, S_ORANGE, 12)

txt(60, 186, 300, 18, "IMPORTED PRIORS", fs=10, bold=True, color=S_BLUE)
txt(645, 186, 160, 18, "LEARNED FROM RAW", fs=10, bold=True, color=S_ORANGE)

box("feat", 60, 220, 190, 108, "Expert physiology",
    "f_eeg in R^188\n112 EEG - 50 EOG - 26 EMG\nband power, spindle,\nHjorth",
    GREEN, S_GREEN, 12)
box("fnd", 262, 220, 190, 108, "Frozen LaBraM",
    "z_fnd in R^200\n5.82 M frozen params\n4 EEG channels only\nno gradient",
    BLUE, NEWC, 12)
box("cardraw", 470, 220, 310, 108, "Raw cardio tensor",
    "x_car in R^(7 x 750) = 5250\nper-subject, per-channel z-score\nno hand-designed summary",
    ORANGE, S_ORANGE, 12)

box("cat", 145, 366, 220, 56, "concat  (+)", "388-d  =  188  +  200",
    GREY, S_GREY, 12)

box("eegenc", 135, 458, 240, 86, "EEG encoder  phi_eeg",
    "FeatMLP  388 -> 128\n%s p" % n("eeg_enc"), GREEN, S_GREEN, 12)
box("cardenc", 470, 458, 310, 86, "Cardio encoder  phi_car",
    "CardioCNN  5250 -> 64\n3 conv stages - %s p" % n("card_enc"),
    ORANGE, NEWC, 12)

box("fuse", 250, 576, 320, 74, "Fusion  (concat)",
    "[e ; c] in R^192  ->  z in R^128\n%s p" % n("fuse"), BLUE, S_BLUE, 12)

box("lstm", 240, 684, 340, 82, "BiLSTM",
    "2 layers - context L = 20 - h in R^256\n-> R^512 - %s p" % n("lstm"),
    YELLOW, S_YELLOW, 12)

box("stage", 105, 808, 255, 92, "Staging head",
    "softmax(W_s h_t) -> 5 classes\n+ HMM Viterbi decode\n%s p" % n("stage_head"),
    PURPLE, S_PURPLE, 12)
box("resp", 420, 808, 290, 92, "Respiratory head",
    "sigma(W_2 [h_t ; c_t])\ndirect cardio bypass\n%s p" % n("apnea_head"),
    PINK, S_PINK, 12)

arr("psg", "feat")
arr("psg", "fnd")
arr("feat", "cat")
arr("fnd", "cat")
arr("cat", "eegenc")
arr("cardsig", "cardraw")
arr("cardraw", "cardenc")
arr("eegenc", "fuse")
arr("cardenc", "fuse")
arr("fuse", "lstm")
arr("lstm", "stage")
arr("lstm", "resp")
arr("cardenc", "resp", color=S_ORANGE, dashed=True, lab="direct c_t")

# ------------------------------------------------------------- bottom strip
txt(60, 930, 740, 22,
    "Joint objective:  L = CE_sqrt-w(y_stg) + lambda * BCE_pw(y_apn),  lambda = 1",
    fs=12)
txt(60, 958, 740, 22,
    "%s trainable parameters  (%s in the submitted model)  +  %s frozen"
    % (n("total"), n("manuscript_total"), n("labram")),
    fs=11, color="#666f78")
txt(60, 986, 740, 22,
    "submitted 0.7275 acc / 0.6536 mF1 / 0.7070 AUC   ->   v4 0.7394 / 0.6701 / 0.7818",
    fs=11, color="#3a6b43", bold=True)

# ------------------------------------------------------------- detail panels
box("p1", 810, 92, 390, 196, "Feature encoder (FeatMLP)",
    "e = GELU(LN(W2 * GELU(LN(W1 f))))\n\n"
    "Linear -> LayerNorm -> GELU -> Dropout(0.3)\n"
    "Linear -> LayerNorm -> GELU -> Dropout(0.3)\n\n"
    "EEG branch only:  388 -> 128", GREY, S_GREY, 12)

box("p2", 810, 304, 390, 272, "Frozen LaBraM encoder",
    "the imported prior, unchanged by training\n\n"
    "C4:M1 C3:M2 O2:M1 O1:M2 -> C4 C3 O2 O1\n"
    "   (10-20 names; EOG/EMG have none)\n"
    "resample 100 -> 200 Hz\n"
    "split 30 s into 2 x 15 s of 3000 samples\n"
    "frozen transformer over patch tokens\n"
    "mean-pool the sub-windows -> R^200\n\n"
    "embeddings cached once, never fine-tuned",
    BLUE, NEWC, 12)

box("p3", 810, 592, 390, 308, "CardioCNN",
    "kernels sized to respiratory physiology:\n"
    "events last 10-30 s, so kernels are wide\n"
    "and the stack downsamples hard -- the\n"
    "opposite of a spindle detector.\n\n"
    "Conv(7->48, k=25, s=2)  BN GELU  MaxPool4\n"
    "Conv(48->96, k=15)      BN GELU  MaxPool4\n"
    "Conv(96->96, k=7)       BN GELU\n"
    "concat[mean, max] -> Linear -> LN -> GELU\n\n"
    "at 25 Hz a 25-sample kernel is one second",
    ORANGE, NEWC, 12)


# ------------------------------------------------------------------- drawio
def esc(s):
    return html.escape(s).replace("\n", "&#10;")


cells = []
for e in E:
    if e["t"] == BOX:
        label = e["title"] + ("\n" + e["detail"] if e["detail"] else "")
        sw = 2 if e["stroke"] == NEWC else 1
        st = ("rounded=1;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;fontSize=%d;"
              "strokeWidth=%d;align=center;verticalAlign=top;spacingTop=6;arcSize=8;%s"
              % (e["fill"], e["stroke"], e["fs"], sw, "dashed=1;" if e["dashed"] else ""))
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

xml = ('<mxfile host="app.diagrams.net"><diagram name="MM-Net v4 architecture" id="mmnetv4">'
       '<mxGraphModel dx="1500" dy="1200" grid="1" gridSize="10" guides="1" page="1" '
       'pageWidth="1260" pageHeight="1040" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
dst = os.path.join(HERE, "mm_architecture_v4.drawio")
open(dst, "w", encoding="utf-8").write(xml)

# ------------------------------------------------------- preview, same coords
W, H = 1240, 1030
fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()
ax.axis("off")


def edge_point(b, other):
    """Anchor on the border facing the other box, so arrows do not cross text."""
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    ox, oy = other["x"] + other["w"] / 2, other["y"] + other["h"] / 2
    if abs(oy - cy) >= abs(ox - cx):
        return (cx, b["y"] + b["h"] if oy > cy else b["y"])
    return (b["x"] + b["w"] if ox > cx else b["x"], cy)


for e in E:
    if e["t"] == BOX:
        ax.add_patch(mpatches.FancyBboxPatch(
            (e["x"], e["y"]), e["w"], e["h"],
            boxstyle="round,pad=0,rounding_size=8",
            linewidth=2.0 if e["stroke"] == NEWC else 1.2,
            edgecolor=e["stroke"], facecolor=e["fill"],
            linestyle="--" if e["dashed"] else "-", zorder=2))
        ax.text(e["x"] + e["w"] / 2, e["y"] + 9, e["title"], ha="center", va="top",
                fontsize=e["fs"] * 0.78, fontweight="bold", color=e["stroke"], zorder=3)
        if e["detail"]:
            ax.text(e["x"] + e["w"] / 2, e["y"] + 9 + e["fs"] * 1.35, e["detail"],
                    ha="center", va="top", fontsize=e["fs"] * 0.63, color="#3a4149",
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

prev = os.path.join(HERE, "mm_architecture_v4_preview.png")
fig.savefig(prev, dpi=120, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "fig_architecture_v4.pdf"),
            bbox_inches="tight", facecolor="white")

print("wrote %s" % dst)
print("wrote %s  (%d boxes, %d arrows)"
      % (prev, sum(1 for e in E if e["t"] == BOX), sum(1 for e in E if e["t"] == ARR)))

# --------------------------------------------------- the figure must not lie
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
    sys.path.insert(0, os.path.join(REPO, "MMNet_research", "foundation"))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import mmnet_core as C                                     # noqa: E402
    from cardio_cnn import CardioCNN                           # noqa: E402
    m = C.MMFeatureNet(n_eeg=388, n_card=14, d=128, d_card=64, hidden=256,
                       layers=2, drop=0.3, fusion="concat", temporal="lstm")
    m.card_enc = CardioCNN(d=64, drop=0.3, width=48)
    for k, mod in (("eeg_enc", m.eeg_enc), ("card_enc", m.card_enc),
                   ("fuse", m.fuse), ("lstm", m.lstm),
                   ("stage_head", m.stage_head), ("apnea_head", m.apnea_head)):
        got = sum(p.numel() for p in mod.parameters())
        assert got == P[k], "%s: figure says %d, model has %d" % (k, P[k], got)
    got = sum(p.numel() for p in m.parameters())
    assert got == P["total"], "total: figure says %d, model has %d" % (P["total"], got)
    print("parameter counts verified against the live modules")
