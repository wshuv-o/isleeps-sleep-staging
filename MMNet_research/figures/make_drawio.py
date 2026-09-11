"""Emit an editable draw.io (diagrams.net) file of the MM-Net architecture.

Labels use HTML formatting (<br>, <sub>, <sup>, <b>) so draw.io renders line
breaks and proper sub/superscripts.

Layout rationale. An earlier revision carried four side panels -- FeatMLP,
CardioCNN, cross-modal fusion, and the temporal decoder. A modular side panel
earns its place only when the block it details is INSTANTIATED MORE THAN ONCE
and the panel saves the reader from reading the same stack twice. None of these
repeats: FeatMLP runs on the neural stream alone, CardioCNN on the
cardiorespiratory stream alone, and there is one temporal decoder. So their
internals are expanded inline as rows of draw.io cube shapes, centred on a
shared horizontal axis so the data path reads as one line, which lays out
horizontally and keeps the figure landscape.

Cross-modal fusion keeps a side panel, and is the only one. Attention does not
inline compactly -- four heads, a residual path and a feed-forward block need
their own two dimensions -- and drawing it in the main column would stretch the
figure past the width of a printed page.

Export (draw.io desktop CLI; --crop trims to the drawing):

    draw.io --no-sandbox --disable-gpu --export --format pdf --crop \\
        --output fig_architecture.pdf mm_architecture.drawio

If any path contains a Windows 8.3 short name (ESMEAB~1), the '~' percent-encodes
and Electron refuses to load its own export3.html with ERR_BLOCKED_BY_CLIENT.
Invoke through the long path.
"""
import base64
import html
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mm_architecture.drawio")

# palette
EEG = ("#D8ECD9", "#4E8A52"); CAR = ("#F7DFC9", "#C4763A")
FUS = ("#D4E2F5", "#3E6FA8"); LSTM = ("#F7ECC6", "#C2A02F")
STG = ("#E5D9F2", "#7A57A8"); RSP = ("#F7D8E6", "#B0517A")
PANEL = ("#F4F4F5", "#CACACA"); WHITE = ("#FFFFFF", "#333333")

cells = []


def _cell(i, label, x, y, w, h, style):
    cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def node(i, label, x, y, w, h, pal, extra=""):
    _cell(i, label, x, y, w, h,
          f"rounded=1;whiteSpace=wrap;html=1;fillColor={pal[0]};strokeColor={pal[1]};"
          f"fontSize=12;spacing=4;shadow=1;{extra}")


def panel(i, label, x, y, w, h):
    _cell(i, label, x, y, w, h,
          f"rounded=1;whiteSpace=wrap;html=1;fillColor={PANEL[0]};strokeColor={PANEL[1]};"
          f"verticalAlign=top;fontStyle=1;fontSize=13;spacingTop=4;")


def small(i, label, x, y, w, h, pal, extra=""):
    _cell(i, label, x, y, w, h,
          f"rounded=1;whiteSpace=wrap;html=1;fillColor={pal[0]};strokeColor={pal[1]};"
          f"fontSize=10;spacing=2;{extra}")


def ell(i, label, x, y, w, h, pal=WHITE):
    _cell(i, label, x, y, w, h,
          f"ellipse;whiteSpace=wrap;html=1;fillColor={pal[0]};strokeColor={pal[1]};"
          f"fontSize=13;fontStyle=1;")


def txt(i, label, x, y, w, h, italic=False, size=11, align="center"):
    _cell(i, label, x, y, w, h,
          f"text;html=1;align={align};verticalAlign=middle;fontSize={size};"
          f"{'fontStyle=2;' if italic else ''}")


def img(i, asset, x, y, w, h):
    """Embed a PNG from arch_assets/ as a base64 data URI.

    The signal thumbnails were originally dropped into draw.io by hand and never
    saved back, so regenerating the file silently lost them. Embedding keeps the
    generator authoritative and the .drawio self-contained.
    """
    d = os.path.join(HERE, "arch_assets", asset)
    b64 = base64.b64encode(open(d, "rb").read()).decode()
    _cell(i, "", x, y, w, h,
          f"shape=image;verticalLabelPosition=bottom;labelBackgroundColor=none;"
          f"imageAspect=0;aspect=fixed;image=data:image/png,{b64};")


def edge(i, s, t, dashed=False, color="#333333", exit_=None, entry=None, w=1.5):
    st = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;"
          f"strokeColor={color};strokeWidth={w};")
    if dashed:
        st += "dashed=1;"
    # Fixed anchors where the router's default is misleading: left to itself it
    # exits the cardio column's LEFT side at the shared centre-line, putting that
    # segment on the neural column's right edge, where it reads as a wire between
    # the two streams.
    if exit_:
        st += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry:
        st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1" source="{s}" target="{t}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')


def stack(prefix, items, x, y, w, pal, h=30, gap=8):
    """A vertical chain of small boxes, wired top to bottom. Returns the ids."""
    ids = []
    for k, lab in enumerate(items):
        cid = f"{prefix}{k}"
        small(cid, lab, x, y + k * (h + gap), w, h, pal)
        ids.append(cid)
    for a, b in zip(ids[:-1], ids[1:]):
        edge(f"e_{a}_{b}", a, b, w=1.1)
    return ids


DASHED_FROZEN = "dashed=1;dashPattern=6 4;"

# ---------------------------------------------------------------- layer blocks
# Drawn with draw.io's own `cube` primitive rather than pasted in as rendered
# images, so every block stays selectable, movable and recolourable in the editor.
C_IN, C_CONV, C_NORM = "#C8CCD2", "#2F4FD0", "#ECEFF2"
C_POOL, C_FC, C_OUT = "#E34EE0", "#3FAE5A", "#F0A03C"
SPINE = ["#D92B2B", "#F0821E", "#F5D020", "#3FAE5A"]     # tinted leading edges
KIND_COLOR = {"in": C_IN, "conv": C_CONV, "norm": C_NORM,
              "pool": C_POOL, "fc": C_FC, "out": C_OUT}


def cube(i, x, y, w, h, fill, size=9):
    """One feature-map sheet: draw.io's cube shape, seen at a slight angle."""
    _cell(i, "", x, y, w, h,
          f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;"
          f"darkOpacity=0.06;darkOpacity2=0.12;fillColor={fill};"
          f"strokeColor=#2B2F33;size={size};")


def layer_row(prefix, stages, x0, ycenter, pitch=74, hmax=104, plane_w=7,
              plane_step=5, main_w=22, flat_w=26):
    """A horizontal run of layer blocks, shape above and operation below.

    stages: (kind, shape_label, op_label, height_frac, n_planes). A conv is a
    BANK -- several thin sheets with tinted edges then the body -- because a
    convolution emits many maps and a single sheet would say it emits one.

    Blocks are CENTRED on ycenter, not stood on a shared baseline: bottom
    alignment made the row read as a skyline of pillars, whereas centring puts
    the data path on one axis and lets the connectors run through the middle of
    every block whatever its height.
    """
    ids = []
    op_baseline = ycenter + hmax // 2 + 8          # one line for every op label
    for k, (kind, shape, op, hf, npl) in enumerate(stages):
        x = x0 + k * pitch
        h = max(16, int(round(hmax * hf)))
        y = ycenter - h // 2
        if kind == "conv":
            for j in range(npl):
                cube(f"{prefix}{k}s{j}", x + j * plane_step, y, plane_w, h,
                     SPINE[j % len(SPINE)], size=5)
            cid = f"{prefix}{k}"
            cube(cid, x + npl * plane_step, y, main_w, h, C_CONV)
            wtot = npl * plane_step + main_w
        else:
            cid = f"{prefix}{k}"
            w = flat_w if kind in ("norm", "pool") else flat_w - 4
            cube(cid, x, y, w, h, KIND_COLOR[kind])
            wtot = w
        ids.append(cid)
        txt(f"{prefix}{k}_sh", f"<b>{shape}</b>", x - 16, y - 19, wtot + 32, 13, size=8)
        txt(f"{prefix}{k}_op", op, x - 18, op_baseline, wtot + 36, 22, size=8)
    for a, b in zip(ids[:-1], ids[1:]):
        edge(f"{prefix}_{a}_{b}", a, b, w=1.0, color="#5A6570",
             exit_=(1, 0.5), entry=(0, 0.5))
    return ids


# ============================================================ how much text
# Calibrated against the reference figures rather than by taste. Those figures
# consistently carry FOUR things and nothing else:
#
#   tensor shapes above each block      operation names below each block
#   a title per grouped stage           a colour legend
#
# and they consistently omit equations, parameter counts on every block, and
# explanatory prose. So the equations that were here (the two head read-outs,
# the tokenizer, the attention formula, the residual identity, the joint
# objective) are gone: they are stated in the paper, where a reader can follow
# them, and on a figure they only compete with the picture. Parameter counts
# survive once, in the footer, because the model's size is a claim the paper
# makes and one line carries it.

def container(cid, title, x, y, w, h, stroke="#8A9099"):
    """A grouped stage: title bar at the top, blocks inside."""
    # solid, deliberately: a dashed outline already means "frozen" in this
    # figure, and dashing the group boxes too would make the legend a lie
    _cell(cid, title, x, y, w, h,
          f"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={stroke};"
          f"verticalAlign=top;fontStyle=1;fontSize=11;spacingTop=3;arcSize=4;"
          f"opacity=70;")


def legend(x, y):
    """Colour key, as the reference figures carry."""
    items = [("input", C_IN), ("conv", C_CONV), ("pool", C_POOL),
             ("linear", C_FC), ("norm", C_NORM), ("output", C_OUT)]
    for k, (name, col) in enumerate(items):
        xx = x + k * 84
        _cell(f"lg{k}", "", xx, y, 15, 15,
              f"rounded=0;whiteSpace=wrap;html=1;fillColor={col};strokeColor=#2B2F33;")
        txt(f"lg{k}t", name, xx + 19, y - 1, 62, 16, size=9, align="left")


# ===================================================================== title
txt("title", "<b>Two-stream multimodal multi-task network</b>", 26, 14, 460, 22,
    size=14, align="left")
txt("subtitle", "f<sub>&#952;</sub> : (f<sup>eeg</sup>, z<sup>fnd</sup>, x<sup>car</sup>) "
    "&#8594; (&#375;<sup>stg</sup>, &#375;<sup>apn</sup>)", 26, 36, 460, 18,
    size=11, align="left")

# ============================================== NEURAL LANE (upper, L to R)
img("eeg_img", "eeg_traces.png", 26, 78, 132, 102)
txt("eeg_cap", "<i>7 ch @ 100 Hz, 30 s</i>", 20, 184, 146, 14, size=8)

node("eeg_feat", "<b>Expert physiology</b><br>f<sup>eeg</sup> &#8712; &#8477;<sup>188</sup>"
     "<br>band power, spindle, Hjorth", 186, 74, 176, 62, EEG)
node("eeg_fnd", "<b>Frozen LaBraM</b> &#10052;<br>z<sup>fnd</sup> &#8712; &#8477;<sup>200</sup>"
     "<br>4 EEG channels, no gradient", 186, 148, 176, 62, EEG, extra=DASHED_FROZEN)
ell("catE", "C", 382, 128, 26, 26)

container("eeg_box", "EEG encoder  &#966;<sub>eeg</sub>  &#183;  FeatMLP 388 &#8594; 128",
          430, 66, 406, 196, EEG[1])
_u = lambda n: math.log10(n) / math.log10(388.0)
layer_row("fe", [("in", "388", "input", _u(388), 1),
                 ("fc", "128", "Linear W&#8321;", _u(128), 1),
                 ("norm", "128", "LN + GELU", _u(128), 1),
                 ("fc", "128", "Linear W&#8322;", _u(128), 1),
                 ("norm", "128", "LN + GELU", _u(128), 1),
                 ("out", "128", "e", _u(128), 1)],
          x0=452, ycenter=176, pitch=62, hmax=88)

# =========================================== CARDIORESPIRATORY LANE (lower)
img("car_img", "cardio_signals.png", 22, 396, 140, 100)
txt("car_cap", "<i>7 ch @ 25 Hz, 30 s</i>", 20, 500, 146, 14, size=8)

node("car_feat", "<b>Raw cardiorespiratory tensor</b><br>"
     "x<sup>car</sup> &#8712; &#8477;<sup>7&#215;750</sup><br>per-subject z-score",
     186, 402, 176, 66, CAR)

container("car_box", "Cardio encoder  &#966;<sub>car</sub>  &#183;  "
          "CardioCNN 7&#215;750 &#8594; 64", 430, 372, 570, 210, CAR[1])
_t = lambda n: math.log10(n) / math.log10(750.0)
layer_row("cc", [("in", "7&#215;750", "raw", _t(750), 1),
                 ("conv", "48&#215;375", "Conv 25, s2<br>BN+GELU", _t(375), 4),
                 ("pool", "48&#215;93", "Pool 4", _t(93), 1),
                 ("conv", "96&#215;93", "Conv 15<br>BN+GELU", _t(93), 5),
                 ("pool", "96&#215;23", "Pool 4", _t(23), 1),
                 ("conv", "96&#215;23", "Conv 7<br>BN+GELU", _t(23), 5),
                 ("pool", "192", "mean &#8853; max", _t(4), 1),
                 ("out", "64", "Linear", _t(3), 1)],
          x0=452, ycenter=482, pitch=67, hmax=96)

# ============================================================ CONVERGE
ell("concatC", "C", 1040, 318, 26, 26)

container("cf_box", "Cross-modal fusion  &#183;  2 tokens &#8594; attention &#8594; "
          "z &#8712; &#8477;<sup>128</sup>", 1096, 232, 500, 200, FUS[1])
# same block language as the encoders, on its own centreline. The four heads are
# PARALLEL, so they stack across the axis rather than along it.
F_TOK, F_HEAD, F_MAIN = "#BBD3F0", "#7FA8DC", "#4A80C8"
CFY, CF_SH, CF_OP = 336, 274, 382


def _cf(cid, x, w, h, fill, shape, op):
    cube(cid, x, CFY - h // 2, w, h, fill)
    txt(cid + "_sh", f"<b>{shape}</b>", x - 24, CF_SH, w + 48, 13, size=8)
    txt(cid + "_op", op, x - 26, CF_OP, w + 52, 15, size=8)


_cf("cf_tok", 1120, 26, 74, F_TOK, "2&#215;128", "tokenize")
for k in range(4):
    cube(f"cf_h{k}", 1214, CFY - 41 + k * 22, 34, 17, F_HEAD, size=5)
txt("cf_h_sh", "<b>4 heads</b>", 1190, CF_SH, 82, 13, size=8)
txt("cf_h_op", "attention", 1188, CF_OP, 86, 15, size=8)
_cf("cf_concat", 1308, 26, 68, F_MAIN, "128", "concat &#183; W<sub>o</sub>")
ell("cf_add", "+", 1388, CFY - 12, 24, 24)
txt("cf_add_op", "residual", 1366, CF_OP, 68, 15, size=8)
_cf("cf_ln", 1452, 22, 62, C_NORM, "128", "LayerNorm")
_cf("cf_fuse", 1528, 24, 56, C_OUT, "128", "feed-forward")

for k in range(4):
    edge(f"cf_t{k}", "cf_tok", f"cf_h{k}", w=1.0, color="#5A6570",
         exit_=(1, 0.5), entry=(0, 0.5))
    edge(f"cf_c{k}", f"cf_h{k}", "cf_concat", w=1.0, color="#5A6570",
         exit_=(1, 0.5), entry=(0, 0.5))
for a, b in (("cf_concat", "cf_add"), ("cf_add", "cf_ln"), ("cf_ln", "cf_fuse")):
    edge(f"cf_{a}_{b}", a, b, w=1.0, color="#5A6570", exit_=(1, 0.5), entry=(0, 0.5))
edge("cf_res", "cf_tok", "cf_add", color="#9AA2AB", dashed=True, w=1.0,
     exit_=(0.5, 1), entry=(0.5, 1))

node("bilstm", "<b>BiLSTM</b>  2 layers, bidirectional<br>context L = 20 epochs (10 min)"
     "<br>h<sub>t</sub> &#8712; &#8477;<sup>512</sup>", 1660, 292, 208, 76, LSTM)
node("stg_head", "<b>Staging head</b><br>512 &#8594; 5<br>+ HMM Viterbi decode",
     1660, 120, 208, 70, STG)
node("rsp_head", "<b>Respiratory head</b><br>[h<sub>t</sub> ; c<sub>t</sub>] &#183; "
     "576 &#8594; 256 &#8594; 1<br>direct cardio bypass", 1660, 470, 208, 70, RSP)

# ==================================================================== wiring
edge("e1a", "eeg_feat", "catE"); edge("e1b", "eeg_fnd", "catE")
edge("e1", "catE", "eeg_box", entry=(0, 0.5))
edge("e2", "car_feat", "car_box", entry=(0, 0.5))
edge("e3", "eeg_box", "concatC", exit_=(1, 0.5), entry=(0, 0.5))
edge("e4", "car_box", "concatC", exit_=(1, 0.5), entry=(0.5, 1))
edge("e5", "concatC", "cf_box", entry=(0, 0.5))
edge("e6", "cf_box", "bilstm", exit_=(1, 0.5), entry=(0, 0.5))
edge("e7", "bilstm", "stg_head")
edge("e8", "bilstm", "rsp_head")
# the bypass: c reaches the respiratory head without passing through fusion
# or the recurrence
edge("e9", "car_box", "rsp_head", dashed=True, color="#C4763A",
     exit_=(1, 0.85), entry=(0, 0.5))
txt("bypass_lbl", "<i>direct c<sub>t</sub></i>", 1470, 560, 90, 16, size=10)

# =================================================================== footer
legend(26, 620)
txt("footer", "<i>dashed outline = frozen (no gradient)  &#183;  2,764,774 trainable + "
    "5,819,936 frozen parameters</i>", 552, 618, 620, 18, size=9, align="left")


# =================================================================== emit
xml = ('<mxfile host="app.diagrams.net">'
       '<diagram name="MM-Net architecture" id="mmnet">'
       '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1900" '
       'pageHeight="670" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')

with open(OUT, "w", encoding="utf-8") as f:
    f.write(xml)
print("wrote", OUT)

import xml.dom.minidom as m
m.parseString(xml)
print("XML OK,", len(cells), "cells")

