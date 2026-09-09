"""Emit an editable draw.io (diagrams.net) file of the MM-Net architecture.

Labels use HTML formatting (<br>, <sub>, <sup>, <b>) so draw.io renders line
breaks and proper sub/superscripts.

Layout rationale. An earlier revision carried four side panels -- FeatMLP,
CardioCNN, cross-modal fusion, and the temporal decoder. A modular side panel
earns its place only when the block it details is INSTANTIATED MORE THAN ONCE
and the panel saves the reader from reading the same stack twice. None of these
repeats: FeatMLP runs on the neural stream alone, CardioCNN on the
cardiorespiratory stream alone, and there is one temporal decoder. So their
internals are expanded inline as isometric layer blocks (make_encoder_blocks.py),
which lay out horizontally and keep the figure landscape.

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

# The generator's img() reads from arch_assets/; the encoder blocks live one
# level up in figures/, so they get their own loader rather than being copied.
def img_here(i, asset, x, y, w, h):
    d = os.path.join(HERE, asset)
    b64 = base64.b64encode(open(d, "rb").read()).decode()
    _cell(i, "", x, y, w, h,
          f"shape=image;verticalLabelPosition=bottom;labelBackgroundColor=none;"
          f"imageAspect=0;aspect=fixed;image=data:image/png,{b64};")


# ===================================================================== title
txt("title", "<b>Two-stream multimodal multi-task network</b>  ·  "
    "f<sub>&#952;</sub> : (f<sup>eeg</sup>, z<sup>fnd</sup>, x<sup>car</sup>) &#8594; "
    "(&#375;<sup>stg</sup>, &#375;<sup>apn</sup>)", 30, 12, 760, 26, size=13, align="left")

# ============================================== NEURAL LANE (upper, L to R)
img("eeg_img", "eeg_traces.png", 34, 74, 128, 100)
txt("eeg_cap", "<i>7 ch @ 100 Hz, 30 s</i>", 24, 178, 150, 14, size=8)

node("eeg_feat", "<b>Expert physiology</b><br>f<sup>eeg</sup> &#8712; &#8477;<sup>188</sup> · "
     "band power,<br>spindle, Hjorth", 190, 66, 172, 66, EEG)
node("eeg_fnd", "<b>Frozen LaBraM</b> &#10052;<br>z<sup>fnd</sup> &#8712; &#8477;<sup>200</sup> · "
     "5.82 M frozen<br>4 EEG channels, no gradient", 190, 146, 172, 66, EEG, extra=DASHED_FROZEN)
ell("catE", "C", 382, 126, 26, 26)
node("eeg_enc", "<b>EEG encoder</b>  &#966;<sub>eeg</sub><br>"
     "FeatMLP: 388 &#8594; e &#8712; &#8477;<sup>128</sup> · 66,816 p", 428, 106, 200, 66, EEG)
img_here("eeg_iso", "eeg_encoder_iso.png", 418, 186, 330, 148)

# =========================================== CARDIORESPIRATORY LANE (lower)
img("car_img", "cardio_signals.png", 30, 388, 140, 104)
txt("car_cap", "<i>7 ch @ 25 Hz, 30 s</i>", 24, 496, 150, 14, size=8)

node("car_feat", "<b>Raw cardiorespiratory tensor</b><br>"
     "x<sup>car</sup> &#8712; &#8477;<sup>7&#215;750</sup> = 5,250<br>"
     "per-subject z-score, no summary", 190, 402, 172, 76, CAR)
node("car_enc", "<b>Cardio encoder</b>  &#966;<sub>car</sub><br>"
     "CardioCNN: 7&#215;750 &#8594; c &#8712; &#8477;<sup>64</sup> · 155,232 p",
     428, 402, 200, 76, CAR)
img_here("car_iso", "cardio_encoder_iso.png", 418, 492, 340, 152)
txt("car_iso_cap", "<i>replaces the 14 engineered cardiorespiratory features</i>",
    418, 648, 340, 14, size=8)

# ================================================ CONVERGE, then L to R
ell("concatC", "C", 786, 282, 26, 26)
node("fusion", "<b>Cross-modal fusion</b><br>2 tokens &#8594; attention<br>"
     "&#8594; z &#8712; &#8477;<sup>128</sup> · 99,456 p", 846, 256, 190, 78, FUS)
node("bilstm", "<b>BiLSTM</b> (2 layers, bidirectional)<br>"
     "L = 20 epochs (10 min) · 128 &#8594; 2 &#215; 256<br>"
     "h<sub>t</sub> &#8712; &#8477;<sup>512</sup> · 2,367,488 p", 1074, 250, 220, 90, LSTM)
node("stg_head", "<b>Staging head</b><br>softmax(W<sub>s</sub> h<sub>t</sub>) · 512 &#8594; 5<br>"
     "+ HMM Viterbi decode · 2,565 p", 1338, 154, 224, 82, STG)
node("rsp_head", "<b>Respiratory head</b><br>"
     "&#963;(W<sub>2</sub> GELU(W<sub>1</sub>[h<sub>t</sub> ; c<sub>t</sub>]))<br>"
     "576 &#8594; 256 &#8594; 1 · direct cardio bypass<br>147,969 p", 1338, 344, 224, 88, RSP)

# ==================================================================== wiring
edge("e1a", "eeg_feat", "catE"); edge("e1b", "eeg_fnd", "catE")
edge("e1", "catE", "eeg_enc")
edge("e2", "car_feat", "car_enc")
edge("e3", "eeg_enc", "concatC", exit_=(1, 0.5), entry=(0, 0.5))
edge("e4", "car_enc", "concatC", exit_=(1, 0.5), entry=(0.5, 1))
edge("e5", "concatC", "fusion")
edge("e6", "fusion", "bilstm")
edge("e7", "bilstm", "stg_head")
edge("e8", "bilstm", "rsp_head")
# the bypass: c_t reaches the respiratory head without passing through fusion
# or the recurrence, so desaturation and effort cues are not diluted
edge("e9", "car_enc", "rsp_head", dashed=True, color="#C4763A",
     exit_=(1, 0.25), entry=(0, 0.5))
txt("bypass_lbl", "<i>direct c<sub>t</sub></i>", 1150, 404, 90, 16, size=10)
# each encoder box is expanded by the isometric block beneath it
txt("eeg_iso_cap", "<i>encoder expanded</i>", 418, 176, 330, 12, size=8)
txt("car_iso_cap2", "<i>encoder expanded</i>", 418, 482, 340, 12, size=8)

# ======================================== the one side panel: fusion detail
panel("cf_panel", "Cross-modal fusion &#8212; the one block detailed separately",
      846, 452, 730, 250)
txt("cf_why", "<i>attention does not expand compactly in line; every other block is "
    "shown inline above</i>", 858, 478, 706, 14, size=8)
small("cf_tok", "tokenize<br>T = [W<sub>e</sub> e ; W<sub>c</sub> c] + M<sub>type</sub>",
      864, 508, 148, 48, FUS)
for k in range(4):
    small(f"cf_h{k}", f"head {k + 1}", 1044, 500 + k * 42, 84, 34, FUS)
txt("cf_headeq", "<i>head<sub>i</sub> = softmax(Q<sub>i</sub>K<sub>i</sub><sup>&#8868;</sup>/"
    "&#8730;d<sub>k</sub>)V<sub>i</sub></i>", 1020, 668, 200, 14, size=8)
small("cf_concat", "concat heads<br>&#183; W<sub>o</sub>", 1164, 546, 116, 48, FUS)
ell("cf_add", "+", 1310, 558, 26, 26)
small("cf_ln", "LayerNorm", 1360, 546, 96, 48, FUS)
small("cf_ff", "Feed-Forward<br>W<sub>f</sub> &#8712; &#8477;<sup>128&#215;256</sup><br>"
      "&#8594; fuse z &#8712; &#8477;<sup>128</sup>", 1466, 540, 100, 60, FUS)
for k in range(4):
    edge(f"cf_t{k}", "cf_tok", f"cf_h{k}", w=1.0)
    edge(f"cf_c{k}", f"cf_h{k}", "cf_concat", w=1.0)
edge("cf_ca", "cf_concat", "cf_add", w=1.0)
edge("cf_al", "cf_add", "cf_ln", w=1.0)
edge("cf_lf", "cf_ln", "cf_ff", w=1.0)
edge("cf_res", "cf_tok", "cf_add", color="#9aa2ab", dashed=True, w=1.0,
     exit_=(0.5, 1), entry=(0.5, 1))
txt("cf_cap", "<i>T&#771; = LN(T + concat<sub>i</sub> head<sub>i</sub>).  Ablations replace "
    "attention with plain concatenation [e ; c] and with a neural-only variant.</i>",
    858, 682, 706, 14, size=8)

# =================================================================== footer
txt("obj", "Joint objective:  L = CE<sub>&#8730;w</sub>(&#375;<sup>stg</sup>, y<sup>stg</sup>) + "
    "&#955; BCE<sub>pw</sub>(&#375;<sup>apn</sup>, y<sup>apn</sup>),  &#955; = 1", 30, 690, 560, 22,
    size=12, align="left")
txt("legend", "<i>dashed outline = frozen (no gradient)  &#183;  2,764,774 trainable + "
    "5,819,936 frozen parameters</i>", 30, 714, 600, 18, size=9, align="left")


# =================================================================== emit
xml = ('<mxfile host="app.diagrams.net">'
       '<diagram name="MM-Net architecture" id="mmnet">'
       '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1620" '
       'pageHeight="760" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')

with open(OUT, "w", encoding="utf-8") as f:
    f.write(xml)
print("wrote", OUT)

import xml.dom.minidom as m
m.parseString(xml)
print("XML OK,", len(cells), "cells")

