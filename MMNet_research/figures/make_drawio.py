"""Emit an editable draw.io (diagrams.net) file of the MM-Net architecture.

Labels use HTML formatting (<br>, <sub>, <sup>, <b>) so draw.io renders line
breaks and proper sub/superscripts.

Layout rationale. An earlier revision carried four side panels -- FeatMLP,
CardioCNN, cross-modal fusion, and the temporal decoder. A modular side panel
earns its place only when the block it details is INSTANTIATED MORE THAN ONCE
and the panel saves the reader from reading the same stack twice. None of these
repeats: FeatMLP runs on the neural stream alone, CardioCNN on the
cardiorespiratory stream alone, and there is one temporal decoder. So their
internals are expanded inline, in the main flow, where the reader meets them.

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

# ===================================================================== title
txt("title", "<b>Two-stream multimodal multi-task network</b><br>"
    "f<sub>&#952;</sub> : (f<sup>eeg</sup>, z<sup>fnd</sup>, x<sup>car</sup>) &#8594; "
    "(&#375;<sup>stg</sup>, &#375;<sup>apn</sup>)", 40, 10, 620, 42, size=13)

# ============================================================ signal previews
img("eeg_img", "eeg_traces.png", 75, 64, 150, 116)
img("car_img", "cardio_signals.png", 445, 62, 175, 122)
txt("eeg_img_cap", "<i>7-channel neural montage, 30 s @ 100 Hz</i>", 40, 188, 220, 16, size=9)
txt("car_img_cap", "<i>7-channel cardiorespiratory, 30 s @ 25 Hz</i>", 430, 190, 220, 16, size=9)

# ================================================================ input row
node("eeg_feat", "<b>Expert physiology</b><br>f<sup>eeg</sup> &#8712; &#8477;<sup>188</sup><br>"
     "band power, spindle, Hjorth", 20, 214, 180, 76, EEG)
node("eeg_fnd", "<b>Frozen LaBraM</b> &#10052;<br>z<sup>fnd</sup> &#8712; &#8477;<sup>200</sup> "
     "&#183; 5.82 M frozen<br>4 EEG channels, no gradient", 210, 214, 180, 76, EEG,
     extra=DASHED_FROZEN)
node("car_feat", "<b>Raw cardiorespiratory tensor</b><br>"
     "x<sup>car</sup> &#8712; &#8477;<sup>7&#215;750</sup> = 5,250<br>"
     "per-subject z-score, no summary", 440, 214, 205, 76, CAR)

ell("catE", "C", 196, 306, 28, 28)

# ============================================================== encoder row
node("eeg_enc", "<b>EEG encoder</b>  &#966;<sub>eeg</sub><br>"
     "FeatMLP: 388 &#8594; e &#8712; &#8477;<sup>128</sup><br>66,816 p", 105, 350, 190, 76, EEG)
node("car_enc", "<b>Cardio encoder</b>  &#966;<sub>car</sub><br>"
     "CardioCNN: 7&#215;750 &#8594; c &#8712; &#8477;<sup>64</sup><br>"
     "3 conv stages &#183; 155,232 p", 440, 350, 205, 76, CAR)

# --- FeatMLP, inlined (was a side panel) ---
txt("fe_eq", "<i>e = GELU(LN(W<sub>2</sub> GELU(LN(W<sub>1</sub> [f<sup>eeg</sup> ; "
    "z<sup>fnd</sup>]))))</i>", 90, 432, 220, 18, size=9)
fe = stack("fe", ["Linear  W<sub>1</sub>  (388 &#8594; 128)", "LayerNorm", "GELU",
                  "Dropout (0.3)", "Linear  W<sub>2</sub>  (128 &#8594; 128)",
                  "LayerNorm", "GELU"],
           115, 456, 170, EEG)

# --- CardioCNN, inlined (was a side panel) ---
txt("cc_note", "<i>kernels sized to respiratory physiology: events last 10&#8211;30 s,<br>"
    "so kernels are wide (25 samples = 1 s at 25 Hz)</i>", 430, 428, 225, 24, size=9)
cc = stack("cc", ["Conv1d 7 &#8594; 48 &#183; k = 25, s = 2 &#183; BN &#183; GELU",
                  "MaxPool 4 &#183; Dropout",
                  "Conv1d 48 &#8594; 96 &#183; k = 15 &#183; BN &#183; GELU",
                  "MaxPool 4 &#183; Dropout",
                  "Conv1d 96 &#8594; 96 &#183; k = 7 &#183; BN &#183; GELU",
                  "global mean &#8853; max pool &#8594; 192",
                  "Linear 192 &#8594; 64 &#183; LayerNorm &#183; GELU"],
           440, 456, 205, CAR)
txt("cc_cap", "<i>replaces the 14 engineered cardiorespiratory features</i>",
    430, 726, 225, 16, size=9)

ell("concatC", "C", 266, 744, 28, 28)

# ================================================================== fusion
node("fusion", "<b>Cross-modal fusion</b><br>2 tokens &#8594; attention &#8594; "
     "z &#8712; &#8477;<sup>128</sup><br>99,456 p", 160, 792, 240, 76, FUS)

# ============================================== temporal decoder, inlined
node("bilstm", "<b>BiLSTM</b>  (2 layers, bidirectional)<br>"
     "context L = 20 epochs (10 min) &#183; h<sub>t</sub> &#8712; &#8477;<sup>512</sup><br>"
     "2,367,488 p", 140, 902, 280, 76, LSTM)
tl = stack("tl", ["BiLSTM layer 1 &#183; 128 &#8594; 2 &#215; 256",
                  "BiLSTM layer 2 &#183; 512 &#8594; 2 &#215; 256",
                  "h<sub>t</sub> = [h<sub>t</sub><sup>&#8594;</sup> ; "
                  "h<sub>t</sub><sup>&#8592;</sup>] &#8712; &#8477;<sup>512</sup>"],
           155, 1002, 250, LSTM)

# =================================================================== heads
node("stg_head", "<b>Staging head</b><br>&#375;<sub>t</sub><sup>stg</sup> = "
     "softmax(W<sub>s</sub> h<sub>t</sub>) &#183; 512 &#8594; 5<br>"
     "+ HMM Viterbi decode (train-label<br>transitions) &#183; 2,565 p",
     20, 1140, 235, 96, STG)
node("rsp_head", "<b>Respiratory head</b><br>&#375;<sub>t</sub><sup>apn</sup> = "
     "&#963;(W<sub>2</sub> GELU(W<sub>1</sub>[h<sub>t</sub> ; c<sub>t</sub>]))<br>"
     "576 &#8594; 256 &#8594; 1 &#183; no smoothing<br>direct cardio bypass &#183; 147,969 p",
     290, 1140, 250, 96, RSP)

# ==================================================================== wiring
edge("e1a", "eeg_feat", "catE"); edge("e1b", "eeg_fnd", "catE")
edge("e1", "catE", "eeg_enc")
edge("e2", "car_feat", "car_enc")
edge("e_enc_fe", "eeg_enc", fe[0], w=1.1)
edge("e_enc_cc", "car_enc", cc[0], w=1.1)
edge("e3", fe[-1], "concatC", w=1.1)
edge("e4", cc[-1], "concatC", w=1.1, exit_=(0.5, 1), entry=(1, 0.5))
edge("e5", "concatC", "fusion")
edge("e6", "fusion", "bilstm")
edge("e_bl_tl", "bilstm", tl[0], w=1.1)
edge("e7", tl[-1], "stg_head"); edge("e8", tl[-1], "rsp_head")
# the bypass: c_t leaves the cardio encoder and reaches the respiratory head
# without passing through fusion or the recurrence
edge("e9", cc[-1], "rsp_head", dashed=True, color="#C4763A",
     exit_=(1, 0.5), entry=(1, 0))
txt("bypass_lbl", "<i>direct c<sub>t</sub></i>", 655, 940, 90, 18, size=10)

# ===================================================== the one side panel
panel("cf_panel", "Cross-modal fusion", 700, 214, 400, 620)
txt("cf_why", "<i>the only block detailed separately: attention does not inline<br>"
    "compactly, and it is instantiated once</i>", 712, 244, 376, 26, size=9)
small("cf_tok", "tokenize:  T = [W<sub>e</sub> e ; W<sub>c</sub> c] + M<sub>type</sub>",
      750, 284, 300, 32, FUS)
txt("cf_headeq", "head<sub>i</sub> = softmax(Q<sub>i</sub> K<sub>i</sub><sup>&#8868;</sup> / "
    "&#8730;d<sub>k</sub>) V<sub>i</sub>", 750, 326, 300, 18, size=10)
heads = []
hx = 726
for k in range(4):
    cid = f"cf_h{k}"
    small(cid, f"head {k + 1}", hx, 358, 78, 32, FUS)
    heads.append(cid); hx += 90
small("cf_concat", "concat heads  &#183;  W<sub>o</sub>", 780, 426, 240, 32, FUS)
ell("cf_add", "+", 887, 478, 26, 26)
small("cf_ln", "LayerNorm", 780, 530, 240, 32, FUS)
small("cf_ff", "Feed-Forward  W<sub>f</sub> &#8712; &#8477;<sup>128&#215;256</sup>",
      780, 582, 240, 32, FUS)
small("cf_fuse", "fuse &#8594; z &#8712; &#8477;<sup>128</sup>", 780, 634, 240, 32, FUS)
txt("cf_cap1", "<i>T&#771; = LN(T + concat<sub>i</sub> head<sub>i</sub>)</i>",
    712, 684, 376, 18, size=10)
txt("cf_cap2", "<i>ablations: plain concatenation [e ; c] and a neural-only variant<br>"
    "are evaluated in place of attention</i>", 712, 708, 376, 28, size=9)
txt("cf_cap3", "<i>c<sub>t</sub> reaches the respiratory head directly, so desaturation and<br>"
    "effort cues are not diluted by a fusion trained mostly for staging</i>",
    712, 752, 376, 28, size=9)
for cid in heads:
    edge(f"cf_t_{cid}", "cf_tok", cid, w=1.1)
    edge(f"cf_c_{cid}", cid, "cf_concat", w=1.1)
edge("cf_c_add", "cf_concat", "cf_add", w=1.1)
edge("cf_add_ln", "cf_add", "cf_ln", w=1.1)
edge("cf_ln_ff", "cf_ln", "cf_ff", w=1.1)
edge("cf_ff_fuse", "cf_ff", "cf_fuse", w=1.1)
edge("cf_resid", "cf_tok", "cf_add", color="#888888", dashed=True, w=1.1,
     exit_=(0, 0.5), entry=(0, 0.5))

# =================================================================== footer
txt("obj", "Joint objective:   L = CE<sub>&#8730;w</sub>(&#375;<sup>stg</sup>, y<sup>stg</sup>) + "
    "&#955; BCE<sub>pw</sub>(&#375;<sup>apn</sup>, y<sup>apn</sup>),   &#955; = 1",
    100, 1268, 600, 24, size=12)
txt("legend", "<i>dashed outline = frozen (no gradient)   &#183;   2,764,774 trainable + "
    "5,819,936 frozen parameters</i>", 80, 1296, 640, 20, size=10)


# =================================================================== emit
xml = ('<mxfile host="app.diagrams.net">'
       '<diagram name="MM-Net architecture" id="mmnet">'
       '<mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1160" '
       'pageHeight="1340" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')

with open(OUT, "w", encoding="utf-8") as f:
    f.write(xml)
print("wrote", OUT)

import xml.dom.minidom as m
m.parseString(xml)
print("XML OK,", len(cells), "cells")
