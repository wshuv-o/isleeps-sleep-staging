"""Emit an editable draw.io (diagrams.net) file of the MM-Net architecture.
Labels use HTML formatting (<br>, <sub>, <sup>, <b>) so draw.io renders line breaks
and proper sub/superscripts."""
import os, html
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_architecture.drawio")

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
          f"rounded=1;whiteSpace=wrap;html=1;fillColor={pal[0]};strokeColor={pal[1]};fontSize=12;spacing=4;shadow=1;{extra}")
def panel(i, label, x, y, w, h):
    _cell(i, label, x, y, w, h,
          f"rounded=1;whiteSpace=wrap;html=1;fillColor={PANEL[0]};strokeColor={PANEL[1]};verticalAlign=top;fontStyle=1;fontSize=13;")
def small(i, label, x, y, w, h, pal):
    _cell(i, label, x, y, w, h,
          f"rounded=1;whiteSpace=wrap;html=1;fillColor={pal[0]};strokeColor={pal[1]};fontSize=11;spacing=3;")
def ell(i, label, x, y, w, h, pal=WHITE):
    _cell(i, label, x, y, w, h,
          f"ellipse;whiteSpace=wrap;html=1;fillColor={pal[0]};strokeColor={pal[1]};fontSize=14;fontStyle=1;")
def img(i, asset, x, y, w, h):
    """Embed a PNG from arch_assets/ as a base64 data URI.

    The signal thumbnails were originally dropped into draw.io by hand and never
    saved back -- mm_architecture.drawio carried zero image references, so
    regenerating it silently lost them. Embedding keeps the .drawio
    self-contained and the generator authoritative.
    """
    import base64
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_assets", asset)
    b64 = base64.b64encode(open(d, "rb").read()).decode()
    _cell(i, "", x, y, w, h,
          f"shape=image;verticalLabelPosition=bottom;labelBackgroundColor=none;"
          f"imageAspect=0;aspect=fixed;image=data:image/png,{b64};")

def txt(i, label, x, y, w, h, italic=False, size=11):
    _cell(i, label, x, y, w, h,
          f"text;html=1;align=center;verticalAlign=middle;fontSize={size};{'fontStyle=2;' if italic else ''}")
def edge(i, s, t, dashed=False, color="#333333", exit_=None, entry=None):
    st = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor={color};strokeWidth=1.5;"
    if dashed: st += "dashed=1;"
    # Fixed anchors where the router's default choice is misleading. Left to
    # itself it exits car_enc's LEFT side at the encoder centre-line, which puts
    # the segment exactly on eeg_enc's right edge and reads as a wire between
    # the two encoders.
    if exit_: st += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry: st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1" source="{s}" target="{t}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')

R = "&#8477;"  # placeholder not used; we use ℝ directly below

# ===== title =====
txt("title", "<b>Two-stream multimodal multi-task network</b><br>"
    "f<sub>θ</sub> : (f<sup>eeg</sup>, z<sup>fnd</sup>, x<sup>car</sup>) → (ŷ<sup>stg</sup>, ŷ<sup>apn</sup>)"
    "   ·   2,764,774 trainable + 5,819,936 frozen",
    40, 8, 620, 44, size=12)
txt("newkey", "<i><font color=\"#C0392B\">red outline = new since submission</font></i>", 1060, 14, 300, 20, size=10)

# ===== PANEL A: main flow =====
# real iSLEEPS signal previews (make_arch_assets.py), one per stream
img("eeg_img", "eeg_traces.png", 60, 62, 150, 116)
img("car_img", "cardio_signals.png", 420, 62, 165, 120)
txt("eeg_img_cap", "<i>7-channel neural montage</i>", 45, 182, 180, 16, size=10)
txt("car_img_cap", "<i>7-channel cardiorespiratory</i>", 415, 186, 180, 16, size=10)

node("eeg_feat", "<b>Expert physiology</b><br>f<sup>eeg</sup> ∈ ℝ<sup>188</sup><br>band power, spindle, Hjorth", 20, 220, 175, 74, EEG)
# NEW: the second, imported prior. Frozen -- red stroke marks it as new since submission.
node("eeg_fnd", "<b>Frozen LaBraM</b><br>z<sup>fnd</sup> ∈ ℝ<sup>200</sup> · 5.82 M frozen<br>4 EEG channels, no gradient",
     205, 220, 175, 74, ("#D4E2F5", "#C0392B"), extra="strokeWidth=2;")
node("car_feat", "<b>Raw cardio tensor</b><br>x<sup>car</sup> ∈ ℝ<sup>7×750</sup> = 5250<br>per-subject z-score, no summary",
     420, 220, 190, 74, CAR)
ell("catE", "C", 185, 308, 30, 30)
node("eeg_enc", "<b>EEG encoder</b>  φ<sub>eeg</sub><br>FeatMLP: 388 → e ∈ ℝ<sup>128</sup><br>66,816 p", 105, 360, 190, 74, EEG)
node("car_enc", "<b>Cardio encoder</b>  φ<sub>car</sub><br>CardioCNN → c ∈ ℝ<sup>64</sup><br>3 conv stages · 155,232 p",
     420, 360, 190, 74, ("#F7DFC9", "#C0392B"), extra="strokeWidth=2;")
ell("concatC", "C", 280, 460, 30, 30)
node("fusion", "<b>Cross-modal fusion</b><br>2 tokens → attention → z ∈ ℝ<sup>128</sup><br>99,456 p", 175, 510, 240, 74, FUS)
node("bilstm", "<b>BiLSTM</b>  (2 layers, bidirectional)<br>context L = 20 epochs · h<sub>t</sub> ∈ ℝ<sup>512</sup><br>2,367,488 p", 155, 640, 280, 74, LSTM)
node("stg_head", "<b>Staging head</b><br>ŷ<sub>t</sub><sup>stg</sup> = softmax(W<sub>s</sub> h<sub>t</sub>)<br>+ HMM Viterbi decode<br>2,565 p", 60, 790, 195, 96, STG)
node("rsp_head", "<b>Respiratory head</b><br>ŷ<sub>t</sub><sup>apn</sup> = σ(W<sub>2</sub>[h<sub>t</sub> ; c<sub>t</sub>])<br>direct cardio bypass<br>147,969 p", 335, 790, 195, 96, RSP)

edge("e1a", "eeg_feat", "catE"); edge("e1b", "eeg_fnd", "catE"); edge("e1", "catE", "eeg_enc")
edge("e2", "car_feat", "car_enc")
edge("e3", "eeg_enc", "fusion")
edge("e4", "car_enc", "fusion", exit_=(0.5, 1), entry=(1, 0.5))
edge("e5", "fusion", "bilstm")
edge("e6", "bilstm", "stg_head"); edge("e7", "bilstm", "rsp_head")
edge("e8", "car_enc", "rsp_head", dashed=True, color="#C4763A")
txt("bypass_lbl", "<i>direct c<sub>t</sub></i>", 548, 580, 70, 20)

# ===== PANEL B: module detail =====
txt("detail_hdr", "<b>Module detail</b>", 700, 12, 320, 26, size=13)

# Feature encoder (FeatMLP)
panel("fe_panel", "Feature encoder  (FeatMLP)", 700, 50, 250, 410)
txt("fe_eq", "e = GELU(LN(W<sub>2</sub> GELU(LN(W<sub>1</sub> f))))", 710, 80, 230, 22)
fe_items = [("fe1", "Linear  W<sub>1</sub>"), ("fe2", "LayerNorm"), ("fe3", "GELU"),
            ("fe4", "Dropout (0.3)"), ("fe5", "Linear  W<sub>2</sub>"), ("fe6", "LayerNorm"), ("fe7", "GELU")]
yy = 112
for cid, lab in fe_items:
    small(cid, lab, 745, yy, 160, 32, EEG); yy += 44
for a, b in zip(fe_items[:-1], fe_items[1:]):
    edge("fe_" + a[0] + b[0], a[0], b[0])
txt("fe_cap", "<i>EEG branch only: 388 → 128<br>cardio uses CardioCNN (panel C)</i>", 710, 418, 230, 32)

# Cross-modal fusion (4 heads drawn)
panel("cf_panel", "Cross-modal fusion", 700, 490, 340, 470)
small("cf_tok", "tokenize:  T = [W<sub>e</sub> e ; W<sub>c</sub> c] + M<sub>type</sub>", 730, 524, 280, 30, FUS)
txt("cf_headeq", "head<sub>i</sub> = softmax(Q<sub>i</sub> K<sub>i</sub><sup>⊤</sup> / √d<sub>k</sub>) V<sub>i</sub>", 730, 560, 280, 18)
heads = [("cf_h1", "head 1"), ("cf_h2", "head 2"), ("cf_h3", "head 3"), ("cf_h4", "head 4")]
hx = 735
for cid, lab in heads:
    small(cid, lab, hx, 584, 64, 30, FUS); hx += 72
small("cf_concat", "concat heads  ·  W<sub>o</sub>", 765, 634, 210, 30, FUS)
ell("cf_add", "+", 857, 676, 26, 26)
small("cf_ln", "LayerNorm", 765, 718, 210, 30, FUS)
small("cf_ff", "Feed-Forward  W<sub>f</sub> ∈ ℝ<sup>128×256</sup>", 765, 758, 210, 30, FUS)
small("cf_fuse", "fuse → z ∈ ℝ<sup>128</sup>", 765, 798, 210, 30, FUS)
txt("cf_cap", "<i>T̃ = LN(T + concat<sub>i</sub> head<sub>i</sub>)   ·   ablatable: concat / eeg-only</i>", 715, 834, 310, 20)
for cid, _ in heads:
    edge("cf_tok_" + cid, "cf_tok", cid); edge(cid + "_concat", cid, "cf_concat")
edge("cf_c_add", "cf_concat", "cf_add")
edge("cf_add_ln", "cf_add", "cf_ln"); edge("cf_ln_ff", "cf_ln", "cf_ff"); edge("cf_ff_fuse", "cf_ff", "cf_fuse")
edge("cf_resid", "cf_tok", "cf_add", color="#666666")

# ===== PANEL C: CardioCNN, the block that replaced the 14 engineered features =====
panel("cc_panel", "CardioCNN", 1060, 50, 300, 410)
txt("cc_note", "<i>replaces the 14 engineered cardio features<br>"
    "kernels sized to respiratory physiology: events<br>"
    "last 10–30 s, so kernels are wide and the<br>stack downsamples hard</i>", 1070, 78, 280, 56)
cc_items = [("cc1", "Conv(7→48, k=25, s=2)  BN  GELU"), ("cc2", "MaxPool 4  ·  Dropout"),
            ("cc3", "Conv(48→96, k=15)  BN  GELU"), ("cc4", "MaxPool 4  ·  Dropout"),
            ("cc5", "Conv(96→96, k=7)  BN  GELU"), ("cc6", "concat [mean , max]"),
            ("cc7", "Linear → LayerNorm → GELU")]
yy = 140
for cid, lab in cc_items:
    small(cid, lab, 1080, yy, 260, 32, CAR); yy += 42
for a, b in zip(cc_items[:-1], cc_items[1:]):
    edge("cc_" + a[0] + b[0], a[0], b[0])
txt("cc_cap", "<i>5250 → c ∈ ℝ<sup>64</sup>  ·  at 25 Hz a 25-sample kernel is one second</i>",
    1070, 436, 280, 20)

# ===== objective footer =====
txt("obj", "Joint objective:   L = CE<sub>√w</sub>(ŷ<sup>stg</sup>, y<sup>stg</sup>) + "
     "λ BCE<sub>pw</sub>(ŷ<sup>apn</sup>, y<sup>apn</sup>),   λ = 1",
    280, 980, 560, 26, size=12)

xml = ('<mxfile host="app.diagrams.net">'
       '<diagram name="MM-Net architecture" id="mmnet">'
       '<mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1420" pageHeight="1190" '
       'math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')

with open(OUT, "w", encoding="utf-8") as f:
    f.write(xml)
print("wrote", OUT)
import xml.dom.minidom as m
m.parseString(xml)
print("XML OK,", len(cells), "cells")
