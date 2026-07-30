"""Emit an editable draw.io (diagrams.net) file of the MM-Net architecture.
Labels use HTML formatting (<br>, <sub>, <sup>, <b>) so draw.io renders line breaks
and proper sub/superscripts."""
import os, html
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures", "mm_architecture.drawio")

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
def txt(i, label, x, y, w, h, italic=False, size=11):
    _cell(i, label, x, y, w, h,
          f"text;html=1;align=center;verticalAlign=middle;fontSize={size};{'fontStyle=2;' if italic else ''}")
def edge(i, s, t, dashed=False, color="#333333"):
    st = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor={color};strokeWidth=1.5;"
    if dashed: st += "dashed=1;"
    cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1" source="{s}" target="{t}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')

R = "&#8477;"  # placeholder not used; we use ℝ directly below

# ===== title =====
txt("title", "<b>Two-stream multimodal multi-task network</b><br>"
    "f<sub>θ</sub> : (f<sup>eeg</sup>, f<sup>car</sup>) → (ŷ<sup>stg</sup>, ŷ<sup>apn</sup>)   ·   856,326 parameters",
    40, 8, 620, 44, size=12)

# ===== PANEL A: main flow =====
node("eeg_feat", "<b>EEG features</b><br>f<sup>eeg</sup> ∈ ℝ<sup>188</sup><br>band power, spindle, Hjorth", 60, 70, 190, 74, EEG)
node("car_feat", "<b>Cardio features</b><br>f<sup>car</sup> ∈ ℝ<sup>14</sup><br>SpO<sub>2</sub>, effort, HRV, airflow", 340, 70, 190, 74, CAR)
node("eeg_enc", "<b>EEG encoder</b>  φ<sub>eeg</sub><br>FeatMLP → e ∈ ℝ<sup>128</sup><br>41,216 p", 60, 210, 190, 74, EEG)
node("car_enc", "<b>Cardio encoder</b>  φ<sub>car</sub><br>FeatMLP → c ∈ ℝ<sup>64</sup><br>5,376 p", 340, 210, 190, 74, CAR)
ell("concatC", "C", 280, 310, 30, 30)
node("fusion", "<b>Cross-modal fusion</b><br>2 tokens → attention → z ∈ ℝ<sup>128</sup><br>99,456 p", 175, 360, 240, 74, FUS)
node("bilstm", "<b>BiLSTM</b>  (2 layers, bidirectional)<br>context L = 20 epochs · h<sub>t</sub> ∈ ℝ<sup>256</sup><br>659,456 p", 155, 490, 280, 74, LSTM)
node("stg_head", "<b>Staging head</b><br>ŷ<sub>t</sub><sup>stg</sup> = softmax(W<sub>s</sub> h<sub>t</sub>)<br>+ HMM Viterbi decode<br>1,285 p", 60, 640, 195, 96, STG)
node("rsp_head", "<b>Respiratory head</b><br>ŷ<sub>t</sub><sup>apn</sup> = σ(W<sub>2</sub>[h<sub>t</sub> ; c<sub>t</sub>])<br>direct cardio bypass<br>41,217 p", 335, 640, 195, 96, RSP)

edge("e1", "eeg_feat", "eeg_enc"); edge("e2", "car_feat", "car_enc")
edge("e3", "eeg_enc", "fusion"); edge("e4", "car_enc", "fusion")
edge("e5", "fusion", "bilstm")
edge("e6", "bilstm", "stg_head"); edge("e7", "bilstm", "rsp_head")
edge("e8", "car_enc", "rsp_head", dashed=True, color="#C4763A")
txt("bypass_lbl", "<i>direct c<sub>t</sub></i>", 548, 430, 70, 20)

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
txt("fe_cap", "<i>EEG: 188 → 128   ·   Cardio: 14 → 64</i>", 710, 422, 230, 20)

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

# ===== objective footer =====
txt("obj", "Joint objective:   L = CE<sub>√w</sub>(ŷ<sup>stg</sup>, y<sup>stg</sup>) + "
     "λ BCE<sub>pw</sub>(ŷ<sup>apn</sup>, y<sup>apn</sup>),   λ = 1",
    280, 980, 560, 26, size=12)

xml = ('<mxfile host="app.diagrams.net">'
       '<diagram name="MM-Net architecture" id="mmnet">'
       '<mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="1040" '
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
