"""Build a real editable draw.io file of the MM-Net architecture in the target style, AND a
matplotlib preview from the SAME layout coordinates (so the layout can be verified here, since
draw.io cannot be rendered in this environment). Mini-plots are embedded as base64 PNGs.
Outputs: mm_architecture_v2.drawio  +  mm_architecture_v2_preview.png"""
import os, base64, html
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(HERE, "arch_assets")
def b64(name):
    with open(os.path.join(A, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

# ---- shared layout model: every element is a dict; both renderers read this ----
BOX, IMG, TXT, ARR = "box", "img", "txt", "arr"
E = []
def box(id, x, y, w, h, text, fill="#ffffff", stroke="#333333", fs=12, bold=True, dashed=False):
    E.append(dict(t=BOX, id=id, x=x, y=y, w=w, h=h, text=text, fill=fill, stroke=stroke, fs=fs, bold=bold, dashed=dashed)); return id
def img(x, y, w, h, name):
    E.append(dict(t=IMG, x=x, y=y, w=w, h=h, name=name))
def txt(x, y, w, h, text, fs=11, bold=False, color="#000000"):
    E.append(dict(t=TXT, x=x, y=y, w=w, h=h, text=text, fs=fs, bold=bold, color=color))
def arr(src, dst, color="#333333", dashed=False, lab=""):
    E.append(dict(t=ARR, src=src, dst=dst, color=color, dashed=dashed, lab=lab))

GREEN, ORANGE, BLUE, YELLOW, PURPLE, RED, GREY, LGREEN, LBLUE = \
    "#e6f4ea", "#fdecdd", "#e7effb", "#fdf6da", "#efe7fb", "#fde7e7", "#f4f5f7", "#d7ecdd", "#dbe6f7"

# EEG stream
box("eegrec", 40, 60, 210, 150, "EEG Recordings", GREEN, "#5a8f66"); img(54, 92, 182, 108, "eeg_traces.png")
box("eegfe", 290, 60, 220, 150, "EEG Feature Extraction\n(band power, spindle, Hjorth)", GREEN, "#5a8f66", 11); img(320, 112, 160, 90, "bandpower.png")
box("eegenc", 550, 80, 150, 110, "EEG Encoder\nphi_eeg (FeatMLP)\n188 to 128", GREEN, "#5a8f66", 11)
txt(705, 120, 90, 20, "z_e in R^128", 11, True, "#2f6f43")

# Cardio stream
box("carsig", 40, 270, 210, 150, "Cardio Signals", ORANGE, "#c9803a"); img(54, 302, 182, 108, "cardio_signals.png")
box("carfe", 290, 270, 220, 150, "Cardio Feature Extraction\n(SpO2, effort, HRV, airflow)", ORANGE, "#c9803a", 11); img(330, 322, 140, 92, "hrv.png")
box("carenc", 550, 290, 150, 110, "Cardio Encoder\nphi_car (FeatMLP)\n14 to 64", ORANGE, "#c9803a", 11)
txt(705, 330, 90, 20, "z_c in R^64", 11, True, "#a85f22")

# Cross-modal fusion
box("fus", 800, 150, 240, 200, "Cross-modal Fusion\n(2 tokens + attention)\nz in R^128", BLUE, "#3a6ea5", 12)
for k in range(4): img(822 + k*54, 250, 46, 46, f"attn_head{k+1}.png")
txt(822, 300, 200, 16, "head 1    head 2    head 3    head 4", 8, False, "#3a6ea5")

# BiLSTM
box("lstm", 300, 470, 500, 110, "BiLSTM  (2 layers, bidirectional)\ncontext L = 20 epochs   -   h_t in R^256", YELLOW, "#b6902a", 12)
txt(320, 520, 460, 40, "-> LSTM -> LSTM -> LSTM -> LSTM ->  ...\n<- LSTM <- LSTM <- LSTM <- LSTM <-  ...", 10, False, "#7a6010")

# heads
box("stage", 250, 640, 300, 170, "Staging Head\ny_stg = softmax(W_s h_t) + HMM Viterbi", PURPLE, "#6b46c1", 12); img(272, 704, 256, 92, "hypnogram_mini.png")
box("resp", 600, 640, 300, 170, "Respiratory Head\ny_apn = sigmoid(W_r [h_t ; c_t])  (direct cardio)", RED, "#c53030", 12); img(628, 714, 246, 78, "apnea_wave.png")

# joint objective
box("obj", 320, 840, 560, 40, "Joint objective:  L = CE_stg(y_stg, target) + lambda BCE_apn(y_apn, target),  lambda = 1", "#ffffff", "#333333", 12, True, True)

# detail panel 1 (FeatMLP)
box("fmlp", 1080, 60, 210, 300, "Feature Encoder (FeatMLP)\ne = GELU(LN(W2 GELU(LN(W1 x))))", GREEN, "#5a8f66", 11)
for j, t in enumerate(["Linear W1", "LayerNorm", "GELU", "Dropout (0.3)", "Linear W2", "LayerNorm", "GELU"]):
    box(f"fm{j}", 1110, 112 + j*32, 150, 26, t, LGREEN, "#5a8f66", 10)
txt(1080, 332, 210, 18, "EEG: 188 to 128  -  Cardio: 14 to 64", 9, False, "#2f6f43")

# detail panel 2 (attention)
box("attn", 1080, 390, 210, 290, "Cross-modal Fusion (Attention)", BLUE, "#3a6ea5", 11)
for j, t in enumerate(["Tokenize T = [W_e z_e ; W_c z_c] + M", "Compute Q, K, V", "4 heads - concat - W_o", "Residual + LayerNorm", "Feed-Forward W_f", "fuse -> z in R^128"]):
    box(f"at{j}", 1098, 422 + j*40, 174, 30, t, LBLUE, "#3a6ea5", 9)

# legend
box("leg", 40, 900, 1000, 30, "Data flow (arrow)   -   Direct cardio bypass (dashed)   -   Residual (+)   -   Multi-head attention   -   Forward/Backward LSTM", "#ffffff", "#999999", 10, False)

# arrows
arr("eegrec", "eegfe"); arr("eegfe", "eegenc"); arr("carsig", "carfe"); arr("carfe", "carenc")
arr("eegenc", "fus"); arr("carenc", "fus"); arr("fus", "lstm"); arr("lstm", "stage"); arr("lstm", "resp")
arr("carenc", "resp", "#c9803a", True, "direct c_t")

# ---------- emit draw.io ----------
def esc(s): return html.escape(s).replace("\n", "<br>")
cells = []; _n = [0]
def nid(): _n[0]+=1; return f"c{_n[0]}"
byid = {}
for e in E:
    if e["t"] == BOX:
        i = e["id"]; byid[i] = e
        st = (f"rounded={0 if e['id'] in ('obj','leg') else 1};whiteSpace=wrap;html=1;fillColor={e['fill']};"
              f"strokeColor={e['stroke']};fontSize={e['fs']};fontStyle={'1' if e['bold'] else '0'};align=center;"
              f"verticalAlign=top;spacingTop=4;arcSize=8;{'dashed=1;' if e['dashed'] else ''}")
        cells.append(f'<mxCell id="{i}" value="{esc(e["text"])}" style="{st}" vertex="1" parent="1"><mxGeometry x="{e["x"]}" y="{e["y"]}" width="{e["w"]}" height="{e["h"]}" as="geometry"/></mxCell>')
    elif e["t"] == IMG:
        i = nid()
        st = f"shape=image;imageAspect=0;aspect=fixed;image={b64(e['name'])};"
        cells.append(f'<mxCell id="{i}" value="" style="{st}" vertex="1" parent="1"><mxGeometry x="{e["x"]}" y="{e["y"]}" width="{e["w"]}" height="{e["h"]}" as="geometry"/></mxCell>')
    elif e["t"] == TXT:
        i = nid()
        st = f"text;html=1;align=center;verticalAlign=middle;fontSize={e['fs']};fontStyle={'1' if e['bold'] else '0'};fontColor={e['color']};"
        cells.append(f'<mxCell id="{i}" value="{esc(e["text"])}" style="{st}" vertex="1" parent="1"><mxGeometry x="{e["x"]}" y="{e["y"]}" width="{e["w"]}" height="{e["h"]}" as="geometry"/></mxCell>')
    elif e["t"] == ARR:
        i = nid()
        st = f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={e['color']};endArrow=block;endFill=1;{'dashed=1;' if e['dashed'] else ''}"
        cells.append(f'<mxCell id="{i}" value="{esc(e["lab"])}" style="{st}" edge="1" parent="1" source="{e["src"]}" target="{e["dst"]}"><mxGeometry relative="1" as="geometry"/></mxCell>')
xml = ('<mxfile host="app.diagrams.net"><diagram name="MM-Net architecture" id="mmnetv2">'
       '<mxGraphModel dx="1400" dy="1000" grid="1" gridSize="10" guides="1" page="1" pageWidth="1360" pageHeight="980" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(os.path.join(HERE, "mm_architecture_v2.drawio"), "w", encoding="utf-8").write(xml)

# ---------- emit matplotlib preview from the SAME coords ----------
W, H = 1360, 980
fig, ax = plt.subplots(figsize=(13.6, 9.8)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis(); ax.axis("off")
for e in E:
    if e["t"] == BOX:
        r = mpatches.FancyBboxPatch((e["x"], e["y"]), e["w"], e["h"],
            boxstyle="round,pad=0,rounding_size=8", linewidth=1.2, edgecolor=e["stroke"], facecolor=e["fill"],
            linestyle="--" if e["dashed"] else "-")
        ax.add_patch(r)
        ax.text(e["x"]+e["w"]/2, e["y"]+6, e["text"].replace("\n", "\n"), ha="center", va="top",
                fontsize=e["fs"]*0.72, fontweight="bold" if e["bold"] else "normal", color=e["stroke"])
    elif e["t"] == TXT:
        ax.text(e["x"]+e["w"]/2, e["y"]+e["h"]/2, e["text"], ha="center", va="center",
                fontsize=e["fs"]*0.72, fontweight="bold" if e["bold"] else "normal", color=e["color"])
    elif e["t"] == IMG:
        im = mpimg.imread(os.path.join(A, e["name"]))
        ax.imshow(im, extent=(e["x"], e["x"]+e["w"], e["y"]+e["h"], e["y"]), aspect="auto", zorder=5)
def center(i): b = byid[i]; return (b["x"]+b["w"]/2, b["y"]+b["h"]/2)
for e in E:
    if e["t"] == ARR:
        (x0, y0), (x1, y1) = center(e["src"]), center(e["dst"])
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                     color=e["color"], linestyle="--" if e["dashed"] else "-", lw=1.2, zorder=1,
                     connectionstyle="arc3,rad=0.0"))
fig.savefig(os.path.join(HERE, "mm_architecture_v2_preview.png"), dpi=110, bbox_inches="tight")
print("wrote mm_architecture_v2.drawio + mm_architecture_v2_preview.png |", len([e for e in E if e['t']==BOX]), "boxes")
