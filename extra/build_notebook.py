"""build_notebook.py -- generate notebooks/experiments.ipynb.

A log of the architectures we BUILT AND TRIED, with their real code, in the order we
tried them. Most of them lost. The notebook keeps the losing code because that is the
record of the search.
"""
import json, os

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").split("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n").split("\n")})

md("""
# Sleep staging on iSLEEPS — what we built, and what happened

Cohort: 99 subacute ischemic-stroke patients, 95,305 epochs, 5 AASM stages.
Protocol everywhere below: **10-fold subject-independent CV** (all epochs of a patient
sit entirely in train or test).

Published baselines to beat: **0.747 acc / 0.677 macro-F1 / 0.640 kappa**.

This notebook is the experiment log. Each section has the architecture we actually
wrote, then the number it scored. Most of these lost. They are kept because the losses
are what told us where the ceiling was coming from.
""")

code('''
import os, sys, json, glob
import numpy as np, torch, torch.nn as nn

ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
RES  = os.path.join(ROOT, "results")
for p in ("utils", "processing", "model", "extra/legacy_models"):
    sys.path.insert(0, os.path.join(ROOT, p))

def load(n):
    p = os.path.join(RES, n if n.endswith(".json") else n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

def triple(d):
    if d is None: return None
    if "mean" in d: d = d["mean"]
    a = d.get("acc", d.get("accuracy"))
    return None if a is None else (a, d.get("macro_f1", d.get("mf1")), d.get("kappa"))

def show(rows, title=""):
    if title: print(title); print("-"*len(title))
    print(f"{'configuration':40s} {'acc':>8s} {'macroF1':>9s} {'kappa':>8s}")
    for n, t in rows:
        if t is None: print(f"{n:40s} {'--':>8s} {'--':>9s} {'--':>8s}"); continue
        a,f,k = t
        f = float('nan') if f is None else f; k = float('nan') if k is None else k
        print(f"{n:40s} {a:8.4f} {f:9.4f} {k:8.4f}")

def params(m): return sum(p.numel() for p in m.parameters())
print("root:", ROOT)
''')

md("""
## 0. The data, and why accuracy is the wrong target here

N2 dominates. A model can look decent on accuracy while being useless on N1/N3, so we
track macro-F1 and kappa alongside it from the start.
""")
code('''
from datasets import DUPLICATE_DROP, CLASS_NAMES
FC = os.path.join(ROOT, "data", "featseq_cache")
subs = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC,"SN*.npz")))
subs = [s for s in subs if s not in DUPLICATE_DROP]
Y = {s: np.load(os.path.join(FC,f"SN{s}.npz"))["y"].astype(int) for s in subs}
cnt = np.bincount(np.concatenate(list(Y.values())), minlength=5)
print(f"subjects {len(subs)}   epochs {cnt.sum():,}   (SN28 dropped: duplicate of SN15)")
for c,n in zip(CLASS_NAMES, cnt): print(f"  {c:3s} {n:7,d}  {100*n/cnt.sum():5.1f}%")
print(f"\\nN2:N3 imbalance = {cnt[2]/cnt[3]:.1f} : 1")
''')

# ---------------------------------------------------------------- TRIAL 1
md("""
## Trial 1 — Raw-signal CNN (`StagingCNN`)

The obvious first deep model: a DeepSleepNet-style 1-D convolutional encoder on the raw
4-channel EEG, per-epoch classification. Large first kernel (fs/2) to capture rhythms,
then stacked small kernels.
""")
code('''
class StagingCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=5, fs=100, dropout=0.5):
        super().__init__()
        k1, s1 = fs // 2, fs // 16 or 1
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 128, k1, stride=s1, padding=k1//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(True), nn.MaxPool1d(8,8), nn.Dropout(dropout),
            nn.Conv1d(128,128,8,padding=4,bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128,128,8,padding=4,bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128,128,8,padding=4,bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.MaxPool1d(4,4), nn.Dropout(dropout))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes))
    def forward(self, x): return self.head(self.pool(self.features(x)))

m = StagingCNN(); print("out", tuple(m(torch.randn(2,4,3000)).shape), "| params", f"{params(m):,}")
show([("Trial 1: StagingCNN (raw, 4ch)", triple(load("cnn4ch_all")))])
''')
md("""
**Verdict: lost.** Per-epoch prediction ignores that sleep is a sequence, and 99 patients
is not enough to learn filters from raw signal.
""")

# ---------------------------------------------------------------- TRIAL 2
md("""
## Trial 2 — CNN + BiLSTM sequence model (`StagingSeqNet`)

Add temporal context: run the CNN encoder per epoch, then a bidirectional LSTM across a
window of epochs so the model can use sleep's transition structure.
""")
code('''
class StagingSeqNet(nn.Module):
    def __init__(self, in_ch=4, n_classes=5, hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()
        enc = StagingCNN(in_ch=in_ch, dropout=dropout)
        self.features, self.pool, self.feat_dim = enc.features, nn.AdaptiveAvgPool1d(1), 128
        self.lstm = nn.LSTM(128, hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers>1 else 0.)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden, n_classes))
    def forward(self, x):                       # x [B, L, C, T]
        B,L,C,T = x.shape
        f = self.pool(self.features(x.reshape(B*L,C,T))).flatten(1).reshape(B,L,self.feat_dim)
        h,_ = self.lstm(f)
        return self.head(h)                     # [B, L, 5]

m = StagingSeqNet(); print("out", tuple(m(torch.randn(2,20,4,3000)).shape), "| params", f"{params(m):,}")
show([("Trial 2: CNN+BiLSTM",              triple(load("seq4ch_all"))),
      ("Trial 2b: + EEG augmentation",     triple(load("seq_aug_all")))])
''')
md("""
**Verdict: better than Trial 1, still lost.** Augmentation (amplitude scaling, noise,
channel dropout, time masking) did not close the gap.
""")

# ---------------------------------------------------------------- TRIAL 3
md("""
## Trial 3 — DeepSleepNet-style dual-resolution encoder (`DeepSleepSeq`)

A faithful reimplementation of the standard architecture: two parallel convolutional
branches (small kernels for fine structure, large kernels for slow rhythms), a residual
shortcut around the BiLSTM, and two-stage training (pretrain the encoder per-epoch with
class-balanced sampling, then train the sequence model).
""")
code('''
def _bn_relu(c): return nn.Sequential(nn.BatchNorm1d(c), nn.ReLU(True))

class DualResCNN(nn.Module):
    def __init__(self, in_ch=4, fs=100, dropout=0.5, feat_pool=4):
        super().__init__()
        self.small = nn.Sequential(                       # fine temporal resolution
            nn.Conv1d(in_ch,64,fs//2,stride=fs//16 or 1,padding=fs//4,bias=False),
            _bn_relu(64), nn.MaxPool1d(8,8), nn.Dropout(dropout),
            nn.Conv1d(64,128,8,padding=4,bias=False), _bn_relu(128),
            nn.Conv1d(128,128,8,padding=4,bias=False), _bn_relu(128),
            nn.Conv1d(128,128,8,padding=4,bias=False), _bn_relu(128), nn.MaxPool1d(4,4))
        self.large = nn.Sequential(                       # coarse / low-frequency
            nn.Conv1d(in_ch,64,fs*4,stride=fs//2 or 1,padding=fs*2,bias=False),
            _bn_relu(64), nn.MaxPool1d(4,4), nn.Dropout(dropout),
            nn.Conv1d(64,128,6,padding=3,bias=False), _bn_relu(128),
            nn.Conv1d(128,128,6,padding=3,bias=False), _bn_relu(128),
            nn.Conv1d(128,128,6,padding=3,bias=False), _bn_relu(128), nn.MaxPool1d(2,2))
        self.gp = nn.AdaptiveAvgPool1d(feat_pool); self.feat_dim = 128*feat_pool*2
    def forward(self, x):
        return torch.cat([self.gp(self.small(x)).flatten(1),
                          self.gp(self.large(x)).flatten(1)], 1)

class DeepSleepSeq(nn.Module):
    def __init__(self, in_ch=4, n_classes=5, hidden=256, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.encoder = DualResCNN(in_ch=in_ch, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(self.encoder.feat_dim,256), nn.ReLU(True), nn.Dropout(dropout))
        self.lstm = nn.LSTM(256,hidden,num_layers=lstm_layers,batch_first=True,
                            bidirectional=True,dropout=dropout if lstm_layers>1 else 0.)
        self.res  = nn.Linear(256, 2*hidden)              # residual shortcut
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden, n_classes))
    def forward(self, x):
        B,L,C,T = x.shape
        f = self.proj(self.encoder(x.reshape(B*L,C,T))).reshape(B,L,-1)
        h,_ = self.lstm(f)
        return self.head(h + self.res(f))

m = DeepSleepSeq(); print("out", tuple(m(torch.randn(2,20,4,3000)).shape), "| params", f"{params(m):,}")
show([("Trial 3: DeepSleepSeq (3.7M params)", triple(load("deepsleep_all")))])
''')
md("""
**Verdict: lost, and this one was decisive.** A 3.7 M-parameter dual-resolution network
was no better than the 0.4 M CNN. That is the signature of a **data-limited** regime:
extra capacity has nothing to feed on. From here we stopped scaling models.
""")

# ---------------------------------------------------------------- TRIAL 4
md("""
## Trial 4 — Transfer learning from healthy sleep (Sleep-EDF)

If in-domain data is the constraint, borrow data. We pretrained the sequence model on
healthy Sleep-EDF recordings and fine-tuned on the stroke cohort.
""")
code('''
show([("Trial 4: Sleep-EDF pretrain -> finetune", triple((load("transfer_all") or {}).get("transfer_hmm"))),
      ("   (no HMM)",                             triple((load("transfer_all") or {}).get("transfer")))])
d = load("domaingap")
if d:
    print("\\nWhy it failed - zero-shot healthy->stroke, per-stage recall:")
    print(f"{'':10s}" + "".join(f"{c:>8s}" for c in CLASS_NAMES))
    for k in ("healthy","stroke"):
        print(f"{k:10s}" + "".join(f"{d[k]['recall'][c]:8.2f}" for c in CLASS_NAMES))
    print(f"\\naccuracy {d['healthy']['acc']:.3f} -> {d['stroke']['acc']:.3f}  "
          f"({d['stroke']['acc']-d['healthy']['acc']:+.3f})")
''')
md("""
**Verdict: worst result of all (0.623).** The per-stage breakdown shows why: REM recall
collapses 0.90 → 0.14 and N3 0.84 → 0.39, while N2 survives. Healthy sleep is not a
useful prior for the stages stroke actually disturbs. Part of the REM drop is also a
montage confound (frontal vs central), which we report as a limitation.
""")

# ---------------------------------------------------------------- TRIAL 5
md("""
## Trial 5 — Sequence model over engineered features (`FeatSeqNet`)

Maybe the problem is raw signal, not depth. Keep the BiLSTM but feed it the 188
engineered physiological features per epoch instead of the waveform.
""")
code('''
class FeatSeqNet(nn.Module):
    """BiLSTM over per-epoch engineered features (no raw signal)."""
    def __init__(self, n_feat=188, n_classes=5, hidden=128, layers=2, dropout=0.5):
        super().__init__()
        self.enc  = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat,256),
                                  nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(256, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers>1 else 0.)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden, n_classes))
    def forward(self, x):                        # [B, L, 188]
        h,_ = self.lstm(self.enc(x)); return self.head(h)

m = FeatSeqNet(); print("out", tuple(m(torch.randn(2,25,188)).shape), "| params", f"{params(m):,}")
show([("Trial 5: FeatSeqNet (BiLSTM on features)", triple(load("featseq_all")))])
''')
md("""
**Verdict: best deep model so far (0.676), still short.** Features beat raw signal, which
is consistent with the data-limited diagnosis, but the sequence model still cannot match
gradient boosting on the same inputs.
""")

# ---------------------------------------------------------------- TRIAL 6
md("""
## Trial 6 — Asymmetry graph + state-space over raw signal (`AsymGraphSSM`)

First attempt at the clinical idea: stroke is lateralized, so model the montage as a
graph and pool the *signed difference* of homologous derivations (C4↔C3, O2↔O1), then
run a Mamba-style selective state-space model across epochs.
""")
code('''
HOMOLOGOUS = [(0,1),(2,3)]                    # C4<->C3, O2<->O1
EDGES = [(0,1),(2,3),(0,2),(1,3),(4,5),(0,4),(1,5),(2,4),(3,5),(6,4),(6,5)]

def build_adj(n=7):
    A = torch.eye(n)
    for i,j in EDGES: A[i,j] = 1; A[j,i] = 1
    return A

class GraphAttention(nn.Module):
    """GAT-style attention over the fixed montage graph."""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d,d,bias=False); self.a = nn.Linear(2*d,1,bias=False)
        self.register_buffer("adj", build_adj()); self.leaky = nn.LeakyReLU(0.2)
    def forward(self, h):                                    # [B, 7, d]
        B,C,d = h.shape; Wh = self.W(h)
        e = self.leaky(self.a(torch.cat([Wh.unsqueeze(2).expand(B,C,C,d),
                                         Wh.unsqueeze(1).expand(B,C,C,d)],-1)).squeeze(-1))
        e = e.masked_fill(~(self.adj>0).unsqueeze(0), float("-inf"))
        return torch.nn.functional.elu(torch.einsum("bij,bjd->bid", torch.softmax(e,-1), Wh)) + h

class AsymPool(nn.Module):
    """The clinical prior: signed difference between homologous derivations."""
    def forward(self, h):                                    # [B,7,d] -> [B,2d]
        return torch.cat([(h[:,i]-h[:,j]).abs() for i,j in HOMOLOGOUS], -1)

g, ap = GraphAttention(48), AsymPool()
z = torch.randn(4,7,48)
print("graph out", tuple(g(z).shape), "| asym out", tuple(ap(g(z)).shape))
show([("Trial 6: AsymGraphSSM (raw signal)", triple((load("kags_all_v1") or {}).get("kags_hmm")))])
''')
md("""
**Verdict: lost (0.715).** The asymmetry idea is clinically right but learning it from raw
signal on 99 patients does not work. This is the trial that reshaped the design: keep the
asymmetry graph, but stop feeding it raw waveforms.
""")

# ---------------------------------------------------------------- TRIAL 7
md("""
## Trial 7 — Connectivity features (inter-hemispheric coherence + PLV)

If asymmetry matters, measure it explicitly: magnitude-squared coherence and sigma-band
phase-locking value between homologous pairs, appended to the feature set (v3).
""")
code('''
from scipy.signal import butter, filtfilt, coherence, hilbert

def plv(a, b, lo, hi, fs=100):
    bb, aa = butter(4, [lo/(fs/2), hi/(fs/2)], btype="band")
    pa = np.angle(hilbert(filtfilt(bb, aa, a)))
    pb = np.angle(hilbert(filtfilt(bb, aa, b)))
    return float(np.abs(np.mean(np.exp(1j*(pa-pb)))))

rng = np.random.RandomState(0); a, b = rng.randn(3000), rng.randn(3000)
f, Cxy = coherence(a, b, fs=100, nperseg=256)
print(f"demo: sigma-band PLV(C4,C3) = {plv(a,b,12,16):.3f} | mean coherence = {Cxy.mean():.3f}")

show([("v2 features (no connectivity)", triple((load("ensemble7_v2_all") or {}).get("ensemble_hmm"))),
      ("v3 features (+ connectivity)",  triple((load("ensemble7_v3_all") or {}).get("ensemble_hmm")))],
     "Does explicit connectivity help?")
''')
md("""
**Verdict: null.** v3 is identical to v2 to three decimals. Explicit connectivity added
dimensionality and no information.
""")

# ---------------------------------------------------------------- TRIAL 8
md("""
## Trial 8 — Stacking and sequence refinement

Two more combination attempts: a meta-learner stacked on the base models' posteriors, and
a sequence refiner trained to correct the ensemble's output.
""")
code('''
show([("Trial 8a: stacking meta-learner", triple((load("stack_all") or {}).get("stack_meta_hmm"))),
      ("Trial 8b: stacked average",       triple((load("stack_all") or {}).get("stack_avg"))),
      ("Trial 8c: sequence refiner",      triple((load("refine_all") or {}).get("refiner_hmm")))],
     "Combination attempts")
''')
md("""
**Verdict: no reliable gain.** Combining weak models with a strong one drags the strong one
down.
""")

# ---------------------------------------------------------------- THE SEARCH
md("""
## Trial 9 — A systematic sweep for anything left

At this point every architecture had lost to gradient boosting, so we stopped changing the
model and swept the things around it. Each lever was tuned on **held-out validation
subjects inside each fold** and applied unchanged to that fold's test subjects.
""")
code('''
d = load("postproc_search")
if d: show([(k,(v["acc"],v["mf1"],v["kappa"])) for k,v in d.items()],
           "A. Post-hoc decoding (no retraining)")
print("\\nprior correction returned exactly 0 (alpha=0 chosen in all 10 folds);")
print("duration-aware HSMM and stacking were WORSE than plain Viterbi.")
''')
code('''
k = load("k_compare")
if k:
    a,b = k["k3"]["hmm"], k["k5"]["hmm"]
    show([("context +-3 epochs", (a["acc"],a["mf1"],a["kappa"])),
          ("context +-5 epochs", (b["acc"],b["mf1"],b["kappa"]))],
         "B. Temporal context width (full 10-fold)")
    print(f"\\ndelta: acc {b['acc']-a['acc']:+.4f}   mF1 {b['mf1']-a['mf1']:+.4f}")
    print("k=5 won on a cheap 3-fold probe and LOST at full scale. Probes lie.")

n = load("n1_bias")
if n:
    B,T = n["baseline"], n["tuned"]
    show([("baseline",              (B["acc"],B["mf1"],B["kappa"])),
          ("tuned per-class bias",  (T["acc"],T["mf1"],T["kappa"]))],
         "\\nC. Attacking N1 (our worst class) with a tuned decision bias")
    print(f"\\nN1 F1 {B['pcf'][1]:.3f} -> {T['pcf'][1]:.3f} ({T['pcf'][1]-B['pcf'][1]:+.3f}), "
          f"but accuracy {T['acc']-B['acc']:+.4f}")
    print("N1 gains are paid for one-for-one elsewhere -> N1 is an INFORMATION problem,")
    print("not a threshold problem. That closes the whole reweighting family.")
''')

# ---------------------------------------------------------------- WHAT WORKED
md("""
## What actually worked

Not a deeper network. Physiological features + gradient boosting + temporal decoding,
with the montage widened to include EOG and EMG so REM and atonia are visible at all.
""")
code('''
show([("4 EEG channels only",                triple((load("ensemble7_all") or {}).get("ensemble_hmm"))),
      ("7 channels + event features",        triple((load("ensemble7_v2_all") or {}).get("ensemble_hmm"))),
      ("   same, without HMM decoding",      triple((load("ensemble7_v2_all") or {}).get("ensemble"))),
      ("--- published LSTM baseline ---",    (0.747, 0.677, 0.640))],
     "Final")
v2 = load("ensemble7_v2_all")
if v2:
    print("\\nper-class F1:", " ".join(f"{c}={v:.3f}" for c,v in zip(CLASS_NAMES, v2["ensemble_hmm"]["per_class_f1"])))
''')
md("""
### Honest reading

Accuracy **0.7464 vs 0.747 is a tie**, not a win: the gap is 0.0006 against a fold standard
deviation of ±0.016. Macro-F1 and kappa differences sit inside the same noise band.

What the system adds that the baselines do not: calibrated uncertainty (conformal coverage
0.900, ECE 0.038), a lesion-severity biomarker (spindle asymmetry vs NIHSS, ρ = 0.41,
p = 0.006), per-patient consistency, and this documented negative search.

The conclusion the failures point to, together: on 99 patients the binding constraint is
**information, not model capacity**. Nine architectures and a nine-lever sweep all land in
the same place.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/experiments.ipynb", "w", encoding="utf-8"), indent=1)
print(f"wrote notebooks/experiments.ipynb ({len(cells)} cells, "
      f"{sum(c['cell_type']=='code' for c in cells)} code)")
