"""MM-Net locked configuration — model, data loader and ten-fold engine.

Extracted verbatim from `1_MM_Net_reproduction.ipynb` (the notebook that produced
the published numbers) so that later experiments train identical code rather than
a re-implementation. The only edit is the data path, which is resolved against the
repository root instead of the caller's working directory.

Reproduces: staging acc 0.7227 +- 0.039, macro-F1 0.6510, kappa 0.6106;
respiratory AUC 0.7111 +- 0.034, AP 0.3367.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))          # -> D:\sleep-staging-psg
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")



# ----------------------------------------------------------------------
# from notebook cell 2
# ------------------------------------------------------------------


import os, sys, glob, json, re, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "preprocessing"))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "utils"))
import torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, NC, EPS = 20, 5, 1e-12
CLS = ["W", "N1", "N2", "N3", "R"]
print("cwd:", os.getcwd(), "| device:", DEV,
      "|", torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU")


# ----------------------------------------------------------------------
# from notebook cell 4
# ------------------------------------------------------------------


FE = os.path.join(REPO, "data", "mm_features")
DUP = {28}
def load_data():
    data = {}
    for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUP: continue
        d = np.load(f)
        Fe = np.nan_to_num(d["Feeg"]).astype(np.float32); Fc = np.nan_to_num(d["Fcard"]).astype(np.float32)
        Fe = (Fe - Fe.mean(0)) / (Fe.std(0) + 1e-6); Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-6)
        data[sid] = (Fe, Fc, d["y"].astype(np.int64), d["apnea"].astype(np.int64))
    return data
DATA = load_data(); SUBS = sorted(DATA)
def make_folds(subs, k=10, seed=42):
    r = np.random.RandomState(seed); s = list(subs); r.shuffle(s)
    folds = [s[i::k] for i in range(k)]
    return [([x for j, f in enumerate(folds) if j != i for x in f], folds[i]) for i in range(k)]
FOLDS = make_folds(SUBS)
n_ep = sum(len(DATA[s][2]) for s in SUBS)
sc = np.bincount(np.concatenate([DATA[s][2] for s in SUBS]), minlength=5)
prev = np.concatenate([DATA[s][3] for s in SUBS]).mean()
print(f"subjects: {len(SUBS)} (SN28 dropped) | epochs: {n_ep:,}")
print("stage %:", {c: round(100*sc[i]/n_ep, 1) for i, c in enumerate(CLS)})
print(f"respiratory-event prevalence: {100*prev:.1f}%")


# ----------------------------------------------------------------------
# from notebook cell 6 (modality masks)
# ------------------------------------------------------------------

from features_v2 import extract_features_v2
_, FEAT_NAMES = extract_features_v2(np.zeros((1, 7, 3000), np.float32))
def eeg_masks():
    eeg, eog, emg = (np.zeros(188, bool) for _ in range(3))
    for i, nm in enumerate(FEAT_NAMES):
        if any(t in nm for t in ("_c0", "_c1", "_c2", "_c3", "spindle", "sw_")): eeg[i] = True
        elif ("_c4" in nm) or ("_c5" in nm) or nm.startswith("eog"): eog[i] = True
        elif ("_c6" in nm) or nm.startswith("emg"): emg[i] = True
    return {"eeg": eeg, "eog": eog, "emg": emg}
EEGM = eeg_masks()
CARD_GROUPS = {"spo2": [0,1,2,3], "pulse_hrv": [4,5], "ecg": [6,7], "airflow": [8,9], "effort": [10,11,12,13]}
print("EEG-feature counts -> EEG:", EEGM["eeg"].sum(), "EOG:", EEGM["eog"].sum(), "EMG:", EEGM["emg"].sum())

# ----------------------------------------------------------------------
# from notebook cell 8
# ------------------------------------------------------------------


class FeatMLP(nn.Module):
    def __init__(self, fin, d, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
    def forward(self, x): return self.net(x)

class CrossFusion(nn.Module):
    def __init__(self, d, heads=4, drop=0.3):
        super().__init__()
        self.mtype = nn.Parameter(torch.randn(2, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Dropout(drop))
    def forward(self, e, c):
        tok = torch.stack([e, c], 1) + self.mtype[None]
        a, _ = self.attn(tok, tok, tok)
        a = self.norm(tok + a)
        return self.ff(a.reshape(a.size(0), -1))

class MMFeatureNet(nn.Module):
    def __init__(self, n_eeg=188, n_card=14, d=128, d_card=64, hidden=128, layers=2,
                 n_cls=5, drop=0.3, fusion="concat"):
        super().__init__()
        self.fusion = fusion
        self.eeg_enc = FeatMLP(n_eeg, d, drop); self.card_enc = FeatMLP(n_card, d_card, drop)
        if fusion == "cross":
            self.card_proj = nn.Linear(d_card, d); self.fuse = CrossFusion(d, drop=drop)
        elif fusion == "concat":
            self.fuse = nn.Sequential(nn.Linear(d + d_card, d), nn.GELU(), nn.Dropout(drop))
        else:
            self.fuse = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop))
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=drop if layers > 1 else 0.0)
        self.uses_cardio = fusion in ("cross", "concat")
        self.stage_head = nn.Linear(2*hidden, n_cls)
        self.apnea_head = nn.Sequential(nn.Linear(2*hidden + d_card, hidden), nn.GELU(),
                                        nn.Dropout(drop), nn.Linear(hidden, 1))
    def forward(self, feeg, fcard):
        B, Ln = feeg.shape[:2]
        e = self.eeg_enc(feeg.reshape(B*Ln, -1)); c = self.card_enc(fcard.reshape(B*Ln, -1))
        if self.fusion == "cross": fz = self.fuse(e, self.card_proj(c))
        elif self.fusion == "concat": fz = self.fuse(torch.cat([e, c], -1))
        else: fz = self.fuse(e)
        h, _ = self.lstm(fz.reshape(B, Ln, -1))
        c_seq = c.reshape(B, Ln, -1)
        if not self.uses_cardio: c_seq = torch.zeros_like(c_seq)
        return self.stage_head(h), self.apnea_head(torch.cat([h, c_seq], -1)).squeeze(-1)

print("parameters (concat):", f"{sum(p.numel() for p in MMFeatureNet(fusion='concat').parameters()):,}")


# ----------------------------------------------------------------------
# from notebook cell 10
# ------------------------------------------------------------------


def apply_drop(Fe, Fc, eeg_drop, card_drop):
    Fe, Fc = Fe.copy(), Fc.copy()
    for m in eeg_drop:
        Fe[:, EEGM[m]] = 0.0
    for g in card_drop:
        if g == "all":
            Fc[:] = 0.0
        else:
            Fc[:, CARD_GROUPS[g]] = 0.0
    return Fe, Fc

def windows(subs, stride, eeg_drop, card_drop):
    Fe, Fc, Y, A, Mk = [], [], [], [], []
    for s in subs:
        fe, fc, y, a = DATA[s]; fe, fc = apply_drop(fe, fc, eeg_drop, card_drop); n = len(y)
        for st in range(0, max(1, n - L + 1), stride):
            e, c, yy, aa = fe[st:st+L], fc[st:st+L], y[st:st+L], a[st:st+L]
            m = np.ones(len(yy), np.float32)
            if len(yy) < L:
                k = L - len(yy)
                e = np.concatenate([e, np.zeros((k,188),np.float32)]); c = np.concatenate([c, np.zeros((k,14),np.float32)])
                yy = np.concatenate([yy, np.zeros(k,np.int64)]); aa = np.concatenate([aa, np.zeros(k,np.int64)]); m = np.concatenate([m, np.zeros(k,np.float32)])
            Fe.append(e); Fc.append(c); Y.append(yy); A.append(aa); Mk.append(m)
    t = lambda a, d: torch.tensor(np.asarray(a), dtype=d, device=DEV)
    return t(Fe,torch.float32), t(Fc,torch.float32), t(Y,torch.long), t(A,torch.float32), t(Mk,torch.float32)

def sqrt_cw(tr):
    cc = np.zeros(5, np.int64)
    for s in tr: cc += np.bincount(DATA[s][2], minlength=5)
    inv = np.sqrt(cc.sum() / (5*np.maximum(cc,1))); return torch.tensor(inv/inv.mean(), dtype=torch.float32, device=DEV)

def hmm(A_log, pi_log, logp):
    T = logp.shape[0]; dp = np.zeros((T,NC)); bp = np.zeros((T,NC),int); dp[0] = pi_log + logp[0]
    for t in range(1,T):
        sc = dp[t-1][:,None] + A_log; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + logp[t]
    p = np.zeros(T,int); p[-1] = dp[-1].argmax()
    for t in range(T-2,-1,-1): p[t] = bp[t+1,p[t+1]]
    return p

def train_fold(tr, va, fusion, eeg_drop, card_drop, epochs=45, patience=8, bs=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    Fe,Fc,Y,A,Mk = windows(tr, L//2, eeg_drop, card_drop); N = Fe.shape[0]
    model = MMFeatureNet(fusion=fusion).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=sqrt_cw(tr), reduction="none")
    ac = np.zeros(2,np.int64)
    for s in tr: ac += np.bincount(DATA[s][3], minlength=2)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([ac[0]/max(1,ac[1])],dtype=torch.float32,device=DEV))
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(N, device=DEV)
        for i in range(0, N-bs+1, bs):
            idx = perm[i:i+bs]; s_o, a_o = model(Fe[idx], Fc[idx]); m = Mk[idx].reshape(-1)
            ls = (ce(s_o.reshape(-1,5), Y[idx].reshape(-1))*m).sum()/m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), A[idx].reshape(-1))*m).sum()/m.sum().clamp(min=1)
            opt.zero_grad(); (ls+la).backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        sch.step()
        acc = np.mean([(subj_infer(model,s,eeg_drop,card_drop)[0].argmax(1)==DATA[s][2]).mean() for s in va])
        if acc > best: best, bad, best_state = acc, 0, {k:v.detach().clone() for k,v in model.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    model.load_state_dict(best_state); return model

@torch.no_grad()
def subj_infer(model, s, eeg_drop, card_drop):
    model.eval(); fe, fc, y, a = DATA[s]; fe, fc = apply_drop(fe, fc, eeg_drop, card_drop)
    n = len(y); pad = (-n) % L
    if pad: fe = np.concatenate([fe, np.zeros((pad,188),np.float32)]); fc = np.concatenate([fc, np.zeros((pad,14),np.float32)])
    fe = torch.tensor(fe.reshape(-1,L,188),device=DEV); fc = torch.tensor(fc.reshape(-1,L,14),device=DEV)
    so, ao = [], []
    for i in range(0,len(fe),32):
        s_o, a_o = model(fe[i:i+32], fc[i:i+32])
        so.append(s_o.softmax(-1).reshape(-1,5).cpu().numpy()); ao.append(torch.sigmoid(a_o).reshape(-1).cpu().numpy())
    return np.concatenate(so)[:n], np.concatenate(ao)[:n]


@torch.no_grad()
def infer_arrays(model, fe, fc, n):
    """Same forward pass as subj_infer, but on caller-supplied feature arrays.

    Used for permutation importance, where a modality's columns are shuffled across
    epochs before inference. Keeping this separate from subj_infer means the published
    inference path is untouched.
    """
    model.eval()
    fe = np.asarray(fe, np.float32).copy()
    fc = np.asarray(fc, np.float32).copy()
    pad = (-n) % L
    if pad:
        fe = np.concatenate([fe, np.zeros((pad, 188), np.float32)])
        fc = np.concatenate([fc, np.zeros((pad, 14), np.float32)])
    fe = torch.tensor(fe.reshape(-1, L, 188), device=DEV)
    fc = torch.tensor(fc.reshape(-1, L, 14), device=DEV)
    so, ao = [], []
    for i in range(0, len(fe), 32):
        s_o, a_o = model(fe[i:i + 32], fc[i:i + 32])
        so.append(s_o.softmax(-1).reshape(-1, 5).cpu().numpy())
        ao.append(torch.sigmoid(a_o).reshape(-1).cpu().numpy())
    return np.concatenate(so)[:n], np.concatenate(ao)[:n]

@torch.no_grad()
def subj_embed(model, s):
    model.eval(); fe, fc, y, a = DATA[s]; n = len(y); pad = (-n) % L
    if pad: fe = np.concatenate([fe, np.zeros((pad,188),np.float32)]); fc = np.concatenate([fc, np.zeros((pad,14),np.float32)])
    fe = torch.tensor(fe.reshape(-1,L,188),device=DEV); fc = torch.tensor(fc.reshape(-1,L,14),device=DEV)
    cap = {}; hk = model.lstm.register_forward_hook(lambda m,i,o: cap.__setitem__("h", o[0].detach()))
    out = []
    for i in range(0,len(fe),32):
        model(fe[i:i+32], fc[i:i+32]); out.append(cap["h"].reshape(-1, cap["h"].shape[-1]).cpu().numpy())
    hk.remove(); return np.concatenate(out)[:n]

def run_10fold(fusion="concat", eeg_drop=(), card_drop=(), keep=False, seed=42):
    per_fold = []; ps = {}; E_h=[]; E_y=[]; E_a=[]; P=[[],[],[],[]]; m0=[None]
    for fi,(tr_all,te) in enumerate(FOLDS):
        rng = np.random.RandomState(100+fi); tr_all=list(tr_all); rng.shuffle(tr_all)
        nv = max(10, len(tr_all)//9); va, tr = tr_all[:nv], tr_all[nv:]
        Am = np.ones((NC,NC)); pi = np.ones(NC)
        for s in tr:
            y = DATA[s][2]; pi[y[0]] += 1
            for x,z in zip(y[:-1],y[1:]): Am[x,z]+=1
        A_log = np.log(Am/Am.sum(1,keepdims=True)); pi_log = np.log(pi/pi.sum())
        model = train_fold(tr, va, fusion, list(eeg_drop), list(card_drop), seed=seed)
        if keep and fi==0: m0[0]=model
        yt,ph,ay,ap = [],[],[],[]
        for s in te:
            sp,apn = subj_infer(model,s,list(eeg_drop),list(card_drop)); y=DATA[s][2]
            pred = hmm(A_log,pi_log,np.log(sp+EPS))
            ps[f"SN{s}"]=dict(acc=float(accuracy_score(y,pred)),kappa=float(cohen_kappa_score(y,pred)),apnea=apn.tolist())
            yt.append(y); ph.append(pred); ay.append(DATA[s][3]); ap.append(apn)
            if keep: E_h.append(subj_embed(model,s)); E_y.append(y); E_a.append(DATA[s][3])
        yt,ph = np.concatenate(yt),np.concatenate(ph); ay,ap = np.concatenate(ay),np.concatenate(ap)
        if keep:
            P[0].append(yt); P[1].append(ph); P[2].append(ay); P[3].append(ap)
        per_fold.append(dict(acc=accuracy_score(yt,ph),mf1=f1_score(yt,ph,average='macro',zero_division=0),
            kappa=cohen_kappa_score(yt,ph),pcf=f1_score(yt,ph,average=None,labels=range(5),zero_division=0).tolist(),
            auc=roc_auc_score(ay,ap) if len(np.unique(ay))>1 else np.nan,
            ap=average_precision_score(ay,ap) if len(np.unique(ay))>1 else np.nan))
    ag = lambda k: (float(np.mean([f[k] for f in per_fold])), float(np.std([f[k] for f in per_fold])))
    out = dict(per_fold=per_fold, per_subject=ps, acc=ag('acc'), mf1=ag('mf1'), kappa=ag('kappa'),
               auc=ag('auc'), ap=ag('ap'), pcf=np.mean([f['pcf'] for f in per_fold],0).tolist())
    if keep:
        out["emb"]=(np.concatenate(E_h),np.concatenate(E_y),np.concatenate(E_a))
        out["pred"]=tuple(np.concatenate(P[i]) for i in range(4)); out["model0"]=m0[0]
    return out
print("training utilities defined.")
