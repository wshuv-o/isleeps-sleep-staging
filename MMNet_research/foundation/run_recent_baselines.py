"""Three recent stagers on iSLEEPS and on healthy sleep, on the standard folds.

Micro SleepNet (2023), 1D-ResNet-SE-LSTM (2023) and AT-BiLSTM (2024) appear in
the literature table with a dash in the iSLEEPS column because none had been run.
They extend the comparison past 2022, which is where the reimplemented set
currently stops.

All three are reimplementations from their published descriptions, not the
authors' code, and the table marks them the same way U-Time, TinySleepNet and
SleepTransformer are marked. Each is run twice, on iSLEEPS ten-fold
patient-independent and on Sleep-EDF ten-fold subject-independent, because a
score on this cohort means nothing until the same code is shown to reach the
published figure on the corpus its authors used.

  micro       depthwise-separable convolutional stack with channel attention,
              per-epoch, no sequence model. Published ~0.833 on SHHS.
  resnetse    1D residual blocks with squeeze-and-excitation, then an LSTM over
              the epoch sequence. Published ~0.864 on Sleep-EDF.
  atbilstm    convolutional epoch encoder, bidirectional LSTM over the sequence,
              additive attention over its outputs. Published ~0.838 on Sleep-EDF.

Mixed precision throughout, so three of these fit on one card at once.

  KMP_DUPLICATE_LIB_OK=TRUE python run_recent_baselines.py micro stroke
  KMP_DUPLICATE_LIB_OK=TRUE python run_recent_baselines.py micro healthy
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score)  # noqa: E402

OUTDIR = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SAMP, N_FOLDS = 3000, 10
PUBLISHED = {"micro": (0.833, "SHHS"), "resnetse": (0.864, "Sleep-EDF"),
             "atbilstm": (0.838, "Sleep-EDF")}
SEQ = {"micro": 1, "resnetse": 20, "atbilstm": 20}   # epochs per training example


# ----------------------------------------------------------------- models
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(4, c // r)), nn.ReLU(),
                                nn.Linear(max(4, c // r), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


def dwsep(cin, cout, k, s=1):
    """Depthwise separable convolution, the unit Micro SleepNet is built from."""
    return nn.Sequential(
        nn.Conv1d(cin, cin, k, stride=s, padding=k // 2, groups=cin, bias=False),
        nn.BatchNorm1d(cin), nn.ReLU(),
        nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU())


class MicroSleepNet(nn.Module):
    """Per-epoch, deliberately small: the paper targets mobile inference."""
    def __init__(self, n_cls=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 49, stride=6, padding=24, bias=False),
                                  nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.body = nn.Sequential(dwsep(32, 64, 9), nn.MaxPool1d(4), SE(64),
                                  dwsep(64, 96, 9), nn.MaxPool1d(4), SE(96),
                                  dwsep(96, 128, 9), SE(128))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):                       # [B, L, SAMP]
        B, L, T = x.shape
        h = self.body(self.stem(x.reshape(B * L, 1, T))).mean(-1)
        return self.head(h).reshape(B, L, -1)


class ResBlockSE(nn.Module):
    def __init__(self, cin, cout, k=7, s=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=s, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout)
        self.skip = (nn.Sequential() if (cin == cout and s == 1) else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride=s, bias=False),
                                   nn.BatchNorm1d(cout)))
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.se(self.b2(self.c2(h)))
        return self.act(h + self.skip(x))


class ResNetSELSTM(nn.Module):
    def __init__(self, n_cls=5, d=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 49, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            ResBlockSE(32, 64, s=2), ResBlockSE(64, 96, s=2), ResBlockSE(96, d, s=2))
        self.lstm = nn.LSTM(d, d, 1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n_cls))

    def forward(self, x):
        B, L, T = x.shape
        e = self.enc(x.reshape(B * L, 1, T)).mean(-1).reshape(B, L, -1)
        return self.head(self.lstm(e)[0])


class ATBiLSTM(nn.Module):
    """Convolutional epoch encoder, BiLSTM, additive attention over its states."""
    def __init__(self, n_cls=5, d=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 64, 49, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(8), nn.Dropout(0.3),
            nn.Conv1d(64, 128, 9, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, d, 9, padding=4, bias=False),
            nn.BatchNorm1d(d), nn.ReLU())
        self.lstm = nn.LSTM(d, d, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att_w = nn.Linear(2 * d, 128)
        self.att_v = nn.Linear(128, 1, bias=False)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * d, n_cls))

    def forward(self, x):
        B, L, T = x.shape
        e = self.enc(x.reshape(B * L, 1, T)).mean(-1).reshape(B, L, -1)
        h, _ = self.lstm(e)                                   # [B, L, 2d]
        a = torch.softmax(self.att_v(torch.tanh(self.att_w(h))), dim=1)
        ctx = (a * h).sum(1, keepdim=True).expand_as(h)       # sequence context
        return self.head(torch.cat([h, ctx], -1))


BUILD = {"micro": MicroSleepNet, "resnetse": ResNetSELSTM, "atbilstm": ATBiLSTM}


# ------------------------------------------------------------------ data
def subject_of(rec):
    return rec[3:5]


def load(corpus):
    data = {}
    if corpus == "stroke":
        src, ch = os.path.join(REPO, "data", "processed7", "SN*.npz"), 0
    else:
        src, ch = os.path.join(REPO, "data", "sleep_edf_proc", "*.npz"), 0
    for f in sorted(glob.glob(src)):
        rec = os.path.basename(f)[:-4]
        if corpus == "stroke":
            sid = int(rec[2:])
            if sid in C.DUP or sid not in C.DATA:
                continue
            rec = sid
        d = np.load(f, allow_pickle=True)
        x = np.asarray(d["x"][:, ch], dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        data[rec] = (x, d["y"].astype(np.int64))
    return data


def seqs(data, subs, L):
    X, Y, M = [], [], []
    for s in subs:
        x, y = data[s]
        for i in range(0, len(y), L):
            xs, ys = x[i:i + L], y[i:i + L]
            k = len(ys)
            if k < L:
                xs = np.concatenate([xs, np.zeros((L - k, SAMP), np.float32)])
                ys = np.concatenate([ys, np.zeros(L - k, np.int64)])
            X.append(xs); Y.append(ys)
            M.append(np.concatenate([np.ones(k, np.float32), np.zeros(L - k, np.float32)]))
    return (torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)),
            torch.tensor(np.stack(M)))


def evaluate(model, X, Y, M, bs):
    model.eval(); yt, yp = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
        for i in range(0, len(X), bs):
            o = model(X[i:i + bs].to(DEV)).argmax(-1).reshape(-1).cpu()
            m = M[i:i + bs].reshape(-1) > 0
            yt.append(Y[i:i + bs].reshape(-1)[m].numpy()); yp.append(o[m].numpy())
    return np.concatenate(yt), np.concatenate(yp)


def run_fold(name, data, tr, va, te, seed, epochs=30, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    L = SEQ[name]
    bs = 256 if L == 1 else 16
    Xtr, Ytr, Mtr = seqs(data, tr, L)
    va_pack = seqs(data, va, L)

    model = BUILD[name]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(np.concatenate([data[s][1] for s in tr]), minlength=5)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32,
                     device=DEV)
    ce = nn.CrossEntropyLoss(weight=w, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                out = model(Xtr[idx].to(DEV))
                m = Mtr[idx].reshape(-1).to(DEV)
                loss = (ce(out.reshape(-1, 5), Ytr[idx].reshape(-1).to(DEV)) * m
                        ).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        acc = accuracy_score(*evaluate(model, *va_pack, bs * 2))
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return evaluate(model, *seqs(data, te, L), bs * 2)


def main(name, corpus):
    data = load(corpus)
    recs = sorted(data, key=lambda r: (str(type(r)), r))
    n_par = sum(p.numel() for p in BUILD[name]().parameters())
    if corpus == "stroke":
        folds = [(list(a), list(b)) for a, b in C.FOLDS]
        label = "10-fold patient-independent"
    else:
        subs = sorted({subject_of(r) for r in recs})
        rng = np.random.RandomState(42); order = list(subs); rng.shuffle(order)
        parts = [order[i::N_FOLDS] for i in range(N_FOLDS)]
        folds = [([r for r in recs if subject_of(r) not in p],
                  [r for r in recs if subject_of(r) in p]) for p in parts]
        label = "10-fold subject-independent"
    print("%s on %s: %d recordings, %d epochs, %d params"
          % (name, corpus, len(recs), sum(len(v[1]) for v in data.values()), n_par),
          flush=True)

    t0 = time.time(); per_fold = []
    for fi, (tr_all, te) in enumerate(folds):
        tr_all = [s for s in tr_all if s in data]; te = [s for s in te if s in data]
        rs = np.random.RandomState(100 + fi)
        order = list(tr_all); rs.shuffle(order)
        nv = max(2, len(order) // 9)
        va, tr = order[:nv], order[nv:]
        yt, yp = run_fold(name, data, tr, va, te, seed=42)
        per_fold.append(dict(fold=fi, n=int(len(yt)),
                             acc=float(accuracy_score(yt, yp)),
                             mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                             kappa=float(cohen_kappa_score(yt, yp))))
        print("  fold %2d  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
              % (fi, len(yt), per_fold[-1]["acc"], per_fold[-1]["mf1"],
                 per_fold[-1]["kappa"], (time.time() - t0) / 60), flush=True)

    agg = {k: (float(np.mean([f[k] for f in per_fold])),
               float(np.std([f[k] for f in per_fold], ddof=1)))
           for k in ("acc", "mf1", "kappa")}
    out = os.path.join(OUTDIR, "%s_%s.json" % (name, corpus))
    json.dump(dict(model=name, corpus=corpus, protocol=label, params=n_par,
                   published=PUBLISHED[name], per_fold=per_fold,
                   **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
                   minutes=(time.time() - t0) / 60), open(out, "w"), indent=1)
    print("\n%s ON %s" % (name.upper(), corpus.upper()))
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, *agg[k]))
    if corpus == "healthy":
        p, corp = PUBLISHED[name]
        print("  published %.3f (%s)   ours %.3f   difference %+.3f"
              % (p, corp, agg["acc"][0], agg["acc"][0] - p))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else "stroke")
