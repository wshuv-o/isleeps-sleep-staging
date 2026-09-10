"""TinySleepNet on iSLEEPS, on the same folds as everything else.

Supratak and Guo, EMBC 2020. A deliberately small single-channel architecture --
four convolutions, one unidirectional LSTM over 15-epoch sequences -- which makes
it a fair test of whether the cohort ceiling is about capacity. If a compact
recent model trained on this cohort lands in the same 0.61-0.69 band as the
others, that band is a property of the data.

This is a reimplementation from the paper, not the authors' code, and the table
says so: a referee is entitled to read it as our version of their architecture.
The hyperparameters below are the paper's own, at our 100 Hz sampling rate:

    conv1     128 filters, kernel Fs/2 = 50, stride Fs/16 = 6
    pool      size 8, stride 8            dropout 0.5
    conv2-4   128 filters, kernel 8, stride 1
    pool      size 4, stride 4            dropout 0.5
    LSTM      1 layer, 128 hidden, unidirectional, sequence length 15
    output    fully connected to 5 stages

Folds, seeds and the class-weighting scheme are taken from mmnet_core so the row
is comparable with the rest of the table rather than merely adjacent to it.

  KMP_DUPLICATE_LIB_OK=TRUE python run_tinysleepnet.py
"""
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

P7 = os.path.join(REPO, "data", "processed7")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "tinysleepnet.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 15                      # the paper's sequence length
EEG_CH = 0                    # C4:M1, the single derivation the model reads
EPOCHS, BS, LR = 30, 16, 1e-3
PATIENCE = 6


class TinySleepNet(nn.Module):
    def __init__(self, n_cls=5, drop=0.5):
        super().__init__()
        self.rep = nn.Sequential(
            nn.Conv1d(1, 128, 50, stride=6, padding=25, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(8, 8), nn.Dropout(drop),
            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4, 4), nn.Dropout(drop))
        self.lstm = nn.LSTM(128 * 15, 128, 1, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):                       # x: [B, SEQ, 3000]
        B, L, T = x.shape
        h = self.rep(x.reshape(B * L, 1, T))    # [B*L, 128, ~15]
        h = nn.functional.adaptive_avg_pool1d(h, 15).reshape(B, L, -1)
        h, _ = self.lstm(h)
        return self.fc(self.drop(h))            # [B, L, 5]


def load():
    """Per-subject single-channel EEG and labels, z-scored per recording."""
    import glob
    data = {}
    for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        d = np.load(f, allow_pickle=True)
        x = np.asarray(d["x"][:, EEG_CH], dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        data[sid] = (x, d["y"].astype(np.int64))
    return data


def seqs(data, subs):
    """Non-overlapping SEQ-length windows, tail padded and masked."""
    X, Y, M = [], [], []
    for s in subs:
        x, y = data[s]
        n = len(y)
        for i in range(0, n, SEQ):
            xs, ys = x[i:i + SEQ], y[i:i + SEQ]
            k = len(ys)
            if k < SEQ:
                xs = np.concatenate([xs, np.zeros((SEQ - k, xs.shape[1]), np.float32)])
                ys = np.concatenate([ys, np.zeros(SEQ - k, np.int64)])
            X.append(xs); Y.append(ys)
            M.append(np.concatenate([np.ones(k, np.float32), np.zeros(SEQ - k, np.float32)]))
    return (torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)),
            torch.tensor(np.stack(M)))


def run_fold(data, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Mtr = [t.to(DEV) for t in seqs(data, tr)]
    Xva, Yva, Mva = [t.to(DEV) for t in seqs(data, va)]

    model = TinySleepNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # same sqrt-inverse-frequency weighting the rest of the paper uses
    cnt = np.bincount(np.concatenate([data[s][1] for s in tr]), minlength=5)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32,
                     device=DEV)
    ce = nn.CrossEntropyLoss(weight=w, reduction="none")

    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            out = model(Xtr[idx])
            m = Mtr[idx].reshape(-1)
            loss = (ce(out.reshape(-1, 5), Ytr[idx].reshape(-1)) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(Xva).argmax(-1).reshape(-1).cpu().numpy()
            yv = Yva.reshape(-1).cpu().numpy(); mv = Mva.reshape(-1).cpu().numpy() > 0
            acc = accuracy_score(yv[mv], p[mv])
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)

    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for s in te:
            X, Y, M = [t.to(DEV) for t in seqs(data, [s])]
            p = model(X).argmax(-1).reshape(-1).cpu().numpy()
            m = M.reshape(-1).cpu().numpy() > 0
            yt.append(Y.reshape(-1).cpu().numpy()[m]); yp.append(p[m])
    return np.concatenate(yt), np.concatenate(yp)


def main():
    data = load()
    print("device %s | %d subjects | %d params"
          % (DEV, len(data), sum(p.numel() for p in TinySleepNet().parameters())),
          flush=True)
    t0 = time.time()
    per_fold = []
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        rng = np.random.RandomState(100 + fi)
        tr_all = [s for s in tr_all if s in data]
        te = [s for s in te if s in data]
        tr_all = list(tr_all); rng.shuffle(tr_all)
        nv = max(5, len(tr_all) // 9)
        va, tr = tr_all[:nv], tr_all[nv:]
        yt, yp = run_fold(data, tr, va, te, seed=42)
        per_fold.append(dict(fold=fi, n=int(len(yt)),
                             acc=float(accuracy_score(yt, yp)),
                             mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                             kappa=float(cohen_kappa_score(yt, yp))))
        print("fold %2d  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
              % (fi, len(yt), per_fold[-1]["acc"], per_fold[-1]["mf1"],
                 per_fold[-1]["kappa"], (time.time() - t0) / 60), flush=True)

    agg = {k: (float(np.mean([f[k] for f in per_fold])),
               float(np.std([f[k] for f in per_fold], ddof=1)))
           for k in ("acc", "mf1", "kappa")}
    res = dict(model="TinySleepNet (reimplementation)",
               reference="Supratak and Guo, EMBC 2020",
               protocol="10-fold patient-independent, seed 42, single EEG derivation",
               params=sum(p.numel() for p in TinySleepNet().parameters()),
               per_fold=per_fold,
               **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
               minutes=(time.time() - t0) / 60)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nTINYSLEEPNET, 10 folds")
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, agg[k][0], agg[k][1]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
