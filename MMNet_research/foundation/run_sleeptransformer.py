"""SleepTransformer on iSLEEPS, on the same folds as everything else.

Phan et al., IEEE TBME 2022. Two stacked transformers over log-magnitude
spectrograms: one attends across the 29 time frames inside a 30 s epoch and
pools them into an epoch vector, the second attends across a 21-epoch sequence
and classifies each position. It is the strongest purely-attentional stager in
the recent literature and shares no inductive bias with the convolutional models
already in the table, which is the reason to run it.

Like the TinySleepNet row this is a reimplementation from the paper rather than
the published code, and the table says so. The paper's own hyperparameters, at
our 100 Hz sampling rate:

    spectrogram   2 s Hamming window, 50% overlap, 256-point FFT
                  -> 29 frames x 128 frequency bins, log magnitude
    epoch trf     4 blocks, 8 heads, d_model 128, d_ff 1024, dropout 0.1
                  attention pooling over the 29 frames
    sequence trf  4 blocks, 8 heads, d_model 128, d_ff 1024, dropout 0.1
    output        1024-unit hidden layer, then 5 stages

  KMP_DUPLICATE_LIB_OK=TRUE python run_sleeptransformer.py
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

P7 = os.path.join(REPO, "data", "processed7")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "sleeptransformer.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 21                      # the paper's sequence length
EEG_CH = 0                    # C4:M1, the single derivation the model reads
NFREQ, NFRAME = 128, 29
EPOCHS, BS, LR = 30, 32, 1e-4
PATIENCE = 6


class AttnPool(nn.Module):
    """The paper's additive attention pooling over the frames of one epoch."""
    def __init__(self, d, hidden=64):
        super().__init__()
        self.w = nn.Linear(d, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h):                       # [N, T, d]
        a = torch.softmax(self.v(torch.tanh(self.w(h))), dim=1)
        return (a * h).sum(1)                   # [N, d]


def block(d, heads, ff, drop):
    return nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                      norm_first=True)


class SleepTransformer(nn.Module):
    def __init__(self, n_cls=5, d=NFREQ, heads=8, ff=1024, n_blk=4, drop=0.1):
        super().__init__()
        self.epoch_trf = nn.TransformerEncoder(block(d, heads, ff, drop), n_blk)
        self.pool = AttnPool(d)
        self.seq_trf = nn.TransformerEncoder(block(d, heads, ff, drop), n_blk)
        self.head = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(1024, n_cls))
        self.pe_f = nn.Parameter(torch.zeros(1, NFRAME, d))
        self.pe_s = nn.Parameter(torch.zeros(1, SEQ, d))
        nn.init.trunc_normal_(self.pe_f, std=0.02)
        nn.init.trunc_normal_(self.pe_s, std=0.02)

    def forward(self, x):                       # x: [B, SEQ, NFRAME, NFREQ]
        B, L, T, F = x.shape
        h = self.epoch_trf(x.reshape(B * L, T, F) + self.pe_f)
        h = self.pool(h).reshape(B, L, F)
        h = self.seq_trf(h + self.pe_s[:, :L])
        return self.head(h)                     # [B, L, 5]


def spectrogram(x):
    """[n, 3000] raw -> [n, 29, 128] log-magnitude, on the GPU."""
    win = torch.hamming_window(200, device=DEV)
    # torch.stft frames by n_fft, not win_length, so a bare 3000-sample epoch
    # yields 28 frames. Pad by (256-200)/2 each side to recover the paper's 29.
    x = torch.nn.functional.pad(x, (28, 28))
    s = torch.stft(x, n_fft=256, hop_length=100, win_length=200, window=win,
                   center=False, return_complex=True)      # [n, 129, 29]
    s = torch.log(s.abs() + 1e-8)[:, :NFREQ].transpose(1, 2)
    return s


def load():
    """Per-subject spectrograms, standardised per recording per frequency bin."""
    data = {}
    for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        d = np.load(f, allow_pickle=True)
        x = torch.tensor(np.asarray(d["x"][:, EEG_CH], dtype=np.float32), device=DEV)
        s = spectrogram(x)
        s = (s - s.mean((0, 1), keepdim=True)) / (s.std((0, 1), keepdim=True) + 1e-6)
        data[sid] = (s.half().cpu(), d["y"].astype(np.int64))
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
                xs = torch.cat([xs, torch.zeros(SEQ - k, NFRAME, NFREQ, dtype=xs.dtype)])
                ys = np.concatenate([ys, np.zeros(SEQ - k, np.int64)])
            X.append(xs); Y.append(torch.tensor(ys))
            M.append(torch.tensor(np.concatenate(
                [np.ones(k, np.float32), np.zeros(SEQ - k, np.float32)])))
    return torch.stack(X), torch.stack(Y), torch.stack(M)


def evaluate(model, X, Y, M, bs=64):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(X[i:i + bs].to(DEV).float()).argmax(-1).reshape(-1).cpu()
            m = M[i:i + bs].reshape(-1) > 0
            yt.append(Y[i:i + bs].reshape(-1)[m].numpy()); yp.append(out[m].numpy())
    return np.concatenate(yt), np.concatenate(yp)


def run_fold(data, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Mtr = seqs(data, tr)
    va_pack = seqs(data, va)

    model = SleepTransformer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    cnt = np.bincount(np.concatenate([data[s][1] for s in tr]), minlength=5)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32,
                     device=DEV)
    ce = nn.CrossEntropyLoss(weight=w, reduction="none")

    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            out = model(Xtr[idx].to(DEV).float())
            m = Mtr[idx].reshape(-1).to(DEV)
            loss = (ce(out.reshape(-1, 5), Ytr[idx].reshape(-1).to(DEV)) * m
                    ).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        acc = accuracy_score(*evaluate(model, *va_pack))
        if acc > best:
            best, best_state, bad = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return evaluate(model, *seqs(data, te))


def main():
    data = load()
    n_par = sum(p.numel() for p in SleepTransformer().parameters())
    print("device %s | %d subjects | %d params" % (DEV, len(data), n_par), flush=True)
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
    res = dict(model="SleepTransformer (reimplementation)",
               reference="Phan et al., IEEE TBME 2022",
               protocol="10-fold patient-independent, seed 42, single EEG derivation",
               params=n_par, per_fold=per_fold,
               **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
               minutes=(time.time() - t0) / 60)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nSLEEPTRANSFORMER, 10 folds")
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, agg[k][0], agg[k][1]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
