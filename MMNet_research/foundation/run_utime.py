"""U-Time on iSLEEPS and on healthy sleep, on the same folds as everything else.

Perslev et al., NeurIPS 2019. A U-Net over the raw signal: the encoder-decoder
runs at sample resolution across a whole multi-epoch segment and only pools down
to one label per 30 s at the very end. It is the one architecture in the table
with no recurrence and no attention over epochs -- context reaches a decision
through the receptive field of the convolutions alone -- which is why it is
worth a row even though it predates TinySleepNet and SleepTransformer.

It is also U-Sleep's direct predecessor, so this row covers that family whether
or not the external API access comes through.

Reimplemented from the paper rather than the published code; the table says so.
The paper's own hyperparameters:

    encoder    4 blocks, two 5-tap convolutions each, BN + ELU
               filters 16 / 32 / 64 / 128, max-pooling 10 / 8 / 6 / 4
    bottleneck two 5-tap convolutions, 256 filters
    decoder    nearest-neighbour upsample, 2-tap conv, skip concat, two 5-tap
    classifier pointwise conv to 5 logits per sample, mean-pooled per 30 s
               epoch, then a pointwise segment classifier

One departure, stated because it is forced rather than chosen: the encoder
divides the segment length by 10*8*6*4 = 1920, so a segment of L epochs needs
L*3000 divisible by 1920, i.e. L a multiple of 16. We use L = 32 epochs (16 min)
where the paper uses 35, which is the nearest faithful value that divides.

  KMP_DUPLICATE_LIB_OK=TRUE python run_utime.py stroke
  KMP_DUPLICATE_LIB_OK=TRUE python run_utime.py healthy
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
SEQ = 32                      # epochs per segment; see the note above
SAMP = 3000                   # samples per 30 s epoch at 100 Hz
POOLS = [10, 8, 6, 4]
FILTERS = [16, 32, 64, 128]
EPOCHS, BS, LR = 25, 8, 1e-3
PATIENCE = 5
N_FOLDS = 10
PUBLISHED_HEALTHY = 0.790     # U-Time on Sleep-EDF-153 in the paper (mean F1 0.76, acc ~0.79)


def conv_block(cin, cout, k=5):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm1d(cout), nn.ELU(),
        nn.Conv1d(cout, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm1d(cout), nn.ELU())


class UTime(nn.Module):
    def __init__(self, n_cls=5, cin=1):
        super().__init__()
        self.enc, self.pool = nn.ModuleList(), nn.ModuleList()
        prev = cin
        for f, p in zip(FILTERS, POOLS):
            self.enc.append(conv_block(prev, f))
            self.pool.append(nn.MaxPool1d(p))
            prev = f
        self.bottom = conv_block(prev, prev * 2)
        prev = prev * 2

        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for f, p in zip(reversed(FILTERS), reversed(POOLS)):
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=p, mode="nearest"),
                nn.Conv1d(prev, f, 2, padding=1, bias=False),
                nn.BatchNorm1d(f), nn.ELU()))
            self.dec.append(conv_block(f * 2, f))   # f from upsample + f from skip
            prev = f
        self.dense = nn.Conv1d(prev, n_cls, 1)
        self.segment = nn.Conv1d(n_cls, n_cls, 1)

    def forward(self, x):                      # x: [B, 1, SEQ*SAMP]
        skips = []
        for blk, pl in zip(self.enc, self.pool):
            x = blk(x); skips.append(x); x = pl(x)
        x = self.bottom(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)[..., :s.shape[-1]]        # the 2-tap conv adds a sample
            x = dec(torch.cat([x, s], dim=1))
        x = self.dense(x)                       # [B, 5, SEQ*SAMP]
        B, K, T = x.shape
        x = x.reshape(B, K, T // SAMP, SAMP).mean(-1)   # one vector per epoch
        return self.segment(torch.tanh(x)).transpose(1, 2)   # [B, SEQ, 5]


def subject_of(rec):
    """SC4ssNE: characters 3:5 are the subject, so SC4001 and SC4002 are one person."""
    return rec[3:5]


def load(corpus):
    """Raw single-derivation EEG per recording, z-scored."""
    data = {}
    if corpus == "stroke":
        src, ch = os.path.join(REPO, "data", "processed7", "SN*.npz"), 0   # C4:M1
    else:
        src, ch = os.path.join(REPO, "data", "sleep_edf_proc", "*.npz"), 0  # Fpz-Cz
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


def seqs(data, subs):
    """Non-overlapping SEQ-epoch segments, tail padded and masked."""
    X, Y, M = [], [], []
    for s in subs:
        x, y = data[s]
        for i in range(0, len(y), SEQ):
            xs, ys = x[i:i + SEQ], y[i:i + SEQ]
            k = len(ys)
            if k < SEQ:
                xs = np.concatenate([xs, np.zeros((SEQ - k, SAMP), np.float32)])
                ys = np.concatenate([ys, np.zeros(SEQ - k, np.int64)])
            X.append(xs.reshape(-1)); Y.append(ys)
            M.append(np.concatenate([np.ones(k, np.float32),
                                     np.zeros(SEQ - k, np.float32)]))
    return (torch.tensor(np.stack(X)).unsqueeze(1),
            torch.tensor(np.stack(Y)), torch.tensor(np.stack(M)))


def evaluate(model, X, Y, M, bs=16):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(X[i:i + bs].to(DEV)).argmax(-1).reshape(-1).cpu()
            m = M[i:i + bs].reshape(-1) > 0
            yt.append(Y[i:i + bs].reshape(-1)[m].numpy()); yp.append(out[m].numpy())
    return np.concatenate(yt), np.concatenate(yp)


def run_fold(data, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Mtr = seqs(data, tr)
    va_pack = seqs(data, va)

    model = UTime().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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
            out = model(Xtr[idx].to(DEV))
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


def main(corpus):
    data = load(corpus)
    recs = sorted(data, key=lambda r: (str(type(r)), r))
    n_par = sum(p.numel() for p in UTime().parameters())
    if corpus == "stroke":
        folds = [(tr, te) for tr, te in C.FOLDS]
        label = "10-fold patient-independent"
    else:
        subs = sorted({subject_of(r) for r in recs})
        rng = np.random.RandomState(42)
        order = list(subs); rng.shuffle(order)
        parts = [order[i::N_FOLDS] for i in range(N_FOLDS)]
        folds = [([r for r in recs if subject_of(r) not in p],
                  [r for r in recs if subject_of(r) in p]) for p in parts]
        label = "10-fold subject-independent"
    print("U-Time on %s: %d recordings, %d epochs, %d params"
          % (corpus, len(recs), sum(len(v[1]) for v in data.values()), n_par),
          flush=True)

    t0 = time.time()
    per_fold = []
    for fi, (tr_all, te) in enumerate(folds):
        tr_all = [s for s in tr_all if s in data]
        te = [s for s in te if s in data]
        rs = np.random.RandomState(100 + fi)
        tr_all = list(tr_all); rs.shuffle(tr_all)
        nv = max(2, len(tr_all) // 9)
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
    out = os.path.join(OUTDIR, "utime_%s.json" % corpus)
    json.dump(dict(model="U-Time (reimplementation)",
                   reference="Perslev et al., NeurIPS 2019",
                   corpus=corpus, protocol="%s, seed 42, single EEG derivation" % label,
                   params=n_par, segment_epochs=SEQ, per_fold=per_fold,
                   **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
                   minutes=(time.time() - t0) / 60), open(out, "w"), indent=1)

    print("\nU-TIME ON %s, %s" % (corpus.upper(), label))
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, agg[k][0], agg[k][1]))
    if corpus == "healthy":
        d = agg["acc"][0] - PUBLISHED_HEALTHY
        print("  published on Sleep-EDF: ~%.3f   ours: %.3f   difference %+.3f"
              % (PUBLISHED_HEALTHY, agg["acc"][0], d))
    print("wrote", out)


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["stroke", "healthy"]):
        main(c)
