"""Reimplementation of the corpus paper's SE-ResNet-LSTM baseline, on our folds.

The benchmark table carries the corpus paper's LSTM at 0.747 accuracy as a quoted
number. It is quoted because it cannot be re-run: the authors' repository
(github.com/suvadeepmaiti/iSLEEPS) contains only `preprocess/`, and the `models/`
directory its README advertises was never committed. So the strongest row in our
own table is the one row nobody can check.

This puts it on the same footing as DeepSleepNet, AttnSleep, TinySleepNet, U-Time
and SleepTransformer: reimplemented from the published description and evaluated
on our ten patient-independent folds, with the same class-balanced loss, early
stopping and HMM Viterbi decode.

What the two papers specify, and what they do not
-------------------------------------------------
Specified (Maiti et al., Sci Data 13:421, 2026; and Sharma et al., arXiv:2309.07156,
which is the architecture it cites):

  * input   raw single-channel EEG (C4-A1 / C4-M1) *and* EOG (EOG1-M1)
  * context 5 consecutive 30 s epochs (150 s); the centre epoch is classified,
            the other four supply context
  * encoder squeeze-and-excitation blocks inside a residual network
  * temporal stacked Bi-LSTM; the *middle cell* output of the final stack is taken
  * training Adam, lr 0.001, batch 128, negative log-likelihood loss
  * their split 80/10/10 patient-wise with 10-fold CV, N=100

Not specified anywhere in either paper: residual block count, channel widths,
kernel sizes, stride, SE reduction ratio, Bi-LSTM hidden size, or the number of
stacks S. Those are chosen here and marked ARCH_UNSPEC below. An exact
reproduction is therefore impossible by construction, and this is a faithful
reimplementation rather than a replication -- which is the honest thing to report
and is why the result should be labelled as such in the table.

Two deliberate departures from their protocol, both toward comparability:

  * our ten patient-independent folds, not their 80/10/10, so the row sits beside
    the other reimplemented baselines rather than in a protocol of its own;
  * N=99, excluding SN28, whose arrays are byte-identical to SN15. Their N=100
    contains both copies.

  KMP_DUPLICATE_LIB_OK=TRUE python run_isleeps_lstm.py [--seeds 42] [--folds 10]
"""
import argparse
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score  # noqa: E402

P7 = os.path.join(REPO, "data", "processed7")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "isleeps_lstm.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

EEG_CH, EOG_CH = 0, 4          # C4:M1 and E1:M2
CONTEXT = 5                    # epochs per window, centre one is the target
EPOCHS, BS, PATIENCE, LR = 30, 128, 6, 1e-3

# ARCH_UNSPEC -- not given in either paper; see module docstring.
STEM, WIDTHS, KERNEL, SE_RATIO = 32, (32, 64, 128, 128), 7, 16
LSTM_HIDDEN, LSTM_STACKS, DROPOUT = 128, 2, 0.3


class SEBlock(nn.Module):
    def __init__(self, ch, r=SE_RATIO):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(ch // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(ch // r, 4), ch), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(-1))
        return x * w.unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        pad = KERNEL // 2
        self.c1 = nn.Conv1d(cin, cout, KERNEL, stride, pad, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, KERNEL, 1, pad, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SEBlock(cout)
        self.skip = (nn.Sequential() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.se(self.b2(self.c2(h)))
        return self.act(h + self.skip(x))


class SEResNetLSTM(nn.Module):
    """Per-epoch SE-ResNet encoder, then a stacked Bi-LSTM over the context window.

    The middle cell of the final stack is the classified representation, which is
    what the source paper specifies: the centre epoch is the prediction target and
    the surrounding epochs are context.
    """

    def __init__(self, n_ch=2, n_cls=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_ch, STEM, 49, 4, 24, bias=False),
                                  nn.BatchNorm1d(STEM), nn.ReLU(inplace=True),
                                  nn.MaxPool1d(4))
        blocks, cin = [], STEM
        for w in WIDTHS:
            blocks.append(ResBlock(cin, w, stride=2))
            cin = w
        self.res = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(cin, LSTM_HIDDEN, num_layers=LSTM_STACKS,
                            batch_first=True, bidirectional=True,
                            dropout=DROPOUT if LSTM_STACKS > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(DROPOUT),
                                  nn.Linear(2 * LSTM_HIDDEN, n_cls))

    def forward(self, x):                      # x: [B, CONTEXT, n_ch, 3000]
        b, t, c, n = x.shape
        e = self.pool(self.res(self.stem(x.reshape(b * t, c, n)))).squeeze(-1)
        h, _ = self.lstm(e.reshape(b, t, -1))
        return self.head(h[:, t // 2])         # middle cell


def load():
    data = {}
    for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        d = np.load(f, allow_pickle=True)
        x = np.asarray(d["x"][:, [EEG_CH, EOG_CH]], dtype=np.float32)
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        data[sid] = (x, d["y"].astype(np.int64))
    return data


def win_index(n):
    """Edge-padded context indices for a recording of n epochs -> [n, CONTEXT]."""
    half = CONTEXT // 2
    return np.clip(np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :],
                   0, n - 1)


def run_fold(data, tr, va, te, seed):
    """Windows are gathered per batch rather than materialised.

    A 5-epoch context turns the training array into a five-fold copy of the
    recording -- 8 GB for one fold at float32 -- and every window overlaps its
    neighbours by four epochs, so all of that is duplication. Batches index into
    the per-subject arrays instead, which keeps the cost at the size of the data.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    widx = {s: win_index(len(data[s][1])) for s in data}
    # global (subject, epoch) index for the training pool
    pool = np.array([(si, i) for si, s in enumerate(tr) for i in range(len(data[s][1]))],
                    dtype=np.int64)
    tr_arr = [data[s][0] for s in tr]
    tr_widx = [widx[s] for s in tr]
    Ytr = np.concatenate([data[s][1] for s in tr])
    cnt = np.bincount(Ytr, minlength=5)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5,
                     dtype=torch.float32, device=DEV)
    Yt = torch.from_numpy(Ytr)
    # offset of each training subject inside the concatenated label vector
    starts = np.cumsum([0] + [len(data[s][1]) for s in tr[:-1]])

    def gather(batch):
        out = np.empty((len(batch), CONTEXT, 2, 3000), dtype=np.float32)
        for k, (si, i) in enumerate(batch):
            out[k] = tr_arr[si][tr_widx[si][i]]
        return torch.from_numpy(out)

    model = SEResNetLSTM().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=w)

    @torch.no_grad()
    def probs(subs):
        model.eval(); out = {}
        for s in subs:
            x, wi = data[s][0], widx[s]
            p = []
            for i in range(0, len(wi), 128):
                xb = torch.from_numpy(x[wi[i:i + 128]])
                p.append(torch.softmax(model(xb.to(DEV)), -1).cpu().numpy())
            out[s] = np.concatenate(p)
        return out

    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(pool))
        for i in range(0, len(perm) - BS + 1, BS):
            sel = pool[perm[i:i + BS]]
            yb = Yt[torch.from_numpy(starts[sel[:, 0]] + sel[:, 1])]
            opt.zero_grad()
            loss = crit(model(gather(sel).to(DEV)), yb.to(DEV))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
        pv = probs(va)
        yv = np.concatenate([data[s][1] for s in va])
        acc = accuracy_score(yv, np.concatenate([pv[s] for s in va]).argmax(1))
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)

    Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
    for s in tr:
        y = data[s][1]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            Am[a, b] += 1
    A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

    pt = probs(te)
    yt = np.concatenate([data[s][1] for s in te])
    yp = np.concatenate([C.hmm(A_log, pi_log, np.log(pt[s] + C.EPS)) for s in te])
    return yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    data = load()
    n_par = sum(p.numel() for p in SEResNetLSTM().parameters())
    print("SE-ResNet-LSTM reimplementation | %d subjects | %d folds x %d seeds | %s"
          % (len(data), a.folds, len(a.seeds), DEV), flush=True)
    print("input C4:M1 + E1:M2, %d-epoch context, %.2f M parameters\n"
          % (CONTEXT, n_par / 1e6), flush=True)

    per_seed, t0 = {}, time.time()
    for seed in a.seeds:
        rows = []
        for fi, (tr_all, te) in enumerate(C.FOLDS[:a.folds]):
            rng = np.random.RandomState(100 + fi)
            tr_all = [s for s in tr_all if s in data]
            rng.shuffle(tr_all)
            nv = max(8, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]
            te = [s for s in te if s in data]
            yt, yp = run_fold(data, tr, va, te, seed)
            r = dict(fold=fi, n=int(len(yt)), acc=float(accuracy_score(yt, yp)),
                     mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                     kappa=float(cohen_kappa_score(yt, yp)))
            rows.append(r)
            print("  seed %-3d fold %d/%d  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
                  % (seed, fi + 1, a.folds, r["acc"], r["mf1"], r["kappa"],
                     (time.time() - t0) / 60), flush=True)
        per_seed["seed|%d" % seed] = {m: [r[m] for r in rows]
                                      for m in ("acc", "mf1", "kappa")}
        acc = [r["acc"] for r in rows]
        print("  -> seed %d: acc %.4f +- %.4f\n" % (seed, np.mean(acc), np.std(acc, ddof=1)),
              flush=True)

    flat = {m: np.concatenate([v[m] for v in per_seed.values()]) for m in ("acc", "mf1", "kappa")}
    summary = {m: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(v.size)}
               for m, v in flat.items()}
    json.dump({"model": "SE-ResNet-LSTM (reimplementation of the corpus paper's LSTM baseline)",
               "reference": "Maiti et al., Sci Data 13:421 (2026); architecture from "
                            "Sharma et al., arXiv:2309.07156",
               "reported_by_authors": {"acc": 0.7470, "protocol": "80/10/10 patient-wise, "
                                       "10-fold CV, N=100, code not released"},
               "protocol": "10-fold patient-independent on mmnet_core.FOLDS, N=%d, "
                           "seeds %s" % (len(data), a.seeds),
               "input": "C4:M1 + E1:M2, %d-epoch context window" % CONTEXT,
               "params_millions": round(n_par / 1e6, 3),
               "arch_unspecified_by_source": {
                   "stem": STEM, "widths": list(WIDTHS), "kernel": KERNEL,
                   "se_ratio": SE_RATIO, "lstm_hidden": LSTM_HIDDEN,
                   "lstm_stacks": LSTM_STACKS, "dropout": DROPOUT},
               "per_seed": per_seed, "summary": summary,
               "minutes": (time.time() - t0) / 60},
              open(a.out, "w", encoding="utf-8"), indent=1)
    print("SUMMARY  acc %.4f +- %.4f | mF1 %.4f | kappa %.4f  (n=%d)"
          % (summary["acc"]["mean"], summary["acc"]["sd"], summary["mf1"]["mean"],
             summary["kappa"]["mean"], summary["acc"]["n"]))
    print("authors report 0.7470 on their own protocol")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
