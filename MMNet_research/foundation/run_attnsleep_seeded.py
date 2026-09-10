"""AttnSleep on iSLEEPS, three seeds on the ten standard folds.

The benchmark table quotes AttnSleep without an interval because the original run
kept only aggregates, and it ran at three folds in a different project directory
rather than the five the caption claims for the retrained baselines. This puts it
on the same footing as everything else: ten patient-independent folds, three
seeds, thirty fold-values, so the row can carry mean and SD like its neighbours.

The architecture is braindecode's reference implementation of Eldele et al.
(IEEE TNSRE 2021), not a reimplementation, on the single C4:M1 derivation the
paper uses. Class-balanced loss, early stopping on a held-out validation split,
and the same HMM Viterbi decode the rest of the benchmark applies.

  KMP_DUPLICATE_LIB_OK=TRUE python run_attnsleep_seeded.py
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
                   "attnsleep_seeded.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EEG_CH = 0                     # C4:M1
SEEDS = [42, 1, 7]
EPOCHS, BS, PATIENCE = 25, 128, 5


def load():
    data = {}
    for f in sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP or sid not in C.DATA:
            continue
        d = np.load(f, allow_pickle=True)
        x = np.asarray(d["x"][:, EEG_CH], dtype=np.float32)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        data[sid] = (x, d["y"].astype(np.int64))
    return data


def run_fold(data, tr, va, te, seed):
    from braindecode.models import AttnSleep
    torch.manual_seed(seed); np.random.seed(seed)

    Xtr = np.concatenate([data[s][0] for s in tr])[:, None, :]
    Ytr = np.concatenate([data[s][1] for s in tr])
    cnt = np.bincount(Ytr, minlength=5)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32,
                     device=DEV)

    model = AttnSleep(n_chans=1, n_outputs=5, n_times=3000, sfreq=100).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=w)

    Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)

    @torch.no_grad()
    def probs(subs):
        model.eval(); out = {}
        for s in subs:
            x = torch.from_numpy(data[s][0][:, None, :])
            p = []
            for i in range(0, len(x), 256):
                p.append(torch.softmax(model(x[i:i + 256].to(DEV)), -1).cpu().numpy())
            out[s] = np.concatenate(p)
        return out

    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            loss = crit(model(Xt[idx].to(DEV)), Yt[idx].to(DEV))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
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

    # the same HMM Viterbi decode the rest of the benchmark uses
    Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
    for s in tr:
        y = data[s][1]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            Am[a, b] += 1
    A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

    pt = probs(te)
    yt, yp = [], []
    for s in te:
        yt.append(data[s][1])
        yp.append(C.hmm(A_log, pi_log, np.log(pt[s] + C.EPS)))
    return np.concatenate(yt), np.concatenate(yp)


def main():
    data = load()
    print("AttnSleep (braindecode) on %d recordings, %d folds x %d seeds"
          % (len(data), len(C.FOLDS), len(SEEDS)), flush=True)
    t0 = time.time()
    res = {}
    for seed in SEEDS:
        per_fold = []
        for fi, (tr_all, te) in enumerate(C.FOLDS):
            tr_all = [s for s in tr_all if s in data]
            te = [s for s in te if s in data]
            rs = np.random.RandomState(100 + fi)
            order = list(tr_all); rs.shuffle(order)
            va, tr = order[:10], order[10:]
            yt, yp = run_fold(data, tr, va, te, seed)
            per_fold.append(dict(
                fold=fi, n=int(len(yt)),
                acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp))))
            print("  seed %2d fold %2d  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
                  % (seed, fi, per_fold[-1]["acc"], per_fold[-1]["mf1"],
                     per_fold[-1]["kappa"], (time.time() - t0) / 60), flush=True)
        res["seed|%d" % seed] = {k: [f[k] for f in per_fold]
                                 for k in ("acc", "mf1", "kappa")}
        json.dump(res, open(OUT, "w"), indent=1)

    print("\nATTNSLEEP, %d fold-values" % sum(len(v["acc"]) for v in res.values()))
    summary = {}
    for m in ("acc", "mf1", "kappa"):
        a = np.array([v for s in res for v in res[s][m]])
        summary[m] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)))
        print("  %-6s %.4f +- %.4f" % (m, a.mean(), a.std(ddof=1)))
    json.dump(dict(model="AttnSleep (braindecode reference implementation)",
                   reference="Eldele et al., IEEE TNSRE 2021",
                   protocol="10-fold patient-independent, seeds %s" % SEEDS,
                   per_seed=res, summary=summary,
                   minutes=(time.time() - t0) / 60), open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
