"""Raw-signal respiratory baselines: a network that reads the belt, not 14 descriptors.

The respiratory baselines in the paper (Table 9) all read the fourteen engineered
cardiorespiratory descriptors, which leaves the representation argument untested
against the obvious alternative: a network trained end to end on the raw signal,
in the style of Nassi et al. (IEEE TBME 2022), who score respiratory events from a
single effort belt. This runs two such detectors on our folds:

  effort   the summed respiratory-effort channel alone (Nassi et al.'s input)
  cardio7  all seven raw cardiorespiratory channels MM-Net's encoder sees
           (ECG, flow, thorax, abdomen, effort, SpO2, pulse)

Architecture: a per-epoch 1-D convolutional encoder over the 30 s signal at 25 Hz,
a bidirectional LSTM across 20-epoch windows, and a per-epoch logit -- the same
epoch-level label and read-out MM-Net's respiratory head uses, so the AUC and AP
are directly comparable. Nassi et al. predict per second; the epoch label here is
ours, which is the only change needed to put the method on the paper's footing.

Protocol: the paper's ten patient-independent folds over the 99 patients, three
seeds, a validation split of training patients for early stopping on AUC, the
positive class weighted by neg/pos as in MM-Net, per-patient z-scoring. Channels a
recording lacks are zero, as in MM-Net.

  KMP_DUPLICATE_LIB_OK=TRUE python run_resp_raw_baseline.py [effort cardio7]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MM = os.path.join(REPO, "data", "multimodal")
P7 = os.path.join(REPO, "data", "processed7")
OUTDIR = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [42, 1, 7]
L, STRIDE, BATCH = 20, 10, 32
EPOCHS, PATIENCE, LR = 30, 5, 1e-3
CHANNELS = {"effort": [4], "cardio7": list(range(7))}
DUP = {28}                                   # SN28 is a byte-identical copy of SN15


def make_folds(subs, k=10, seed=42):
    """Identical to mmnet_core.make_folds; copied so the folds do not depend on
    which feature caches happen to be on disk."""
    r = np.random.RandomState(seed); s = list(subs); r.shuffle(s)
    folds = [s[i::k] for i in range(k)]
    return [([x for j, f in enumerate(folds) if j != i for x in f], folds[i]) for i in range(k)]


SUBJECTS = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(P7, "SN*.npz"))
                  if int(os.path.basename(p)[2:-4]) not in DUP)
FOLDS = make_folds(SUBJECTS)


def load(which):
    data = {}
    for sid in SUBJECTS:
        d = np.load(os.path.join(MM, "SN%d.npz" % sid), allow_pickle=True)
        c = d["card"][:, CHANNELS[which]].astype(np.float32)
        c = (c - c.mean((0, 2), keepdims=True)) / (c.std((0, 2), keepdims=True) + 1e-6)
        c[:, ~d["cvalid"][CHANNELS[which]]] = 0.0
        # half precision in RAM (the seven-channel windows otherwise need ~6 GB)
        data[sid] = (np.nan_to_num(c).astype(np.float16), d["apnea"].astype(np.float32))
    return data


class Detector(nn.Module):
    def __init__(self, n_ch, d=64, hidden=64, drop=0.3):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2, bias=False),
                                 nn.BatchNorm1d(o), nn.ReLU(inplace=True))
        self.enc = nn.Sequential(blk(n_ch, 32, 7, 2), blk(32, 32, 7, 1), nn.MaxPool1d(2),
                                 blk(32, 64, 5, 1), blk(64, 64, 5, 1), nn.MaxPool1d(2),
                                 blk(64, d, 3, 1), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Dropout(drop))
        self.rnn = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))

    def forward(self, x):                    # [B, L, C, 750] -> logits [B, L]
        b, t, c, n = x.shape
        e = self.enc(x.reshape(b * t, c, n)).reshape(b, t, -1)
        return self.head(self.rnn(e)[0]).squeeze(-1)


def windows(data, subs, stride):
    X, Y, M = [], [], []
    for s in subs:
        x, y = data[s]
        n = len(y)
        for a in range(0, n, stride):
            xs, ys = x[a:a + L], y[a:a + L]
            k = len(ys)
            if stride < L and k < L and a > 0:
                continue                     # training: skip the short tail chunks
            m = np.zeros(L, np.float32); m[:k] = 1
            if k < L:
                xs = np.concatenate([xs, np.zeros((L - k,) + xs.shape[1:], xs.dtype)])
                ys = np.concatenate([ys, np.zeros(L - k, np.float32)])
            X.append(xs); Y.append(ys); M.append(m)
    return np.stack(X), np.stack(Y), np.stack(M)


@torch.no_grad()
def predict(model, data, subs):
    model.eval(); yt, pp = [], []
    for s in subs:
        X, Y, M = windows(data, [s], L)
        p = []
        for i in range(0, len(X), 64):
            p.append(torch.sigmoid(model(torch.from_numpy(X[i:i + 64]).to(DEV).float())).cpu().numpy())
        p = np.concatenate(p)[M > 0]
        yt.append(Y[M > 0]); pp.append(p)
    return np.concatenate(yt), np.concatenate(pp)


def run_fold(data, n_ch, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y, M = (torch.from_numpy(a) for a in windows(data, tr, STRIDE))
    pos = float((Y * M).sum()); neg = float(M.sum()) - pos
    bce = nn.BCEWithLogitsLoss(reduction="none",
                               pos_weight=torch.tensor(neg / max(pos, 1.0), device=DEV))
    model = Detector(n_ch).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best, best_state, bad = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            m = M[b].to(DEV)
            loss = (bce(model(X[b].to(DEV).float()), Y[b].to(DEV)) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        v = roc_auc_score(*predict(model, data, va))
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    yt, p = predict(model, data, te)
    return dict(auc=float(roc_auc_score(yt, p)), ap=float(average_precision_score(yt, p)),
                prevalence=float(yt.mean()), n=int(len(yt)))


def main(which, seeds=SEEDS):
    data = load(which)
    n_ch = len(CHANNELS[which])
    n_par = sum(p.numel() for p in Detector(n_ch).parameters())
    out = os.path.join(OUTDIR, "resp_raw_%s.json" % which)
    print("raw-signal respiratory detector [%s] | %d patients | %d params | %s"
          % (which, len(data), n_par, DEV), flush=True)
    # resume: keep seeds an earlier run completed on all ten folds
    res, t0 = {}, time.time()
    if os.path.exists(out):
        prev = json.load(open(out)).get("per_seed", {})
        res = {k: v for k, v in prev.items() if len(v) == len(FOLDS)}
    for seed in seeds:
        if "seed|%d" % seed in res:
            print("  [skip] seed %d already complete" % seed, flush=True)
            continue
        rows = []
        for fi, (tr_all, te) in enumerate(FOLDS):
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all); rng.shuffle(tr_all)
            nv = max(8, len(tr_all) // 9)
            r = run_fold(data, n_ch, tr_all[nv:], tr_all[:nv], te, seed)
            r["fold"] = fi; rows.append(r)
            print("  seed %-2d fold %d  AUC %.4f  AP %.4f  [%.1f min]"
                  % (seed, fi, r["auc"], r["ap"], (time.time() - t0) / 60), flush=True)
            res["seed|%d" % seed] = rows
            json.dump(dict(partial=True, per_seed=res), open(out, "w"), indent=1)
    summ = {}
    for m in ("auc", "ap"):
        v = np.array([r[m] for rows in res.values() for r in rows])
        fm = np.array([np.mean([res[s][f][m] for s in res]) for f in range(len(FOLDS))])
        summ[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(v.size),
                       fold_means=fm.tolist())
    json.dump(dict(model="raw-signal respiratory detector (%s)" % which,
                   reference="input and task after Nassi et al., IEEE TBME 69(6), 2022",
                   channels=which, params=n_par,
                   protocol="10-fold patient-independent over the 99 patients "
                            "(mmnet_core.make_folds), seeds %s, early stopping on "
                            "validation AUC" % sorted(int(k.split("|")[1]) for k in res),
                   per_seed=res, summary=summ, minutes=(time.time() - t0) / 60),
              open(out, "w"), indent=1)
    print("SUMMARY [%s] AUC %.4f +- %.4f | AP %.4f +- %.4f"
          % (which, summ["auc"]["mean"], summ["auc"]["sd"], summ["ap"]["mean"], summ["ap"]["sd"]))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("channels", nargs="*", default=["effort", "cardio7"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    for w in a.channels:
        main(w, a.seeds)
