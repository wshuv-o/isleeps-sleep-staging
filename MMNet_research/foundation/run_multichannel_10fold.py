"""The two multi-channel baselines at ten folds, like every other row.

The four-EEG CNN (the "CNN + BiLSTM" row, from HAGNet_research train.py, cnn4ch) and
the raw fourteen-channel multimodal CNN (train/mm_train.py, fusion="cross") are the
like-for-like comparisons for MM-Net, which reads more than one channel. Both were
run at five folds or on a single split, and the four-EEG CNN chose its epoch on test
macro-F1. This reruns both on the paper's ten patient-independent folds over the 99
patients, three seeds, with the checkpoint chosen on a validation split of training
patients. Training recipes are the originals'; only the protocol changes.

  cnn4ch  StagingCNN(in_ch=4) on C4:M1, C3:M2, O2:M1, O1:M2; 20 epochs, batch 1024,
          Adam 1.5e-3, cosine, class-weighted CE, per-recording channel z-score
  rawmm   MultimodalSleepNet(fusion="cross") on 7 neural + 7 cardiorespiratory
          raw channels, 20-epoch windows (stride 10 in training), AdamW 1e-3, up to
          40 epochs, patience 6, staging CE + 0.5 x respiratory BCE (pos-weighted)

Staging is reported raw (the original decode) and with the HMM Viterbi decode the
other rows use; the raw CNN also reports respiratory AUC and AP.

  KMP_DUPLICATE_LIB_OK=TRUE python run_multichannel_10fold.py [cnn4ch rawmm] [--seeds 42 1 7]
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
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             f1_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "train", "legacy_models"))
from run_resp_raw_baseline import FOLDS, SUBJECTS   # noqa: E402  (same 99-patient folds)

MM = os.path.join(REPO, "data", "multimodal")
P7 = os.path.join(REPO, "data", "processed7")
OUTDIR = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NC, EPS, L = 5, 1e-12, 20


def hmm_decode(tr_labels, logp):
    """Viterbi with transition and initial probabilities counted on training labels."""
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for y in tr_labels:
        pi[y[0]] += 1
        np.add.at(A, (y[:-1], y[1:]), 1)
    A = np.log(A / A.sum(1, keepdims=True)); pi = np.log(pi / pi.sum())
    n = len(logp); dp = np.zeros((n, NC)); bp = np.zeros((n, NC), np.int64)
    dp[0] = pi + logp[0]
    for t in range(1, n):
        s = dp[t - 1][:, None] + A
        bp[t] = s.argmax(0); dp[t] = s.max(0) + logp[t]
    path = np.zeros(n, np.int64); path[-1] = dp[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def stage_metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)),
                mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y, p)))


# ------------------------------------------------------------------ cnn4ch
def load_cnn4ch():
    data = {}
    for sid in SUBJECTS:
        d = np.load(os.path.join(P7, "SN%d.npz" % sid), allow_pickle=True)
        x = d["x"][:, :4].astype(np.float32)
        x = (x - x.mean((0, 2), keepdims=True)) / (x.std((0, 2), keepdims=True) + 1e-6)
        data[sid] = (np.nan_to_num(x).astype(np.float16), d["y"].astype(np.int64))
    return data                          # float16 in RAM, cast to float32 on the GPU


def cnn4ch_fold(data, tr, va, te, seed, epochs=20, bs=1024, lr=1.5e-3):
    from staging_cnn import StagingCNN
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.from_numpy(np.concatenate([data[s][0] for s in tr]))
    Y = np.concatenate([data[s][1] for s in tr])
    cnt = np.bincount(Y, minlength=5)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)),
                                                   dtype=torch.float32, device=DEV))
    Y = torch.from_numpy(Y)
    model = StagingCNN(in_ch=4, dropout=0.5).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    @torch.no_grad()
    def probs(subs):
        model.eval(); out = {}
        for s in subs:
            x = data[s][0]; p = []
            for i in range(0, len(x), 1024):
                with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                    p.append(model(torch.from_numpy(x[i:i + 1024]).to(DEV).float()).float()
                             .softmax(-1).cpu().numpy())
            out[s] = np.concatenate(p)
        return out

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                loss = crit(model(X[b].to(DEV).float()), Y[b].to(DEV))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        pv = probs(va)
        v = f1_score(np.concatenate([data[s][1] for s in va]),
                     np.concatenate([pv[s].argmax(1) for s in va]), average="macro")
        if v > best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt = probs(te)
    yt = np.concatenate([data[s][1] for s in te])
    raw = np.concatenate([pt[s].argmax(1) for s in te])
    hmm = np.concatenate([hmm_decode([data[s][1] for s in tr], np.log(pt[s] + EPS)) for s in te])
    return dict(raw=stage_metrics(yt, raw), hmm=stage_metrics(yt, hmm), n=int(len(yt)))


# ------------------------------------------------------------------- rawmm
def load_rawmm():
    data = {}
    for sid in SUBJECTS:
        d = np.load(os.path.join(MM, "SN%d.npz" % sid), allow_pickle=True)
        e = d["eeg"].astype(np.float32); c = d["card"].astype(np.float32)
        e = (e - e.mean((0, 2), keepdims=True)) / (e.std((0, 2), keepdims=True) + 1e-6)
        c = (c - c.mean((0, 2), keepdims=True)) / (c.std((0, 2), keepdims=True) + 1e-6)
        data[sid] = (np.nan_to_num(e).astype(np.float16), np.nan_to_num(c).astype(np.float16),
                     d["y"].astype(np.int64), d["apnea"].astype(np.float32))
    return data


def rawmm_fold(data, tr, va, te, seed, epochs=40, patience=6, bs=16, lr=1e-3, apnea_w=0.5):
    from multimodal_net import MultimodalSleepNet
    torch.manual_seed(seed); np.random.seed(seed)
    idx = [(s, st) for s in tr for st in range(0, max(1, len(data[s][2]) - L + 1), L // 2)]
    cnt = np.bincount(np.concatenate([data[s][2] for s in tr]), minlength=5)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)),
                                                 dtype=torch.float32, device=DEV), reduction="none")
    a = np.concatenate([data[s][3] for s in tr])
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(
        [(len(a) - a.sum()) / max(1.0, a.sum())], dtype=torch.float32, device=DEV))
    model = MultimodalSleepNet(fusion="cross").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    def batch(items):
        E, Cd, Y, A, M = [], [], [], [], []
        for s, st in items:
            e, c, y, ap = data[s]
            k = len(y[st:st + L]); pad = L - k
            E.append(np.pad(e[st:st + L], ((0, pad), (0, 0), (0, 0))))
            Cd.append(np.pad(c[st:st + L], ((0, pad), (0, 0), (0, 0))))
            Y.append(np.pad(y[st:st + L], (0, pad))); A.append(np.pad(ap[st:st + L], (0, pad)))
            M.append(np.pad(np.ones(k, np.float32), (0, pad)))
        f = lambda z, t: torch.from_numpy(np.stack(z).astype(t)).to(DEV)   # noqa: E731
        return (f(E, np.float32), f(Cd, np.float32), f(Y, np.int64), f(A, np.float32),
                f(M, np.float32))

    @torch.no_grad()
    def infer(subs):
        model.eval(); out = {}
        for s in subs:
            n = len(data[s][2])
            items = [(s, st) for st in range(0, n, L)]
            so, ao = [], []
            for i in range(0, len(items), 8):
                e, c, _, _, m = batch(items[i:i + 8])
                with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                    s_o, a_o = model(e, c)
                mk = m.reshape(-1) > 0
                so.append(s_o.float().softmax(-1).reshape(-1, 5)[mk].cpu().numpy())
                ao.append(torch.sigmoid(a_o.float()).reshape(-1)[mk].cpu().numpy())
            out[s] = (np.concatenate(so), np.concatenate(ao))
        return out

    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(idx))
        for i in range(0, len(perm) - bs + 1, bs):
            e, c, y, ap, m = batch([idx[j] for j in perm[i:i + bs]])
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                s_o, a_o = model(e, c)
                mm = m.reshape(-1)
                ls = (ce(s_o.reshape(-1, 5).float(), y.reshape(-1)) * mm).sum() / mm.sum().clamp(min=1)
                la = (bce(a_o.reshape(-1).float(), ap.reshape(-1)) * mm).sum() / mm.sum().clamp(min=1)
                loss = ls + apnea_w * la
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        sch.step()
        pv = infer(va)
        v = f1_score(np.concatenate([data[s][2] for s in va]),
                     np.concatenate([pv[s][0].argmax(1) for s in va]), average="macro")
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    pt = infer(te)
    yt = np.concatenate([data[s][2] for s in te])
    raw = np.concatenate([pt[s][0].argmax(1) for s in te])
    hmm = np.concatenate([hmm_decode([data[s][2] for s in tr], np.log(pt[s][0] + EPS))
                          for s in te])
    at = np.concatenate([data[s][3] for s in te]); ap = np.concatenate([pt[s][1] for s in te])
    return dict(raw=stage_metrics(yt, raw), hmm=stage_metrics(yt, hmm), n=int(len(yt)),
                auc=float(roc_auc_score(at, ap)), ap=float(average_precision_score(at, ap)))


def deepsleep_fold(data, tr, va, te, seed):
    """The two-stage DeepSleepNet of the table row (HAGNet train_deepsleep.py), via the
    same in-memory loop the Sleep-EDF control uses, on the four EEG derivations. The
    loop selects on validation macro-F1 and decodes by argmax, as the original does."""
    import run_healthy_extra as H
    yt, yp = H.deepsleep_fold(data, tr, va, te, seed)
    return dict(raw=stage_metrics(yt, yp), n=int(len(yt)))


MODELS = {"deepsleep": (load_cnn4ch, deepsleep_fold, "DeepSleepNet two-stage, 4 EEG"),
          "cnn4ch": (load_cnn4ch, cnn4ch_fold, "4-EEG CNN (cnn4ch, the CNN+BiLSTM row)"),
          "rawmm": (load_rawmm, rawmm_fold, "raw multimodal CNN, 14 channels, cross fusion")}


def main(which, seeds):
    loader, fold_fn, name = MODELS[which]
    data = loader()
    out = os.path.join(OUTDIR, "multichannel10_%s.json" % which)
    print("%s | %d patients | 10 folds x seeds %s | %s" % (name, len(data), seeds, DEV), flush=True)
    # resume: keep seeds already completed (all ten folds) in an earlier run
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
            r = fold_fn(data, tr_all[nv:], tr_all[:nv], te, seed)
            r["fold"] = fi; rows.append(r)
            extra = ("  AUC %.4f AP %.4f" % (r["auc"], r["ap"])) if "auc" in r else ""
            dec = r.get("hmm", r["raw"])
            print("  seed %-2d fold %d  raw acc %.4f mF1 %.4f | %s acc %.4f mF1 %.4f k %.4f%s  [%.1f min]"
                  % (seed, fi, r["raw"]["acc"], r["raw"]["mf1"], "hmm" if "hmm" in r else "raw",
                     dec["acc"], dec["mf1"], dec["kappa"], extra, (time.time() - t0) / 60), flush=True)
            res["seed|%d" % seed] = rows
            json.dump(dict(partial=True, per_seed=res), open(out, "w"), indent=1)
    summ = {}
    first = next(iter(res.values()))
    for dec in [d for d in ("raw", "hmm") if d in first[0]]:
        summ[dec] = {m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
                     for m in ("acc", "mf1", "kappa")
                     for v in [[r[dec][m] for rows in res.values() for r in rows]]}
    if which == "rawmm":
        for m in ("auc", "ap"):
            v = [r[m] for rows in res.values() for r in rows]
            summ[m] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
    json.dump(dict(model=name, protocol="10-fold patient-independent over the 99 patients "
                   "(mmnet_core.make_folds), seeds %s, checkpoint on a validation split of "
                   "training patients" % sorted(int(k.split("|")[1]) for k in res), per_seed=res, summary=summ,
                   minutes=(time.time() - t0) / 60), open(out, "w"), indent=1)
    print("SUMMARY [%s] %s" % (which, json.dumps(summ)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["cnn4ch", "rawmm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    a = ap.parse_args()
    for w in a.models:
        main(w, a.seeds)
