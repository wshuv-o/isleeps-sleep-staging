"""Fine-tune LaBraM on raw stroke-sleep EEG, one fold at a time.

The frozen LaBraM embedding MM-Net uses was never trained on sleep. Here the same
pretrained encoder (braindecode/labram-pretrained) is trained further, on the fold's
TRAINING patients, to classify the sleep stage of each 30 s epoch from the four EEG
derivations alone. Early stopping uses the fold's validation patients; the test
patients are only scored. Folds and the train/validation split are mmnet_core's.

Input exactly as build_labram_cache.py prepares it: C4:M1 C3:M2 O2:M1 O1:M2 named
C4 C3 O2 O1, 100 -> 200 Hz, uV/100, the 30 s epoch split into two 15 s windows of 3000
samples, each encoded, the two embeddings averaged; a linear head gives the stage.

Reported per fold (per-epoch argmax, no temporal model, no HMM):
    fine-tuned LaBraM vs the FROZEN LaBraM embedding with a logistic-regression head
    trained on the same patients -- the question is how much fine-tuning adds.
Saves the fine-tuned 200-d embeddings for every patient of the fold to
data/labram_ft/fold<k>/SN<n>.npz (training patients' embeddings are in-sample).

Runs with the prep interpreter (braindecode needs its torchaudio):
    G:/codes/isleeps/prepenv/Scripts/python finetune_labram.py --fold 0 [--passes 3]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import mmnet_core as C

P7 = os.path.join(REPO, "data", "processed7")
EMB = os.path.join(REPO, "data", "cbramod_emb", "labram")
OUTD = os.path.join(REPO, "data", "labram_ft")
RES = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "labram_ft")
CH = ["C4", "C3", "O2", "O1"]
NCH = 4          # --eog: + E1:M2, E2:M2 placed at F7 / F8 (frontal sites that carry eye movements)
DEV = "cuda"


def split(fi, tr_all):
    rng = np.random.RandomState(100 + fi)
    tr_all = list(tr_all)
    rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    return tr_all[nv:], tr_all[:nv]


def load_eeg():
    X, Y = {}, {}
    for s in C.SUBS:
        d = np.load(os.path.join(P7, "SN%d.npz" % s))
        X[s] = d["x"][:, :NCH].astype(np.float16)          # [n, NCH, 3000] uV @ 100 Hz
        Y[s] = d["y"].astype(np.int64)
        assert np.array_equal(Y[s], C.DATA[s][2]), s
    return X, Y


class Stager(nn.Module):
    def __init__(self):
        super().__init__()
        from braindecode.models import Labram
        self.enc = Labram.from_pretrained("braindecode/labram-pretrained")
        self.head = nn.Linear(200, 5)

    def embed(self, x):
        """x [b, 4, 3000] uV @ 100 Hz -> [b, 200]."""
        x = AF.resample(x.float(), 100, 200) / 100.0     # [b, 4, 6000]
        b = x.shape[0]
        c = x.shape[1]
        sub = x.reshape(b, c, 2, 3000).permute(0, 2, 1, 3).reshape(b * 2, c, 3000)
        return self.enc(sub, ch_names=CH).float().reshape(b, 2, -1).mean(1)

    def forward(self, x):
        return self.head(self.embed(x))


@torch.no_grad()
def predict(model, X, subs, bs=128):
    model.eval()
    P, E = {}, {}
    for s in subs:
        ps, es = [], []
        for i in range(0, len(X[s]), bs):
            x = torch.from_numpy(X[s][i:i + bs]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                e = model.embed(x)
            ps.append(model.head(e.float()).softmax(-1).cpu().numpy())
            es.append(e.cpu().numpy())
        P[s], E[s] = np.concatenate(ps), np.concatenate(es)
    return P, E


def metrics(Y, P, subs):
    yt = np.concatenate([Y[s] for s in subs])
    yp = np.concatenate([P[s].argmax(1) for s in subs])
    return dict(acc=float(accuracy_score(yt, yp)),
                mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(yt, yp)))


def frozen_baseline(Y, tr, va, te):
    def load(s):
        e = np.load(os.path.join(EMB, "SN%d.npz" % s))["emb"].astype(np.float32)
        return (e - e.mean(0)) / (e.std(0) + 1e-6)
    Xtr = np.concatenate([load(s) for s in tr]); ytr = np.concatenate([Y[s] for s in tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return {sp: metrics(Y, {s: clf.predict_proba(load(s)) for s in subs}, subs)
            for sp, subs in (("va", va), ("te", te))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--passes", type=float, default=3.0, help="passes over the training epochs")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--evals-per-pass", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # cross-fitting: train on the fold's training patients minus part k of K, and save
    # embeddings only for part k (so each training patient's embedding is out-of-sample)
    ap.add_argument("--cf-part", type=int, default=None)
    ap.add_argument("--cf-parts", type=int, default=3)
    ap.add_argument("--eog", action="store_true", help="also feed E1/E2 to LaBraM as F7/F8")
    a = ap.parse_args()
    global CH, NCH
    if a.eog:
        CH, NCH = CH + ["F7", "F8"], 6
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(RES, exist_ok=True)

    tr_all, te = C.FOLDS[a.fold]
    tr, va = split(a.fold, tr_all)
    te = list(te)
    held = []
    if a.cf_part is not None:
        rng = np.random.RandomState(1000 + a.fold)
        order = sorted(tr)
        rng.shuffle(order)
        held = order[a.cf_part::a.cf_parts]
        tr = [s for s in tr if s not in held]
        print("cross-fit part %d/%d: %d patients held out of training" % (a.cf_part, a.cf_parts, len(held)), flush=True)
    X, Y = load_eeg()
    idx = np.array([(s, i) for s in tr for i in range(len(Y[s]))])
    print("fold %d: %d train patients (%d epochs), %d validation, %d test"
          % (a.fold, len(tr), len(idx), len(va), len(te)), flush=True)

    t0 = time.time()
    base = frozen_baseline(Y, tr, va, te) if a.cf_part is None else None
    if base:
        print("frozen LaBraM + logistic regression: val acc %.4f | test acc %.4f  (%.1f min)"
              % (base["va"]["acc"], base["te"]["acc"], (time.time() - t0) / 60), flush=True)

    model = Stager().to(DEV)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps = int(a.passes * len(idx) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss()
    eval_every = max(1, int(len(idx) / a.bs / a.evals_per_pass))
    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    for step in range(steps):
        model.train()
        b = idx[np.random.randint(0, len(idx), a.bs)]
        x = torch.from_numpy(np.stack([X[int(s)][int(i)] for s, i in b])).to(DEV)
        y = torch.tensor([Y[int(s)][int(i)] for s, i in b], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ce(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            Pv, _ = predict(model, X, va)
            m = metrics(Y, Pv, va)
            hist.append(dict(step=step + 1, loss=float(loss), **m))
            print("  step %5d/%d  loss %.3f  val acc %.4f  mf1 %.4f  [%.1f min]"
                  % (step + 1, steps, float(loss), m["acc"], m["mf1"], (time.time() - t0) / 60), flush=True)
            if m["acc"] > best:
                best = m["acc"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    if a.cf_part is not None:
        P, E = predict(model, X, held)
        od = os.path.join(REPO, "data", "labram_cf" + ("_eog" if a.eog else ""), "fold%d" % a.fold)
        os.makedirs(od, exist_ok=True)
        for s in held:
            np.savez_compressed(os.path.join(od, "SN%d.npz" % s), emb=E[s].astype(np.float16),
                                prob=P[s].astype(np.float32), y=Y[s])
        m = metrics(Y, P, held)
        print("cross-fit part %d: held-out training patients acc %.4f (best val %.4f)"
              % (a.cf_part, m["acc"], best), flush=True)
        return

    P, E = predict(model, X, C.SUBS)
    ft = {"va": metrics(Y, P, va), "te": metrics(Y, P, te)}
    sfx = ("" if a.seed == 42 else "_s%d" % a.seed) + ("_eog" if a.eog else "")   # own folders
    od = os.path.join(OUTD + sfx, "fold%d" % a.fold)
    os.makedirs(od, exist_ok=True)
    for s in C.SUBS:
        np.savez_compressed(os.path.join(od, "SN%d.npz" % s), emb=E[s].astype(np.float16),
                            prob=P[s].astype(np.float32), y=Y[s])
    out = dict(fold=a.fold, args=vars(a), frozen=base, finetuned=ft, history=hist,
               minutes=(time.time() - t0) / 60)
    json.dump(out, open(os.path.join(RES, "fold%d%s.json" % (a.fold, sfx)), "w"), indent=1)
    print("\nfold %d, per-epoch argmax (no temporal model):" % a.fold)
    for sp in ("va", "te"):
        print("  %s  frozen acc %.4f mf1 %.4f | fine-tuned acc %.4f mf1 %.4f  (%+.4f)"
              % (sp, base[sp]["acc"], base[sp]["mf1"], ft[sp]["acc"], ft[sp]["mf1"],
                 ft[sp]["acc"] - base[sp]["acc"]), flush=True)


if __name__ == "__main__":
    main()
