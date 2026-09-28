"""The remaining retrained baselines on healthy sleep, for the Healthy column of Table 5.

run_healthy_baselines.py filled the column for TinySleepNet and SleepTransformer.
This does the same for the four other rows whose code we have: the LSTM
reimplementation, AttnSleep, DeepSleepNet and the four-channel CNN. Same corpus
(Sleep-EDF Expanded, sleep-cassette), same subject-level folds, same validation
split rule, so the rows are comparable with each other and with the ones above.

Each model's own `run_fold` is imported where one exists, so it is the stroke
training loop that is validated, not a copy. DeepSleepNet and the CNN come from
HAGNet_research, whose loops read iSLEEPS from disk; they are reproduced here with
the same hyper-parameters over in-memory arrays.

Sleep-EDF has two EEG derivations, not four, and that forces two departures that
belong in the table caption:

  * DeepSleepNet and the CNN ran on C4:M1, C3:M2, O2:M1, O1:M2 on stroke; here
    they take Fpz-Cz and Pz-Oz. The input stem is the only layer that changes.
  * The LSTM reimplementation's EOG input is Sleep-EDF's horizontal EOG.

The CNN's stroke run selected its epoch on test macro-F1 (train.py); here it
selects on the validation split like every other row, which can only lower it.

  KMP_DUPLICATE_LIB_OK=TRUE python run_healthy_extra.py [lstm attnsleep deepsleep cnn]
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "train", "legacy_models"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score  # noqa: E402

PROC = os.path.join(REPO, "data", "sleep_edf_proc")
OUTDIR = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 10
FPZ, PZ, EOG = 0, 1, 2

MODELS = {
    "lstm": ("LSTM (our reimplementation of the corpus paper's SE-ResNet-LSTM)",
             "Fpz-Cz + horizontal EOG, 5-epoch context"),
    "attnsleep": ("AttnSleep (braindecode reference implementation)", "Fpz-Cz"),
    "deepsleep": ("DeepSleepNet (two-stage, HAGNet_research/train_deepsleep.py)",
                  "Fpz-Cz + Pz-Oz (4 EEG on stroke)"),
    "cnn": ("CNN, the 4-channel CNN+BiLSTM row (HAGNet_research/train.py, cnn4ch)",
            "Fpz-Cz + Pz-Oz (4 EEG on stroke)"),
    "bose": ("SE-ResNet-18 + BiLSTM (Bose et al., released code; run_bose_seresnet.py)",
             "Fpz-Cz, 9-epoch window, decoded as released"),
}


def subject_of(rec):
    """SC4ssNE: characters 3:5 are the subject, so SC4001 and SC4002 are one person."""
    return rec[3:5]


def zscore(x, axis):
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-6)


def load(which):
    """Per recording, in the representation and normalisation each stroke loader uses."""
    data = {}
    for f in sorted(glob.glob(os.path.join(PROC, "*.npz"))):
        d = np.load(f)
        x, y = d["x"].astype(np.float32), d["y"].astype(np.int64)
        if which == "lstm":            # per epoch, per channel
            x = zscore(x[:, [FPZ, EOG]], -1)
        elif which == "attnsleep":     # per epoch
            x = zscore(x[:, FPZ], -1)
        elif which == "bose":          # per recording, as in run_bose_seresnet.load
            x = zscore(x[:, FPZ], None)
        else:                          # per recording, per channel
            x = zscore(x[:, [FPZ, PZ]], (0, 2))
        data[os.path.basename(f)[:-4]] = (np.ascontiguousarray(x), y)
    return data


# ---------------------------------------------------------------- DeepSleepNet
def seq_chunks(data, subs, L, stride):
    """HAGNet's SequenceDataset over in-memory arrays: [N, L, C, T], labels, mask.

    Evaluation (stride == L) pads the tail instead of adding an overlapping last
    chunk, which in the original scored some epochs twice.
    """
    X, Y, M = [], [], []
    for s in subs:
        x, y = data[s]
        n = len(y)
        if stride == L:
            starts = list(range(0, n, L))
        else:
            starts = [0] if n <= L else list(range(0, n - L + 1, stride))
            if n > L and starts[-1] + L < n:
                starts.append(n - L)
        for a in starts:
            xs, ys = x[a:a + L], y[a:a + L]
            k = len(ys)
            m = np.ones(L, np.float32); m[k:] = 0
            if k < L:
                xs = np.concatenate([xs, np.zeros((L - k,) + xs.shape[1:], xs.dtype)])
                ys = np.concatenate([ys, np.zeros(L - k, np.int64)])
            X.append(xs); Y.append(ys); M.append(m)
    return np.stack(X), np.stack(Y), np.stack(M)


def balanced_w(counts):
    return torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                        dtype=torch.float32, device=DEV)


def deepsleep_fold(data, tr, va, te, seed, L=20, pre_epochs=10, epochs=35,
                   bs=64, bs_pre=512, lr=1e-3, dropout=0.5):
    from deepsleep import DeepSleepSeq
    torch.manual_seed(seed); np.random.seed(seed)
    model = DeepSleepSeq(in_ch=data[tr[0]][0].shape[1], dropout=dropout).to(DEV)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    # stage 1: encoder on single epochs, class-balanced sampling
    Xe = torch.from_numpy(np.concatenate([data[s][0] for s in tr]))
    Ye = np.concatenate([data[s][1] for s in tr])
    cnt = np.bincount(Ye, minlength=5)
    p = (cnt.sum() / (5 * np.maximum(cnt, 1)))[Ye]
    Ye = torch.from_numpy(Ye)
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.proj.parameters())
                           + list(model.epoch_head.parameters()), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(pre_epochs):
        model.train()
        idx = torch.multinomial(torch.tensor(p, dtype=torch.double), len(p), replacement=True)
        for i in range(0, len(idx), bs_pre):
            b = idx[i:i + bs_pre]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                loss = ce(model.classify_epoch(Xe[b].to(DEV).float()), Ye[b].to(DEV))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    del Xe

    # stage 2: full sequence model, masked weighted loss, select on val macro-F1
    Xtr, Ytr, Mtr = (torch.from_numpy(a) for a in seq_chunks(data, tr, L, L // 2))
    vpack = seq_chunks(data, va, L, L)
    w = balanced_w(np.bincount(Ytr.numpy()[Mtr.numpy() > 0], minlength=5))
    enc = list(model.encoder.parameters()) + list(model.proj.parameters())
    new = (list(model.lstm.parameters()) + list(model.res.parameters())
           + list(model.head.parameters()))
    opt = torch.optim.Adam([{"params": enc, "lr": lr * 0.3}, {"params": new, "lr": lr}],
                           weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=w, reduction="none")

    @torch.no_grad()
    def predict(pack):
        model.eval()
        X, Y, M = pack
        yt, yp = [], []
        for i in range(0, len(X), 64):
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                o = model(torch.from_numpy(X[i:i + 64]).to(DEV).float()).argmax(-1).cpu().numpy()
            m = M[i:i + 64] > 0
            yt.append(Y[i:i + 64][m]); yp.append(o[m])
        return np.concatenate(yt), np.concatenate(yp)

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x, y, m = Xtr[b].to(DEV).float(), Ytr[b].to(DEV), Mtr[b].to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                loss = crit(model(x).reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        sch.step()
        v = f1_score(*predict(vpack), average="macro", zero_division=0)
        if v > best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(seq_chunks(data, te, L, L))


# ------------------------------------------------------------------------- CNN
def cnn_fold(data, tr, va, te, seed, epochs=20, bs=1024, lr=1.5e-3, dropout=0.5):
    from staging_cnn import StagingCNN
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.from_numpy(np.concatenate([data[s][0] for s in tr]))
    Y = np.concatenate([data[s][1] for s in tr])
    crit = nn.CrossEntropyLoss(weight=balanced_w(np.bincount(Y, minlength=5)))
    Y = torch.from_numpy(Y)
    model = StagingCNN(in_ch=2, dropout=dropout).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    @torch.no_grad()
    def predict(subs):
        model.eval()
        x = np.concatenate([data[s][0] for s in subs])
        p = []
        for i in range(0, len(x), 512):
            p.append(model(torch.from_numpy(x[i:i + 512]).to(DEV)).argmax(1).cpu().numpy())
        return np.concatenate([data[s][1] for s in subs]), np.concatenate(p)

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                loss = crit(model(X[b].to(DEV)), Y[b].to(DEV))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        v = f1_score(*predict(va), average="macro", zero_division=0)
        if v > best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(te)


def fold_fn(which):
    if which == "lstm":
        import run_isleeps_lstm as M
        return M.run_fold
    if which == "attnsleep":
        import run_attnsleep_seeded as M
        return M.run_fold
    if which == "bose":
        import run_bose_seresnet as M

        def bose_fold(data, tr, va, te, seed):
            yt, raw, _ = M.run_fold(data, tr, va, te, seed)   # raw decode, as on stroke
            return yt, raw
        return bose_fold
    return deepsleep_fold if which == "deepsleep" else cnn_fold


def main(which):
    name, inp = MODELS[which]
    run_fold = fold_fn(which)
    data = load(which)
    recs = sorted(data)
    subs = sorted({subject_of(r) for r in recs})
    print("%s on sleep-edf: %d recordings, %d subjects, %d epochs"
          % (name, len(recs), len(subs), sum(len(v[1]) for v in data.values())), flush=True)

    rng = np.random.RandomState(42)
    order = list(subs); rng.shuffle(order)
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]

    t0 = time.time()
    per_fold = []
    for fi, te_subs in enumerate(folds):
        te = [r for r in recs if subject_of(r) in te_subs]
        tr_all = [r for r in recs if subject_of(r) not in te_subs]
        rs = np.random.RandomState(100 + fi)
        tr_all = list(tr_all); rs.shuffle(tr_all)
        nv = max(2, len(tr_all) // 9)
        va, tr = tr_all[:nv], tr_all[nv:]
        yt, yp = run_fold(data, tr, va, te, seed=42)
        per_fold.append(dict(fold=fi, n_test_subjects=len(te_subs), n=int(len(yt)),
                             acc=float(accuracy_score(yt, yp)),
                             mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                             kappa=float(cohen_kappa_score(yt, yp))))
        print("fold %2d  %d subj  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
              % (fi, len(te_subs), len(yt), per_fold[-1]["acc"], per_fold[-1]["mf1"],
                 per_fold[-1]["kappa"], (time.time() - t0) / 60), flush=True)

    agg = {k: (float(np.mean([f[k] for f in per_fold])),
               float(np.std([f[k] for f in per_fold], ddof=1)))
           for k in ("acc", "mf1", "kappa")}
    out = os.path.join(OUTDIR, "healthy_%s.json" % which)
    json.dump(dict(model=name, corpus="sleep_edf", input=inp,
                   protocol="%d-fold, subject-independent, seed 42" % N_FOLDS,
                   n_recordings=len(recs), n_subjects=len(subs), per_fold=per_fold,
                   **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
                   minutes=(time.time() - t0) / 60), open(out, "w"), indent=1)
    print("\n%s ON HEALTHY SLEEP" % name.upper())
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, agg[k][0], agg[k][1]))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["cnn", "attnsleep", "lstm", "deepsleep"]):
        main(w)
