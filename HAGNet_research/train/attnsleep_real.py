"""
attnsleep_real.py -- run the REAL published AttnSleep (Eldele et al., IEEE TNSRE 2021)
on iSLEEPS, via braindecode's reference implementation. Single-channel EEG (C4:M1),
subject-independent folds, class-balanced loss, early stopping, HMM decoding.

This is the actual published architecture (MRCNN + SE recalibration + transformer
encoder), not a reimplementation. It reports ~0.84 accuracy on healthy Sleep-EDF.
Question: what does it score on 99 stroke patients?
"""
import os, sys, glob, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("utils",): sys.path.insert(0, os.path.join(ROOT, p))
from datasets import make_folds, DUPLICATE_DROP  # noqa
P7 = os.path.join(ROOT, "data", "processed7"); RES = os.path.join(ROOT, "results")
FS, NC, EPS, CH = 100, 5, 1e-12, 0                       # channel 0 = C4:M1
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

subs = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(P7, "SN*.npz")))
subs = [s for s in subs if s not in DUPLICATE_DROP]
X, Y = {}, {}
for s in subs:
    d = np.load(os.path.join(P7, f"SN{s}.npz"), allow_pickle=True)
    X[s] = d["x"][:, CH, :].astype(np.float32)           # single channel, [n,3000]
    Y[s] = d["y"].astype(np.int64)
print(f"loaded {len(subs)} subjects, single channel C4:M1 | device {DEV}", flush=True)


class DS(Dataset):
    def __init__(self, ss):
        self.it = [(s, i) for s in ss for i in range(len(Y[s]))]
    def __len__(self): return len(self.it)
    def __getitem__(self, k):
        s, i = self.it[k]; x = X[s][i]
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x[None, :]), int(Y[s][i])


def cw(ss):
    c = np.bincount(np.concatenate([Y[s] for s in ss]), minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=DEV)


def transmat(ss):
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for s in ss:
        y = Y[s]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A + EPS), np.log(pi + EPS)


def viterbi(le, lA, lpi):
    T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


def met(y, p):
    return dict(acc=float(accuracy_score(y, p)), mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y, p)),
                pcf=f1_score(y, p, average=None, labels=range(5), zero_division=0).tolist())


def main(folds=3, epochs=30, patience=5, batch=128):
    from braindecode.models import AttnSleep
    F = make_folds(subs, folds, seed=42); rng = np.random.RandomState(0)
    raw, hmm = [], []
    for k, (tr, te) in enumerate(F):
        vs = sorted(rng.choice(tr, size=12, replace=False).tolist())
        tc = [s for s in tr if s not in vs]
        model = AttnSleep(n_chans=1, n_outputs=5, n_times=3000, sfreq=FS).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss(weight=cw(tc))
        dl = DataLoader(DS(tc), batch_size=batch, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

        @torch.no_grad()
        def sp(s):
            x = X[s]; x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
            out = []
            for i in range(0, len(x), 256):
                out.append(torch.softmax(model(torch.from_numpy(x[i:i+256, None, :]).to(DEV)), -1).cpu().numpy())
            return np.concatenate(out)

        best_v, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad(); loss = crit(model(xb), yb)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sch.step()
            model.eval()
            vy = np.concatenate([Y[s] for s in vs]); vp = np.concatenate([sp(s).argmax(1) for s in vs])
            v = f1_score(vy, vp, average="macro", zero_division=0)
            if v > best_v: best_v, bad = v, 0; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else: bad += 1
            print(f"  f{k} ep{ep:02d} val_mF1={v:.4f} (best {best_v:.4f})", flush=True)
            if bad >= patience: print(f"  f{k} early stop ep{ep}", flush=True); break
        model.load_state_dict(best_state)
        lA, lpi = transmat(tc); model.eval()
        ty, pr, ph = [], [], []
        for s in te:
            p = sp(s); ty.append(Y[s]); pr.append(p.argmax(1)); ph.append(viterbi(np.log(p + EPS), lA, lpi))
        ty = np.concatenate(ty); raw.append(met(ty, np.concatenate(pr))); hmm.append(met(ty, np.concatenate(ph)))
        print(f"FOLD {k}: raw acc={raw[-1]['acc']:.4f} mF1={raw[-1]['mf1']:.4f} | "
              f"+HMM acc={hmm[-1]['acc']:.4f} mF1={hmm[-1]['mf1']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    agg = lambda rs, k: float(np.mean([r[k] for r in rs]))
    out = dict(model="AttnSleep (real, braindecode) single-ch C4:M1",
               raw=dict(acc=agg(raw,"acc"), mf1=agg(raw,"mf1"), kappa=agg(raw,"kappa")),
               hmm=dict(acc=agg(hmm,"acc"), mf1=agg(hmm,"mf1"), kappa=agg(hmm,"kappa")))
    json.dump(out, open(os.path.join(RES, "attnsleep_real.json"), "w"), indent=2)
    print("\n==== REAL AttnSleep on iSLEEPS ====")
    print(f"  raw : acc {out['raw']['acc']:.4f}  mF1 {out['raw']['mf1']:.4f}  kappa {out['raw']['kappa']:.4f}")
    print(f"  +HMM: acc {out['hmm']['acc']:.4f}  mF1 {out['hmm']['mf1']:.4f}  kappa {out['hmm']['kappa']:.4f}")
    print(f"  vs boosting ensemble: 0.7464 / 0.6753 / 0.6415")
    print(f"  vs AttnSleep on healthy Sleep-EDF (published): ~0.84 acc")


if __name__ == "__main__":
    main()
