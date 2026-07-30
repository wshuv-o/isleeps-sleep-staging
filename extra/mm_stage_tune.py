"""
mm_stage_tune.py -- close the deep joint model's staging gap to the boosting ensemble.

Diagnosis of why the deep model stages at 0.68 while boosting gets 0.746 on the SAME
features:
  (1) class-weighted CE upweights rare N1/N3 -> trades overall accuracy for macro-F1
  (2) L=20 (10 min) context is short; sleep cycles are ~90 min
  (3) no transition smoothing; boosting uses an HMM Viterbi pass that removes implausible
      1-epoch stage flips

Fixes, tested against the baseline on ONE subject-independent split (fast) before we spend
a 10-fold run:
  - gentler class weighting (sqrt of inverse-freq, or none)
  - longer context L
  - HMM Viterbi smoothing of the model's per-epoch stage posteriors (post-hoc, train-fit A)

Reports staging acc/mF1/kappa raw and +HMM, for cross fusion (the joint model).

  KMP_DUPLICATE_LIB_OK=TRUE python extra/mm_stage_tune.py
"""
import os, sys, glob, time, itertools
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model")); sys.path.insert(0, os.path.join(ROOT, "utils"))
from mm_feature_net import MMFeatureNet  # noqa
try:
    from datasets import DUPLICATE_DROP  # noqa
except Exception:
    DUPLICATE_DROP = {28}
FE = os.path.join(ROOT, "data", "mm_features")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NC, EPS = 5, 1e-12


def load_all():
    data = {}
    for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUPLICATE_DROP:
            continue
        d = np.load(f)
        Fe = np.nan_to_num(d["Feeg"]).astype(np.float32); Fc = np.nan_to_num(d["Fcard"]).astype(np.float32)
        Fe = (Fe - Fe.mean(0)) / (Fe.std(0) + 1e-6); Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-6)
        data[sid] = (Fe, Fc, d["y"].astype(np.int64), d["apnea"].astype(np.int64))
    return data


class WinDS(Dataset):
    def __init__(self, data, subs, L, stride):
        self.data, self.L, self.idx = data, L, []
        for s in subs:
            n = len(data[s][2])
            for st in range(0, max(1, n - L + 1), stride):
                self.idx.append((s, st))

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, st = self.idx[i]; L = self.L; fe, fc, y, a = self.data[s]
        fe, fc, yy, aa = fe[st:st+L], fc[st:st+L], y[st:st+L], a[st:st+L]
        m = np.ones(len(yy), np.float32)
        if len(yy) < L:
            k = L - len(yy)
            fe = np.concatenate([fe, np.zeros((k, fe.shape[1]), np.float32)])
            fc = np.concatenate([fc, np.zeros((k, fc.shape[1]), np.float32)])
            yy = np.concatenate([yy, np.zeros(k, np.int64)]); aa = np.concatenate([aa, np.zeros(k, np.int64)])
            m = np.concatenate([m, np.zeros(k, np.float32)])
        return (torch.from_numpy(fe), torch.from_numpy(fc), torch.from_numpy(yy),
                torch.from_numpy(aa.astype(np.float32)), torch.from_numpy(m))


def hmm(A_log, pi_log, logprob):
    T = logprob.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = pi_log + logprob[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + A_log; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + logprob[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


@torch.no_grad()
def infer_probs(model, data, subs, L):
    out = {}
    model.eval()
    for s in subs:
        fe, fc, y, a = data[s]; n = len(y); pad = (-n) % L
        fe2, fc2 = fe, fc
        if pad:
            fe2 = np.concatenate([fe, np.zeros((pad, fe.shape[1]), np.float32)])
            fc2 = np.concatenate([fc, np.zeros((pad, fc.shape[1]), np.float32)])
        fe2 = fe2.reshape(-1, L, fe.shape[1]); fc2 = fc2.reshape(-1, L, fc.shape[1])
        so = []
        for i in range(0, len(fe2), 16):
            s_o, _ = model(torch.from_numpy(fe2[i:i+16]).to(DEV), torch.from_numpy(fc2[i:i+16]).to(DEV))
            so.append(s_o.softmax(-1).reshape(-1, 5).cpu().numpy())
        out[s] = (np.concatenate(so)[:n], y)
    return out


def cw(tr, data, mode):
    cc = np.zeros(5, np.int64)
    for s in tr: cc += np.bincount(data[s][2], minlength=5)
    if mode == "none":
        return torch.ones(5, device=DEV)
    inv = cc.sum() / (5 * np.maximum(cc, 1))
    if mode == "sqrt":
        inv = np.sqrt(inv)
    return torch.tensor(inv / inv.mean(), dtype=torch.float32, device=DEV)


def run(data, tr, va, te, L, cwmode, epochs=40, patience=7):
    torch.manual_seed(42); np.random.seed(42)
    dl = DataLoader(WinDS(data, tr, L, L // 2), batch_size=32, shuffle=True, drop_last=True)
    model = MMFeatureNet(fusion="cross", drop=0.3).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=cw(tr, data, cwmode), reduction="none")
    ac = np.zeros(2, np.int64)
    for s in tr: ac += np.bincount(data[s][3], minlength=2)
    pw = torch.tensor([ac[0] / max(1, ac[1])], dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    # transition + prior for HMM (from train labels)
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for s in tr:
        y = data[s][2]; pi[y[0]] += 1
        for x, z in zip(y[:-1], y[1:]): A[x, z] += 1
    A_log = np.log(A / A.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for fe, fc, y, apn, m in dl:
            fe, fc, y, apn, m = fe.to(DEV), fc.to(DEV), y.to(DEV), apn.to(DEV), m.to(DEV)
            opt.zero_grad(); s_o, a_o = model(fe, fc)
            ls = (ce(s_o.reshape(-1, 5), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            la = (bce(a_o.reshape(-1), apn.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            (ls + 0.5 * la).backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        pr = infer_probs(model, data, va, L)
        y = np.concatenate([pr[s][1] for s in va]); p = np.concatenate([pr[s][0].argmax(1) for s in va])
        vacc = accuracy_score(y, p)
        if vacc > best:
            best, bad = vacc, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    pr = infer_probs(model, data, te, L)
    yt = np.concatenate([pr[s][1] for s in te])
    praw = np.concatenate([pr[s][0].argmax(1) for s in te])
    phmm = np.concatenate([hmm(A_log, pi_log, np.log(pr[s][0] + EPS)) for s in te])
    def M(p): return (accuracy_score(yt, p), f1_score(yt, p, average="macro", zero_division=0), cohen_kappa_score(yt, p))
    return M(praw), M(phmm)


def main():
    data = load_all(); subs = sorted(data)
    rng = np.random.RandomState(42); order = subs[:]; rng.shuffle(order)
    n = len(order); te = order[:n//5]; va = order[n//5:n//5+12]; tr = order[n//5+12:]
    print(f"{len(subs)} subj | train {len(tr)} val {len(va)} test {len(te)} | {DEV}", flush=True)
    print(f"{'config':28s} {'raw acc/mF1/k':>22s}   {'+HMM acc/mF1/k':>22s}", flush=True)
    for L, cwmode in itertools.product([20, 40], ["balanced", "sqrt", "none"]):
        t0 = time.time()
        raw, hm = run(data, tr, va, te, L, cwmode)
        print(f"L={L:<3d} cw={cwmode:9s}          "
              f"{raw[0]:.3f}/{raw[1]:.3f}/{raw[2]:.3f}   {hm[0]:.3f}/{hm[1]:.3f}/{hm[2]:.3f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("\nboosting ensemble ref: acc 0.7464 / mF1 0.6753 / kappa 0.6415")


if __name__ == "__main__":
    main()
