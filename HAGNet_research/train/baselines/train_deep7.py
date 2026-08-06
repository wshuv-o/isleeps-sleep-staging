"""
train_deep7.py — deep CNN+BiLSTM on the 7-channel montage (GPU), + HMM, + saves
per-subject test probabilities so we can STACK with the classical ensemble.

Loads data/processed7 (7 ch: 4 EEG + 2 EOG + 1 EMG). Subject-independent CV. Saves
results/deep7_probs.npz (per-subject softmax probs, keyed by subject) for stacking.
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa
from staging_seq import StagingSeqNet  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
RESULTS = os.path.join(HERE, "results")
torch.backends.cudnn.benchmark = True
L = 20


def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load7(sid):
    d = np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)
    x = d["x"].astype(np.float32); y = d["y"].astype(np.int64)     # [n,7,3000]
    mu = x.mean(axis=(0, 2), keepdims=True); sd = x.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32), y


def list7():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def _starts(n, stride):
    if n <= L: return [0]
    st = list(range(0, n - L + 1, stride))
    if st[-1] + L < n: st.append(n - L)
    return st


class SeqDS(Dataset):
    """Memory-efficient: preallocate float16, fill subject-by-subject (no np.stack spike)."""
    def __init__(self, subs, stride):
        counts = [(s, len(np.load(os.path.join(PROC7, f"SN{s}.npz"))["y"])) for s in subs]
        total = sum(len(_starts(n, stride)) for _, n in counts)
        self.X = np.zeros((total, L, 7, 3000), np.float16)
        self.Y = np.zeros((total, L), np.int64)
        self.M = np.zeros((total, L), np.float32)
        idx = 0
        for s, n in counts:
            x, y = load7(s)
            for a in _starts(n, stride):
                xs, ys = x[a:a + L], y[a:a + L]; ln = len(ys)
                self.X[idx, :ln] = xs; self.Y[idx, :ln] = ys; self.M[idx, :ln] = 1.0
                idx += 1
            del x, y

    def counts(self): return np.bincount(self.Y[self.M.astype(bool)], minlength=5)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i].astype(np.float32), self.Y[i], self.M[i]


def augment(x, noise=0.05, scale=0.15, ch_drop=0.1, tmask=0.10):
    B, Ln, C, T = x.shape
    x = x * torch.empty(B, 1, C, 1, device=x.device).uniform_(1 - scale, 1 + scale)
    x = x + torch.randn_like(x) * noise
    keep = (torch.rand(B, 1, C, 1, device=x.device) > ch_drop).float()
    keep = torch.where(keep.sum(2, keepdim=True) == 0, torch.ones_like(keep), keep); x = x * keep
    w = int(T * tmask)
    if w > 0:
        st = torch.randint(0, T - w + 1, (1,), device=x.device).item(); x[..., st:st + w] = 0.0
    return x


def transition(seqs, n=5, eps=1.0):
    A = np.full((n, n), eps); pi = np.full(n, eps)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return A, pi


def viterbi(le, lA, lpi):
    T, S = le.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): p[t] = bp[t + 1, p[t + 1]]
    return p


@torch.no_grad()
def subj_probs(model, sid, device):
    x, y = load7(sid); n = len(y); pad = (-n) % L
    xp = np.concatenate([x, np.zeros((pad,) + x.shape[1:], np.float32)]) if pad else x
    seqs = xp.reshape(-1, L, x.shape[1], x.shape[2]); out = []
    for i in range(0, len(seqs), 16):
        xb = torch.tensor(seqs[i:i + 16], device=device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            lo = model(xb)
        out.append(torch.softmax(lo.float(), -1).reshape(-1, 5).cpu().numpy())
    return np.concatenate(out)[:n], y


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, p)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-subj", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = list7(); folds = make_folds(subs, args.folds, seed=args.seed); rng = np.random.RandomState(args.seed)
    print(f"device={device} | 7-ch deep CNN+BiLSTM | subjects={len(subs)} | {args.folds}-fold")
    deep_res, hmm_res = [], []; all_probs = {}
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        tr = SeqDS(tc, L)  # non-overlap stride = memory-safe for 7-channel sequences
        w = torch.tensor(tr.counts().sum() / (5 * np.maximum(tr.counts(), 1)),
                                                 dtype=torch.float32, device=device)
        tl = DataLoader(tr, batch_size=args.batch, shuffle=True, pin_memory=True)
        model = StagingSeqNet(in_ch=7, dropout=0.5).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(weight=w, reduction="none"); sc = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        A, pi = transition([load7(s)[1] for s in tc]); lA, lpi = np.log(A + 1e-12), np.log(pi + 1e-12)
        best_val = -1; best = None
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y, m in tqdm(tl, desc=f"fold{k} ep{ep:02d}", leave=False, ncols=80, mininterval=0.5):
                x, y, m = x.to(device), y.to(device), m.to(device); x = augment(x); opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    lo = model(x); loss = crit(lo.reshape(-1, 5), y.reshape(-1))
                    loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
                sc.scale(loss).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                sc.step(opt); sc.update()
            sch.step()
            vy = [load7(s)[1] for s in vs]; vp = [subj_probs(model, s, device)[0].argmax(1) for s in vs]
            vmf1 = f1_score(np.concatenate(vy), np.concatenate(vp), average="macro", zero_division=0)
            if vmf1 > best_val: best_val = vmf1; best = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best)
        ty, pd_, ph_ = [], [], []
        for s in te_s:
            pr, yy = subj_probs(model, s, device); all_probs[str(s)] = pr
            ty.append(yy); pd_.append(pr.argmax(1)); ph_.append(viterbi(np.log(pr + 1e-12), lA, lpi))
        ty = np.concatenate(ty); md = metrics(ty, np.concatenate(pd_)); mh = metrics(ty, np.concatenate(ph_))
        deep_res.append(md); hmm_res.append(mh)
        print(f"Fold {k}: deep acc={md['acc']:.3f} mF1={md['macro_f1']:.3f} | +HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")

    def summ(name, res):
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res]); kap = np.array([r["kappa"] for r in res])
        print(f"{name:14s} acc={acc.mean():.4f}+-{acc.std():.4f} mF1={mf1.mean():.4f} kappa={kap.mean():.4f}")
        return {"acc": float(acc.mean()), "acc_std": float(acc.std()), "macro_f1": float(mf1.mean()), "kappa": float(kap.mean())}
    print("\n===== 7-ch deep =====")
    out = {"deep": summ("deep", deep_res), "deep_hmm": summ("deep+HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, "deep7_all.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "deep7_probs.npz"), **all_probs)   # for stacking
    print("saved -> results/deep7_all.json, results/deep7_probs.npz")


if __name__ == "__main__":
    main()
