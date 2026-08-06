"""
train_deep7_strong.py — a PROPER strong deep model on all 99 subjects, 7-channel montage.

Key fixes vs earlier attempts:
  * LAZY loading: all epochs held once as compact float16 in RAM; sequence windows are
    SLICED on the fly in __getitem__ (no giant pre-expanded array, no OOM). The 6 GB GPU
    processes one batch at a time -> we use ALL the data, batch-wise.
  * STRONG architecture: DeepSleepNet-style dual-resolution CNN encoder + residual BiLSTM
    (deepsleep.DeepSleepSeq), in_ch=7, with two-stage training (pretrain encoder, then seq),
    EEG augmentation, cosine schedule, and HMM (Viterbi) decoding.
  * Saves per-subject out-of-fold probabilities (results/deep7_probs.npz) for stacking.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_deep7_strong.py --folds 10 --epochs 50
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa
from deepsleep import DeepSleepSeq  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
RESULTS = os.path.join(HERE, "results")
torch.backends.cudnn.benchmark = True
L = 20
DATA = {}   # sid -> (x_float16 [n,7,3000], y [n]); loaded once, shared across folds
NORM = {}   # sid -> (mu[1,7,1], sd[1,7,1])


def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def list7():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load_all(subs):
    for s in subs:
        if s in DATA: continue
        d = np.load(os.path.join(PROC7, f"SN{s}.npz"), allow_pickle=True)
        xf = d["x"].astype(np.float32); y = d["y"].astype(np.int64)
        mu = xf.mean((0, 2), keepdims=True); sd = xf.std((0, 2), keepdims=True) + 1e-6
        # store PRE-NORMALISED float16 -> __getitem__ is just a slice (keeps GPU fed)
        DATA[s] = (((xf - mu) / sd).astype(np.float16), y)


def _starts(n, stride):
    if n <= L: return [0]
    st = list(range(0, n - L + 1, stride))
    if st[-1] + L < n: st.append(n - L)
    return st


class LazySeq(Dataset):
    """Slices L-epoch windows from in-RAM float16 arrays on the fly. Tiny memory footprint."""
    def __init__(self, subs, stride):
        self.index = [(s, a) for s in subs for a in _starts(len(DATA[s][1]), stride)]

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        s, a = self.index[i]; x, y = DATA[s]
        xs = x[a:a + L].astype(np.float32); ys = y[a:a + L].copy(); m = np.ones(len(ys), np.float32)
        if len(ys) < L:
            p = L - len(ys)
            xs = np.concatenate([xs, np.zeros((p, 7, 3000), np.float32)])
            ys = np.concatenate([ys, np.zeros(p, np.int64)]); m = np.concatenate([m, np.zeros(p, np.float32)])
        return xs, ys, m


class LazyEpoch(Dataset):
    """Per-epoch (for stage-1 encoder pretraining), sliced from RAM."""
    def __init__(self, subs):
        self.index = [(s, e) for s in subs for e in range(len(DATA[s][1]))]
        self.y = np.array([DATA[s][1][e] for s, e in self.index])

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        s, e = self.index[i]; x, _ = DATA[s]
        return x[e].astype(np.float32), self.y[i]


def class_weights(subs, device):
    c = np.zeros(5, np.int64)
    for s in subs: c += np.bincount(DATA[s][1], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=device), c


def augment(x, noise=0.05, scale=0.15, ch_drop=0.1, tmask=0.10):
    B = x.shape[0]; C = x.shape[2]; T = x.shape[3]
    x = x * torch.empty(B, 1, C, 1, device=x.device).uniform_(1 - scale, 1 + scale)
    x = x + torch.randn_like(x) * noise
    keep = (torch.rand(B, 1, C, 1, device=x.device) > ch_drop).float()
    keep = torch.where(keep.sum(2, keepdim=True) == 0, torch.ones_like(keep), keep); x = x * keep
    w = int(T * tmask)
    if w > 0:
        st = torch.randint(0, T - w + 1, (1,), device=x.device).item(); x[..., st:st + w] = 0.0
    return x


def transition(subs, n=5, eps=1.0):
    A = np.full((n, n), eps); pi = np.full(n, eps)
    for s in subs:
        y = DATA[s][1]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A + 1e-12), np.log(pi + 1e-12)


def viterbi(le, lA, lpi):
    T, S = le.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): p[t] = bp[t + 1, p[t + 1]]
    return p


@torch.no_grad()
def subj_probs(model, sid, device):
    x, y = DATA[sid]; n = len(y); pad = (-n) % L
    xf = x.astype(np.float32)
    if pad: xf = np.concatenate([xf, np.zeros((pad, 7, 3000), np.float32)])
    seqs = xf.reshape(-1, L, 7, 3000); out = []
    for i in range(0, len(seqs), 8):
        xb = torch.tensor(seqs[i:i + 8], device=device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            lo = model(xb)
        out.append(torch.softmax(lo.float(), -1).reshape(-1, 5).cpu().numpy())
    return np.concatenate(out)[:n], y


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, p)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist()}


def train_fold(k, tc, vs, te, args, device):
    w, counts = class_weights(tc, device)
    model = DeepSleepSeq(in_ch=7, dropout=args.dropout).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ---- Stage 1: pretrain dual-res encoder on per-epoch classification (balanced) ----
    ep_ds = LazyEpoch(tc); sw = (counts.sum() / (5 * np.maximum(counts, 1)))[ep_ds.y]
    sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(ep_ds.y), replacement=True)
    el = DataLoader(ep_ds, batch_size=args.batch_pre, sampler=sampler, num_workers=0, pin_memory=True)
    opt1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.proj.parameters())
                            + list(model.epoch_head.parameters()), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, args.pre_epochs + 1):
        model.train()
        for x, y in tqdm(el, desc=f"f{k} pre{ep}", leave=False, ncols=80, mininterval=1):
            x, y = x.to(device), y.to(device); opt1.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = ce(model.classify_epoch(augment(x.unsqueeze(1)).squeeze(1)), y)
            scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()

    # ---- Stage 2: full sequence model ----
    tl = DataLoader(LazySeq(tc, L // 2), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    enc_p = list(model.encoder.parameters()) + list(model.proj.parameters())
    new_p = list(model.lstm.parameters()) + list(model.res.parameters()) + list(model.head.parameters())
    opt = torch.optim.Adam([{"params": enc_p, "lr": args.lr * 0.3}, {"params": new_p, "lr": args.lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w, reduction="none")
    best_val = -1; best = None
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y, m in tqdm(tl, desc=f"f{k} ep{ep:02d}/{args.epochs}", leave=False, ncols=88, mininterval=1):
            x, y, m = x.to(device), y.to(device), m.to(device); x = augment(x); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                lo = model(x); loss = crit(lo.reshape(-1, 5), y.reshape(-1))
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        sch.step()
        vy = np.concatenate([DATA[s][1] for s in vs])
        vp = np.concatenate([subj_probs(model, s, device)[0].argmax(1) for s in vs])
        vmf1 = f1_score(vy, vp, average="macro", zero_division=0)
        if vmf1 > best_val: best_val = vmf1; best = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        print(f"  f{k} ep{ep:02d} val_mF1={vmf1:.3f}{' *' if best_val == vmf1 else ''}")
    model.load_state_dict(best)

    lA, lpi = transition(tc); probs = {}
    ty, pd_, ph_ = [], [], []
    for s in te:
        pr, yy = subj_probs(model, s, device); probs[str(s)] = pr
        ty.append(yy); pd_.append(pr.argmax(1)); ph_.append(viterbi(np.log(pr + 1e-12), lA, lpi))
    ty = np.concatenate(ty)
    md = metrics(ty, np.concatenate(pd_)); mh = metrics(ty, np.concatenate(ph_))
    print(f"Fold {k}: deep acc={md['acc']:.3f} mF1={md['macro_f1']:.3f} | +HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")
    return md, mh, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pre-epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batch-pre", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5); ap.add_argument("--val-subj", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42); args = ap.parse_args(); set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = list7(); print(f"loading all {len(subs)} subjects into RAM (float16)...")
    load_all(subs)
    folds = make_folds(subs, args.folds, seed=args.seed); rng = np.random.RandomState(args.seed)
    nparams = sum(p.numel() for p in DeepSleepSeq(in_ch=7).parameters())
    print(f"device={device} | STRONG DeepSleepNet(7ch) | params={nparams:,} | {args.folds}-fold | lazy batch={args.batch}")
    deep_res, hmm_res, all_probs = [], [], {}
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        print(f"\n=== Fold {k} | train {len(tc)} / val {len(vs)} / test {len(te_s)} ===")
        md, mh, probs = train_fold(k, tc, vs, te_s, args, device)
        deep_res.append(md); hmm_res.append(mh); all_probs.update(probs)

    def summ(name, res):
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res]); kap = np.array([r["kappa"] for r in res])
        pcf = np.array([r["per_class_f1"] for r in res]).mean(0)
        print(f"{name:12s} acc={acc.mean():.4f}+-{acc.std():.4f} mF1={mf1.mean():.4f} kappa={kap.mean():.4f}  "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, pcf)))
        return {"acc": float(acc.mean()), "acc_std": float(acc.std()), "macro_f1": float(mf1.mean()), "kappa": float(kap.mean())}
    print("\n===== STRONG 7-ch deep =====")
    out = {"deep": summ("deep", deep_res), "deep_hmm": summ("deep+HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, "deep7_strong.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "deep7_probs.npz"), **all_probs)
    print("saved -> results/deep7_strong.json, results/deep7_probs.npz")


if __name__ == "__main__":
    main()
