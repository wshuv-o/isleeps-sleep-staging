"""
train_asym.py -- train/eval AsymGraphSSM (graph-attention + selective-SSM hybrid) on
iSLEEPS 7-channel, subject-independent CV, class-balanced loss, augmentation, HMM decode.
  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_asym.py --folds 5
"""
import os, sys, json, argparse, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP  # noqa
from asym_graph_ssm import AsymGraphSSM  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
RESULTS = os.path.join(HERE, "results")
torch.backends.cudnn.benchmark = True
L = 20
DATA = {}


def seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load7(sid):
    d = np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)
    x = d["x"].astype(np.float32); y = d["y"].astype(np.int64)
    mu = x.mean((0, 2), keepdims=True); sd = x.std((0, 2), keepdims=True) + 1e-6
    DATA[sid] = (((x - mu) / sd).astype(np.float16), y)


def list7():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def _starts(n, stride):
    if n <= L: return [0]
    st = list(range(0, n - L + 1, stride))
    if st[-1] + L < n: st.append(n - L)
    return st


class LazySeq(Dataset):
    def __init__(self, subs, stride):
        self.idx = [(s, a) for s in subs for a in _starts(len(DATA[s][1]), stride)]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, a = self.idx[i]; x, y = DATA[s]
        xs = x[a:a + L].astype(np.float32); ys = y[a:a + L].copy(); m = np.ones(len(ys), np.float32)
        if len(ys) < L:
            p = L - len(ys)
            xs = np.concatenate([xs, np.zeros((p, 7, 3000), np.float32)])
            ys = np.concatenate([ys, np.zeros(p, np.int64)]); m = np.concatenate([m, np.zeros(p, np.float32)])
        return xs, ys, m


def cw(subs, device):
    c = np.zeros(5, np.int64)
    for s in subs: c += np.bincount(DATA[s][1], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=device)


def augment(x, noise=0.05, scale=0.15, tmask=0.10):
    B, Ln, C, T = x.shape
    x = x * torch.empty(B, 1, C, 1, device=x.device).uniform_(1 - scale, 1 + scale)
    x = x + torch.randn_like(x) * noise
    w = int(T * tmask)
    if w > 0:
        st = torch.randint(0, T - w + 1, (1,), device=x.device).item(); x[..., st:st + w] = 0.0
    return x


def transition(subs):
    A = np.ones((5, 5)); pi = np.ones(5)
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
            "kappa": float(cohen_kappa_score(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--val-subj", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = list7()
    print(f"AsymGraphSSM | {len(subs)} subj | {args.folds}-fold | device={device}")
    for s in subs: load7(s)
    folds = make_folds(subs, args.folds, seed=args.seed); rng = np.random.RandomState(args.seed)
    deep_res, hmm_res, all_probs = [], [], {}
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        dl = DataLoader(LazySeq(tc, L // 2), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
        model = AsymGraphSSM(in_ch=7, dropout=0.3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(weight=cw(tc, device), reduction="none")
        lA, lpi = transition(tc)
        print(f"\n=== Fold {k} | train {len(tc)} / val {len(vs)} / test {len(te_s)} ===")
        best_val = -1; best = None
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y, m in tqdm(dl, desc=f"f{k} ep{ep:02d}/{args.epochs}", leave=False, ncols=84, mininterval=1):
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
            if ep % 5 == 0 or ep == args.epochs: print(f"  ep{ep:02d} val_mF1={vmf1:.3f} (best {best_val:.3f})")
        model.load_state_dict(best)
        ty, pd_, ph_ = [], [], []
        for s in te_s:
            pr, yy = subj_probs(model, s, device); all_probs[str(s)] = pr
            ty.append(yy); pd_.append(pr.argmax(1)); ph_.append(viterbi(np.log(pr + 1e-12), lA, lpi))
        ty = np.concatenate(ty); md = metrics(ty, np.concatenate(pd_)); mh = metrics(ty, np.concatenate(ph_))
        deep_res.append(md); hmm_res.append(mh)
        print(f"Fold {k}: AsymGraphSSM acc={md['acc']:.3f} mF1={md['macro_f1']:.3f} | +HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")

    def summ(name, res):
        a = np.array([r["acc"] for r in res]); f = np.array([r["macro_f1"] for r in res]); kp = np.array([r["kappa"] for r in res])
        print(f"{name:16s} acc={a.mean():.4f}+-{a.std():.4f} mF1={f.mean():.4f} kappa={kp.mean():.4f}")
        return {"acc": float(a.mean()), "acc_std": float(a.std()), "macro_f1": float(f.mean()), "kappa": float(kp.mean())}
    print("\n===== AsymGraphSSM (graph-attention + selective-SSM hybrid) =====")
    out = {"asym": summ("AsymGraphSSM", deep_res), "asym_hmm": summ("AsymGraphSSM+HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, "asym_all.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "asym_probs.npz"), **all_probs)
    print("saved -> results/asym_all.json, results/asym_probs.npz")


if __name__ == "__main__":
    main()
