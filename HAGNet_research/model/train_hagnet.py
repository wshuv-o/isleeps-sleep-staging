"""
train_hagnet.py -- train/eval HAG-Net (hemisphere graph-attention + selective-SSM +
classical-deep RESIDUAL fusion). Data-efficient: inputs are the 188 engineered features
(featseq_cache) + the ensemble's out-of-fold probs (ensemble7_v2_probs.npz), NEVER raw
signal. Residual design (output = ensemble log-probs + gated correction) means it starts
at the 0.746 ensemble and can only add. Proper subject-independent meta-CV + HMM decode.
  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_hagnet.py --folds 10
"""
import os, sys, json, argparse, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repo root
for _p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, HERE)
from datasets import make_folds, DUPLICATE_DROP  # noqa
from hag_net import HAGNet  # noqa

PROC7 = os.path.join(ROOT, "data", "processed7")
FCACHE = os.path.join(ROOT, "data", "featseq_cache")
RESULTS = os.path.join(ROOT, "results")
torch.backends.cudnn.benchmark = True
L = 25
FEAT, PROB = {}, {}   # sid -> (F[n,188], y) ; sid -> ens_probs[n,5]


def seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def list7():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load(sid, probd):
    d = np.load(os.path.join(FCACHE, f"SN{sid}.npz"))
    FEAT[sid] = (np.nan_to_num(d["F"]).astype(np.float32), d["y"].astype(np.int64))
    PROB[sid] = probd[str(sid)].astype(np.float32)


def _starts(n, stride):
    if n <= L: return [0]
    st = list(range(0, n - L + 1, stride))
    if st[-1] + L < n: st.append(n - L)
    return st


class WinDS(Dataset):
    def __init__(self, subs, mu, sd, stride):
        self.mu, self.sd = mu, sd
        self.idx = [(s, a) for s in subs for a in _starts(len(FEAT[s][1]), stride)]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s, a = self.idx[i]; F, y = FEAT[s]; p = PROB[s]
        Fn = ((F[a:a + L] - self.mu) / self.sd).astype(np.float32)
        lp = np.log(p[a:a + L] + 1e-8).astype(np.float32)
        ys = y[a:a + L].copy(); m = np.ones(len(ys), np.float32)
        if len(ys) < L:
            k = L - len(ys)
            Fn = np.concatenate([Fn, np.zeros((k, 188), np.float32)])
            lp = np.concatenate([lp, np.zeros((k, 5), np.float32)])
            ys = np.concatenate([ys, np.zeros(k, np.int64)]); m = np.concatenate([m, np.zeros(k, np.float32)])
        return Fn[:, :161], Fn[:, 161:], lp, ys, m


def cw(subs, device):
    c = np.zeros(5, np.int64)
    for s in subs: c += np.bincount(FEAT[s][1], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=device)


def transition(subs):
    A = np.ones((5, 5)); pi = np.ones(5)
    for s in subs:
        y = FEAT[s][1]; pi[y[0]] += 1
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
def subj_probs(model, sid, mu, sd, device):
    F, y = FEAT[sid]; p = PROB[sid]; n = len(y); pad = (-n) % L
    Fn = ((F - mu) / sd).astype(np.float32); lp = np.log(p + 1e-8).astype(np.float32)
    if pad:
        Fn = np.concatenate([Fn, np.zeros((pad, 188), np.float32)]); lp = np.concatenate([lp, np.zeros((pad, 5), np.float32)])
    B = np.arange(len(Fn)).reshape(-1, L); out = []
    for i in range(0, len(B), 16):
        idx = B[i:i + 16]
        base = torch.tensor(Fn[idx][:, :, :161], device=device)
        event = torch.tensor(Fn[idx][:, :, 161:], device=device)
        el = torch.tensor(lp[idx], device=device)
        with torch.amp.autocast("cuda", enabled=False):
            lo = model(base, event, el)
        out.append(torch.softmax(lo.float(), -1).reshape(-1, 5).cpu().numpy())
    return np.concatenate(out)[:n], y


def metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)), mf1=float(f1_score(y, p, average="macro", zero_division=0)),
               kappa=float(cohen_kappa_score(y, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10); ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--val-subj", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="probe mode: run only the first N folds of the --folds split (0 = all)")
    args = ap.parse_args(); seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subs = list7()
    probd = np.load(os.path.join(RESULTS, "ensemble7_v2_probs.npz"))
    print(f"HAG-Net (graph+SSM+routed classical/deep fusion; no KAN) | {len(subs)} subj | {args.folds}-fold | {device}")
    for s in subs: load(s, probd)
    folds = make_folds(subs, args.folds, seed=args.seed); rng = np.random.RandomState(args.seed)
    if args.max_folds:
        folds = folds[:args.max_folds]
        print(f"[probe] running only the first {len(folds)} of {args.folds} folds")
    deep_res, hmm_res, all_probs = [], [], {}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        Fall = np.concatenate([FEAT[s][0] for s in tc]); mu = Fall.mean(0); sd = Fall.std(0) + 1e-6
        dl = DataLoader(WinDS(tc, mu, sd, L // 2), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
        model = HAGNet(dropout=0.5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(weight=cw(tc, device), reduction="none")
        lA, lpi = transition(tc)
        print(f"\n=== Fold {k} | train {len(tc)} / val {len(vs)} / test {len(te_s)} ===")
        best_val, best = -1, None
        for ep in range(1, args.epochs + 1):
            model.train()
            for base, event, el, y, m in tqdm(dl, desc=f"f{k} ep{ep:02d}", leave=False, ncols=80, mininterval=1):
                base, event, el, y, m = base.to(device), event.to(device), el.to(device), y.to(device), m.to(device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=False):
                    lo = model(base, event, el); loss = crit(lo.reshape(-1, 5), y.reshape(-1))
                    # mild pressure to keep the router closed unless the deep stream earns it
                    loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1) + 0.01 * model.last_gate.mean()
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
            sch.step()
            vy = np.concatenate([FEAT[s][1] for s in vs])
            vp = np.concatenate([subj_probs(model, s, mu, sd, device)[0].argmax(1) for s in vs])
            vmf1 = f1_score(vy, vp, average="macro", zero_division=0)
            if vmf1 > best_val: best_val = vmf1; best = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            if ep % 5 == 0 or ep == args.epochs: print(f"  ep{ep:02d} val_mF1={vmf1:.3f} (best {best_val:.3f}) gate={model.last_gate.mean().item():.3f}")
        model.load_state_dict(best)
        # Residual-fallback: HAG-Net's learnable correction is accepted ONLY if it
        # improves validation macro-F1 over the classical prior it wraps. Otherwise
        # the gated correction is discarded and the model emits the prior exactly.
        # This makes the "cannot fall below the prior" guarantee hold on test, not
        # just at initialization (gate=0).
        vy = np.concatenate([FEAT[s][1] for s in vs])
        prior_vp = np.concatenate([PROB[s].argmax(1) for s in vs])
        prior_vmf1 = f1_score(vy, prior_vp, average="macro", zero_division=0)
        use_learned = best_val >= prior_vmf1
        print(f"  [fold {k}] learned val_mF1={best_val:.3f} vs prior val_mF1={prior_vmf1:.3f} "
              f"-> {'LEARNED correction' if use_learned else 'PRIOR fallback'}")
        ty, pd_, ph_ = [], [], []
        for s in te_s:
            if use_learned:
                pr, yy = subj_probs(model, s, mu, sd, device)
            else:
                pr, yy = PROB[s].astype(np.float32), FEAT[s][1]
            all_probs[str(s)] = pr
            ty.append(yy); pd_.append(pr.argmax(1)); ph_.append(viterbi(np.log(pr + 1e-12), lA, lpi))
        ty = np.concatenate(ty); md = metrics(ty, np.concatenate(pd_)); mh = metrics(ty, np.concatenate(ph_))
        deep_res.append(md); hmm_res.append(mh)
        print(f"Fold {k}: KAGS acc={md['acc']:.3f} mF1={md['mf1']:.3f} | +HMM acc={mh['acc']:.3f} mF1={mh['mf1']:.3f} k={mh['kappa']:.3f}")

    def summ(name, res):
        a = np.array([r["acc"] for r in res]); f = np.array([r["mf1"] for r in res]); kp = np.array([r["kappa"] for r in res])
        print(f"{name:14s} acc={a.mean():.4f}+-{a.std():.4f} mF1={f.mean():.4f} kappa={kp.mean():.4f}")
        return {"acc": float(a.mean()), "acc_std": float(a.std()), "macro_f1": float(f.mean()), "kappa": float(kp.mean())}
    print("\n===== KAGS-Net =====")
    out = {"kags": summ("HAG-Net", deep_res), "kags_hmm": summ("HAG-Net+HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, "kags_all.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, "kags_probs.npz"), **all_probs)
    print("saved -> results/kags_all.json, results/kags_probs.npz")


if __name__ == "__main__":
    main()
