"""
run_standalone_gate.py -- the go/no-go for the architecture paper.

Question: does STANDALONE HAG-Net (no boosting prior fed in) beat the best other
standalone deep model trained on the same 188 features?

    target to beat:  FeatSeq BiLSTM  =  0.676 acc / 0.639 macro-F1 / 0.566 kappa

Identical protocol to everything else: subject-independent folds, class-balanced loss,
early stopping on validation macro-F1, HMM decoding at test. Runs the full model and
the three ablations so that, if the gate passes, the ablation table already exists.

  KMP_DUPLICATE_LIB_OK=TRUE python extra/run_standalone_gate.py --folds 3
"""
import os, sys, glob, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, p))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES          # noqa
from hag_net_standalone import VARIANTS                               # noqa

FC = os.path.join(ROOT, "data", "featseq_cache")
RES = os.path.join(ROOT, "results")
L, NC, EPS = 25, 5, 1e-12
FEAT = {}
SAVED_PROBS = {}
TARGET = dict(name="FeatSeq BiLSTM (best standalone deep)", acc=0.676, mf1=0.639, kappa=0.566)


def subjects():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load(sid):
    d = np.load(os.path.join(FC, f"SN{sid}.npz"))
    FEAT[sid] = (np.nan_to_num(d["F"]).astype(np.float32), d["y"].astype(np.int64))


def starts(n, stride):
    if n <= L: return [0]
    st = list(range(0, n - L + 1, stride))
    if st[-1] + L < n: st.append(n - L)
    return st


class WinDS(Dataset):
    def __init__(self, subs, mu, sd, stride):
        self.mu, self.sd = mu, sd
        self.idx = [(s, a) for s in subs for a in starts(len(FEAT[s][1]), stride)]
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        s, a = self.idx[i]; F, y = FEAT[s]
        Fn = ((F[a:a+L] - self.mu) / self.sd).astype(np.float32)
        ys = y[a:a+L].copy(); m = np.ones(len(ys), np.float32)
        if len(ys) < L:
            k = L - len(ys)
            Fn = np.concatenate([Fn, np.zeros((k, 188), np.float32)])
            ys = np.concatenate([ys, np.zeros(k, np.int64)]); m = np.concatenate([m, np.zeros(k, np.float32)])
        return Fn[:, :161], Fn[:, 161:], ys, m


def cw(subs, dev):
    c = np.zeros(5, np.int64)
    for s in subs: c += np.bincount(FEAT[s][1], minlength=5)
    return torch.tensor(c.sum() / (5 * np.maximum(c, 1)), dtype=torch.float32, device=dev)


def transitions(subs):
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for s in subs:
        y = FEAT[s][1]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum()
    return np.log(A + EPS), np.log(pi + EPS)


def viterbi(le, lA, lpi):
    T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


@torch.no_grad()
def subj_probs(model, sid, mu, sd, dev):
    F, y = FEAT[sid]; n = len(y); pad = (-n) % L
    Fn = ((F - mu) / sd).astype(np.float32)
    if pad: Fn = np.concatenate([Fn, np.zeros((pad, 188), np.float32)])
    B = np.arange(len(Fn)).reshape(-1, L); out = []
    for i in range(0, len(B), 16):
        idx = B[i:i+16]
        lo = model(torch.tensor(Fn[idx][:, :, :161], device=dev),
                   torch.tensor(Fn[idx][:, :, 161:], device=dev))
        out.append(torch.softmax(lo.float(), -1).reshape(-1, 5).cpu().numpy())
    return np.concatenate(out)[:n], y


def met(y, p):
    return dict(acc=float(accuracy_score(y, p)), mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y, p)),
                pcf=f1_score(y, p, average=None, labels=range(5), zero_division=0).tolist())


def run_variant(name, cls, folds, args, dev):
    rng = np.random.RandomState(args.seed); raw, hmm = [], []
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=args.val_subj, replace=False).tolist())
        tc = [s for s in tr_s if s not in vs]
        Fall = np.concatenate([FEAT[s][0] for s in tc]); mu = Fall.mean(0); sd = Fall.std(0) + 1e-6
        dl = DataLoader(WinDS(tc, mu, sd, L // 2), batch_size=args.batch, shuffle=True, num_workers=0)
        model = cls(dropout=args.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(weight=cw(tc, dev), reduction="none")
        lA, lpi = transitions(tc)
        best_val, best_state, bad = -1, None, 0
        for ep in range(1, args.epochs + 1):
            model.train()
            for base, event, y, m in dl:
                base, event, y, m = base.to(dev), event.to(dev), y.to(dev), m.to(dev)
                opt.zero_grad()
                lo = model(base, event)
                loss = (crit(lo.reshape(-1, 5), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sch.step()
            vy = np.concatenate([FEAT[s][1] for s in vs])
            vp = np.concatenate([subj_probs(model, s, mu, sd, dev)[0].argmax(1) for s in vs])
            v = f1_score(vy, vp, average="macro", zero_division=0)
            if v > best_val:
                best_val, bad = v, 0
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else:
                bad += 1
            if ep % 5 == 0 or bad >= args.patience:
                print(f"    [{name}] f{k} ep{ep:02d} val_mF1={v:.4f} (best {best_val:.4f})", flush=True)
            if bad >= args.patience:
                print(f"    [{name}] f{k} early stop at ep{ep}", flush=True); break
        model.load_state_dict(best_state)
        ty, pr, ph = [], [], []
        for s in te_s:
            p, yy = subj_probs(model, s, mu, sd, dev)
            SAVED_PROBS[str(s)] = p.astype(np.float32)          # keep for complementarity analysis
            ty.append(yy); pr.append(p.argmax(1)); ph.append(viterbi(np.log(p + EPS), lA, lpi))
        ty = np.concatenate(ty)
        raw.append(met(ty, np.concatenate(pr))); hmm.append(met(ty, np.concatenate(ph)))
        print(f"  [{name}] fold {k}: +HMM acc={hmm[-1]['acc']:.4f} mF1={hmm[-1]['mf1']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    agg = lambda rs, k: float(np.mean([r[k] for r in rs]))
    return dict(name=name,
                raw=dict(acc=agg(raw,"acc"), mf1=agg(raw,"mf1"), kappa=agg(raw,"kappa")),
                hmm=dict(acc=agg(hmm,"acc"), mf1=agg(hmm,"mf1"), kappa=agg(hmm,"kappa"),
                         pcf=np.mean([r["pcf"] for r in hmm], axis=0).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--val-subj", type=int, default=12); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-full", action="store_true", help="gate only; skip ablations")
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subs = subjects()
    for s in subs: load(s)
    folds = make_folds(subs, a.folds, seed=a.seed)
    print(f"STANDALONE HAG-Net gate | {len(subs)} subj | {a.folds}-fold | early stop "
          f"(max {a.epochs}, patience {a.patience}) | {dev}")
    print(f"target to beat: {TARGET['name']} = {TARGET['acc']:.3f} acc / {TARGET['mf1']:.3f} mF1\n")
    todo = [("HAG-Net (full)", VARIANTS["HAG-Net (full)"])] if a.only_full else list(VARIANTS.items())
    out = []
    for name, cls in todo:
        print(f"--- {name} ---", flush=True)
        t0 = time.time(); r = run_variant(name, cls, folds, a, dev); r["minutes"] = (time.time()-t0)/60
        out.append(r)
        print(f"  == {name}: +HMM acc={r['hmm']['acc']:.4f} mF1={r['hmm']['mf1']:.4f} "
              f"kappa={r['hmm']['kappa']:.4f}  ({r['minutes']:.1f} min)\n", flush=True)
        json.dump(out, open(os.path.join(RES, "standalone_gate.json"), "w"), indent=2)
        if name == "HAG-Net (full)":
            np.savez_compressed(os.path.join(RES, "hagnet_standalone_probs.npz"), **SAVED_PROBS)
            print(f"  saved {len(SAVED_PROBS)} subject prob arrays", flush=True)

    print("=" * 74)
    print(f"{'variant':24s} {'acc':>8s} {'macroF1':>9s} {'kappa':>8s}   N1")
    print("-" * 74)
    for r in out:
        print(f"{r['name']:24s} {r['hmm']['acc']:8.4f} {r['hmm']['mf1']:9.4f} "
              f"{r['hmm']['kappa']:8.4f}   {r['hmm']['pcf'][1]:.3f}")
    print("-" * 74)
    print(f"{TARGET['name']:24s} {TARGET['acc']:8.4f} {TARGET['mf1']:9.4f} {TARGET['kappa']:8.4f}")
    full = out[0]
    da, dm = full["hmm"]["acc"] - TARGET["acc"], full["hmm"]["mf1"] - TARGET["mf1"]
    print(f"\nDELTA vs target: acc {da:+.4f}   macro-F1 {dm:+.4f}")
    print("\nGATE: " + ("PASS - architecture beats the best standalone deep model; "
                        "proceed to full benchmark + ablations"
                        if (da > 0.01 or dm > 0.01) else
                        "FAIL - does not clearly beat it; fall back to the current paper"))
    print("(a difference under ~0.016 is inside fold noise and should be read as a tie)")


if __name__ == "__main__":
    main()
