"""
refine_seq.py -- the strongest shot: a BiLSTM over the WHOLE NIGHT whose input is the
concatenated per-epoch probabilities of the base models (ensemble + featseq + asym).
This unifies (a) stacked generalization (diverse base models decorrelate) and
(b) learned temporal refinement (richer than a first-order HMM), while staying tiny/
data-efficient (input is ~5*M numbers/epoch). Proper subject-independent meta-CV.
"""
import os, sys, json, glob
import numpy as np
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
RESULTS = os.path.join(HERE, "results")
CANDIDATES = {"ensemble": "ensemble7_v2_probs.npz", "featseq": "featseq_probs.npz", "asym": "asym_probs.npz"}


def seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def labels(sid):
    return np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)["y"].astype(np.int64)


def metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)), mf1=float(f1_score(y, p, average="macro", zero_division=0)),
               kappa=float(cohen_kappa_score(y, p)))


def transition(subs, Y):
    A = np.ones((5, 5)); pi = np.ones(5)
    for s in subs:
        y = Y[s]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A + 1e-12), np.log(pi + 1e-12)


def viterbi(le, lA, lpi):
    T, S = le.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): p[t] = bp[t + 1, p[t + 1]]
    return p


class Refiner(nn.Module):
    def __init__(self, in_dim, hid=96, n_cls=5, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hid, hid, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(2 * hid), nn.Dropout(dropout), nn.Linear(2 * hid, n_cls))

    def forward(self, x, lengths):
        h = self.proj(x)
        pk = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        return self.head(o)


def main():
    seed(42); device = "cuda" if torch.cuda.is_available() else "cpu"
    P = {}
    for name, fn in CANDIDATES.items():
        fp = os.path.join(RESULTS, fn)
        if os.path.exists(fp):
            d = np.load(fp); P[name] = {k: d[k] for k in d.files}; print(f"loaded {name}: {len(P[name])} subj")
    names = list(P.keys())
    if not names: print("no base probs found"); return
    subs = sorted(set.intersection(*[set(int(k) for k in P[n]) for n in names]) - DUPLICATE_DROP)
    Y = {s: labels(s) for s in subs}
    X = {s: np.concatenate([np.log(P[n][str(s)] + 1e-8) for n in names], axis=1).astype(np.float32) for s in subs}
    in_dim = X[subs[0]].shape[1]
    print(f"refiner input dim={in_dim} ({names}) over {len(subs)} subjects on {device}")

    folds = make_folds(subs, 10, seed=42); rng = np.random.RandomState(42)
    yt, pr_, ph_ = [], [], []
    for k, (tr_s, te_s) in enumerate(folds):
        vs = sorted(rng.choice(tr_s, size=6, replace=False).tolist()); tc = [s for s in tr_s if s not in vs]
        cnt = np.zeros(5); [cnt.__iadd__(np.bincount(Y[s], minlength=5)) for s in tc]
        w = torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
        model = Refiner(in_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
        best, bv = None, -1
        for ep in range(1, 81):
            model.train(); order = rng.permutation(len(tc))
            for i in range(0, len(tc), 8):
                bs = [tc[j] for j in order[i:i + 8]]; mx = max(len(Y[s]) for s in bs)
                xb = np.zeros((len(bs), mx, in_dim), np.float32); yb = np.full((len(bs), mx), -100, np.int64)
                ln = []
                for r, s in enumerate(bs):
                    xb[r, :len(Y[s])] = X[s]; yb[r, :len(Y[s])] = Y[s]; ln.append(len(Y[s]))
                xb = torch.tensor(xb, device=device); yb = torch.tensor(yb, device=device)
                opt.zero_grad(); lo = model(xb, torch.tensor(ln))
                loss = crit(lo.reshape(-1, 5), yb.reshape(-1)); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            sch.step()
            if ep % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vy = np.concatenate([Y[s] for s in vs])
                    vp = np.concatenate([model(torch.tensor(X[s][None], device=device), torch.tensor([len(Y[s])]))[0].argmax(1).cpu().numpy() for s in vs])
                vmf1 = f1_score(vy, vp, average="macro", zero_division=0)
                if vmf1 > bv: bv = vmf1; best = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best); model.eval()
        lA, lpi = transition(tc, Y)
        with torch.no_grad():
            for s in te_s:
                lo = model(torch.tensor(X[s][None], device=device), torch.tensor([len(Y[s])]))[0]
                pp = torch.softmax(lo.float(), -1).cpu().numpy()
                yt.append(Y[s]); pr_.append(pp.argmax(1)); ph_.append(viterbi(np.log(pp + 1e-12), lA, lpi))
        print(f"fold {k} done")
    y = np.concatenate(yt)
    out = {"refiner": metrics(y, np.concatenate(pr_)), "refiner_hmm": metrics(y, np.concatenate(ph_))}
    print("\n===== SEQUENCE REFINER on base probs =====")
    for kk, m in out.items():
        print(f"  {kk:13s} acc={m['acc']:.4f} mF1={m['mf1']:.4f} kappa={m['kappa']:.4f}")
    print("  (best single = ensemble+HMM 0.746/0.675/0.642 | published 0.747/0.677/0.640)")
    json.dump({"models": names, **out}, open(os.path.join(RESULTS, "refine_all.json"), "w"), indent=2)
    print("saved -> results/refine_all.json")


if __name__ == "__main__":
    main()
