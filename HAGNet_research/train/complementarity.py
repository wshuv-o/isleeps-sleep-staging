"""
complementarity.py -- what else is inside HAG-Net?

Asks whether HAG-Net's errors are DIFFERENT from the boosting ensemble's. If the two
models fail on different epochs, the architecture is seeing something the engineered
features miss, and combining them may beat either alone.

  1. error overlap / disagreement rate
  2. per-class comparison: where does each win?
  3. oracle ceiling: accuracy if we always picked the correct one of the two
  4. actual blends: probability average, weighted, and confidence-routed

Baseline to beat: ensemble + HMM = 0.7464 acc / 0.6753 macro-F1
"""
import os, sys, glob
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "utils"))
from datasets import make_folds, DUPLICATE_DROP  # noqa
FC = os.path.join(ROOT, "data", "featseq_cache"); RES = os.path.join(ROOT, "results")
NC, EPS = 5, 1e-12
CLS = ["W", "N1", "N2", "N3", "R"]


def viterbi(le, lA, lpi):
    T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


def transmat(seqs):
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A+EPS), np.log(pi+EPS)


def met(y, p):
    return (accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0),
            cohen_kappa_score(y, p))


def main():
    hp = os.path.join(RES, "hagnet_standalone_probs.npz")
    if not os.path.exists(hp):
        print("HAG-Net probabilities not found yet - run the gate with --only-full first."); return
    H = np.load(hp); E = np.load(os.path.join(RES, "ensemble7_v2_probs.npz"))
    subs = sorted(int(k) for k in H.files if k in E.files)
    Y = {s: np.load(os.path.join(FC, f"SN{s}.npz"))["y"].astype(int) for s in subs}
    HP = {s: H[str(s)].astype(np.float64)[:len(Y[s])] for s in subs}
    EP = {s: E[str(s)].astype(np.float64)[:len(Y[s])] for s in subs}
    for s in subs:
        HP[s] /= HP[s].sum(1, keepdims=True).clip(EPS); EP[s] /= EP[s].sum(1, keepdims=True).clip(EPS)
    print(f"{len(subs)} subjects with both models' predictions\n")

    folds = make_folds(sorted(Y), 3, seed=42)
    fold_of = {s: k for k, (tr, te) in enumerate(folds) for s in te}
    LA = {}
    for k, (tr, te) in enumerate(folds):
        LA[k] = transmat([Y[s] for s in tr if s in Y])

    def decode(P):
        yt, yp = [], []
        for s in subs:
            lA, lpi = LA[fold_of[s]]
            yt.append(Y[s]); yp.append(viterbi(np.log(P[s] + EPS), lA, lpi))
        return np.concatenate(yt), np.concatenate(yp)

    yt, hag = decode(HP)
    _,  ens = decode(EP)

    print("=== 1. standalone performance (+HMM, same folds) ===")
    for n, p in (("HAG-Net", hag), ("Ensemble", ens)):
        a, f, k = met(yt, p); print(f"  {n:10s} acc {a:.4f}  mF1 {f:.4f}  kappa {k:.4f}")

    print("\n=== 2. do they make the SAME mistakes? ===")
    he, ee = hag != yt, ens != yt
    both = (he & ee).sum(); only_h = (he & ~ee).sum(); only_e = (~he & ee).sum()
    print(f"  both wrong          : {both:6d}  ({100*both/len(yt):.1f}%)")
    print(f"  only HAG-Net wrong  : {only_h:6d}  ({100*only_h/len(yt):.1f}%)")
    print(f"  only Ensemble wrong : {only_e:6d}  ({100*only_e/len(yt):.1f}%)")
    print(f"  disagreement rate   : {100*(hag != ens).mean():.1f}%")
    ov = both / max(1, (he | ee).sum())
    print(f"  error overlap (Jaccard): {ov:.3f}   (1.0 = identical errors, 0 = fully complementary)")

    print("\n=== 3. oracle ceiling (if we always picked the right model) ===")
    oracle = np.where(~he, hag, np.where(~ee, ens, hag))
    a, f, k = met(yt, oracle)
    print(f"  oracle acc {a:.4f}  mF1 {f:.4f}   <- upper bound on any combination")

    print("\n=== 4. per-class: who wins where? ===")
    fh = f1_score(yt, hag, average=None, labels=range(5), zero_division=0)
    fe = f1_score(yt, ens, average=None, labels=range(5), zero_division=0)
    print(f"  {'stage':6s} {'HAG-Net':>9s} {'Ensemble':>9s} {'diff':>8s}")
    for i, c in enumerate(CLS):
        print(f"  {c:6s} {fh[i]:9.3f} {fe[i]:9.3f} {fh[i]-fe[i]:+8.3f}")

    print("\n=== 5. actual blends ===")
    base_a, base_f, _ = met(yt, ens)
    rows = []
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        B = {s: (1-w)*EP[s] + w*HP[s] for s in subs}
        _, p = decode(B); a, f, k = met(yt, p)
        rows.append((f"blend {1-w:.1f}*ens + {w:.1f}*hag", a, f, k))
    # confidence routing: use HAG-Net only where the ensemble is unsure
    for thr in (0.5, 0.6, 0.7):
        B = {}
        for s in subs:
            conf = EP[s].max(1, keepdims=True)
            B[s] = np.where(conf < thr, HP[s], EP[s])
        _, p = decode(B); a, f, k = met(yt, p)
        rows.append((f"route to HAG when ens conf<{thr}", a, f, k))
    print(f"  {'combination':34s} {'acc':>8s} {'mF1':>8s} {'d(acc)':>8s}")
    for n, a, f, k in sorted(rows, key=lambda z: -z[1]):
        flag = "  <-- beats ensemble" if a > base_a + 0.002 else ""
        print(f"  {n:34s} {a:8.4f} {f:8.4f} {a-base_a:+8.4f}{flag}")
    print(f"  {'ensemble alone (baseline)':34s} {base_a:8.4f} {base_f:8.4f} {0.0:+8.4f}")
    print("\nnote: a real win must clear +0.016 acc (fold noise).")


if __name__ == "__main__":
    main()
