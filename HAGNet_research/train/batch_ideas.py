"""
batch_ideas.py -- screen many post-hoc ideas at FULL 99-subject scale, instantly.

Every idea is a transform on the SAVED out-of-fold ensemble probabilities
(results/ensemble7_v2_probs.npz). No training, so each runs in seconds on all 99
subjects under the exact per-fold protocol (transition matrices fit on each fold's
TRAIN subjects, applied to its TEST subjects). Nothing is tuned on test.

Baseline to beat:  ensemble + HMM = 0.7464 acc / 0.6753 macro-F1 / 0.6415 kappa
"""
import os, sys, glob
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("utils",): sys.path.insert(0, os.path.join(ROOT, p))
from datasets import make_folds, DUPLICATE_DROP  # noqa
FC = os.path.join(ROOT, "data", "featseq_cache")
RES = os.path.join(ROOT, "results")
NC, EPS = 5, 1e-12


def subjects():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


subs = subjects()
Y = {s: np.load(os.path.join(FC, f"SN{s}.npz"))["y"].astype(int) for s in subs}
D = np.load(os.path.join(RES, "ensemble7_v2_probs.npz"))
P = {int(k): (D[k].astype(np.float64)[:len(Y[int(k)])]) for k in D.files if int(k) in Y}
for s in P: P[s] = P[s] / P[s].sum(1, keepdims=True).clip(EPS)
folds = make_folds(subs, 10, seed=42)


# ---------- decoders ----------------------------------------------------------
def viterbi(le, lA, lpi):
    T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t-1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p


def fwd_bwd(le, lA, lpi):
    """posterior (sum-product) decoding instead of best-path Viterbi"""
    T = le.shape[0]
    a = np.zeros((T, NC)); a[0] = lpi + le[0]
    for t in range(1, T):
        a[t] = le[t] + logsumexp(a[t-1][:, None] + lA, 0)
    b = np.zeros((T, NC))
    for t in range(T-2, -1, -1):
        b[t] = logsumexp(lA + (le[t+1] + b[t+1])[None, :], 1)
    return (a + b).argmax(1)


def logsumexp(x, axis):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def transmat(seqs, beta=1.0, self_mul=1.0):
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True)
    if beta != 1.0: A = np.power(A, beta)
    if self_mul != 1.0: A[np.arange(NC), np.arange(NC)] *= self_mul
    A /= A.sum(1, keepdims=True); pi /= pi.sum()
    return np.log(A + EPS), np.log(pi + EPS)


def med_filter(seq, w):
    T = len(seq); out = seq.copy()
    for t in range(T):
        lo, hi = max(0, t-w//2), min(T, t+w//2+1)
        vals = seq[lo:hi]; out[t] = np.bincount(vals, minlength=NC).argmax()
    return out


def min_run(seq, k=3):
    """merge runs shorter than k into the previous stage"""
    out = seq.copy(); T = len(out); i = 0
    while i < T:
        j = i
        while j < T and out[j] == out[i]: j += 1
        if (j - i) < k and i > 0: out[i:j] = out[i-1]
        i = j
    return out


def neigh_smooth(prob, w=1):
    T = len(prob); acc = prob.copy()
    for d in range(1, w+1):
        acc[d:] += prob[:-d]; acc[:-d] += prob[d:]
    return acc / acc.sum(1, keepdims=True).clip(EPS)


def temp(prob, T):
    q = np.power(prob.clip(EPS, 1), 1.0/T); return q / q.sum(1, keepdims=True)


# ---------- the idea list -----------------------------------------------------
IDEAS = {}
def idea(name):
    def deco(f): IDEAS[name] = f; return f
    return deco


@idea("00 raw argmax")
def _(prob, lA, lpi): return prob.argmax(1)
@idea("01 HMM (baseline)")
def _(prob, lA, lpi): return viterbi(np.log(prob+EPS), lA, lpi)
@idea("02 forward-backward posterior")
def _(prob, lA, lpi): return fwd_bwd(np.log(prob+EPS), lA, lpi)
@idea("03 sharpen T=0.5 + HMM")
def _(prob, lA, lpi): return viterbi(np.log(temp(prob,0.5)+EPS), lA, lpi)
@idea("04 soften T=1.5 + HMM")
def _(prob, lA, lpi): return viterbi(np.log(temp(prob,1.5)+EPS), lA, lpi)
@idea("05 power posterior ^1.5 + HMM")
def _(prob, lA, lpi):
    q = prob**1.5; q/=q.sum(1,keepdims=True); return viterbi(np.log(q+EPS), lA, lpi)
@idea("06 neighbour-smooth w1 + HMM")
def _(prob, lA, lpi): return viterbi(np.log(neigh_smooth(prob,1)+EPS), lA, lpi)
@idea("07 neighbour-smooth w2 + HMM")
def _(prob, lA, lpi): return viterbi(np.log(neigh_smooth(prob,2)+EPS), lA, lpi)
@idea("08 median filter w3 (on raw)")
def _(prob, lA, lpi): return med_filter(prob.argmax(1), 3)
@idea("09 median filter w5 (on raw)")
def _(prob, lA, lpi): return med_filter(prob.argmax(1), 5)
@idea("10 min-run>=3 (on HMM)")
def _(prob, lA, lpi): return min_run(viterbi(np.log(prob+EPS), lA, lpi), 3)
@idea("11 confidence-gated HMM (thr .85)")
def _(prob, lA, lpi):
    raw = prob.argmax(1); vit = viterbi(np.log(prob+EPS), lA, lpi)
    conf = prob.max(1); return np.where(conf > 0.85, raw, vit)
@idea("12 prior-blend 0.85 + HMM")
def _(prob, lA, lpi):
    gp = np.exp(lpi); q = 0.85*prob + 0.15*gp[None,:]; q/=q.sum(1,keepdims=True)
    return viterbi(np.log(q+EPS), lA, lpi)
# transition-shape variants use their own lA (handled in the loop)


def met(y, p):
    return (accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0),
            cohen_kappa_score(y, p), f1_score(y, p, average=None, labels=range(5), zero_division=0)[1])


def run(idea_fn, beta=1.0, self_mul=1.0):
    yt, yp = [], []
    for tr, te in folds:
        lA, lpi = transmat([Y[s] for s in tr], beta=beta, self_mul=self_mul)
        for s in te:
            yt.append(Y[s]); yp.append(idea_fn(P[s], lA, lpi))
    return met(np.concatenate(yt), np.concatenate(yp))


def main():
    print(f"screening {len(IDEAS)+2} post-hoc ideas on {len(subs)} subjects, per-fold protocol\n")
    rows = []
    for name, fn in IDEAS.items():
        rows.append((name, *run(fn)))
    # two transition-shape variants (need beta/self_mul, so run baseline decoder with them)
    rows.append(("13 self-loop x3 + HMM", *run(IDEAS["01 HMM (baseline)"], self_mul=3.0)))
    rows.append(("14 transition beta=0.7 + HMM", *run(IDEAS["01 HMM (baseline)"], beta=0.7)))

    base = next(r for r in rows if r[0] == "01 HMM (baseline)")
    print(f"{'idea':32s} {'acc':>7s} {'mF1':>7s} {'kappa':>7s} {'N1':>6s}   {'d(acc)':>7s} {'d(mF1)':>7s}")
    print("-"*84)
    for r in sorted(rows, key=lambda z: -z[2]):
        da, dm = r[1]-base[1], r[2]-base[2]
        flag = "  <-- beats baseline" if (da > 0.002 and dm > 0.002) else ""
        print(f"{r[0]:32s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.4f} {r[4]:6.3f}   {da:+7.4f} {dm:+7.4f}{flag}")
    print("-"*84)
    print(f"baseline (01 HMM): acc {base[1]:.4f}  mF1 {base[2]:.4f}  kappa {base[3]:.4f}")
    best = max(rows, key=lambda z: z[2])
    print(f"\nbest macro-F1: {best[0]}  =  {best[2]:.4f}  (baseline {base[2]:.4f}, "
          f"delta {best[2]-base[2]:+.4f})")
    print("note: differences under ~0.005 are inside fold noise; a real win must clear +0.016 acc.")


if __name__ == "__main__":
    main()
