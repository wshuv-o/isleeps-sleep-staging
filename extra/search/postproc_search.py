"""
postproc_search.py -- exhaust the post-hoc decoding space on the SAVED out-of-fold
posteriors. No retraining required.

Levers
  1. class-prior correction   p' ~ p * pi^alpha      (boosters were trained balanced)
  2. temperature calibration  p' ~ p^(1/T)
  3. transition tuning        A' ~ A^beta, self-loop x s     (first-order Viterbi)
  4. duration-aware decoding  explicit-duration HSMM Viterbi
  5. stacking                 multinomial LR over {ensemble, featseq, kags, transfer}

EVERY hyperparameter is chosen on held-out VALIDATION SUBJECTS drawn from the
training side of each fold, then applied unchanged to that fold's TEST subjects.
Nothing is tuned on test, so the reported deltas are honest.

  d:/EEG-TransNet/testenv/python.exe postproc_search.py
"""
import os, sys, glob, json, itertools
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP  # noqa

RES = os.path.join(HERE, "results")
FC = os.path.join(HERE, "data", "featseq_cache")
NC = 5
EPS = 1e-12


def subjects():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load_labels(subs):
    return {s: np.load(os.path.join(FC, f"SN{s}.npz"))["y"].astype(int) for s in subs}


def load_probs(fn, Y):
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        return None
    d = np.load(p); out = {}
    for k in d.files:
        s = int(k)
        if s in Y:
            q = d[k].astype(np.float64)
            out[s] = q[:len(Y[s])] / q[:len(Y[s])].sum(1, keepdims=True).clip(EPS)
    return out


# ----------------------------------------------------------------- transforms
def apply_prior(p, pri, alpha):
    if alpha == 0.0:
        return p
    q = p * (pri[None, :] ** alpha)
    return q / q.sum(1, keepdims=True).clip(EPS)


def apply_temp(p, T):
    if T == 1.0:
        return p
    q = np.power(p.clip(EPS, 1.0), 1.0 / T)
    return q / q.sum(1, keepdims=True).clip(EPS)


def transitions(seqs, beta=1.0, self_mul=1.0, eps=1.0):
    A = np.full((NC, NC), eps); pi = np.full(NC, eps)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            A[a, b] += 1
    A = A / A.sum(1, keepdims=True)
    if beta != 1.0:
        A = np.power(A, beta)
    if self_mul != 1.0:
        A[np.arange(NC), np.arange(NC)] *= self_mul
    A = A / A.sum(1, keepdims=True)
    pi = pi / pi.sum()
    return np.log(A + EPS), np.log(pi + EPS)


def viterbi(le, lA, lpi):
    T = le.shape[0]
    dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA
        bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    path = np.zeros(T, int); path[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def duration_logpmf(seqs, dmax=60, smooth=1.0):
    """empirical per-stage segment-duration distribution from training hypnograms"""
    cnt = np.full((NC, dmax + 1), smooth)
    for y in seqs:
        cur, run = y[0], 1
        for v in y[1:]:
            if v == cur:
                run += 1
            else:
                cnt[cur, min(run, dmax)] += 1; cur, run = v, 1
        cnt[cur, min(run, dmax)] += 1
    cnt[:, 0] = 0
    cnt = cnt / cnt.sum(1, keepdims=True).clip(EPS)
    return np.log(cnt + EPS)


def hsmm_viterbi(le, lA, lpi, lD, dmax=60):
    """explicit-duration HSMM Viterbi. le [T,NC] log-emissions, lD [NC,dmax+1]."""
    T = le.shape[0]
    C = np.vstack([np.zeros((1, NC)), np.cumsum(le, axis=0)])       # C[t] = sum_{u<t} le[u]
    dp = np.full((T + 1, NC), -np.inf)
    bp_d = np.zeros((T + 1, NC), int); bp_i = np.zeros((T + 1, NC), int)
    dp[0] = 0.0
    for t in range(1, T + 1):
        dmx = min(dmax, t)
        ds = np.arange(1, dmx + 1)
        seg = C[t][None, :] - C[t - ds]                              # [d,NC] segment emission
        dur = lD[:, ds].T                                            # [d,NC]
        prev = dp[t - ds]                                            # [d,NC] score at segment start
        # transition into j from best previous state i (or initial prior at t-d==0)
        best_prev = np.empty((dmx, NC)); best_i = np.zeros((dmx, NC), int)
        for k, d in enumerate(ds):
            s0 = t - d
            if s0 == 0:
                best_prev[k] = lpi; best_i[k] = -1
            else:
                sc = prev[k][:, None] + lA                           # [NC_from, NC_to]
                best_i[k] = sc.argmax(0); best_prev[k] = sc.max(0)
        tot = best_prev + dur + seg                                  # [d,NC]
        k = tot.argmax(0)
        dp[t] = tot[k, np.arange(NC)]
        bp_d[t] = ds[k]; bp_i[t] = best_i[k, np.arange(NC)]
    path = np.zeros(T, int); t = T; j = int(dp[T].argmax())
    while t > 0:
        d = bp_d[t, j]; path[t - d:t] = j
        j2 = bp_i[t, j]; t -= d; j = j2 if j2 >= 0 else j
    return path


def met(y, p):
    return dict(acc=float(accuracy_score(y, p)),
                mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y, p)))


def score(Y, P, subs, fn):
    yt, yp = [], []
    for s in subs:
        yt.append(Y[s]); yp.append(fn(s, P[s]))
    return met(np.concatenate(yt), np.concatenate(yp))


def main():
    subs = subjects(); Y = load_labels(subs)
    ENS = load_probs("ensemble7_v2_probs.npz", Y)
    OTHER = {n: load_probs(f, Y) for n, f in
             [("featseq", "featseq_probs.npz"), ("kags", "kags_probs.npz"),
              ("transfer", "transfer_probs.npz")]}
    OTHER = {k: v for k, v in OTHER.items() if v}
    print(f"subjects={len(subs)}  stack sources={['ensemble'] + list(OTHER)}")
    folds = make_folds(subs, 10, seed=42)
    rng = np.random.RandomState(0)

    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    TEMPS = [0.75, 1.0, 1.25, 1.5, 2.0]
    BETAS = [0.6, 0.8, 1.0, 1.3]
    SELFM = [1.0, 2.0, 4.0, 8.0]

    acc_rows = {k: [] for k in ["raw", "hmm(base)", "L1 prior", "L2 +temp", "L3 +trans", "L4 hsmm", "L5 stack"]}

    for k, (tr_s, te_s) in enumerate(folds):
        val_s = sorted(rng.choice(tr_s, size=min(20, len(tr_s)), replace=False).tolist())
        fit_s = [s for s in tr_s if s not in val_s]
        pri = np.bincount(np.concatenate([Y[s] for s in fit_s]), minlength=NC).astype(float)
        pri /= pri.sum()
        lA0, lpi0 = transitions([Y[s] for s in fit_s])
        lD = duration_logpmf([Y[s] for s in fit_s])

        # ---- baselines
        acc_rows["raw"].append(score(Y, ENS, te_s, lambda s, p: p.argmax(1)))
        acc_rows["hmm(base)"].append(score(Y, ENS, te_s,
                                     lambda s, p: viterbi(np.log(p + EPS), lA0, lpi0)))

        # ---- L1: prior correction (tuned on val, criterion = macro-F1)
        best = max(ALPHAS, key=lambda a: score(Y, ENS, val_s,
                   lambda s, p, a=a: viterbi(np.log(apply_prior(p, pri, a) + EPS), lA0, lpi0))["mf1"])
        a1 = best
        acc_rows["L1 prior"].append(score(Y, ENS, te_s,
                   lambda s, p: viterbi(np.log(apply_prior(p, pri, a1) + EPS), lA0, lpi0)))

        # ---- L2: + temperature
        bestT = max(TEMPS, key=lambda T: score(Y, ENS, val_s,
                    lambda s, p, T=T: viterbi(np.log(apply_temp(apply_prior(p, pri, a1), T) + EPS), lA0, lpi0))["mf1"])
        acc_rows["L2 +temp"].append(score(Y, ENS, te_s,
                   lambda s, p: viterbi(np.log(apply_temp(apply_prior(p, pri, a1), bestT) + EPS), lA0, lpi0)))

        # ---- L3: + transition tuning
        bestBS, bestv = (1.0, 1.0), -1
        for b, sm in itertools.product(BETAS, SELFM):
            lA, lpi = transitions([Y[s] for s in fit_s], beta=b, self_mul=sm)
            v = score(Y, ENS, val_s, lambda s, p: viterbi(
                np.log(apply_temp(apply_prior(p, pri, a1), bestT) + EPS), lA, lpi))["mf1"]
            if v > bestv:
                bestv, bestBS = v, (b, sm)
        lA3, lpi3 = transitions([Y[s] for s in fit_s], beta=bestBS[0], self_mul=bestBS[1])
        acc_rows["L3 +trans"].append(score(Y, ENS, te_s, lambda s, p: viterbi(
            np.log(apply_temp(apply_prior(p, pri, a1), bestT) + EPS), lA3, lpi3)))

        # ---- L4: duration-aware HSMM
        acc_rows["L4 hsmm"].append(score(Y, ENS, te_s, lambda s, p: hsmm_viterbi(
            np.log(apply_temp(apply_prior(p, pri, a1), bestT) + EPS), lA0, lpi0, lD)))

        # ---- L5: stacking over models (LR fit on val subjects), then best decoder
        try:
            from sklearn.linear_model import LogisticRegression
            names = [n for n in OTHER if all(s in OTHER[n] for s in val_s + te_s)]
            def feats(s):
                v = [np.log(ENS[s] + EPS)] + [np.log(OTHER[n][s][:len(Y[s])] + EPS) for n in names]
                return np.concatenate(v, axis=1)
            Xv = np.concatenate([feats(s) for s in val_s]); yv = np.concatenate([Y[s] for s in val_s])
            lr = LogisticRegression(max_iter=400, C=1.0, multi_class="multinomial")
            lr.fit(Xv, yv)
            ST = {s: lr.predict_proba(feats(s)) for s in te_s}
            acc_rows["L5 stack"].append(score(Y, ST, te_s, lambda s, p: viterbi(
                np.log(apply_temp(apply_prior(p, pri, a1), bestT) + EPS), lA3, lpi3)))
        except Exception as e:
            print("  stack failed:", type(e).__name__, e)

        print(f"fold {k}: alpha={a1} T={bestT} beta,self={bestBS} "
              f"| base={acc_rows['hmm(base)'][-1]['mf1']:.3f} "
              f"L3={acc_rows['L3 +trans'][-1]['mf1']:.3f} "
              f"L4={acc_rows['L4 hsmm'][-1]['mf1']:.3f}")

    print("\n================ POST-HOC SEARCH (10-fold, tuned on val subjects) ============")
    print(f"{'config':12s} {'acc':>8s} {'macroF1':>9s} {'kappa':>8s}")
    out = {}
    for name, rows in acc_rows.items():
        if not rows:
            continue
        a = np.mean([r["acc"] for r in rows]); f = np.mean([r["mf1"] for r in rows])
        kp = np.mean([r["kappa"] for r in rows])
        out[name] = dict(acc=float(a), mf1=float(f), kappa=float(kp))
        print(f"{name:12s} {a:8.4f} {f:9.4f} {kp:8.4f}")
    json.dump(out, open(os.path.join(RES, "postproc_search.json"), "w"), indent=2)
    print("\nsaved -> results/postproc_search.json")


if __name__ == "__main__":
    main()
