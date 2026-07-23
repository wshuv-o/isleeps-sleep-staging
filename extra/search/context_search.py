"""
context_search.py -- lever 6: what the model SEES. Post-hoc decoding is exhausted,
so probe the input representation instead. Fast single-booster (LightGBM) probe over
context width and long-range night features, 3 folds, so we get a signal in minutes
before committing the full 4-booster ensemble to a 10-fold run.

variants
  A  k=3   (current baseline)
  B  k=5
  C  k=7
  D  k=3 + position-in-night + rolling mean/std over +-10 epochs
  E  k=5 + position-in-night + rolling mean/std over +-10 epochs
"""
import os, sys, glob, json, time
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP  # noqa

FC = os.path.join(HERE, "data", "featseq_cache")
RES = os.path.join(HERE, "results")
NC, EPS = 5, 1e-12
BASE = {}


def subjects():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load(subs):
    for s in subs:
        d = np.load(os.path.join(FC, f"SN{s}.npz"))
        F = np.nan_to_num(d["F"]).astype(np.float32)
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        BASE[s] = (F, d["y"].astype(int))


def roll(F, w=10):
    """centred rolling mean/std over +-w epochs (prefix-sum needs a leading zero row)"""
    n, d = F.shape
    Fp = np.pad(F, ((w, w), (0, 0)), mode="edge")            # n + 2w
    z = np.zeros((1, d), Fp.dtype)
    cs = np.concatenate([z, np.cumsum(Fp, 0)])               # n + 2w + 1
    cs2 = np.concatenate([z, np.cumsum(Fp ** 2, 0)])
    W = 2 * w + 1
    m = (cs[W:W + n] - cs[0:n]) / W
    v = (cs2[W:W + n] - cs2[0:n]) / W - m ** 2
    return m.astype(np.float32), np.sqrt(np.clip(v, 0, None)).astype(np.float32)


def build(s, k, longfeat):
    """longfeat: False | 'pos' (position-in-night) | 'full' (rolling mean/std + pos)"""
    F, y = BASE[s]
    parts = [F]
    if longfeat:
        pos = (np.arange(len(F)) / max(1, len(F) - 1)).astype(np.float32)[:, None]
        if longfeat == "full":
            m, sd = roll(F, 10)
            parts += [m, sd]
        parts += [pos]
    G = np.concatenate(parts, 1)
    Gp = np.pad(G, ((k, k), (0, 0)), mode="edge")
    X = np.concatenate([Gp[i:i + len(G)] for i in range(2 * k + 1)], 1)
    return X.astype(np.float32), y


def transitions(seqs):
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum()
    return np.log(A + EPS), np.log(pi + EPS)


def viterbi(le, lA, lpi):
    T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        p[t] = bp[t + 1, p[t + 1]]
    return p


def met(y, p):
    return (accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0),
            cohen_kappa_score(y, p))


def run(name, k, longfeat, folds, n_folds=3):
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    raw, hmm = [], []
    for i, (tr_s, te_s) in enumerate(folds[:n_folds]):
        Xtr = np.concatenate([build(s, k, longfeat)[0] for s in tr_s])
        ytr = np.concatenate([BASE[s][1] for s in tr_s])
        # identical row budget across variants keeps the comparison fair and fits 6 GB VRAM
        rs = np.random.RandomState(0)
        if len(ytr) > 60000:
            idx = rs.choice(len(ytr), 60000, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.06,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            device="cuda", n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        lA, lpi = transitions([BASE[s][1] for s in tr_s])
        yt, pr, ph = [], [], []
        for s in te_s:
            X, y = build(s, k, longfeat); p = clf.predict_proba(X)
            yt.append(y); pr.append(p.argmax(1)); ph.append(viterbi(np.log(p + EPS), lA, lpi))
        yt = np.concatenate(yt)
        raw.append(met(yt, np.concatenate(pr))); hmm.append(met(yt, np.concatenate(ph)))
        print(f"    {name} fold{i}: raw mF1={raw[-1][1]:.4f}  +HMM acc={hmm[-1][0]:.4f} mF1={hmm[-1][1]:.4f}", flush=True)
    r = np.array(raw).mean(0); h = np.array(hmm).mean(0)
    print(f"  == {name:34s} dim={Xtr.shape[1]:5d} | raw {r[0]:.4f}/{r[1]:.4f} | +HMM {h[0]:.4f}/{h[1]:.4f}/{h[2]:.4f}", flush=True)
    return dict(name=name, k=k, longfeat=longfeat, dim=int(Xtr.shape[1]),
                raw_acc=float(r[0]), raw_mf1=float(r[1]),
                hmm_acc=float(h[0]), hmm_mf1=float(h[1]), hmm_kappa=float(h[2]))


def main():
    subs = subjects(); load(subs)
    folds = make_folds(subs, 10, seed=42)
    # A/B/C already measured: k=5 won (0.7450/0.6851 vs k=3 0.7419/0.6806).
    # Now test the long-range features on top of the k=5 winner.
    variants = [("F k=5 +pos", 5, "pos"), ("G k=5 +roll+pos", 5, "full")]
    out = []
    for nm, k, lf in variants:
        t0 = time.time()
        out.append(run(nm, k, lf, folds))
        print(f"     ({time.time()-t0:.0f}s)", flush=True)
    print("\n=========== CONTEXT SEARCH (LightGBM probe, 3 folds) ===========")
    print(f"{'variant':24s} {'dim':>6s} {'rawmF1':>8s} {'hmmAcc':>8s} {'hmmmF1':>8s}")
    for r in sorted(out, key=lambda z: -z["hmm_mf1"]):
        print(f"{r['name']:24s} {r['dim']:6d} {r['raw_mf1']:8.4f} {r['hmm_acc']:8.4f} {r['hmm_mf1']:8.4f}")
    json.dump(out, open(os.path.join(RES, "context_search.json"), "w"), indent=2)
    print("saved -> results/context_search.json")


if __name__ == "__main__":
    main()
