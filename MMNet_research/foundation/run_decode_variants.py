"""Decoding variants on top of the blended stager, chosen on validation patients.

Input per fold: MM-Net probabilities (run_seed_ensemble.py, configs given by --mmnet)
and boosting probabilities (run_boost_blend.py, folders given by --boost), blended
p = w * mean(MM-Net) + (1 - w) * mean(boosters), for w in a small grid.

Decoders, all fitted on the fold's TRAINING patients' labels only:
    hmm1      the published first-order HMM (posterior used as emission)
    hybrid    first-order HMM with emission p(y|x) / p(y)^beta (beta = 0, 0.5, 1), the
              standard hybrid classifier-HMM, where p(y) is the training class prior
    temp      emission log-posterior multiplied by alpha (trust in the classifier vs
              the transition model)
    hmm2      second-order HMM: state = (previous stage, stage), transitions counted on
              the training hypnograms with add-one smoothing

Every (w, decoder, beta, alpha) combination is scored on the fold's validation
patients and test patients; the combination with the best mean validation accuracy
over the ten folds is the one reported. All test scores are listed as well.

    python run_decode_variants.py [--mmnet base base-nocw] [--boost boost]
"""
import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import mmnet_core as C

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
NC, EPS = C.NC, C.EPS
WS = (0.5, 0.6, 0.75, 0.9)
BETAS = (0.0,)
ALPHAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def split(fi, tr_all):
    rng = np.random.RandomState(100 + fi)
    tr_all = list(tr_all)
    rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    return tr_all[nv:], tr_all[:nv]


def fit(tr):
    """First- and second-order transition tables and class prior from training labels."""
    A1 = np.ones((NC, NC)); pi1 = np.ones(NC); prior = np.ones(NC)
    A2 = np.ones((NC, NC, NC)); pi2 = np.ones((NC, NC))
    for s in tr:
        y = C.DATA[s][2]
        prior += np.bincount(y, minlength=NC)
        pi1[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            A1[a, b] += 1
        if len(y) > 1:
            pi2[y[0], y[1]] += 1
        for a, b, c in zip(y[:-2], y[1:-1], y[2:]):
            A2[a, b, c] += 1
    return dict(A1=np.log(A1 / A1.sum(1, keepdims=True)), pi1=np.log(pi1 / pi1.sum()),
                A2=np.log(A2 / A2.sum(2, keepdims=True)), pi2=np.log(pi2 / pi2.sum()),
                lprior=np.log(prior / prior.sum()))


def viterbi2(A2, pi2, le):
    """Second-order Viterbi: state (a, b) = (stage t-1, stage t)."""
    T = le.shape[0]
    if T == 1:
        return np.array([int(le[0].argmax())])
    dp = pi2 + le[0][:, None] + le[1][None, :]          # [a, b] for t = 1
    bps = []
    for t in range(2, T):
        sc = dp[:, :, None] + A2                          # [a, b, c]
        bp = sc.argmax(0)                                 # best a for each (b, c)
        dp = sc.max(0) + le[t][None, :]                   # [b, c]
        bps.append(bp)
    b, c = np.unravel_index(dp.argmax(), dp.shape)
    path = [c, b]
    for bp in reversed(bps):
        a = bp[b, c]
        path.append(a)
        b, c = a, b
    return np.array(path[::-1])


EXTRA, EXTRA_WS = None, (0.0,)


def load_fold(fi, mmnet, boosts, seeds):
    e = {c: np.load(os.path.join(OUT, "seed_ensemble", "%s_fold%d.npz" % (c, fi))) for c in mmnet}
    b = [np.load(os.path.join(OUT, d, "fold%d.npz" % fi)) for d in boosts]
    return e, b


def run(mmnet, boosts, seeds):
    rows = {}
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        tr, va = split(fi, tr_all)
        P = fit(tr)
        e, b = load_fold(fi, mmnet, boosts, seeds)
        for sp, subs in (("va", va), ("te", te)):
            yt = np.concatenate([C.DATA[s][2] for s in subs])
            pm = {s: np.mean([e[c]["SN%d|%d" % (s, sd)] for c in mmnet for sd in seeds], 0) for s in subs}
            pb = {s: np.mean([x["SN%d" % s] for x in b], 0) for s in subs}
            if EXTRA:      # an extra stager: per-fold, per-patient 'prob' arrays (data/<EXTRA>/fold<k>/)
                px = {s: np.mean([np.load(os.path.join(REPO, "data", d, "fold%d" % fi, "SN%d.npz" % s))["prob"]
                                   for d in EXTRA.split(",")], 0) for s in subs}
            for w, wx in itertools.product(WS, EXTRA_WS):
                lp = {s: np.log((1 - wx) * (w * pm[s] + (1 - w) * pb[s])
                                + (wx * px[s] if wx else 0) + EPS) for s in subs}
                for dec, beta, alpha in itertools.product(("hmm1",), BETAS, ALPHAS):
                    yp = []
                    for s in subs:
                        if np.isinf(alpha):           # no HMM: per-epoch argmax
                            yp.append((lp[s] - beta * P["lprior"]).argmax(1))
                            continue
                        le = alpha * (lp[s] - beta * P["lprior"])
                        yp.append(C.hmm(P["A1"], P["pi1"], le) if dec == "hmm1"
                                  else viterbi2(P["A2"], P["pi2"], le))
                    yp = np.concatenate(yp)
                    key = "w=%.2f %s beta=%.1f alpha=%.2f" % (w, dec, beta, alpha)
                    if EXTRA:
                        key = "wx=%.2f " % wx + key
                    rows.setdefault(key, {"va": [], "te": []})[sp].append(
                        (accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0),
                         cohen_kappa_score(yt, yp)))
        print("fold %d done" % fi, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmnet", nargs="+", default=["base", "base-nocw"])
    ap.add_argument("--boost", nargs="+", default=["boost"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--name", default="decode_variants")
    ap.add_argument("--alphas", type=float, nargs="+", default=None, help="'inf' = no HMM")
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--extra", default=None, help="folder(s) under data/ with fold<k>/SN<n>.npz 'prob'; comma-separated folders are averaged")
    ap.add_argument("--extra-weights", type=float, nargs="+", default=[0.0, 0.15, 0.25, 0.35, 0.5])
    a = ap.parse_args()
    global ALPHAS, WS, EXTRA, EXTRA_WS
    if a.extra:
        EXTRA, EXTRA_WS = a.extra, tuple(a.extra_weights)
    if a.alphas:
        ALPHAS = tuple(a.alphas)
    if a.weights:
        WS = tuple(a.weights)
    rows = run(a.mmnet, a.boost, a.seeds)
    mean = {k: {sp: np.mean(v[sp], 0).tolist() for sp in v} for k, v in rows.items()}
    json.dump({"mmnet": a.mmnet, "boost": a.boost, "folds": {k: {sp: [list(x) for x in v[sp]] for sp in v} for k, v in rows.items()}},
              open(os.path.join(OUT, a.name + ".json"), "w"))
    order = sorted(mean, key=lambda k: -mean[k]["va"][0])
    ref = "w=0.75 hmm1 beta=0.0 alpha=1.00"
    if EXTRA:   # reference: the same search without the extra stager
        ref = max((k for k in mean if k.startswith("wx=0.00 ")), key=lambda k: mean[k]["va"][0])
    lines = ["MM-Net %s + boosters %s; ranked by VALIDATION accuracy (test only scored)." % (a.mmnet, a.boost),
             "%-34s %22s   %22s" % ("", "VALIDATION acc/mf1/k", "TEST acc/mf1/k")]
    for k in order[:15] + ([ref] if ref in mean and ref not in order[:15] else []):
        lines.append("%-34s " % k + " ".join("%7.4f" % x for x in mean[k]["va"]) + "   "
                     + " ".join("%7.4f" % x for x in mean[k]["te"]))
    best = order[0]
    if ref in rows:
        d = np.array([x[0] for x in rows[best]["te"]]) - np.array([x[0] for x in rows[ref]["te"]])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        lines.append("\nchosen on validation: %s -> TEST acc %.4f; vs %s %+.4f (p=%.3f, %d/10 folds up)"
                     % (best, mean[best]["te"][0], ref, d.mean(), p, int((d > 0).sum())))
    text = "\n".join(lines)
    open(os.path.join(OUT, a.name + ".txt"), "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
