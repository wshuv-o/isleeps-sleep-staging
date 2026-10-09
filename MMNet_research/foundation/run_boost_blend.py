"""Blend MM-Net's seed ensemble with a gradient-boosting stager, choices made on validation patients.

Boosting stager: sklearn HistGradientBoostingClassifier on the same per-patient
z-scored features MM-Net reads (188 engineered EEG/EOG/EMG + 14 cardiorespiratory),
with +/-3 epochs of context, as in preprocessing/cardio_features.py. For each of the
paper's ten folds it is trained on exactly the patients MM-Net trains on (the fold's
validation and test patients both held out), so its validation probabilities are
out-of-sample, like MM-Net's.

Blend: p = w * MM-Net 3-seed ensemble + (1 - w) * boosting, w in {0, .25, .5, .75, 1},
optionally followed by the class-weight correction of run_seed_ensemble.py, then the
same train-fold HMM. The (w, correction) pair is chosen by mean validation accuracy
over the ten folds; every pair's test score is reported as well.

Needs the MM-Net probabilities from run_seed_ensemble.py (default: config "base").
Boosting probabilities: results/revision/runs/lesion/boost/fold<k>.npz (resumes).
Summary: boost_blend.json / boost_blend.txt in results/revision/runs/lesion/.

    python run_boost_blend.py [--mmnet base] [--threads 8]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import mmnet_core as C

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
ENS = os.path.join(OUT, "seed_ensemble")
BOOST = os.path.join(OUT, "boost")
CTX = 3
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("acc", "mf1", "kappa")


def context(F, k=None):
    k = CTX if k is None else k
    Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([Fp[i:i + len(F)] for i in range(2 * k + 1)], axis=1)


LABRAM = False


def X(s):
    fe, fc, _, _ = C.DATA[s]           # already z-scored per patient by load_data
    parts = [fe, fc]
    if LABRAM:                         # frozen LaBraM embedding, z-scored per patient as arms.py does
        e = np.load(os.path.join(REPO, "data", "cbramod_emb", "labram", "SN%d.npz" % s))["emb"].astype(np.float32)
        parts.append((e - e.mean(0)) / (e.std(0) + 1e-6))
    return context(np.concatenate(parts, 1))


def split(fi, tr_all):
    rng = np.random.RandomState(100 + fi)
    tr_all = list(tr_all)
    rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    return tr_all[nv:], tr_all[:nv]


def train_boost(max_features=1.0, seed=42):
    os.makedirs(BOOST, exist_ok=True)
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        path = os.path.join(BOOST, "fold%d.npz" % fi)
        tr, va = split(fi, tr_all)
        if os.path.exists(path):
            try:
                with np.load(path) as d:
                    if all("SN%d" % s in d for s in list(va) + list(te)):
                        print("[skip] boost fold %d" % fi, flush=True)
                        continue
            except Exception:
                pass
        t0 = time.time()
        Xtr = np.concatenate([X(s) for s in tr])
        ytr = np.concatenate([C.DATA[s][2] for s in tr])
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=seed,
                                             max_features=max_features)
        clf.fit(Xtr, ytr)
        probs = {"SN%d" % s: clf.predict_proba(X(s)).astype(np.float32) for s in list(va) + list(te)}
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **probs)
        os.replace(tmp, path)
        print("[ok] boost fold %d  (%.1f min, %d training epochs)"
              % (fi, (time.time() - t0) / 60, len(ytr)), flush=True)


def evaluate(mmnet):
    rows = {}
    for fi in range(len(C.FOLDS)):
        e = np.load(os.path.join(ENS, "%s_fold%d.npz" % (mmnet, fi)))
        b = np.load(os.path.join(BOOST, "fold%d.npz" % fi))
        A_log, pi_log, cw, seeds = e["A_log"], e["pi_log"], e["cw"], [int(x) for x in e["seeds"]]
        for sp in ("va", "te"):
            subs = [int(s) for s in e[sp]]
            yt = np.concatenate([C.DATA[s][2] for s in subs])
            for w in WEIGHTS:
                for corr in (False, True):
                    yp = []
                    for s in subs:
                        pm = np.mean([e["SN%d|%d" % (s, sd)] for sd in seeds], 0)
                        p = w * pm + (1 - w) * b["SN%d" % s]
                        if corr:
                            p = p / cw
                            p = p / p.sum(1, keepdims=True)
                        yp.append(C.hmm(A_log, pi_log, np.log(p + C.EPS)))
                    yp = np.concatenate(yp)
                    key = "w=%.2f%s" % (w, " cwcorr" if corr else "")
                    rows.setdefault(key, {"va": [], "te": []})[sp].append(dict(
                        acc=float(accuracy_score(yt, yp)),
                        mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                        kappa=float(cohen_kappa_score(yt, yp))))
    return {k: {sp: {m: [f[m] for f in v[sp]] for m in METRICS} for sp in v} for k, v in rows.items()}


def report(res, mmnet):
    from scipy.stats import wilcoxon
    lines = ["MM-Net (%s, 3-seed ensemble) blended with boosting; w = MM-Net weight." % mmnet,
             "Validation patients choose; test folds are only scored.",
             "%-16s %24s   %24s" % ("", "VALIDATION acc/mf1/kappa", "TEST acc/mf1/kappa")]
    for k, v in res.items():
        lines.append("%-16s " % k + " ".join("%7.4f" % np.mean(v["va"][m]) for m in METRICS)
                     + "    " + " ".join("%7.4f" % np.mean(v["te"][m]) for m in METRICS))
    best = max(res, key=lambda k: np.mean(res[k]["va"]["acc"]))
    ref = "w=1.00"                                   # MM-Net ensemble alone
    d = np.array(res[best]["te"]["acc"]) - np.array(res[ref]["te"]["acc"])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    lines.append("\nchosen on validation: %s -> TEST acc %.4f (vs MM-Net ensemble alone %+.4f, "
                 "Wilcoxon p=%.3f, %d/10 folds up)"
                 % (best, np.mean(res[best]["te"]["acc"]), d.mean(), p, int((d > 0).sum())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmnet", default="base")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boost-only", action="store_true")
    # a second, more diverse booster: wider context, feature subsampling, own folder
    ap.add_argument("--tag", default="", help="saves to boost<_tag>/ instead of boost/")
    ap.add_argument("--ctx", type=int, default=3)
    ap.add_argument("--max-features", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labram", action="store_true", help="also feed the LaBraM embedding")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    global BOOST, CTX, LABRAM
    CTX, LABRAM = a.ctx, a.labram
    if a.tag:
        BOOST = os.path.join(OUT, "boost_" + a.tag)
    train_boost(a.max_features, a.seed)
    if a.boost_only:
        return
    res = evaluate(a.mmnet)
    json.dump(res, open(os.path.join(OUT, "boost_blend.json"), "w"), indent=1)
    text = report(res, a.mmnet)
    open(os.path.join(OUT, "boost_blend.txt"), "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
