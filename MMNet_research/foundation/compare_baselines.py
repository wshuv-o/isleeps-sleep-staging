"""MM-Net variants against every reimplemented baseline, fold by fold, on the paper's ten folds.

Baseline per-fold accuracies are read from results/revision/runs/final/ (seed-averaged where
several seeds exist). MM-Net variants come from run_seed_ensemble.py's per-fold probabilities
(seed-averaged single-model accuracy, and the 3-seed ensemble), decoded with the train-fold HMM
exactly as run_10fold does. Each MM-Net variant is compared with each baseline by a paired
Wilcoxon test on the ten fold accuracies, Holm-corrected across the baselines.

    python compare_baselines.py [--mmnet base-nocw-cross base-nocw-cross-ftl ...]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score

import mmnet_core as C

FINAL = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
ENS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "seed_ensemble")
BASELINES = {
    "AttnSleep": "attnsleep_seeded.json",
    "LSTM (corpus, reimpl.)": "isleeps_lstm.json",
    "1D-ResNet-SE-LSTM": "resnetse_stroke.json",
    "AT-BiLSTM": "atbilstm_stroke.json",
    "SleepTransformer": "sleeptransformer.json",
    "TinySleepNet": "tinysleepnet.json",
    "U-Time": "utime_stroke.json",
    "Huttunen et al.": "huttunen.json",
    "StagingSeqNet": "stagingseqnet_seeded.json",
}


def _fold_accs(obj):
    """Ten fold accuracies from one seed's record, whatever its layout."""
    if isinstance(obj, dict) and "acc" in obj and isinstance(obj["acc"], list):
        return [float(x) for x in obj["acc"]]
    if isinstance(obj, dict) and "per_fold" in obj:
        obj = obj["per_fold"]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for k in ("acc", "hmm", "raw"):
            if k in obj[0]:
                v = [f[k] for f in obj]
                return [float(x["acc"]) if isinstance(x, dict) else float(x) for x in v]
    return None


def baseline_folds(fname):
    r = json.load(open(os.path.join(FINAL, fname)))
    seeds = []
    if isinstance(r, dict) and "per_seed" in r and isinstance(r["per_seed"], dict):
        seeds = [_fold_accs(v) for v in r["per_seed"].values()]
    elif isinstance(r, dict) and any(k.startswith("seed|") for k in r):
        seeds = [_fold_accs(v) for k, v in r.items() if k.startswith("seed|")]
    else:
        seeds = [_fold_accs(r)]
    seeds = [s for s in seeds if s is not None and len(s) == 10]
    return np.mean(seeds, 0) if seeds else None, len(seeds)


def mmnet_folds(cfg, seeds=(42, 1, 7)):
    single, ens = [], []
    for fi in range(10):
        d = np.load(os.path.join(ENS, "%s_fold%d.npz" % (cfg, fi)))
        A, pi, te = d["A_log"], d["pi_log"], [int(s) for s in d["te"]]
        yt = np.concatenate([C.DATA[s][2] for s in te])
        accs = [accuracy_score(yt, np.concatenate([C.hmm(A, pi, np.log(d["SN%d|%d" % (s, sd)] + C.EPS))
                                                   for s in te])) for sd in seeds]
        single.append(np.mean(accs))
        p = np.concatenate([C.hmm(A, pi, np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in seeds], 0) + C.EPS))
                            for s in te])
        ens.append(accuracy_score(yt, p))
    return np.array(single), np.array(ens)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmnet", nargs="+", default=["base-nocw-cross"])
    a = ap.parse_args()
    base = {}
    print("baselines (10-fold mean accuracy, seed-averaged):")
    for name, f in BASELINES.items():
        v, n = baseline_folds(f)
        if v is None:
            print("  %-24s unreadable" % name)
            continue
        base[name] = v
        print("  %-24s %.4f  (%d seed%s)" % (name, v.mean(), n, "" if n == 1 else "s"))
    for cfg in a.mmnet:
        if not all(os.path.exists(os.path.join(ENS, "%s_fold%d.npz" % (cfg, fi))) for fi in range(10)):
            print("\n%s: not complete" % cfg)
            continue
        single, ens = mmnet_folds(cfg)
        for label, v in (("single model, seed mean", single), ("3-seed ensemble", ens)):
            print("\n== MM-Net %s, %s: %.4f" % (cfg, label, v.mean()))
            names = list(base)
            ps = np.array([wilcoxon(v - base[n]).pvalue for n in names])
            adj = holm(ps)
            for n, p, q in zip(names, ps, adj):
                d = v - base[n]
                print("   vs %-24s %+.4f  %d/10 folds up  p=%.4f  Holm p=%.4f%s"
                      % (n, d.mean(), (d > 0).sum(), p, q, "  *" if q < 0.05 else ""))


if __name__ == "__main__":
    main()
