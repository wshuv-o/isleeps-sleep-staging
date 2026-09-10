"""TinySleepNet and SleepTransformer on healthy sleep -- validating the reimplementations.

The two literature rows in Table~\\ref{tab:bench} are our reimplementations, not
the published code, which leaves an obvious objection: they may score 0.61--0.69
on iSLEEPS because they were rebuilt badly rather than because the cohort is
hard. This answers it. The same code, on Sleep-EDF Expanded with the single
Fpz-Cz derivation both papers use, should land near the accuracies their authors
report (TinySleepNet 0.854, SleepTransformer 0.848 on Sleep-EDF-78). If it does,
the drop on stroke is a property of the cohort; if it does not, the rows should
not be in the table at all.

It deliberately imports `run_fold` from each script rather than reimplementing
the loop -- validating a different training loop would validate nothing. Only
the corpus changes.

Folds are split by SUBJECT, not recording: Sleep-EDF Cassette records each
subject on two nights, so SC4001 and SC4002 are one person and splitting on
filenames would put them on both sides of a fold.

  KMP_DUPLICATE_LIB_OK=TRUE python run_healthy_baselines.py [tiny|transformer|both]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import run_sleeptransformer as ST    # noqa: E402
import run_tinysleepnet as TSN       # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score)  # noqa: E402

PROC = os.path.join(REPO, "data", "sleep_edf_proc")
OUTDIR = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
EEG_CH = 0                   # Fpz-Cz, the derivation both papers use on Sleep-EDF
N_FOLDS = 10

# what the authors report on Sleep-EDF, for the sanity check printed at the end
PUBLISHED = {"tiny": ("TinySleepNet", 0.854, "Supratak and Guo, EMBC 2020"),
             "transformer": ("SleepTransformer", 0.848, "Phan et al., TBME 2022")}


def subject_of(rec):
    """SC4ssNE: characters 3:5 are the subject, so SC4001 and SC4002 are one person."""
    return rec[3:5]


def load(which):
    """Raw Fpz-Cz per recording, in whichever representation the model expects."""
    data = {}
    for f in sorted(glob.glob(os.path.join(PROC, "*.npz"))):
        rec = os.path.basename(f)[:-4]
        d = np.load(f)
        y = d["y"].astype(np.int64)
        x = np.asarray(d["x"][:, EEG_CH], dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        if which == "tiny":
            data[rec] = (x, y)
        else:
            s = ST.spectrogram(torch.tensor(x, device=ST.DEV))
            s = (s - s.mean((0, 1), keepdim=True)) / (s.std((0, 1), keepdim=True) + 1e-6)
            data[rec] = (s.half().cpu(), y)
    return data


def main(which):
    mod = TSN if which == "tiny" else ST
    name, published, ref = PUBLISHED[which]
    data = load(which)
    recs = sorted(data)
    subs = sorted({subject_of(r) for r in recs})
    print("%s on sleep-edf: %d recordings, %d subjects, %d epochs"
          % (name, len(recs), len(subs), sum(len(v[1]) for v in data.values())),
          flush=True)

    rng = np.random.RandomState(42)
    order = list(subs); rng.shuffle(order)
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]

    t0 = time.time()
    per_fold = []
    for fi, te_subs in enumerate(folds):
        te = [r for r in recs if subject_of(r) in te_subs]
        tr_all = [r for r in recs if subject_of(r) not in te_subs]
        rs = np.random.RandomState(100 + fi)
        tr_all = list(tr_all); rs.shuffle(tr_all)
        nv = max(2, len(tr_all) // 9)
        va, tr = tr_all[:nv], tr_all[nv:]
        yt, yp = mod.run_fold(data, tr, va, te, seed=42)
        per_fold.append(dict(fold=fi, n_test_subjects=len(te_subs), n=int(len(yt)),
                             acc=float(accuracy_score(yt, yp)),
                             mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                             kappa=float(cohen_kappa_score(yt, yp))))
        print("fold %2d  %d subj  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
              % (fi, len(te_subs), len(yt), per_fold[-1]["acc"], per_fold[-1]["mf1"],
                 per_fold[-1]["kappa"], (time.time() - t0) / 60), flush=True)

    agg = {k: (float(np.mean([f[k] for f in per_fold])),
               float(np.std([f[k] for f in per_fold], ddof=1)))
           for k in ("acc", "mf1", "kappa")}
    out = os.path.join(OUTDIR, "healthy_%s.json" % which)
    json.dump(dict(model="%s (reimplementation)" % name, reference=ref,
                   corpus="sleep_edf", protocol="%d-fold, subject-independent, "
                   "single Fpz-Cz derivation" % N_FOLDS,
                   n_recordings=len(recs), n_subjects=len(subs),
                   published_accuracy=published, per_fold=per_fold,
                   **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
                   minutes=(time.time() - t0) / 60), open(out, "w"), indent=1)

    print("\n%s ON HEALTHY SLEEP, %d-fold subject-independent" % (name.upper(), N_FOLDS))
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, agg[k][0], agg[k][1]))
    delta = agg["acc"][0] - published
    print("  published on Sleep-EDF: %.3f   ours: %.3f   difference %+.3f"
          % (published, agg["acc"][0], delta))
    print("  -> reimplementation %s"
          % ("looks faithful" if abs(delta) < 0.04 else
             "does NOT reproduce the published number; do not report the stroke row "
             "without saying so"))
    print("wrote", out)


if __name__ == "__main__":
    arg = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    for w in (["tiny", "transformer"] if arg == "both" else [arg]):
        main(w)
