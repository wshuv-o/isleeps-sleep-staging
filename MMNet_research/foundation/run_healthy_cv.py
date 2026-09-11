"""MM-Net trained AND evaluated on healthy sleep -- the missing control.

The paper argues that 0.739 staging accuracy is a property of this cohort rather
than a weakness of the model, and supports it three ways: many architectures land
in the same band here, the learning curve is flat, and healthy-trained stagers
collapse on stroke. None of those shows the proposed model performing well on
easy data, which is the first thing a reader will ask for. This runs that cell:
the same architecture, cross-validated on Sleep-EDF Expanded.

Two departures from the main experiment, both forced by the corpus and both
stated in the table caption rather than buried:

  * Sleep-EDF carries no cardiorespiratory channels, so this is the neural
    branch alone and `task="stage"` drops the respiratory loss entirely. The
    comparison row is therefore MM-Net (neural only), not the full model.
  * Folds are split by SUBJECT, not by recording. Sleep-EDF Cassette records
    each subject on two nights (SC4001 and SC4002 are one person), so splitting
    on filenames would put the same subject on both sides and inflate the
    result -- the exact failure the iSLEEPS protocol is designed to avoid.

  KMP_DUPLICATE_LIB_OK=TRUE python run_healthy_cv.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score,        # noqa: E402
                             f1_score)

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "healthy_cv.json")
FINAL = dict(arm="A+labram", cardio="raw_cnn", temporal="lstm",
             hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
N_FOLDS = 10
STAGES = ["W", "N1", "N2", "N3", "R"]


def subject_of(rec):
    """SC4ssNE: characters 3:5 are the subject, so SC4001 and SC4002 are one person."""
    return rec[3:5]


def load(n_card, labram_dir):
    base = os.path.join(REPO, "data", "sleep_edf_mm")
    data = {}
    for f in sorted(glob.glob(os.path.join(base, "*.npz"))):
        rec = os.path.basename(f)[:-4]
        d = np.load(f)
        fe = np.nan_to_num(d["Feeg"]).astype(np.float32)
        fe = (fe - fe.mean(0)) / (fe.std(0) + 1e-6)
        e = np.load(os.path.join(labram_dir, rec + ".npz"))["emb"].astype(np.float32)
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)
        y = d["y"].astype(np.int64)
        # no cardiorespiratory channels in this corpus, and no respiratory labels
        fc = np.zeros((len(y), n_card), np.float32)
        a = np.zeros(len(y), np.int64)
        data[rec] = (np.concatenate([fe, e], axis=1), fc, y, a)
    return data


def main():
    t0 = time.time()
    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode="concat") as h:
        data = load(h.n_card, os.path.join(REPO, "data", "labram_ext", "sleep_edf"))
        recs = sorted(data)
        subs = sorted({subject_of(r) for r in recs})
        print("sleep-edf: %d recordings, %d subjects, %d epochs (eeg %d-d)"
              % (len(recs), len(subs), sum(len(v[2]) for v in data.values()), h.dim),
              flush=True)

        rng = np.random.RandomState(42)
        order = list(subs); rng.shuffle(order)
        folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]

        saved = dict(C.DATA)
        per_fold = []
        try:
            # the arm's recompiled subj_infer reads a SNAPSHOT of C.__dict__, so the
            # dict object has to be mutated in place rather than rebound
            C.DATA.clear(); C.DATA.update(data)
            for fi, te_subs in enumerate(folds):
                te = [r for r in recs if subject_of(r) in te_subs]
                tr_all = [r for r in recs if subject_of(r) not in te_subs]
                rs = np.random.RandomState(100 + fi)
                tr_all = list(tr_all); rs.shuffle(tr_all)
                nv = max(2, len(tr_all) // 9)
                va, tr = tr_all[:nv], tr_all[nv:]

                model = C.train_fold(tr, va, "concat", [], [], seed=42,
                                     temporal=FINAL["temporal"], task="stage")
                Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
                for s in tr:
                    y = C.DATA[s][2]; pi[y[0]] += 1
                    for x, z in zip(y[:-1], y[1:]): Am[x, z] += 1
                A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

                yt, yp = [], []
                for s in te:
                    sp, _ = C.subj_infer(model, s, [], [])
                    yt.append(C.DATA[s][2])
                    yp.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
                yt, yp = np.concatenate(yt), np.concatenate(yp)
                per_fold.append(dict(
                    fold=fi, n_test_subjects=len(te_subs), n_epochs=int(len(yt)),
                    acc=float(accuracy_score(yt, yp)),
                    mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                    kappa=float(cohen_kappa_score(yt, yp)),
                    pcf=[float(v) for v in f1_score(yt, yp, average=None,
                                                    labels=range(5), zero_division=0)]))
                print("fold %2d  %d subj  %5d ep  acc %.4f  mF1 %.4f  kappa %.4f  [%.1f min]"
                      % (fi, len(te_subs), len(yt), per_fold[-1]["acc"],
                         per_fold[-1]["mf1"], per_fold[-1]["kappa"],
                         (time.time() - t0) / 60), flush=True)
        finally:
            C.DATA.clear(); C.DATA.update(saved)

    agg = {k: (float(np.mean([f[k] for f in per_fold])),
               float(np.std([f[k] for f in per_fold], ddof=1)))
           for k in ("acc", "mf1", "kappa")}
    pcf = np.mean([f["pcf"] for f in per_fold], axis=0)
    res = dict(corpus="sleep_edf", protocol="%d-fold, subject-independent" % N_FOLDS,
               n_recordings=len(recs), n_subjects=len(subs),
               per_fold=per_fold, **{k: dict(mean=v[0], sd=v[1]) for k, v in agg.items()},
               per_class_f1={STAGES[i]: float(pcf[i]) for i in range(5)},
               minutes=(time.time() - t0) / 60)
    json.dump(res, open(OUT, "w"), indent=1)

    print("\nMM-NET ON HEALTHY SLEEP, %d-fold subject-independent" % N_FOLDS)
    for k in ("acc", "mf1", "kappa"):
        print("  %-6s %.4f +- %.4f" % (k, agg[k][0], agg[k][1]))
    print("  per-class F1:", {STAGES[i]: round(float(pcf[i]), 3) for i in range(5)})
    print("\n  for comparison, same architecture on stroke: 0.741 +- 0.017")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
