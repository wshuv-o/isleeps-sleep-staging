"""Learning curve for the final model -- does more data still help?

The manuscript's data-efficiency curve (notebook 8, results/revision/runs/
data_efficiency.json) was measured on the submitted configuration: 188
engineered features and 14 engineered cardiorespiratory features. The final
model replaced both branches, so that curve no longer describes it.

The question the curve answers is the one that decides where effort goes next.
On the submitted model the staging curve was flat -- more patients did not help,
which is why the manuscript argues the ceiling is the cohort rather than the
architecture -- while the respiratory head was still climbing. If the learned
cardio CNN has changed that, the recommendation changes with it: a branch that
is still climbing at 99 patients is an argument for collecting more, and a flat
one is an argument that the representation, not the sample, is the limit.

Design copied from notebook 8 so the two curves are comparable:

  * training patients are subsampled, validation and test folds are untouched,
    so every fraction is scored on exactly the same held-out patients
  * the HMM transition matrix is rebuilt from the reduced training set, since
    letting it see all patients would leak
  * one seed (42), ten patient-independent folds per fraction

Sharded per fraction, checkpointed per fold. Four power cuts in one day made
the case for never losing more than one fold of work.

  KMP_DUPLICATE_LIB_OK=TRUE python run_learning_curve.py --workers 2
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
SH = os.path.join(OUT, "lc_shards")

FRACTIONS = [0.25, 0.50, 0.75, 1.00]

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, r"{repo}\MMNet_research\model")
sys.path.insert(0, r"{repo}\MMNet_research\foundation")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, f1_score, roc_auc_score)
import mmnet_core as C
import sweep

frac, dst = {frac}, r"{dst}"
done = json.load(open(dst)) if os.path.exists(dst) else {{}}

with sweep.config(C, "A+labram", hidden=256, drop=0.3, lr=3e-4, wd=1e-4,
                  cardio="raw_cnn", cardio_mode="concat") as h:
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        if str(fi) in done:
            print("[skip] fold %d" % fi, flush=True); continue
        t0 = time.time()
        # identical split to run_10fold, so folds line up with every other run
        rng = np.random.RandomState(100 + fi)
        tr_all = list(tr_all); rng.shuffle(tr_all)
        nv = max(10, len(tr_all) // 9)
        va, tr_full = tr_all[:nv], tr_all[nv:]

        # subsample TRAINING patients only; va and te are untouched
        k = max(5, int(round(frac * len(tr_full))))
        sub = np.random.RandomState(500 + fi).permutation(len(tr_full))[:k]
        tr = [tr_full[i] for i in sorted(sub)]

        model = C.train_fold(tr, va, "concat", [], [], seed=42, temporal="lstm")

        # HMM rebuilt from the REDUCED training set -- using all patients here
        # would leak data the model at this fraction never saw
        Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
        for s in tr:
            y = C.DATA[s][2]; pi[y[0]] += 1
            for x, z in zip(y[:-1], y[1:]): Am[x, z] += 1
        A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

        yt, ph, ay, ap = [], [], [], []
        for s in te:
            sp, apn = C.subj_infer(model, s, [], [])
            y = C.DATA[s][2]
            yt.append(y); ph.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
            ay.append(C.DATA[s][3]); ap.append(apn)
        yt, ph = np.concatenate(yt), np.concatenate(ph)
        ay, ap = np.concatenate(ay), np.concatenate(ap)
        two = len(np.unique(ay)) > 1
        done[str(fi)] = dict(
            n_train=len(tr), frac=frac,
            acc=float(accuracy_score(yt, ph)),
            mf1=float(f1_score(yt, ph, average="macro", zero_division=0)),
            kappa=float(cohen_kappa_score(yt, ph)),
            auc=float(roc_auc_score(ay, ap)) if two else float("nan"),
            ap=float(average_precision_score(ay, ap)) if two else float("nan"),
            minutes=(time.time() - t0) / 60)
        json.dump(done, open(dst, "w"), indent=1)
        print("fold %d  n_train %2d  acc %.4f  auc %.4f  [%.1f min]"
              % (fi, len(tr), done[str(fi)]["acc"], done[str(fi)]["auc"],
                 done[str(fi)]["minutes"]), flush=True)
print("FRACTION %.2f COMPLETE  acc %.4f  auc %.4f"
      % (frac, np.mean([v["acc"] for v in done.values()]),
         np.nanmean([v["auc"] for v in done.values()])), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    # two workers: raw cardio windows are ~2.9 GB of HOST RAM each, and three
    # exhausted it during the earlier CNN runs
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(SH, exist_ok=True)

    jobs = []
    for frac in a.fractions:
        dst = os.path.join(SH, "lc_f%03d.json" % int(round(frac * 100)))
        n = len(json.load(open(dst))) if os.path.exists(dst) else 0
        if n >= 10:
            print("[skip] fraction %.2f complete" % frac, flush=True)
            continue
        jobs.append((frac, dst, n))

    print("%d fractions to run, %d workers" % (len(jobs), a.workers), flush=True)
    t0 = time.time(); running = []
    while jobs or running:
        while jobs and len(running) < a.workers:
            frac, dst, n = jobs.pop(0)
            src = WORKER.format(repo=REPO, frac=frac, dst=dst)
            log = open(dst[:-5] + ".log", "a")
            p = subprocess.Popen([sys.executable, "-c", src], stdout=log,
                                 stderr=subprocess.STDOUT)
            running.append((p, frac, log))
            print("[start] fraction %.2f  (%d folds already done)" % (frac, n), flush=True)
        time.sleep(5)
        for it in list(running):
            p, frac, log = it
            if p.poll() is not None:
                log.close(); running.remove(it)
                print("[done ] fraction %.2f  rc=%d  [%.1f min elapsed]"
                      % (frac, p.returncode, (time.time() - t0) / 60), flush=True)
    print("\nlearning curve finished in %.2f h" % ((time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
