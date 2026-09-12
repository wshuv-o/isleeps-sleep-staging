"""Review item 11: how much does the transductive normalisation buy?

Features are currently standardised per recording, using that recording's own mean
and standard deviation. The labels are never touched, but the test night's own
distribution is, which the paper already calls mildly transductive. A reviewer
asked what a deployable pipeline would cost, where the statistics have to come from
the training population instead.

This runs the same model and the same folds with the only change being where the
standardisation statistics come from: pooled over the training patients of each
fold, then applied unchanged to the held-out patients. Nothing else differs.

  --folds N   probe on the first N folds before committing to all ten.

  KMP_DUPLICATE_LIB_OK=TRUE python run_train_pop_norm.py --folds 2
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
FE = os.path.join(REPO, "data", "mm_features")
FINAL = dict(arm="A+labram", cardio="raw_cnn", temporal="lstm",
             hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
EPS = 1e-6


def load_raw():
    """Every subject's features with NO standardisation applied."""
    raw = {}
    for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in C.DUP:
            continue
        d = np.load(f)
        raw[sid] = (np.nan_to_num(d["Feeg"]).astype(np.float32),
                    np.nan_to_num(d["Fcard"]).astype(np.float32),
                    d["y"].astype(np.int64), d["apnea"].astype(np.int64))
    return raw


def standardise(raw, train_subs):
    """Statistics pooled over the TRAINING patients, applied to everyone."""
    Fe = np.concatenate([raw[s][0] for s in train_subs], 0)
    Fc = np.concatenate([raw[s][1] for s in train_subs], 0)
    mu_e, sd_e = Fe.mean(0), Fe.std(0) + EPS
    mu_c, sd_c = Fc.mean(0), Fc.std(0) + EPS
    return {s: ((v[0] - mu_e) / sd_e, (v[1] - mu_c) / sd_c, v[2], v[3])
            for s, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw = load_raw()
    saved = C.DATA
    rows = []
    t0 = time.time()

    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode="concat"):
        for fi, (tr_all, te) in enumerate(C.FOLDS[:a.folds]):
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all)
            rng.shuffle(tr_all)
            nv = max(10, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]

            # the whole point: statistics from the training patients only
            C.DATA = standardise(raw, tr)

            Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
            for s in tr:
                y = C.DATA[s][2]; pi[y[0]] += 1
                for x, z in zip(y[:-1], y[1:]):
                    Am[x, z] += 1
            A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())

            model = C.train_fold(tr, va, "concat", [], [], seed=a.seed,
                                 bypass=True, temporal=FINAL["temporal"])
            yt, ph, ay, apn = [], [], [], []
            for s in te:
                sp, ap_s = C.subj_infer(model, s, [], [])
                y = C.DATA[s][2]
                yt.append(y); ph.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
                ay.append(C.DATA[s][3]); apn.append(ap_s)
            yt, ph = np.concatenate(yt), np.concatenate(ph)
            ay, apn = np.concatenate(ay), np.concatenate(apn)
            rows.append(dict(fold=fi,
                             acc=float(accuracy_score(yt, ph)),
                             kappa=float(cohen_kappa_score(yt, ph)),
                             auc=float(roc_auc_score(ay, apn))))
            print("  fold %d  acc %.4f  kappa %.4f  auc %.4f"
                  % (fi, rows[-1]["acc"], rows[-1]["kappa"], rows[-1]["auc"]), flush=True)

    C.DATA = saved
    acc = np.mean([r["acc"] for r in rows])
    auc = np.mean([r["auc"] for r in rows])
    print("\ntraining-population normalisation, %d folds" % len(rows))
    print("  accuracy %.4f   kappa %.4f   respiratory AUC %.4f   [%.1f min]"
          % (acc, np.mean([r["kappa"] for r in rows]), auc, (time.time() - t0) / 60))
    print("  per-recording normalisation, same folds: see final_model.json")

    dst = os.path.join(OUT, "train_pop_norm.json")
    json.dump({"seed": a.seed, "n_folds": len(rows), "per_fold": rows},
              open(dst, "w"), indent=1)
    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
