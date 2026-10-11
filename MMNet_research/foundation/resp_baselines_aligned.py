"""Respiratory-event baselines on the aligned timeline, same ten patient-independent folds as MM-Net.

Input: the 14 engineered cardiorespiratory descriptors per epoch (data/aligned/mm_features Fcard),
z-scored per patient; label: aligned epoch event label. Train on each fold's training side (training +
validation patients), score the test patients; AUC and AP are fold means, as for MM-Net.
    desaturation rule    SpO2 desaturation depth (median - minimum within the epoch)
    logistic regression  14 descriptors
    gradient boosting    14 descriptors, HistGradientBoosting
Output: results/revision/runs/lesion/resp_baselines_aligned.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MMNET_FEATURES", "aligned/mm_features")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import mmnet_core as C

FE = os.path.join(REPO, "data", "aligned", "mm_features")


def load(s):
    d = np.load(os.path.join(FE, "SN%d.npz" % s))
    f = d["Fcard"].astype(np.float32)
    return (f - f.mean(0)) / (f.std(0) + 1e-6), d["apnea"].astype(int)


def main():
    D = {s: load(s) for s in C.SUBS}
    res = {m: {"auc": [], "ap": []} for m in ("desaturation rule", "logistic regression", "gradient boosting")}
    for fi, (tr, te) in enumerate(C.FOLDS):
        Xtr = np.concatenate([D[s][0] for s in tr]); ytr = np.concatenate([D[s][1] for s in tr])
        Xte = np.concatenate([D[s][0] for s in te]); yte = np.concatenate([D[s][1] for s in te])
        scores = {"desaturation rule": Xte[:, 3],
                  "logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr).predict_proba(Xte)[:, 1],
                  "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42).fit(Xtr, ytr).predict_proba(Xte)[:, 1]}
        for m, sc in scores.items():
            res[m]["auc"].append(float(roc_auc_score(yte, sc))); res[m]["ap"].append(float(average_precision_score(yte, sc)))
        print("fold %d done" % fi, flush=True)
    out = {m: dict(auc=float(np.mean(v["auc"])), auc_sd=float(np.std(v["auc"])), ap=float(np.mean(v["ap"])), folds=v) for m, v in res.items()}
    json.dump(out, open(os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion", "resp_baselines_aligned.json"), "w"), indent=1)
    for m, v in out.items():
        print("%-20s AUC %.4f +- %.4f  AP %.4f" % (m, v["auc"], v["auc_sd"], v["ap"]))


if __name__ == "__main__":
    main()
