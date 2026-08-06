"""
resp_baselines.py -- respiratory-event detection baselines (revision brief 4.1).

The paper benchmarks fifteen architectures on staging and zero on the respiratory task,
which is the task the contribution rests on. Here we give the respiratory head real
competition, on the SAME folds, SAME 14 cardiorespiratory features (per-subject z-scored),
N=96:
    * Desaturation rule  -- score = SpO2 desaturation depth (feature 3), no training
    * Logistic regression on the 14 cardiorespiratory features
    * Gradient boosting   on the 14 cardiorespiratory features
Report AUC and AP per fold (mean +- std), plus the AP of a random classifier at the event
prevalence so the MM-Net AP is interpretable. If MM-Net does not beat gradient boosting on
cardio-only features, the framing shifts to the joint single-pass argument.

  KMP_DUPLICATE_LIB_OK=TRUE python revision/resp_baselines.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmnet_repro import load_data, make_folds, REV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DESAT_IDX = 3   # SpO2 desaturation depth in the 14-feature cardio vector


def main():
    data = load_data(); subs = sorted(data)
    folds = make_folds(subs, 10, seed=42)
    Xc = {s: data[s][1] for s in subs}          # 14 cardio features (per-subject z-scored)
    A = {s: data[s][3] for s in subs}           # binary apnea label
    prev = float(np.concatenate([A[s] for s in subs]).mean())
    res = {k: {"auc": [], "ap": []} for k in ["desat_rule", "logreg", "gboost"]}
    for tr, te in folds:
        Xtr = np.concatenate([Xc[s] for s in tr]); ytr = np.concatenate([A[s] for s in tr])
        Xte = np.concatenate([Xc[s] for s in te]); yte = np.concatenate([A[s] for s in te])
        # desaturation rule: score = desaturation depth (no training)
        sd = Xte[:, DESAT_IDX]
        res["desat_rule"]["auc"].append(roc_auc_score(yte, sd)); res["desat_rule"]["ap"].append(average_precision_score(yte, sd))
        # logistic regression
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        p = lr.predict_proba(Xte)[:, 1]
        res["logreg"]["auc"].append(roc_auc_score(yte, p)); res["logreg"]["ap"].append(average_precision_score(yte, p))
        # gradient boosting
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                            class_weight="balanced", random_state=42).fit(Xtr, ytr)
        p = gb.predict_proba(Xte)[:, 1]
        res["gboost"]["auc"].append(roc_auc_score(yte, p)); res["gboost"]["ap"].append(average_precision_score(yte, p))
    out = {"prevalence": prev, "random_ap": prev}
    print(f"respiratory-event detection baselines (10-fold, N={len(subs)}, prevalence {prev:.3f})\n")
    print(f"  {'random classifier':22s} AP={prev:.4f} (= prevalence)")
    for k in ["desat_rule", "logreg", "gboost"]:
        au = np.array(res[k]["auc"]); ap = np.array(res[k]["ap"])
        out[k] = {"auc": [float(au.mean()), float(au.std())], "ap": [float(ap.mean()), float(ap.std())]}
        print(f"  {k:22s} AUC={au.mean():.4f}+-{au.std():.3f}  AP={ap.mean():.4f}+-{ap.std():.3f}")
    json.dump(out, open(os.path.join(REV, "runs", "resp_baselines.json"), "w"), indent=2)
    print("\n  (MM-Net headline: apnea AUC ~0.69-0.70, AP ~0.31-0.33 -- compare above)")


if __name__ == "__main__":
    main()
