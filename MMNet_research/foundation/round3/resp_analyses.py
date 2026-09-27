"""Items 6K, 10b, 12b, 15: respiratory re-analyses of the stored seed-42 outputs.

Inputs: per-epoch respiratory probabilities for all 99 patients
(results/revision/runs/final/per_subject_seed42.json), stage labels (processed7),
event annotations (Flow Events sheets), AHI (subject_description.xlsx).

  * 6K  pooled AUC/AP (one curve over all epochs) and per-fold AUC, which must
        reproduce final_model.json seed 42 (fold assignment = mmnet_core.make_folds)
  * 10b burden-vs-AHI correlation with the burden averaged over all epochs (as
        published) and over sleep epochs only
  * 12b respiratory AUC separately on complete- and incomplete-montage recordings
  * 15  the same predictions scored against alternative epoch labels: any overlap,
        overlap > 5 s, overlap >= 15 s (half an epoch), and the epoch holding the
        majority of the event. The model is NOT retrained on these labels.

  python resp_analyses.py      (CPU, ~1 min)
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

import _common as C


def folds():
    subs = sorted(int(s[2:]) for s in C.SUBJECTS)
    r = np.random.RandomState(42)
    s = list(subs)
    r.shuffle(s)
    return [["SN%d" % x for x in s[i::10]] for i in range(10)]


def score(sids, P, Y):
    y = np.concatenate([Y[s] for s in sids])
    p = np.concatenate([P[s] for s in sids])
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                prevalence=float(y.mean()), n_epochs=int(len(y)), n_patients=len(sids))


def per_fold(P, Y, F):
    a = [roc_auc_score(np.concatenate([Y[s] for s in f]), np.concatenate([P[s] for s in f])) for f in F]
    return float(np.mean(a)), [float(x) for x in a]


def main():
    P, raw = C.probs()
    F = folds()
    ST = {s: C.stages(s) for s in C.SUBJECTS}
    rules = {
        "onset (published)": dict(rule="onset"),
        "any overlap": dict(rule="overlap", min_s=0.0),
        "overlap > 5 s": dict(rule="overlap", min_s=5.0),
        "overlap >= 15 s": dict(rule="overlap", min_s=14.999),
        "majority of event": dict(rule="overlap", frac=0.5),
    }
    Y = {name: {s: C.labels(s, len(ST[s]), **kw) for s in C.SUBJECTS} for name, kw in rules.items()}
    Yo = Y["onset (published)"]

    fm = json.load(open(os.path.join(C.FINAL, "final_model.json")))["final|42"]["auc"]
    mean_auc, fold_auc = per_fold(P, Yo, F)
    res = {"seed": 42,
           "fold_check": {"per_fold_auc_recomputed": fold_auc,
                          "per_fold_auc_final_model_json": fm,
                          "max_abs_diff": float(np.max(np.abs(np.subtract(fold_auc, fm))))},
           "pooled_onset": score(C.SUBJECTS, P, Yo),
           "fold_mean_auc_seed42": mean_auc}

    # 15 -- label-rule sensitivity
    sens = {}
    for name, Yr in Y.items():
        m, fa = per_fold(P, Yr, F)
        sens[name] = dict(score(C.SUBJECTS, P, Yr), fold_mean_auc=m)
    res["label_rule_sensitivity"] = sens

    # 12b -- montage subgroups
    comp = C.montage_complete()
    groups = {"complete": [s for s in C.SUBJECTS if comp[s] is True],
              "incomplete": [s for s in C.SUBJECTS if comp[s] is False],
              "status_not_on_disk": [s for s in C.SUBJECTS if comp[s] is None]}
    sub = {g: score(v, P, Yo) for g, v in groups.items() if v}
    per_pt = {}
    for s in C.SUBJECTS:
        if Yo[s].min() != Yo[s].max():
            per_pt[s] = float(roc_auc_score(Yo[s], P[s]))
    for g, v in groups.items():
        vals = [per_pt[s] for s in v if s in per_pt]
        if vals:
            sub[g]["per_patient_auc_median"] = float(np.median(vals))
            sub[g]["per_patient_auc_mean"] = float(np.mean(vals))
    a = [per_pt[s] for s in groups["complete"] if s in per_pt]
    b = [per_pt[s] for s in groups["incomplete"] if s in per_pt]
    sub["mannwhitney_per_patient_auc_p"] = float(mannwhitneyu(a, b).pvalue)
    sub["members"] = groups
    sub["pooled_all_but_incomplete"] = score(groups["complete"] + groups["status_not_on_disk"], P, Yo)
    res["montage_subgroups"] = sub

    # 10b -- burden vs AHI, all epochs vs sleep epochs
    A = C.ahi()
    ids = [s for s in C.SUBJECTS if s in A]
    all_b = [P[s].mean() for s in ids]
    sleep_b = [P[s][ST[s] != 0].mean() for s in ids]
    ahi = [A[s] for s in ids]
    r1 = spearmanr(all_b, ahi)
    r2 = spearmanr(sleep_b, ahi)
    # events per hour of sleep from the model's thresholded output is in event_level_final.json;
    # here we also give the scored-epoch fraction vs AHI as the label-side ceiling.
    lab_b = [Yo[s][ST[s] != 0].mean() for s in ids]
    r3 = spearmanr(lab_b, ahi)
    res["ahi"] = {"n": len(ids),
                  "rho_burden_all_epochs": float(r1.statistic), "p_all": float(r1.pvalue),
                  "rho_burden_sleep_epochs": float(r2.statistic), "p_sleep": float(r2.pvalue),
                  "rho_scored_label_fraction_sleep_vs_ahi": float(r3.statistic), "p_label": float(r3.pvalue)}
    C.save("resp_analyses.json", res)
    print(json.dumps({k: res[k] for k in ("fold_check", "pooled_onset", "ahi")}, indent=1)[:1500])
    for k, v in sens.items():
        print("%-20s prev %.3f  pooled AUC %.4f  AP %.4f  fold-mean AUC %.4f"
              % (k, v["prevalence"], v["auc"], v["ap"], v["fold_mean_auc"]))
    for g in ("complete", "incomplete", "status_not_on_disk"):
        if g in sub:
            print(g, {k: sub[g][k] for k in sub[g] if k != "n_epochs"})
    print("MW p", sub["mannwhitney_per_patient_auc_p"])


if __name__ == "__main__":
    main()
