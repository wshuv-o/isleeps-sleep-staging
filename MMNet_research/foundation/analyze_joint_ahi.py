"""AHI from the joint final MM-Net vs AHI from its two single-task controls combined.

Same estimator as ahi_final.py (three-seed scores summed over decoded sleep, per-fold scale k
fitted on the fold's validation patients); the single-task estimate takes the decoded stages
of the staging-only model and the scores of the respiratory-only model. Nothing is trained.

    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal \
    MMNET_PFLOW=aligned/pressure_flow python analyze_joint_ahi.py
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

import ahi_final as AF

C, SEEDS, RUNS, DATA = AF.C, [42, 1, 7], AF.RUNS, AF.DATA
JOINT = "base-nocw-cross-ftlae-p8-al"


def estimate(stg_cfg, apn_cfg, ann):
    out = {}
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        _, va = AF.split(fi, tr_all)
        ds = np.load(os.path.join(RUNS, "seed_ensemble", "%s_fold%d.npz" % (stg_cfg, fi)))
        da = np.load(os.path.join(RUNS, "seed_ensemble", "%s_fold%d.npz" % (apn_cfg, fi)))
        raw = {}
        for s in list(va) + list(te):
            st = C.hmm(ds["A_log"], ds["pi_log"],
                       np.log(np.mean([ds["SN%d|%d" % (s, sd)] for sd in SEEDS], 0) + C.EPS))
            apr = np.mean([da["A:SN%d|%d" % (s, sd)] for sd in SEEDS], 0)
            sl = st != 0
            raw[s] = apr[sl].sum() / max(sl.sum() * 30 / 3600, 0.5)
        k = np.median([ann[s] / max(raw[s], 1e-3) for s in va])
        for s in te:
            out[s] = raw[s] * k
    return out


def main():
    ann = {s: float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%d.npz" % s))["ahi_ann"])
           for s in C.SUBS}
    j = estimate(JOINT, JOINT, ann)
    sp = estimate(JOINT + "-stgonly", JOINT + "-apnonly", ann)
    subs = sorted(j)
    a = np.array([ann[s] for s in subs])
    ej, es = np.array([j[s] for s in subs]), np.array([sp[s] for s in subs])
    band = lambda x: np.digitize(x, [5, 15, 30])
    res = {}
    for name, e in (("joint", ej), ("single-task", es)):
        res[name] = {"rho": float(spearmanr(e, a)[0]), "mae": float(np.mean(np.abs(e - a))),
                     "severity_acc": float(np.mean(band(e) == band(a)))}
    res["patients_lower_error_joint"] = int((np.abs(ej - a) < np.abs(es - a)).sum())
    res["n"] = len(subs)
    res["abs_error_wilcoxon_p"] = float(wilcoxon(np.abs(ej - a), np.abs(es - a)).pvalue)
    dst = os.path.join(AF.RUNS, "joint_ahi_analysis.json")
    json.dump(res, open(dst, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
