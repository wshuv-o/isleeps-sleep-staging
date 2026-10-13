"""Single-task controls for the final MM-Net (base-nocw-cross-ftlae-p8-al).

The same configuration, folds, validation split, seeds, budget and train-fold HMM as the
final model in run_seed_ensemble.py; only the loss changes: task="stage" trains the staging
head alone, task="apnea" the respiratory head alone (mmnet_core.train_fold). Fold
probabilities go to results/revision/runs/lesion/seed_ensemble/ as
<config>-stgonly_fold<k>.npz and <config>-apnonly_fold<k>.npz, beside the joint model's
files, and compare_final_single_task() scores all three with the same decoding.

Run with the aligned caches, as the final model was:
    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal
    MMNET_PFLOW=aligned/pressure_flow python run_final_single_task.py
"""
import functools
import json
import os
import sys

import numpy as np

import run_seed_ensemble as SE

C, sweep = SE.C, SE.sweep
FINAL_CFG = "base-nocw-cross-ftlae-p8-al"
TASKS = {"stgonly": "stage", "apnonly": "apnea"}


def run_task(suffix, task, seeds):
    """run_config with train_fold's loss restricted to one head."""
    enter = sweep.config.__enter__

    def enter_with_task(self):
        h = enter(self)
        C.train_fold = functools.partial(C.train_fold, task=task)
        return h

    sweep.config.__enter__ = enter_with_task
    try:
        SE.run_config("%s-%s" % (FINAL_CFG, suffix), seeds)
    finally:
        sweep.config.__enter__ = enter


def _resp_auc_ap(d, subs, seeds):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.concatenate([C.DATA[s][3] for s in subs]).astype(int)
    out = {}
    for sd in seeds:
        p = np.concatenate([d["A:SN%d|%d" % (s, sd)] for s in subs])
        out[sd] = (roc_auc_score(y, p), average_precision_score(y, p))
    return out


def compare(seeds):
    """Ten fold values per metric (seed mean) for joint, staging-only and respiratory-only."""
    from scipy.stats import wilcoxon
    names = {"joint": FINAL_CFG, "stgonly": FINAL_CFG + "-stgonly", "apnonly": FINAL_CFG + "-apnonly"}
    res = {k: {"acc": [], "kappa": [], "mf1": [], "auc": [], "ap": []} for k in names}
    for fi in range(len(C.FOLDS)):
        for k, cfg in names.items():
            d = np.load(os.path.join(SE.OUT, "%s_fold%d.npz" % (cfg, fi)))
            subs = [int(s) for s in d["te"]]
            yt = np.concatenate([C.DATA[s][2] for s in subs])
            m = [SE._metrics(yt, np.concatenate([SE._decode({cfg: d}, d["A_log"], d["pi_log"], s, [(cfg, sd)])
                                                 for s in subs])) for sd in seeds]
            for met in ("acc", "kappa", "mf1"):
                res[k][met].append(float(np.mean([x[met] for x in m])))
            ra = _resp_auc_ap(d, subs, seeds)
            res[k]["auc"].append(float(np.mean([v[0] for v in ra.values()])))
            res[k]["ap"].append(float(np.mean([v[1] for v in ra.values()])))
    lines = ["Final MM-Net (%s): joint vs single-task, ten folds, seeds %s" % (FINAL_CFG, seeds)]
    for met, single in (("acc", "stgonly"), ("kappa", "stgonly"), ("mf1", "stgonly"),
                        ("auc", "apnonly"), ("ap", "apnonly")):
        j, s = np.array(res["joint"][met]), np.array(res[single][met])
        lines.append("%-6s single %.4f  joint %.4f  diff %+.4f  folds up %d/10  Wilcoxon p=%.3f"
                     % (met, s.mean(), j.mean(), j.mean() - s.mean(), int((j > s).sum()),
                        wilcoxon(j, s).pvalue))
    return res, "\n".join(lines)


def main():
    seeds = [42, 1, 7]
    for suffix, task in TASKS.items():
        run_task(suffix, task, seeds)
    res, text = compare(seeds)
    out = os.path.join(SE.RL.OUT, "final_single_task")
    json.dump(res, open(out + ".json", "w"), indent=1)
    open(out + ".txt", "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
