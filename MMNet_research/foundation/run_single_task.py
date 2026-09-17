"""Train each task alone, so the joint model can be compared against matched
single-task models instead of only against other people's architectures.

Both reviews ask for this and call it the missing control. The paper shows that
one network can carry both outputs; it does not show what the second output
costs the first, or what the joint objective buys either of them. Without a
single-task baseline the multi-task claim is an assertion.

Everything is held fixed except the loss. Same configuration, same ten folds,
same three seeds, same optimiser and budget as run_final_model.py, and the same
architecture including the cardiorespiratory bypass -- only `task` changes, which
selects which term of L = L_stg + L_apn is kept (mmnet_core.train_fold, line 217).
The staging-only model therefore still builds the cardiorespiratory stream and
still feeds the respiratory head, it just never receives a respiratory gradient,
which is the comparison that isolates the objective rather than the architecture.

Read the metrics that belong to each row: accuracy and kappa are meaningless for
the respiratory-only model, whose staging head is untrained, and AUC and average
precision are meaningless for the staging-only model. Those cells are recorded
anyway so the asymmetry is visible rather than hidden.

  KMP_DUPLICATE_LIB_OK=TRUE python run_single_task.py
"""
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

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
FINAL = dict(arm="A+labram", cardio="raw_cnn", cardio_mode="concat",
             temporal="lstm", hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
SEEDS = [42, 1, 7]
TASKS = ["apnea", "stage"]


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "single_task.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()

    for task in TASKS:
        for seed in SEEDS:
            key = "%s|%d" % (task, seed)
            if key in results:
                print("[skip] %s" % key, flush=True)
                continue
            t1 = time.time()
            with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                              lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                              cardio_mode=FINAL["cardio_mode"]) as h:
                r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"],
                                 seed=seed, task=task)
            results[key] = {
                "task": task, "seed": seed,
                "acc":   [f["acc"]   for f in r["per_fold"]],
                "kappa": [f["kappa"] for f in r["per_fold"]],
                "mf1":   [f["mf1"]   for f in r["per_fold"]],
                "auc":   [f["auc"]   for f in r["per_fold"]],
                "ap":    [f["ap"]    for f in r["per_fold"]],
                "minutes": round((time.time() - t1) / 60, 2),
            }
            json.dump(results, open(path, "w"), indent=1)
            print("%-12s seed %2d  acc %.4f  kappa %.4f  auc %.4f  ap %.4f  [%.1f min]"
                  % (task, seed, np.mean(results[key]["acc"]), np.mean(results[key]["kappa"]),
                     np.mean(results[key]["auc"]), np.mean(results[key]["ap"]),
                     results[key]["minutes"]), flush=True)

    # ---- fold-mean comparison against the joint model -----------------------
    joint = json.load(open(os.path.join(OUT, "final_model.json")))
    jkeys = [k for k in joint if k.startswith("final|")]

    def fold_means(rows, metric):
        """Average the seeds within each fold, the unit every paired test uses."""
        per_seed = [r[metric] for r in rows]
        return np.mean(np.asarray(per_seed, float), axis=0)

    summary = {}
    for task in TASKS:
        rows = [results["%s|%d" % (task, s)] for s in SEEDS]
        for metric in ("acc", "kappa", "auc", "ap"):
            single = fold_means(rows, metric)
            j = fold_means([joint[k] for k in jkeys], metric)
            d = single - j
            summary["%s|%s" % (task, metric)] = dict(
                single=round(float(np.mean(single)), 4),
                joint=round(float(np.mean(j)), 4),
                delta=round(float(np.mean(d)), 4),
                folds_single_wins=int((d > 0).sum()))
    results["_summary"] = summary
    results["_note"] = ("accuracy and kappa are undefined for the apnea-only model and "
                        "AUC and AP for the stage-only model; both are stored so the "
                        "asymmetry is visible. Deltas are single minus joint on ten fold-means.")
    json.dump(results, open(path, "w"), indent=1)

    print("\n%-18s %8s %8s %8s  %s" % ("", "single", "joint", "delta", "folds single wins"))
    for k, v in summary.items():
        print("%-18s %8.4f %8.4f %+8.4f  %d/10" % (k, v["single"], v["joint"], v["delta"],
                                                   v["folds_single_wins"]))
    print("\nsaved -> %s  [%.1f min]" % (path, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
