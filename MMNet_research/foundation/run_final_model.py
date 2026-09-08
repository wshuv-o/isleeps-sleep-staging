"""Run the final model and save every artefact the manuscript's numbers derive from.

Final configuration (chosen by the ablations, see TODO_AFTER_FINAL_MODEL.md):

    EEG     188 engineered features  +  frozen LaBraM embedding (200-d)
    cardio  learned CNN over the raw 7-channel signal (0.155 M params)
    temporal BiLSTM, fusion concat, validated bypass, HMM Viterbi decode

Most of the paper's numbers are not separate experiments -- the confusion matrix,
per-class F1, AHI correlation, severity table, per-event-type AUC, calibration and
the ROC/PR curves are all derived from the same pooled test predictions. So this
runs the model once per seed with keep=True and saves those predictions, rather
than re-running the model once per figure.

Outputs, under results/revision/runs/final/:
    final_model.json          per-fold metrics for every seed
    final_per_fold.csv        the per-fold table the submission bundle ships
    predictions_seed42.npz    pooled y_true / y_pred / apnea_true / apnea_score
    per_subject_seed42.json   per-patient accuracy and mean apnea score
"""
import argparse
import csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--keep-seed", type=int, default=42,
                    help="the seed whose predictions are saved for the derived analyses")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "final_model.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    for seed in a.seeds:
        key = "final|%d" % seed
        if key in results:
            print("[skip] %s" % key, flush=True)
            continue
        t0 = time.time()
        keep = (seed == a.keep_seed)
        with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                          lr=FINAL["lr"], wd=FINAL["wd"],
                          cardio=FINAL["cardio"], cardio_mode=FINAL["cardio_mode"]) as h:
            r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"],
                             seed=seed, keep=keep)
        results[key] = {
            "seed": seed, "eeg_dim": h.dim, "card_dim": h.n_card,
            "acc":   [f["acc"]   for f in r["per_fold"]],
            "mf1":   [f["mf1"]   for f in r["per_fold"]],
            "kappa": [f["kappa"] for f in r["per_fold"]],
            "pcf":   [f["pcf"]   for f in r["per_fold"]],
            "auc":   [f["auc"]   for f in r["per_fold"]],
            "ap":    [f["ap"]    for f in r["per_fold"]],
            "minutes": (time.time() - t0) / 60,
        }
        json.dump(results, open(path, "w"), indent=1)
        print("[ok] seed %d  acc %.4f  mf1 %.4f  auc %.4f  [%.1f min]"
              % (seed, np.mean(results[key]["acc"]), np.mean(results[key]["mf1"]),
                 np.nanmean(results[key]["auc"]), results[key]["minutes"]), flush=True)

        if keep:
            # run_10fold(keep=True) returns pooled arrays: y_true, y_pred,
            # apnea_true, apnea_score. Everything downstream reads these.
            # run_10fold stores the pooled arrays under "pred":
            # (y_true, y_pred, apnea_true, apnea_score)
            P = r.get("pred")
            if P is not None:
                np.savez_compressed(
                    os.path.join(OUT, "predictions_seed%d.npz" % seed),
                    y_true=np.asarray(P[0]), y_pred=np.asarray(P[1]),
                    apnea_true=np.asarray(P[2]), apnea_score=np.asarray(P[3]))
                print("   saved pooled predictions (%d epochs)" % len(np.asarray(P[0])), flush=True)
            if "per_subject" in r:
                json.dump(r["per_subject"],
                          open(os.path.join(OUT, "per_subject_seed%d.json" % seed), "w"), indent=1)
                print("   saved per-subject results (%d patients)" % len(r["per_subject"]), flush=True)

    # per-fold CSV in the same schema the submission bundle ships
    csv_path = os.path.join(OUT, "final_per_fold.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "fold", "acc", "mf1", "kappa", "apnea_auc", "apnea_ap"])
        for k in sorted(results):
            r = results[k]
            for i in range(len(r["acc"])):
                w.writerow([r["seed"], i, r["acc"][i], r["mf1"][i], r["kappa"][i],
                            r["auc"][i], r["ap"][i]])
    print("\nwrote %s" % csv_path)

    acc = np.concatenate([results[k]["acc"] for k in results])
    mf1 = np.concatenate([results[k]["mf1"] for k in results])
    kap = np.concatenate([results[k]["kappa"] for k in results])
    auc = np.concatenate([results[k]["auc"] for k in results])
    apr = np.concatenate([results[k]["ap"] for k in results])
    print("\nFINAL MODEL, %d fold-values" % len(acc))
    for n, v in (("accuracy", acc), ("macro-F1", mf1), ("kappa", kap),
                 ("resp AUC", auc), ("resp AP", apr)):
        print("  %-10s %.4f +- %.4f" % (n, np.nanmean(v), np.nanstd(v, ddof=1)))


if __name__ == "__main__":
    main()
