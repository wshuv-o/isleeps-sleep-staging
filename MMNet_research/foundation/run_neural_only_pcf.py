"""Neural-only model, retaining per-class F1 -- the missing half of Table V.

The manuscript compares per-class staging F1 for the neural-only model against
the full model, and the same comparison is drawn as fig_mm_perclass. The
ablation grid already ran the neural-only condition (card_drop='all'), but its
worker stored only acc/mF1/kappa/AUC/AP, so the per-class breakdown does not
exist for the final model and both the table row and the figure were left
marked stale.

This reruns exactly that condition and keeps `pcf`, which run_10fold already
computes per fold. Nothing else differs from the grid's '-all cardio' row, so
the aggregate metrics here should reproduce it (0.741 acc) -- and that is worth
checking, because if they do not, the two runs disagree about what
'neural-only' means.

  KMP_DUPLICATE_LIB_OK=TRUE python run_neural_only_pcf.py
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
import ablation_remap           # noqa: E402
import sweep                    # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
FINAL = dict(arm="A+labram", cardio="raw_cnn", temporal="lstm",
             hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
STAGES = ["W", "N1", "N2", "N3", "R"]


def main():
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "neural_only_pcf.json")
    res = json.load(open(dst)) if os.path.exists(dst) else {}

    for seed in (42, 1, 7):
        key = "neural_only|%d" % seed
        if key in res:
            print("[skip] %s" % key, flush=True); continue
        t0 = time.time()
        with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                          lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                          cardio_mode="concat") as h:
            restore = ablation_remap.apply(C, n_eeg_total=h.dim)
            try:
                r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"],
                                 seed=seed, card_drop=("all",))
            finally:
                restore()
        res[key] = {"seed": seed,
                    "acc": [f["acc"] for f in r["per_fold"]],
                    "mf1": [f["mf1"] for f in r["per_fold"]],
                    "kappa": [f["kappa"] for f in r["per_fold"]],
                    "pcf": [f["pcf"] for f in r["per_fold"]],
                    "auc": [f["auc"] for f in r["per_fold"]],
                    "minutes": (time.time() - t0) / 60}
        json.dump(res, open(dst, "w"), indent=1)
        print("[ok] seed %d  acc %.4f  pcf %s  [%.1f min]"
              % (seed, np.mean(res[key]["acc"]),
                 np.round(np.mean(res[key]["pcf"], axis=0), 3),
                 res[key]["minutes"]), flush=True)

    pcf = np.concatenate([res[k]["pcf"] for k in sorted(res)], axis=0)
    acc = np.concatenate([res[k]["acc"] for k in sorted(res)])
    print("\nNEURAL-ONLY, %d fold-values" % len(acc))
    print("  accuracy %.4f  (grid's '-all cardio' row: 0.7409)" % np.mean(acc))
    print("  per-class F1: %s"
          % {STAGES[i]: round(float(v), 4) for i, v in enumerate(pcf.mean(0))})
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
