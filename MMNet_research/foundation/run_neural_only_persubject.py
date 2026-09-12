"""Neural-only MM-Net with per-subject output, as the control for review item 10.

A reviewer asked whether zero-filling the absent cardiorespiratory channels lets
the network learn missingness instead of physiology. The sensitivity analysis on
the full model answers most of it already: dropping the twelve incomplete-montage
patients moves pooled respiratory AUC by -0.003, and predicted burden is
independent of missingness (p = 0.905).

What it does not settle is staging, which is lower on those twelve (0.670 vs
0.743, p = 0.017). The full model's ablation says that cannot be the zero-filling,
since deleting the whole cardiorespiratory stream leaves staging unchanged at
0.741. This run makes the argument directly instead of by inference.

Neural-only never receives a cardiorespiratory channel at all, zero-filled or
otherwise. So if it ALSO scores lower on those twelve patients, the gap belongs to
the recordings and not to how missing channels are handled. If it does not, the
handling is implicated and the paper should say so.

Same configuration and folds as run_neural_only_pcf.py, which already ran this
condition; the only change is that run_10fold's per-subject dict is kept.

  KMP_DUPLICATE_LIB_OK=TRUE python run_neural_only_persubject.py
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


def main():
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "neural_only_per_subject.json")
    res = json.load(open(dst)) if os.path.exists(dst) else {}

    for seed in (42,):
        key = "neural_only|%d" % seed
        if key in res:
            print("[skip] %s" % key, flush=True)
            continue
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
        res[key] = {
            "seed": seed,
            "acc": [f["acc"] for f in r["per_fold"]],
            "kappa": [f["kappa"] for f in r["per_fold"]],
            # the point of this run: staging per held-out patient
            "per_subject": {k: {"acc": v["acc"], "kappa": v["kappa"]}
                            for k, v in r["per_subject"].items()},
            "minutes": (time.time() - t0) / 60,
        }
        json.dump(res, open(dst, "w"), indent=1)
        print("[ok] seed %d  acc %.4f  over %d patients  [%.1f min]"
              % (seed, float(np.mean(res[key]["acc"])),
                 len(res[key]["per_subject"]), res[key]["minutes"]), flush=True)

    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
