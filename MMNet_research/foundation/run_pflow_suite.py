"""The paper's model-level analyses, re-run for the eight-channel model (+ nasal pressure).

The eight-channel configuration adds nasal pressure ("Pressure Flow") to the seven raw
cardiorespiratory channels of the final model. Everything else is the final
configuration. This runs, on the same ten folds, the analyses the manuscript reports for
the main model:

  full        the model itself; seed 42 also saves pooled predictions and per-patient
              results for the derived tables and figures
  ablations   the modality-ablation grid: each modality removed and the model retrained.
              'airflow' removes both airflow sensors (thermal and nasal pressure), and
              '-nasal pressure' removes that sensor alone
  joint       the matched single-task controls (staging-only, respiratory-only loss)
  bypass      the respiratory bypass removed (cardiorespiratory block zeroed at the head)

Seed 42 runs for every condition first, then seeds 1 and 7, so a complete single-seed
picture exists early. Results accumulate in results/revision/runs/final/pflow_suite.json
and the script resumes where it stopped.

The training loop is the host-memory variant from run_missing_channel.py (identical
batches; windows kept in system RAM, cut per batch).

  KMP_DUPLICATE_LIB_OK=TRUE python run_pflow_suite.py [--seeds 42 1 7] [--only full ...]
                                                      [--res other_machine.json]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import ablation_remap
import run_missing_channel as R
import run_mmnet_pflow as P

C, sweep, cardio_cnn = R.C, R.sweep, R.cardio_cnn
OUT = os.path.join(R.REPO, "MMNet_research", "results", "revision", "runs", "final")
RES = os.path.join(OUT, "pflow_suite.json")
FINAL = R.FINAL

N_T = 750
RAW8 = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse", "PressureFlow"]
GROUPS8 = {"spo2": ["SpO2"], "pulse_hrv": ["Pulse"], "ecg": ["ECG"],
           "airflow": ["Flow", "PressureFlow"], "effort": ["Thorax", "Abdomen", "Effort"],
           "pressure": ["PressureFlow"]}

# condition -> run_10fold keyword arguments
CONDITIONS = {
    "full":            {},
    "-EEG":            dict(eeg_drop=("eeg",)),
    "-EOG":            dict(eeg_drop=("eog",)),
    "-EMG":            dict(eeg_drop=("emg",)),
    "-SpO2":           dict(card_drop=("spo2",)),
    "-pulse/HRV":      dict(card_drop=("pulse_hrv",)),
    "-ECG":            dict(card_drop=("ecg",)),
    "-airflow":        dict(card_drop=("airflow",)),
    "-nasal pressure": dict(card_drop=("pressure",)),
    "-effort":         dict(card_drop=("effort",)),
    "-all cardio":     dict(card_drop=("all",)),
    "stage-only":      dict(task="stage"),
    "resp-only":       dict(task="apnea"),
    "no-bypass":       dict(bypass=False),
}


def card_groups8():
    out = {}
    for name, chans in GROUPS8.items():
        idx = []
        for c in chans:
            k = RAW8.index(c)
            idx.extend(range(k * N_T, (k + 1) * N_T))
        out[name] = idx
    out["all"] = list(range(len(RAW8) * N_T))
    return out


def run_one(cond, seed):
    kw = dict(CONDITIONS[cond])
    for k in ("eeg_drop", "card_drop"):
        kw[k] = list(kw.get(k, ()))
    keep = (cond == "full" and seed == 42)
    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode=FINAL["cardio_mode"]) as cfg:
        old_groups, old_eegm = C.CARD_GROUPS, C.EEGM
        C.CARD_GROUPS = card_groups8()
        C.EEGM = ablation_remap.widened_eeg_masks(C, cfg.dim)
        orig_tf = C.train_fold
        tf = R.train_fold_host(C, cfg)

        def train_fold_fresh(*a, **k):
            torch.cuda.empty_cache()
            return tf(*a, **k)
        C.__dict__["train_fold"] = train_fold_fresh
        try:
            r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"], seed=seed,
                             keep=keep, **kw)
        finally:
            C.__dict__["train_fold"] = orig_tf
            C.CARD_GROUPS, C.EEGM = old_groups, old_eegm
        if keep:
            order = [s for _, te in C.FOLDS for s in te]
            Pr = r["pred"]
            np.savez_compressed(os.path.join(OUT, "predictions_8ch_seed42.npz"),
                                y_true=Pr[0], y_pred=Pr[1], apnea_true=Pr[2],
                                apnea_score=Pr[3], subjects=np.array(order),
                                n_epochs=np.array([len(C.DATA[s][2]) for s in order]))
            json.dump(r["per_subject"], open(os.path.join(OUT, "per_subject_8ch_seed42.json"),
                                             "w"), indent=1)
    return r


def main(seeds, only, res_path=RES):
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    # the full model at seeds 1 and 7 was already run by run_mmnet_pflow.py
    done = os.path.join(OUT, "mmnet_pflow.json")
    if os.path.exists(done):
        for k, v in json.load(open(done)).items():
            s_ = v["seed"]
            if s_ != 42 and "full|%d" % s_ not in res:   # seed 42 is re-run to save predictions
                res["full|%d" % s_] = dict(cond="full", seed=s_, source="mmnet_pflow.json",
                                           minutes=v["minutes"],
                                           **{m: v[m] for m in ("acc", "mf1", "kappa", "auc", "ap")})
    conds = [c for c in CONDITIONS if not only or c in only]
    cardio_cnn.raw_cardio_table = P.table_with_pressure
    cardio_cnn.N_CH = 8
    t_all = time.time()
    try:
        # seed-major: every condition at the first seed before any second seed
        for seed in seeds:
            for cond in conds:
                key = "%s|%d" % (cond, seed)
                if key in res:
                    continue
                t0 = time.time()
                r = run_one(cond, seed)
                res[key] = dict(cond=cond, seed=seed,
                                **{m: [float(f[m]) for f in r["per_fold"]]
                                   for m in ("acc", "mf1", "kappa", "auc", "ap")},
                                minutes=(time.time() - t0) / 60)
                json.dump(res, open(res_path, "w"), indent=1)
                print("[ok] %-16s seed %-2d acc %.4f mf1 %.4f kappa %.4f auc %.4f ap %.4f "
                      "[%.1f min, %.1f h total]"
                      % (cond, seed, *(np.nanmean(res[key][m]) for m in
                                       ("acc", "mf1", "kappa", "auc", "ap")),
                         res[key]["minutes"], (time.time() - t_all) / 3600), flush=True)
    finally:
        cardio_cnn.raw_cardio_table, cardio_cnn.N_CH = P._raw_table, 7


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--res", default=RES,
                    help="results file; give each machine its own, then merge_pflow_suite.py")
    a = ap.parse_args()
    main(a.seeds, a.only, os.path.abspath(a.res))
