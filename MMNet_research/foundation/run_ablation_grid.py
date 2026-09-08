"""Modality-ablation grid for the final model -- the evidence for C4.

Each row removes one modality and retrains. The masks are remapped first (see
ablation_remap.py): the manuscript's indices address the 14 engineered
cardiorespiratory features and the 188-d neural block, both of which the final
model replaced, and using them unpatched would silently produce plausible but
meaningless numbers.

Nine conditions, three seeds, ten patient-independent folds each. The 'full'
baseline is not re-run here -- it is the final model itself, already in
results/revision/runs/final/final_model.json.

  KMP_DUPLICATE_LIB_OK=TRUE python run_ablation_grid.py --workers 2
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
SH = os.path.join(OUT, "ablation_shards")

# manuscript condition name -> (eeg_drop, card_drop)
CONDITIONS = {
    "-EEG":        (("eeg",), ()),
    "-EOG":        (("eog",), ()),
    "-EMG":        (("emg",), ()),
    "-SpO2":       ((), ("spo2",)),
    "-pulse/HRV":  ((), ("pulse_hrv",)),
    "-ECG":        ((), ("ecg",)),
    "-airflow":    ((), ("airflow",)),
    "-effort":     ((), ("effort",)),
    "-all cardio": ((), ("all",)),
}

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, r"{repo}\MMNet_research\model")
sys.path.insert(0, r"{repo}\MMNet_research\foundation")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import mmnet_core as C
import sweep, ablation_remap

cond, seed, dst = "{cond}", {seed}, r"{dst}"
eeg_drop, card_drop = {eeg_drop}, {card_drop}
t0 = time.time()
with sweep.config(C, "A+labram", hidden=256, drop=0.3, lr=3e-4, wd=1e-4,
                  cardio="raw_cnn", cardio_mode="concat") as h:
    # remap AFTER the arm is entered: the masks depend on the widened input
    restore = ablation_remap.apply(C, n_eeg_total=h.dim)
    try:
        r = C.run_10fold(fusion="concat", temporal="lstm", seed=seed,
                         eeg_drop=eeg_drop, card_drop=card_drop)
    finally:
        restore()
rec = {{"condition": cond, "seed": seed, "eeg_drop": list(eeg_drop),
       "card_drop": list(card_drop), "eeg_dim": h.dim, "card_dim": h.n_card,
       "acc":   [f["acc"]   for f in r["per_fold"]],
       "mf1":   [f["mf1"]   for f in r["per_fold"]],
       "kappa": [f["kappa"] for f in r["per_fold"]],
       "auc":   [f["auc"]   for f in r["per_fold"]],
       "ap":    [f["ap"]    for f in r["per_fold"]],
       "minutes": (time.time() - t0) / 60}}
json.dump({{"%s|%d" % (cond, seed): rec}}, open(dst, "w"), indent=1)
print("DONE %-12s seed %-3d acc %.4f auc %.4f [%.1f min]"
      % (cond, seed, np.mean(rec["acc"]), np.nanmean(rec["auc"]), rec["minutes"]), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    # two workers, not three: raw cardio makes each window tensor ~2.9 GB of HOST
    # RAM, and three exhausted it during the earlier CNN runs
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(SH, exist_ok=True)

    jobs = []
    for cond, (ed, cd) in CONDITIONS.items():
        for seed in a.seeds:
            tag = cond.replace("-", "no").replace("/", "_").replace(" ", "_")
            dst = os.path.join(SH, "%s__s%d.json" % (tag, seed))
            if os.path.exists(dst):
                print("[skip] %s seed %d" % (cond, seed), flush=True)
                continue
            jobs.append((cond, ed, cd, seed, dst))

    print("%d runs, %d workers  (~%.1f h estimated)"
          % (len(jobs), a.workers, len(jobs) * 14.0 / a.workers / 60), flush=True)
    t0 = time.time(); running = []
    while jobs or running:
        while jobs and len(running) < a.workers:
            cond, ed, cd, seed, dst = jobs.pop(0)
            src = WORKER.format(repo=REPO, cond=cond, seed=seed, dst=dst,
                                eeg_drop=repr(ed), card_drop=repr(cd))
            log = open(dst[:-5] + ".log", "w")
            p = subprocess.Popen([sys.executable, "-c", src], stdout=log,
                                 stderr=subprocess.STDOUT)
            running.append((p, cond, seed, log))
            print("[start] %-12s seed %d" % (cond, seed), flush=True)
        time.sleep(5)
        for it in list(running):
            p, cond, seed, log = it
            if p.poll() is not None:
                log.close(); running.remove(it)
                print("[done ] %-12s seed %d  rc=%d  [%.1f min elapsed]"
                      % (cond, seed, p.returncode, (time.time() - t0) / 60), flush=True)
    print("\ngrid finished in %.1f h" % ((time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
