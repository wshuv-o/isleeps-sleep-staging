"""Complete the leave-one-modality-out grid so it covers every modality that
permutation importance covers.

Referee round-2 Finding 2 asks for an attribution that could have come out
otherwise: does per-feature permutation importance implicate the same channels
the retrain-ablation does? Testing that needs both methods measured over the same
modality set. Permutation importance covers eight (EEG, EOG, EMG, SpO2,
pulse/HRV, ECG, airflow, effort); the ablation grid covers seven -- '-EEG' was
never run. With seven the rank correlation is rho = 0.750, p = 0.052 on the
respiratory head, which is underpowered rather than negative.

This adds the missing condition. It is completeness, not a search for
significance: '-EEG' is the modality permutation importance ranks FIRST for
staging, so omitting it left the most important channel untested by one of the
two methods.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "foundation")
SH = os.path.join(OUT, "ablation_shards")

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, r"{repo}\MMNet_research\model")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import mmnet_core as C

name, seed, dst = "{name}", {seed}, r"{dst}"
eeg_drop, card_drop = {eeg_drop}, {card_drop}
t0 = time.time()
r = C.run_10fold(fusion="concat", eeg_drop=eeg_drop, card_drop=card_drop, seed=seed)
rec = {{"condition": name, "seed": seed, "eeg_drop": list(eeg_drop), "card_drop": list(card_drop),
       "acc":   [f["acc"]   for f in r["per_fold"]],
       "mf1":   [f["mf1"]   for f in r["per_fold"]],
       "kappa": [f["kappa"] for f in r["per_fold"]],
       "auc":   [f["auc"]   for f in r["per_fold"]],
       "ap":    [f["ap"]    for f in r["per_fold"]],
       "minutes": (time.time() - t0) / 60}}
json.dump({{"%s|%d" % (name, seed): rec}}, open(dst, "w"), indent=1)
print("DONE %s seed %d acc %.4f auc %.4f [%.1f min]"
      % (name, seed, np.mean(rec["acc"]), np.mean(rec["auc"]), rec["minutes"]), flush=True)
'''

CONDITIONS = {
    "-EEG": (("eeg",), ()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(SH, exist_ok=True)

    jobs = []
    for name, (ed, cd) in CONDITIONS.items():
        for seed in a.seeds:
            dst = os.path.join(SH, "%s_%d.json" % (name.replace("-", "no"), seed))
            if os.path.exists(dst):
                print("[skip] %s seed %d" % (name, seed), flush=True)
                continue
            jobs.append((name, ed, cd, seed, dst))

    print("%d jobs, %d workers" % (len(jobs), a.workers), flush=True)
    t0 = time.time()
    running = []
    while jobs or running:
        while jobs and len(running) < a.workers:
            name, ed, cd, seed, dst = jobs.pop(0)
            src = WORKER.format(repo=REPO, name=name, seed=seed, dst=dst,
                                eeg_drop=repr(ed), card_drop=repr(cd))
            log = open(dst[:-5] + ".log", "w")
            p = subprocess.Popen([sys.executable, "-c", src], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, name, seed, log))
            print("[start] %s seed %d" % (name, seed), flush=True)
        time.sleep(5)
        for it in list(running):
            p, name, seed, log = it
            if p.poll() is not None:
                log.close(); running.remove(it)
                print("[done ] %s seed %d rc=%d [%.1f min]"
                      % (name, seed, p.returncode, (time.time() - t0) / 60), flush=True)
    print("finished in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
