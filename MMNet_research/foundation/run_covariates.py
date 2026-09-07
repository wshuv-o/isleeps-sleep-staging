"""Do clinical covariates improve the respiratory head?

The paper's learning curve shows staging has reached this cohort's ceiling while
respiratory detection has not. This tests the head with headroom, using the
clinical predictors of sleep-disordered breathing that the metadata carries at
79-100% completeness -- and never AHI, which is derived from the events being
detected.

Runs the published configuration (temporal='lstm', which is the one with the
stronger respiratory AUC) with and without covariates, three seeds each, so the
comparison is paired over the same folds.
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
SH = os.path.join(OUT, "covar_shards")

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, r"{repo}\MMNet_research\model")
sys.path.insert(0, r"{repo}\MMNet_research\foundation")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import mmnet_core as C
import arms

variant, covars, temporal, seed = "{variant}", {covars}, "{temporal}", {seed}
key, dst = "{key}", r"{dst}"
t0 = time.time()
with arms.arm(C, variant, covars=covars) as a:
    r = C.run_10fold(fusion="concat", temporal=temporal, seed=seed)
rec = {{"variant": variant, "covars": covars, "temporal": temporal, "seed": seed,
       "eeg_dim": a.dim, "card_dim": a.n_card,
       "acc":   [f["acc"]   for f in r["per_fold"]],
       "mf1":   [f["mf1"]   for f in r["per_fold"]],
       "kappa": [f["kappa"] for f in r["per_fold"]],
       "auc":   [f["auc"]   for f in r["per_fold"]],
       "ap":    [f["ap"]    for f in r["per_fold"]],
       "minutes": (time.time() - t0) / 60}}
json.dump({{key: rec}}, open(dst, "w"), indent=1)
print("DONE %s acc %.4f auc %.4f ap %.4f [%.1f min]"
      % (key, np.mean(rec["acc"]), np.mean(rec["auc"]), np.mean(rec["ap"]), rec["minutes"]), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temporal", default="lstm")
    ap.add_argument("--variant", default="A")
    a = ap.parse_args()
    os.makedirs(SH, exist_ok=True)
    sys.path.insert(0, HERE)
    import clinical

    allc = list(clinical.COVARIATES)
    conditions = [("no_covars", None), ("all_covars", allc),
                  ("anthro_only", ["bmi", "neck", "abd"])]

    jobs = []
    for cname, cov in conditions:
        for seed in a.seeds:
            key = "%s|%s|t=%s|%d" % (a.variant, cname, a.temporal, seed)
            dst = os.path.join(SH, key.replace("|", "__").replace("=", "") + ".json")
            if os.path.exists(dst):
                print("[skip] %s" % key, flush=True); continue
            jobs.append((a.variant, cov, a.temporal, seed, key, dst))

    print("%d jobs, %d workers" % (len(jobs), a.workers), flush=True)
    t0 = time.time(); running = []
    while jobs or running:
        while jobs and len(running) < a.workers:
            variant, cov, temporal, seed, key, dst = jobs.pop(0)
            src = WORKER.format(repo=REPO, variant=variant, covars=repr(cov),
                                temporal=temporal, seed=seed, key=key, dst=dst)
            log = open(dst[:-5] + ".log", "w")
            p = subprocess.Popen([sys.executable, "-c", src], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, key, log)); print("[start] %s" % key, flush=True)
        time.sleep(5)
        for it in list(running):
            p, key, log = it
            if p.poll() is not None:
                log.close(); running.remove(it)
                print("[done ] %s rc=%d [%.1f min]" % (key, p.returncode, (time.time()-t0)/60), flush=True)
    print("finished in %.1f min" % ((time.time()-t0)/60))


if __name__ == "__main__":
    main()
