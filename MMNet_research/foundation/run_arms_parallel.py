"""Run the frozen arms (A, B, D) x seeds concurrently.

One 10-fold of the frozen arms uses ~1.35 GB of VRAM and about 29% of the GPU:
the models are small (0.77-0.95 M parameters) and the wall-clock is dominated by
Python overhead, the per-subject validation loops and the CPU-side HMM Viterbi
decode. Running the jobs one after another therefore leaves most of both the GPU
and the CPU idle.

The (arm, seed) jobs are completely independent, so they are run as separate
processes. Each writes its OWN shard under runs/foundation/shards/ -- concurrent
writers must never share a JSON file -- and merge_shards() collects them into the
arms_frozen.json that notebook 15 reads. The notebook still recomputes anything
missing, so deleting the cache reproduces everything from the notebook alone.

Batch size is deliberately NOT increased: it is part of the published training
configuration and changing it would make these arms incomparable with arm A.
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
SHARDS = os.path.join(OUT, "shards")

ARMS = {"A_features": "A", "B_frozen": "pretrained", "D_random": "random"}

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, r"{repo}\MMNet_research\model")
sys.path.insert(0, r"{repo}\MMNet_research\foundation")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import mmnet_core as C
import arms

name, variant, seed, dst = "{name}", "{variant}", {seed}, r"{dst}"
t0 = time.time()
with arms.arm(C, variant) as a:
    r = C.run_10fold(fusion="concat", seed=seed)
rec = {{"arm": name, "seed": seed, "eeg_dim": a.dim,
       "acc":   [f["acc"]   for f in r["per_fold"]],
       "mf1":   [f["mf1"]   for f in r["per_fold"]],
       "kappa": [f["kappa"] for f in r["per_fold"]],
       "auc":   [f["auc"]   for f in r["per_fold"]],
       "ap":    [f["ap"]    for f in r["per_fold"]],
       "minutes": (time.time() - t0) / 60}}
json.dump({{"%s|%d" % (name, seed): rec}}, open(dst, "w"), indent=1)
print("DONE %s seed %d  acc %.4f  auc %.4f  [%.1f min]"
      % (name, seed, np.mean(rec["acc"]), np.mean(rec["auc"]), rec["minutes"]), flush=True)
'''


def merge_shards():
    merged = {}
    for f in sorted(os.listdir(SHARDS)) if os.path.isdir(SHARDS) else []:
        if f.endswith(".json"):
            merged.update(json.load(open(os.path.join(SHARDS, f))))
    dst = os.path.join(OUT, "arms_frozen.json")
    json.dump(merged, open(dst, "w"), indent=1)
    return dst, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    os.makedirs(SHARDS, exist_ok=True)

    jobs = []
    for name in a.arms:
        for seed in a.seeds:
            dst = os.path.join(SHARDS, "%s_%d.json" % (name, seed))
            if os.path.exists(dst):
                print("[skip] %s seed %d" % (name, seed), flush=True)
                continue
            jobs.append((name, ARMS[name], seed, dst))

    print("%d jobs, %d workers" % (len(jobs), a.workers), flush=True)
    t0 = time.time()
    running, logs = [], {}
    while jobs or running:
        while jobs and len(running) < a.workers:
            name, variant, seed, dst = jobs.pop(0)
            src = WORKER.format(repo=REPO, name=name, variant=variant, seed=seed, dst=dst)
            log = open(os.path.join(SHARDS, "%s_%d.log" % (name, seed)), "w")
            p = subprocess.Popen([sys.executable, "-c", src], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, name, seed, log))
            print("[start] %s seed %d (pid %d)" % (name, seed, p.pid), flush=True)
        time.sleep(5)
        for item in list(running):
            p, name, seed, log = item
            if p.poll() is not None:
                log.close(); running.remove(item)
                tag = "ok" if p.returncode == 0 else "FAILED rc=%d" % p.returncode
                print("[done ] %s seed %d  %s  [%.1f min elapsed]"
                      % (name, seed, tag, (time.time() - t0) / 60), flush=True)

    dst, merged = merge_shards()
    print("\nmerged %d runs -> %s  [%.1f min total]" % (len(merged), dst, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
