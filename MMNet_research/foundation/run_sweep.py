"""Screening sweep over the axes this project never searched.

Stage 1 screens many configurations at a single seed; promising ones are then
confirmed at three seeds, because a one-seed difference is not evidence -- fold
SD here is ~0.017 and seed-to-seed spread is comparable.

Axes, and why each is here:
  temporal  'none' beat 'lstm' by +0.0148 in 13_architecture_ablation, so the
            published recurrent decoder is not the best known option.
  arm       'A' is the 188 engineered features; 'pretrained' is frozen CBraMod,
            which reached statistical parity with A. Their combination with
            temporal='none' has never been tried.
  hidden / drop / lr / wd  never tuned anywhere in this repository.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "foundation")
SH = os.path.join(OUT, "sweep_shards")

WORKER = r'''
import json, os, sys, time
import numpy as np
sys.path.insert(0, r"{repo}\MMNet_research\model")
sys.path.insert(0, r"{repo}\MMNet_research\foundation")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import mmnet_core as C
import sweep

cfg = {cfg}
key, dst = "{key}", r"{dst}"
t0 = time.time()
with sweep.config(C, cfg["arm"], hidden=cfg["hidden"], drop=cfg["drop"],
                  lr=cfg["lr"], wd=cfg["wd"],
                  cardio=cfg.get("cardio"),
                  cardio_mode=cfg.get("cardio_mode", "concat")) as h:
    r = C.run_10fold(fusion="concat", temporal=cfg["temporal"], seed=cfg["seed"])
rec = dict(cfg)
rec.update({{"eeg_dim": h.dim, "card_dim": h.n_card,
            "acc":   [f["acc"]   for f in r["per_fold"]],
            "mf1":   [f["mf1"]   for f in r["per_fold"]],
            "kappa": [f["kappa"] for f in r["per_fold"]],
            "auc":   [f["auc"]   for f in r["per_fold"]],
            "ap":    [f["ap"]    for f in r["per_fold"]],
            "minutes": (time.time() - t0) / 60}})
json.dump({{key: rec}}, open(dst, "w"), indent=1)
print("DONE %s acc %.4f auc %.4f [%.1f min]"
      % (key, np.mean(rec["acc"]), np.mean(rec["auc"]), rec["minutes"]), flush=True)
'''


def key_of(c):
    base = "%s|t=%s|h%d|dr%g|lr%g|wd%g|s%d" % (
        c["arm"], c["temporal"], c["hidden"], c["drop"], c["lr"], c["wd"], c["seed"])
    if c.get("cardio"):
        base += "|c=%s-%s" % (c["cardio"], c.get("cardio_mode", "concat"))
    return base


def grid(seeds):
    for arm, temporal, hidden, drop, lr, wd, seed in itertools.product(
            ["A", "pretrained"], ["none"], [128, 256], [0.3, 0.15],
            [1e-3, 3e-4], [1e-4], seeds):
        yield dict(arm=arm, temporal=temporal, hidden=hidden, drop=drop,
                   lr=lr, wd=wd, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--configs", default=None,
                    help="JSON file with an explicit list of configs (for the confirm stage)")
    a = ap.parse_args()
    os.makedirs(SH, exist_ok=True)

    cfgs = json.load(open(a.configs)) if a.configs else list(grid(a.seeds))
    jobs = []
    for c in cfgs:
        k = key_of(c)
        dst = os.path.join(SH, k.replace("|", "__").replace("=", "") + ".json")
        if os.path.exists(dst):
            print("[skip] %s" % k, flush=True)
            continue
        jobs.append((c, k, dst))

    print("%d configs, %d workers" % (len(jobs), a.workers), flush=True)
    t0 = time.time()
    running = []
    while jobs or running:
        while jobs and len(running) < a.workers:
            c, k, dst = jobs.pop(0)
            src = WORKER.format(repo=REPO, cfg=repr(c), key=k, dst=dst)
            log = open(dst[:-5] + ".log", "w")
            p = subprocess.Popen([sys.executable, "-c", src], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, k, log))
            print("[start] %s" % k, flush=True)
        time.sleep(5)
        for it in list(running):
            p, k, log = it
            if p.poll() is not None:
                log.close(); running.remove(it)
                print("[done ] %s  rc=%d  [%.1f min elapsed]"
                      % (k, p.returncode, (time.time() - t0) / 60), flush=True)
    print("\nsweep finished in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
