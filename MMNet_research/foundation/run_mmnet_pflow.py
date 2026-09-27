"""MM-Net with nasal pressure as an eighth cardiorespiratory channel.

iSLEEPS records two airflow sensors. The published model reads the oronasal thermal
sensor ("Flow Th") and not nasal pressure ("Pressure Flow"), the signal hypopneas are
scored from under AASM rules. Huttunen et al.'s five-signal configuration reads both,
so a fair comparison with it needs the same input on this side. This runs the final
configuration unchanged except that the raw cardiorespiratory tensor gains the nasal
pressure channel (data/pressure_flow, built by preprocessing/build_pressure_cache.py),
z-scored per patient like the other seven; recordings without the channel (SN1) get
zeros, the convention used for every missing channel.

The training loop is the host-memory variant from run_missing_channel.py (identical
batches, windows kept in system RAM) because the widened tensor does not fit the card.

  KMP_DUPLICATE_LIB_OK=TRUE python run_mmnet_pflow.py [--seeds 42 1 7]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import run_missing_channel as R

C, sweep, cardio_cnn = R.C, R.sweep, R.cardio_cnn
REPO = R.REPO
PFLOW = os.path.join(REPO, "data", "pressure_flow")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "mmnet_pflow.json")
FINAL = R.FINAL
_raw_table = cardio_cnn.raw_cardio_table


def table_with_pressure(Cm, mm_dir):
    out = {}
    for sid, x in _raw_table(Cm, mm_dir).items():          # [n, 7*750], z-scored
        z = np.load(os.path.join(PFLOW, "SN%d.npz" % sid))
        p = z["pflow"].astype(np.float32)
        if bool(z["valid"]):
            p = (p - p.mean()) / (p.std() + 1e-6)
        else:
            p = np.zeros_like(p)
        if len(p) != len(x):
            raise ValueError("SN%d: %d pressure epochs vs %d" % (sid, len(p), len(x)))
        out[sid] = np.concatenate([x, np.nan_to_num(p)], axis=1)   # [n, 8*750]
    return out


def main(seeds):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cardio_cnn.raw_cardio_table = table_with_pressure
    cardio_cnn.N_CH = 8                     # CardioCNN reads N_CH when built and reshaping
    try:
        for seed in seeds:
            key = "pflow|%d" % seed
            if key in res:
                print("[skip]", key); continue
            t0 = time.time()
            with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                              lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                              cardio_mode=FINAL["cardio_mode"]) as cfg:
                orig_tf = C.train_fold
                tf = R.train_fold_host(C, cfg)

                def train_fold_fresh(*a, **kw):
                    torch.cuda.empty_cache()
                    return tf(*a, **kw)
                C.__dict__["train_fold"] = train_fold_fresh
                r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"], seed=seed)
                C.__dict__["train_fold"] = orig_tf
            res[key] = dict(seed=seed, **{m: [f[m] for f in r["per_fold"]]
                                          for m in ("acc", "mf1", "kappa", "auc", "ap")},
                            minutes=(time.time() - t0) / 60)
            json.dump(res, open(OUT, "w"), indent=1)
            print("[ok] pflow seed %-2d acc %.4f mf1 %.4f kappa %.4f auc %.4f ap %.4f  [%.1f min]"
                  % (seed, *(np.nanmean(res[key][m]) for m in ("acc", "mf1", "kappa", "auc", "ap")),
                     res[key]["minutes"]), flush=True)
    finally:
        cardio_cnn.raw_cardio_table, cardio_cnn.N_CH = _raw_table, 7


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    main(ap.parse_args().seeds)
