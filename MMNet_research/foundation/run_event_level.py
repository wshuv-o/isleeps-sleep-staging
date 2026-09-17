"""Event-level respiratory scoring for the final model.

The paper reports detection per 30-second epoch and lists event-level scoring as
a limitation, with a threshold sweep and a per-patient event-count correlation.
Those numbers came from a notebook that called train_fold directly, without the
sweep.config wrapper, so they describe mmnet_core's default configuration -- a
773k-parameter feature-only network -- and not the 2.76M model every other number
in the paper describes. Same folds, same seed, different model.

This runs the same analysis on the published configuration. The definitions are
the notebook's, restated here so the rule is in the repository rather than in a
notebook:

  * within a patient, an event is a maximal run of consecutive positive epochs,
    with no gap tolerance: one negative epoch ends a run;
  * a scored event counts as detected when the model flags any epoch inside it,
    so there is no onset tolerance in time, only overlap;
  * a false alarm is a maximal run of flagged epochs overlapping no scored
    positive epoch;
  * the rate denominator is the whole recording, wake included, at 120 epochs
    per hour.

The asymmetry is deliberate and is what the undercount in the paper comes from:
one flagged run spanning several scored events marks each of them detected while
contributing at most one false alarm, and several flagged runs inside one event
count that event once.

  KMP_DUPLICATE_LIB_OK=TRUE python run_event_level.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

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
SEED = 42
EPOCH_H = 120.0                 # 30-second epochs per hour
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def runs_of_ones(x):
    """Maximal runs of 1s in a binary vector, as (start, end_exclusive) pairs."""
    x = np.asarray(x).astype(bool)
    if not x.any():
        return []
    d = np.diff(np.concatenate(([0], x.view(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    subj = {}

    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode=FINAL["cardio_mode"]) as h:
        print("configuration: eeg_dim %d, n_card %d" % (h.dim, h.n_card), flush=True)
        for fi, (tr_all, te) in enumerate(C.FOLDS):
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all)
            rng.shuffle(tr_all)
            nv = max(10, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]
            model = C.train_fold(tr, va, "concat", [], [], seed=SEED,
                                 temporal=FINAL["temporal"])
            n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if fi == 0:
                print("  trainable parameters: %s" % format(n_par, ","), flush=True)
                assert n_par == 2764774, ("this is not the published model: %d parameters"
                                          % n_par)
            for s in te:
                _, apn = C.subj_infer(model, s, [], [])
                subj[s] = (C.DATA[s][3].astype(int), apn)
            print("  fold %d done (%.1f min)" % (fi, (time.time() - t0) / 60), flush=True)

    n_ep = sum(len(v[0]) for v in subj.values())
    n_ev = sum(len(runs_of_ones(v[0])) for v in subj.values())
    hours = n_ep / EPOCH_H
    print("\npatients %d | epochs %d | scored events %d | %.1f h"
          % (len(subj), n_ep, n_ev, hours))

    def evaluate(th):
        det = tot = fa = 0
        ep_tp = ep_pos = ep_flag = 0
        for yt, sc in subj.values():
            pred = (sc >= th).astype(int)
            true_runs = runs_of_ones(yt)
            tot += len(true_runs)
            det += sum(1 for a, b in true_runs if pred[a:b].any())
            for a, b in runs_of_ones(pred):
                if not yt[a:b].any():
                    fa += 1
            ep_tp += int(((pred == 1) & (yt == 1)).sum())
            ep_pos += int((yt == 1).sum())
            ep_flag += int((pred == 1).sum())
        return dict(th=float(th),
                    event_sens=det / tot if tot else float("nan"),
                    fa_per_hour=fa / hours,
                    epoch_sens=ep_tp / ep_pos if ep_pos else float("nan"),
                    epoch_prec=ep_tp / ep_flag if ep_flag else float("nan"),
                    flagged_frac=ep_flag / n_ep)

    rows = [evaluate(t) for t in THRESHOLDS]
    print("\n%-8s %12s %14s %12s %12s %10s"
          % ("thresh", "event sens", "false alarms/h", "epoch sens", "epoch prec", "% flagged"))
    print("-" * 74)
    for r in rows:
        print("%-8.2f %12.3f %14.2f %12.3f %12.3f %10.1f"
              % (r["th"], r["event_sens"], r["fa_per_hour"], r["epoch_sens"],
                 r["epoch_prec"], 100 * r["flagged_frac"]))

    # ---- per-patient event count against the scored count -------------------
    TH = 0.5
    true_idx, pred_idx = [], []
    for yt, sc in subj.values():
        h_ = len(yt) / EPOCH_H
        true_idx.append(len(runs_of_ones(yt)) / h_)
        pred_idx.append(len(runs_of_ones((sc >= TH).astype(int))) / h_)
    true_idx, pred_idx = np.array(true_idx), np.array(pred_idx)
    rho, p = spearmanr(true_idx, pred_idx)
    print("\nper-patient event rate at threshold %.2f" % TH)
    print("  scored    mean %.1f/h   predicted mean %.1f/h" % (true_idx.mean(), pred_idx.mean()))
    print("  Spearman rho %.3f (p = %.4f, n = %d), MAE %.1f/h"
          % (rho, p, len(true_idx), np.abs(true_idx - pred_idx).mean()))

    out = {"config": FINAL, "seed": SEED, "params": 2764774,
           "patients": len(subj), "epochs": n_ep, "scored_events": n_ev,
           "hours": round(hours, 1),
           "threshold_sweep": rows,
           "event_rate": {"rho": float(rho), "p": float(p), "n": len(true_idx),
                          "scored_mean": float(true_idx.mean()),
                          "predicted_mean": float(pred_idx.mean()),
                          "mae": float(np.abs(true_idx - pred_idx).mean())},
           "_note": ("the published configuration, unlike the earlier notebook run which "
                     "used mmnet_core defaults (773,254 parameters, feature-only)"),
           "minutes": round((time.time() - t0) / 60, 2)}
    path = os.path.join(OUT, "event_level_final.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\nsaved -> %s  [%.1f min]" % (path, out["minutes"]))


if __name__ == "__main__":
    main()
