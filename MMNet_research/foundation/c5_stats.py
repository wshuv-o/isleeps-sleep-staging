"""Paired tests for the two comparison families the manuscript does not report.

Both families hold every hyperparameter fixed and vary one thing, so the
comparison is clean; mixing hidden sizes or temporal settings across rows would
not be. Fold-values are paired across the identical 3 seeds x 10 folds, and
Holm corrects within each family.

  family 1  cardio representation    A+labram, h=256, BiLSTM
  family 2  EEG representation       14 engineered cardio features, h=256, no BiLSTM

  KMP_DUPLICATE_LIB_OK=TRUE python c5_stats.py
"""
import glob
import json
import os
import re

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SH = os.path.join(REPO, "MMNet_research", "results", "revision", "runs",
                  "foundation", "sweep_shards")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs",
                   "foundation", "c5_stats.json")


def load(arm, cardio, mode, hidden, temporal):
    """Pool the seeds for one configuration into fold-values."""
    acc, auc = [], []
    for s in (42, 1, 7):
        tag = "%s__t%s__h%d__dr0.3__lr0.0003__wd0.0001__s%d" % (arm, temporal, hidden, s)
        if cardio != "features14":
            tag += "__c%s-%s" % (cardio, mode)
        p = os.path.join(SH, tag + ".json")
        if not os.path.exists(p):
            return None, None
        d = json.load(open(p))
        rec = d if "acc" in d else next(iter(d.values()))
        acc.append(rec["acc"]); auc.append(rec.get("auc", []))
    return np.concatenate(acc), np.concatenate(auc)


def family(name, base, others, metric, hidden, temporal, cardio_of):
    print("\n=== %s (%s) ===" % (name, metric))
    b_acc, b_auc = load(*base, hidden, temporal)
    if b_acc is None:
        print("  base missing:", base); return []
    ref = b_acc if metric == "acc" else b_auc
    print("  baseline %-24s %.4f" % (base[0] + "/" + base[1], np.nanmean(ref)))
    rows, ps = [], []
    for o in others:
        a_acc, a_auc = load(*o, hidden, temporal)
        if a_acc is None:
            print("  missing:", o); continue
        v = a_acc if metric == "acc" else a_auc
        p = wilcoxon(v, ref, nan_policy="omit").pvalue
        rows.append(dict(config=o[0] + "/" + o[1], value=float(np.nanmean(v)),
                         delta=float(np.nanmean(v) - np.nanmean(ref)), p=float(p)))
        ps.append(p)
    if ps:
        holm = multipletests(ps, method="holm")[1]
        for r, h in zip(rows, holm):
            r["p_holm"] = float(h)
    for r in rows:
        print("  %-26s %.4f  %+.4f   p %.4f  Holm %.4f %s"
              % (r["config"], r["value"], r["delta"], r["p"], r["p_holm"],
                 "sig" if r["p_holm"] < 0.05 else "ns"))
    return rows


out = {}

# --- family 1: what the cardiorespiratory stream is represented as -----------
out["cardio_ladder"] = family(
    "cardio representation", ("A+labram", "features14", ""),
    [("A+labram", "randproj_feat", "concat"),
     ("A+labram", "randproj_raw", "concat"),
     ("A+labram", "raw_cnn", "concat"),
     ("A+labram", "moment_random", "concat"),
     ("A+labram", "moment", "replace"),
     ("A+labram", "moment", "concat")],
    "auc", 256, "lstm", None)

# --- family 2: what the neural stream is represented as ----------------------
out["eeg_arms"] = family(
    "EEG representation", ("A", "features14", ""),
    [("labram", "features14", ""),
     ("pretrained", "features14", ""),
     ("A+labram", "features14", ""),
     ("A+pretrained", "features14", ""),
     ("A+pretrained+labram", "features14", "")],
    "acc", 256, "none", None)

# does a SECOND pretrained encoder add anything to the first?
print("\n=== does a second pretrained encoder add? ===")
a1, _ = load("A+labram", "features14", "", 256, "none")
a2, _ = load("A+pretrained+labram", "features14", "", 256, "none")
if a1 is not None and a2 is not None:
    p = wilcoxon(a2, a1, nan_policy="omit").pvalue
    print("  A+labram            %.4f" % np.mean(a1))
    print("  A+pretrained+labram %.4f  (%+.4f)  p %.4f  -> %s"
          % (np.mean(a2), np.mean(a2) - np.mean(a1), p,
             "adds nothing" if p >= 0.05 else "adds"))
    out["second_encoder"] = dict(one=float(np.mean(a1)), two=float(np.mean(a2)),
                                 delta=float(np.mean(a2) - np.mean(a1)), p=float(p))

json.dump(out, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
