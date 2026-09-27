"""Merge per-machine eight-channel suite results into pflow_suite.json and summarise.

  python merge_pflow_suite.py pflow_suite_machineB.json [more.json ...]

Keys are "condition|seed"; a key present in two files keeps the first one seen
(pflow_suite.json first). Prints mean over the available seeds per condition.
"""
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "revision", "runs", "final")
MAIN = os.path.join(OUT, "pflow_suite.json")

res = json.load(open(MAIN)) if os.path.exists(MAIN) else {}
for f in sys.argv[1:]:
    f = f if os.path.isabs(f) else os.path.join(OUT, f)
    for k, v in json.load(open(f)).items():
        res.setdefault(k, v)
json.dump(res, open(MAIN, "w"), indent=1)

conds = sorted({v["cond"] for v in res.values()}, key=lambda c: (c != "full", c))
print("%-16s %-8s %7s %7s %7s %7s %7s" % ("condition", "seeds", "acc", "mf1", "kappa", "auc", "ap"))
for c in conds:
    rows = [v for v in res.values() if v["cond"] == c]
    seeds = sorted(v["seed"] for v in rows)
    print("%-16s %-8s " % (c, ",".join(map(str, seeds))) + " ".join(
        "%7.4f" % np.nanmean(np.concatenate([r[m] for r in rows]))
        for m in ("acc", "mf1", "kappa", "auc", "ap")))
print("merged -> %s (%d runs)" % (MAIN, len(res)))
