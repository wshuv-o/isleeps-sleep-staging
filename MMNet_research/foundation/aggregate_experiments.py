"""Aggregate every foundation-arm and cardio-representation run into one table.

122 experiment shards accumulated under runs/foundation/sweep_shards/ during the
revision and none of them reached the manuscript. This collects them by
configuration, pools the seeds, and tests each against the configuration the
paper reports, so the ones worth reporting can be chosen on evidence rather
than recollection.

Grouping key is (eeg arm, cardio representation, cardio mode); seeds are pooled
into 30 fold-values wherever three seeds ran. Wilcoxon is paired on fold-values
against the final model, and Holm-corrected within each family of comparisons.

  KMP_DUPLICATE_LIB_OK=TRUE python aggregate_experiments.py
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
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs")
SH = os.path.join(RUNS, "foundation", "sweep_shards")
OUT = os.path.join(RUNS, "foundation", "experiment_table.json")


def parse(name):
    """A shard filename encodes arm, hyperparameters, seed and cardio variant."""
    base = os.path.basename(name)[:-5]
    m = re.match(r"(.+?)__t(\w+)__h(\d+)__dr([\d.]+)__lr([\d.e-]+)__wd([\d.e-]+)__s(\d+)"
                 r"(?:__c(.+))?$", base)
    if not m:
        return None
    arm, temporal, hidden, drop, lr, wd, seed, cardio = m.groups()
    if cardio:
        cardio, _, mode = cardio.partition("-")
    else:
        cardio, mode = "features14", ""
    return dict(arm=arm, temporal=temporal, hidden=int(hidden), drop=float(drop),
                lr=float(lr), wd=float(wd), seed=int(seed),
                cardio=cardio, mode=mode)


def main():
    rows = {}
    bad = 0
    for f in sorted(glob.glob(os.path.join(SH, "*.json"))):
        meta = parse(f)
        if meta is None:
            bad += 1
            continue
        try:
            d = json.load(open(f))
        except ValueError:
            bad += 1
            continue
        rec = d if isinstance(d, dict) and "acc" in d else next(iter(d.values()), None)
        if not isinstance(rec, dict) or "acc" not in rec:
            bad += 1
            continue
        key = (meta["arm"], meta["cardio"], meta["mode"], meta["hidden"],
               meta["temporal"])
        rows.setdefault(key, {"meta": meta, "seeds": {}})
        rows[key]["seeds"][meta["seed"]] = rec

    fm = json.load(open(os.path.join(RUNS, "final", "final_model.json")))
    f_acc = np.concatenate([fm[k]["acc"] for k in sorted(fm)])
    f_auc = np.concatenate([fm[k]["auc"] for k in sorted(fm)])

    table = []
    for key, v in rows.items():
        seeds = sorted(v["seeds"])
        acc = np.concatenate([v["seeds"][s]["acc"] for s in seeds])
        auc = np.concatenate([v["seeds"][s].get("auc", [np.nan]) for s in seeds])
        table.append(dict(
            arm=key[0], cardio=key[1], mode=key[2], hidden=key[3], temporal=key[4],
            n_seeds=len(seeds), n_folds=len(acc),
            acc=float(np.mean(acc)), acc_sd=float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            auc=float(np.nanmean(auc)),
            auc_sd=float(np.nanstd(auc, ddof=1)) if len(auc) > 1 else 0.0,
            _acc=acc, _auc=auc))

    # paired tests, but only where the fold count matches the reference
    for metric, ref, field in (("acc", f_acc, "p_acc"), ("auc", f_auc, "p_auc")):
        idx = [i for i, r in enumerate(table) if len(r["_" + metric]) == len(ref)]
        ps = []
        for i in idx:
            a = table[i]["_" + metric]
            ps.append(1.0 if np.allclose(np.nan_to_num(a), np.nan_to_num(ref))
                      else wilcoxon(a, ref, nan_policy="omit").pvalue)
        if ps:
            holm = multipletests(ps, method="holm")[1]
            for i, p, h in zip(idx, ps, holm):
                table[i][field] = float(p)
                table[i][field + "_holm"] = float(h)

    table.sort(key=lambda r: -r["auc"])
    for r in table:
        r.pop("_acc"); r.pop("_auc")
    json.dump(table, open(OUT, "w"), indent=1)

    print("%d configurations from %d shards (%d unparsable)\n"
          % (len(table), sum(len(v["seeds"]) for v in rows.values()), bad))
    print("%-16s %-14s %-8s %2s %8s %8s %9s %9s"
          % ("arm", "cardio", "mode", "s", "acc", "resp AUC", "p(acc)", "p(AUC)"))
    print("-" * 92)
    for r in table:
        print("%-16s %-14s %-8s %2d %8.4f %8.4f %9s %9s"
              % (r["arm"], r["cardio"], r["mode"] or "-", r["n_seeds"],
                 r["acc"], r["auc"],
                 ("%.4f" % r["p_acc_holm"]) if "p_acc_holm" in r else "-",
                 ("%.4f" % r["p_auc_holm"]) if "p_auc_holm" in r else "-"))
    print("\nreference (final model): acc %.4f  resp AUC %.4f  (%d fold-values)"
          % (f_acc.mean(), np.nanmean(f_auc), len(f_acc)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
