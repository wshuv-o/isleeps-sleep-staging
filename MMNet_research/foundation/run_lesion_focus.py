"""Healthy-hemisphere emphasis (lesion_focus.py), compared with the published model.

Same ten folds, same three seeds, same training as run_lesion_aware.py; the reference
is that script's 'base' configuration, read from lesion_runs.json. Both modes are
fixed in advance and each is run once per seed; nothing is selected on test folds.

Outputs, under results/revision/runs/lesion/:
    lesion_focus_runs.json      per-fold metrics and per-patient accuracy, mode x seed
    lesion_focus_summary.txt    means, paired Wilcoxon on the ten fold-means vs base,
                                and the lateralised-patient comparison vs base

    python run_lesion_focus.py [--modes healthy-copy damaged-drop] [--seeds 42 1 7]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon

import run_lesion_aware as RL
import lesion_focus as LF

C, sweep, arms, FINAL = RL.C, RL.sweep, RL.arms, RL.FINAL
PATH = os.path.join(RL.OUT, "lesion_focus_runs.json")
BASE = os.path.join(RL.OUT, "lesion_runs.json")


def run_one(mode, seed):
    undo = LF.install(arms, C, mode)
    try:
        with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                          lr=FINAL["lr"], wd=FINAL["wd"],
                          cardio=FINAL["cardio"], cardio_mode=FINAL["cardio_mode"]) as h:
            r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"], seed=seed)
            dim = h.dim
    finally:
        undo()
    rec = {m: [float(f[m]) for f in r["per_fold"]] for m in RL.METRICS}
    rec["eeg_dim"] = int(dim)
    rec["per_subject"] = {k: v["acc"] for k, v in r["per_subject"].items()}
    return rec


def save(res):
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, PATH)


def summarise(res, seeds):
    if not os.path.exists(BASE):
        return "no base results yet (%s)" % BASE
    base = json.load(open(BASE))
    if not all("base|%d" % s in base for s in seeds):
        return "base is not complete for seeds %s" % seeds
    table = LF.lesion.lesion_table()
    lat = {"SN%d" % s for s, (side, _) in table.items() if side in ("L", "R")}
    fm = lambda d, key, m: np.mean([d["%s|%d" % (key, s)][m] for s in seeds], axis=0)
    lines = ["%-14s %8s %8s %8s %8s %8s" % (("config",) + RL.METRICS)]
    lines.append("%-14s " % "base" + " ".join("%8.4f" % np.nanmean(fm(base, "base", m)) for m in RL.METRICS))
    for mode in LF.MODES:
        if not all("%s|%d" % (mode, s) in res for s in seeds):
            continue
        lines.append("%-14s " % mode + " ".join("%8.4f" % np.nanmean(fm(res, mode, m)) for m in RL.METRICS))
    lines.append("\npaired Wilcoxon on ten fold-means (seed-averaged), vs base")
    for mode in LF.MODES:
        if not all("%s|%d" % (mode, s) in res for s in seeds):
            continue
        parts = []
        for m in RL.METRICS:
            d = fm(res, mode, m) - fm(base, "base", m)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            parts.append("%s %+.4f (p=%.3f, %d/10 up)" % (m, np.nanmean(d), p, int((d > 0).sum())))
        lines.append("%-14s " % mode + "; ".join(parts))
        keys = [k for k in base["base|%d" % seeds[0]]["per_subject"] if k in lat]
        a = np.array([np.mean([res["%s|%d" % (mode, s)]["per_subject"][k] for s in seeds]) for k in keys])
        b = np.array([np.mean([base["base|%d" % s]["per_subject"][k] for s in seeds]) for k in keys])
        lines.append("  lateralised patients: base %.4f -> %s %.4f (%+.4f), Wilcoxon p=%.4f, %d/%d up"
                     % (b.mean(), mode, a.mean(), (a - b).mean(), wilcoxon(a - b).pvalue,
                        int((a > b).sum()), len(keys)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(LF.MODES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    a = ap.parse_args()
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for mode in a.modes:
        for seed in a.seeds:
            key = "%s|%d" % (mode, seed)
            if key in res:
                print("[skip] %s" % key, flush=True)
                continue
            t0 = time.time()
            res[key] = run_one(mode, seed)
            res[key]["minutes"] = (time.time() - t0) / 60
            save(res)
            print("[ok] %-12s seed %d  acc %.4f  kappa %.4f  auc %.4f  [%.1f min]"
                  % (mode, seed, np.mean(res[key]["acc"]), np.mean(res[key]["kappa"]),
                     np.nanmean(res[key]["auc"]), res[key]["minutes"]), flush=True)
    text = summarise(res, [42, 1, 7])
    open(os.path.join(RL.OUT, "lesion_focus_summary.txt"), "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
