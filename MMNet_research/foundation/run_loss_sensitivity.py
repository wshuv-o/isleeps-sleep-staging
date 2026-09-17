"""Sensitivity of both heads to the two loss constants the paper fixes by fiat.

The paper trains with L = L_stg + lambda*L_apn at lambda = 1 and weights the
respiratory term by pi = n_-/n_+, roughly 5.3 on this corpus, and neither value is
justified by an experiment. The review is right that equal coefficients do not
establish equal influence: the two terms have different curvatures and scales, so
lambda = 1 is a choice about arithmetic and not about task balance.

This sweeps each constant with everything else held at the published configuration
and reports both heads for every value, because the interesting failure is not a
lower number on one head but a trade: if the two curves peak in different places,
lambda = 1 is a compromise and should be described as one.

train_fold hardcodes both constants, so both are patched by recompiling the
function's source with the literals substituted, the technique sweep.py uses, with
asserted substitution counts so a future edit to mmnet_core fails loudly instead of
silently sweeping nothing.

  KMP_DUPLICATE_LIB_OK=TRUE python run_loss_sensitivity.py
"""
import inspect
import json
import os
import re
import sys
import textwrap
import time

import numpy as np

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
SEEDS = [42, 1, 7]

# lambda = 1 and the prevalence-matched pi are the published settings and are read
# from final_model.json instead of being re-run.
LAMBDAS = [0.25, 0.5, 2.0, 4.0]
PIS = [1.0, 2.5, 8.0, 11.0]

LOSS_LINE = "loss = ls if task == \"stage\" else la if task == \"apnea\" else ls + la"
PI_EXPR = "pos_weight=torch.tensor([ac[0]/max(1,ac[1])]"
LR_EXPR = "lr=1e-3"
WD_EXPR = "weight_decay=1e-4"

# Captured now, while train_fold is still the function mmnet_core defined. Inside a
# sweep.config block it has been replaced by one compiled from a string, and a
# string-compiled function has no source file for inspect to read -- which is what
# broke the first attempt at this sweep. Patching this text means the lr and weight
# decay have to be substituted here too, since sweep's own patch is bypassed.
ORIG_SRC = textwrap.dedent(inspect.getsource(C.train_fold))


def patched(lam=None, pi=None, lr=None, wd=None):
    """train_fold with the loss constants, lr and weight decay substituted."""
    src = ORIG_SRC
    if lr is not None:
        src, n = re.subn(re.escape(LR_EXPR), "lr=%g" % lr, src)
        if n != 1:
            raise RuntimeError("lr: replaced %d, expected 1" % n)
    if wd is not None:
        src, n = re.subn(re.escape(WD_EXPR), "weight_decay=%g" % wd, src)
        if n != 1:
            raise RuntimeError("weight_decay: replaced %d, expected 1" % n)
    if lam is not None:
        new = LOSS_LINE.replace("ls + la", "ls + %g*la" % lam)
        src, n = re.subn(re.escape(LOSS_LINE), new, src)
        if n != 1:
            raise RuntimeError("loss line: replaced %d, expected 1" % n)
    if pi is not None:
        src, n = re.subn(re.escape(PI_EXPR),
                         "pos_weight=torch.tensor([%g]" % pi, src)
        if n != 1:
            raise RuntimeError("pos_weight: replaced %d, expected 1" % n)
    ns = dict(C.__dict__)
    exec(compile(src, "<train_fold:lam=%s,pi=%s>" % (lam, pi), "exec"), ns)
    return ns["train_fold"]


def run(key, lam, pi, res, path):
    if key in res:
        print("[skip] %s" % key, flush=True)
        return
    t1 = time.time()
    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode=FINAL["cardio_mode"]):
        keep = C.train_fold
        # sweep.config has already baked lr and wd into its own copy; this replaces
        # that copy, so both are substituted here as well and the configuration is
        # the published one in every respect but the constant under test.
        C.train_fold = patched(lam, pi, FINAL["lr"], FINAL["wd"])
        try:
            r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"],
                             seed=int(key.split("|")[-1]))
        finally:
            C.train_fold = keep
    res[key] = {k: [float(f[k]) for f in r["per_fold"]]
                for k in ("acc", "kappa", "auc", "ap")}
    res[key]["minutes"] = round((time.time() - t1) / 60, 2)
    json.dump(res, open(path, "w"), indent=1)
    print("%-16s acc %.4f  kappa %.4f  auc %.4f  ap %.4f  [%.1f min]"
          % (key, np.mean(res[key]["acc"]), np.mean(res[key]["kappa"]),
             np.mean(res[key]["auc"]), np.mean(res[key]["ap"]),
             res[key]["minutes"]), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "loss_sensitivity.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()

    # the patch must actually bite; check before spending hours on it
    probe = inspect.getsource(C.train_fold)
    assert probe.count(LOSS_LINE.strip()) == 1, "loss line not found; mmnet_core changed"
    assert probe.count(PI_EXPR) == 1, "pos_weight expression not found; mmnet_core changed"
    print("both literals located in train_fold\n")

    for lam in LAMBDAS:
        for seed in SEEDS:
            run("lam=%g|%d" % (lam, seed), lam, None, res, path)
    for pi in PIS:
        for seed in SEEDS:
            run("pi=%g|%d" % (pi, seed), None, pi, res, path)

    # ---- curves, with the published point read from final_model.json --------
    fm = json.load(open(os.path.join(OUT, "final_model.json")))
    base = {k: np.mean(np.asarray([fm["final|%d" % s][k] for s in SEEDS], float), axis=0)
            for k in ("acc", "kappa", "auc", "ap")}

    def curve(prefix, values, published):
        rows = []
        for v in values:
            f = {k: np.mean(np.asarray([res["%s=%g|%d" % (prefix, v, s)][k]
                                        for s in SEEDS], float), axis=0)
                 for k in ("acc", "auc")}
            rows.append((v, f["acc"].mean(), f["auc"].mean()))
        rows.append((published, base["acc"].mean(), base["auc"].mean()))
        rows.sort()
        return rows

    summary = {}
    for name, prefix, values, published in (("lambda", "lam", LAMBDAS, 1.0),
                                            ("pi", "pi", PIS, 5.35)):
        rows = curve(prefix, values, published)
        print("\n%s sweep (published value marked *)" % name)
        print("  %-10s %10s %10s" % (name, "staging acc", "resp AUC"))
        for v, a, u in rows:
            mark = " *" if abs(v - published) < 1e-9 else "  "
            print("  %-10g %10.4f %10.4f%s" % (v, a, u, mark))
        summary[name] = [dict(value=v, acc=round(a, 4), auc=round(u, 4)) for v, a, u in rows]
        best_acc = max(rows, key=lambda r: r[1])
        best_auc = max(rows, key=lambda r: r[2])
        summary[name + "_best"] = dict(acc_at=best_acc[0], auc_at=best_auc[0])
        print("  best staging at %g, best respiratory at %g" % (best_acc[0], best_auc[0]))

    res["_summary"] = summary
    res["_note"] = ("lambda=1 and pi=n_-/n_+ (about 5.35) are the published settings and are "
                    "taken from final_model.json, not re-run")
    json.dump(res, open(path, "w"), indent=1)
    print("\nsaved -> %s  [%.1f min]" % (path, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
