"""Is the ISRUC change between the submitted and the final model real?

The external-validation table reports ISRUC night-one accuracy falling 0.664 ->
0.649 and respiratory AUC rising 0.721 -> 0.780 when the submitted model is
replaced by the final one. Both were read off pooled point estimates, which on
eight recordings says very little: per-recording accuracy on this corpus spans
0.37 to 0.79, so a pooled shift of 0.015 is far inside the spread.

The two models were evaluated on the *same eight recordings*, so the comparison
can be paired, which removes between-recording variance -- the dominant term --
and asks the only question that matters: did the same recording get better or
worse under the new model?

Submitted-model per-recording values are recovered from the saved outputs of
notebook 11 (`11_external_validation_isruc.ipynb`, cell 7), which is the only
place they survive; that run wrote pooled figures to its JSON and nothing else.
Final-model values come from `runs/final/external_isruc.json`.

Usage:  python isruc_paired_check.py [--out <json>]
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FINAL = os.path.join(REPO, "MMNet_research", "results", "revision", "runs",
                     "final", "external_isruc.json")

# Submitted model, ISRUC session 1, from notebook 11 cell 7 saved output.
SUBMITTED = {
    "S1_1": dict(n=933, acc=0.458, kappa=0.323, auc=0.681, prev=0.085),
    "S2_1": dict(n=851, acc=0.682, kappa=0.549, auc=0.848, prev=0.224),
    "S3_1": dict(n=871, acc=0.727, kappa=0.617, auc=0.807, prev=0.071),
    "S4_1": dict(n=932, acc=0.682, kappa=0.546, auc=0.825, prev=0.163),
    "S5_1": dict(n=814, acc=0.748, kappa=0.654, auc=0.515, prev=0.014),
    "S6_1": dict(n=965, acc=0.675, kappa=0.547, auc=0.649, prev=0.064),
    "S7_1": dict(n=941, acc=0.629, kappa=0.532, auc=0.525, prev=0.134),
    "S8_1": dict(n=815, acc=0.739, kappa=0.635, auc=0.682, prev=0.045),
}


def paired(label, old, new, higher_is_better=True):
    old, new = np.asarray(old, float), np.asarray(new, float)
    d = new - old
    t, pt = stats.ttest_rel(new, old)
    try:
        _, pw = stats.wilcoxon(new, old)
    except ValueError:
        pw = float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), se)
    better = int((d > 0).sum()) if higher_is_better else int((d < 0).sum())
    print("%-22s submitted %.4f -> final %.4f   (pooled shift %+.4f)"
          % (label, old.mean(), new.mean(), new.mean() - old.mean()))
    print("%-22s paired diff %+.4f, 95%% CI [%+.4f, %+.4f]"
          % ("", d.mean(), lo, hi))
    print("%-22s improved on %d/%d recordings | t p=%.4f | wilcoxon p=%.4f"
          % ("", better, len(d), pt, pw))
    verdict = ("CHANGE IS REAL" if pt < 0.05
               else "WITHIN NOISE - not distinguishable")
    print("%-22s -> %s" % ("", verdict))
    print()
    return dict(submitted=float(old.mean()), final=float(new.mean()),
                paired_diff=float(d.mean()), ci95=[float(lo), float(hi)],
                improved=better, n=len(d), p_ttest=float(pt),
                p_wilcoxon=float(pw), significant=bool(pt < 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO, "MMNet_research", "results", "revision", "runs", "final",
        "isruc_paired_check.json"))
    a = ap.parse_args()

    final = json.load(open(FINAL, encoding="utf-8"))["per_recording"]
    tags = sorted(SUBMITTED)
    missing = [t for t in tags if t not in final]
    if missing:
        raise SystemExit("final run is missing recordings: %s" % missing)

    print("ISRUC session 1, %d recordings, the same ones under both models\n" % len(tags))
    out = {"n_recordings": len(tags), "recordings": tags}
    out["accuracy"] = paired("accuracy",
                             [SUBMITTED[t]["acc"] for t in tags],
                             [final[t]["acc"] for t in tags])
    out["kappa"] = paired("kappa",
                          [SUBMITTED[t]["kappa"] for t in tags],
                          [final[t]["kappa"] for t in tags])
    auc_tags = [t for t in tags if final[t].get("auc") is not None]
    out["respiratory_auc"] = paired("respiratory AUC",
                                    [SUBMITTED[t]["auc"] for t in auc_tags],
                                    [final[t]["auc"] for t in auc_tags])

    # spread of the corpus itself, for reporting alongside any point estimate
    acc = np.array([final[t]["acc"] for t in tags])
    se = acc.std(ddof=1) / np.sqrt(len(acc))
    lo, hi = stats.t.interval(0.95, len(acc) - 1, acc.mean(), se)
    out["final_accuracy_ci95"] = [float(lo), float(hi)]
    print("Final-model accuracy across these recordings: %.4f, 95%% CI [%.4f, %.4f]"
          % (acc.mean(), lo, hi))
    print("Half-width +-%.4f. Any pooled shift smaller than that is not a finding."
          % ((hi - lo) / 2))

    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
