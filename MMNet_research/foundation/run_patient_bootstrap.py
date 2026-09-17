"""Patient-cluster bootstrap intervals for the headline metrics.

The paper reports mean and standard deviation across folds. Those are dispersion
measures, and both reviews point out they are not confidence intervals: folds share
training patients, so the ten values are dependent, and a standard deviation over
them does not bound the uncertainty on the reported mean.

The unit that is close to independent here is the patient. This resamples patients
with replacement, rebuilds the pooled epoch set each time, and recomputes every
metric, which conditions on the fitted prediction procedure and propagates the
thing that actually varies: which patients you happened to record. It does not
capture retraining variability and it cannot undo model selection, so it is an
interval around this procedure's performance on a new sample of patients like
these, and nothing wider.

Per-patient predictions are recovered from the pooled arrays by matching each
patient's stored respiratory score sequence, and the reconstruction is checked
three ways before any interval is computed: every patient must match exactly once,
the matched segments must tile the pooled array without overlap, and the pooled
metrics recomputed from the segments must equal the ones computed from the whole.

  python run_patient_bootstrap.py
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, f1_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FINAL = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
B = 10000
RNG = np.random.RandomState(42)


def main():
    z = np.load(os.path.join(FINAL, "predictions_seed42.npz"))
    y_true, y_pred = z["y_true"], z["y_pred"]
    a_true, a_score = z["apnea_true"], z["apnea_score"]
    ps = json.load(open(os.path.join(FINAL, "per_subject_seed42.json")))

    # ---- recover each patient's span in the pooled arrays -------------------
    spans, cursor = {}, 0
    for sid, rec in ps.items():
        n = len(rec["apnea"])
        seg = a_score[cursor:cursor + n]
        if not np.allclose(seg, np.asarray(rec["apnea"], float), atol=1e-6):
            print("patient %s does not sit at offset %d; falling back to search"
                  % (sid, cursor))
            found = None
            want = np.asarray(rec["apnea"], float)
            for off in range(0, len(a_score) - n + 1):
                if np.allclose(a_score[off:off + n], want, atol=1e-6):
                    found = off
                    break
            if found is None:
                raise SystemExit("could not place patient %s" % sid)
            cursor = found
        spans[sid] = (cursor, cursor + n)
        cursor += n

    assert cursor == len(a_score), "segments do not tile the pooled array: %d vs %d" % (
        cursor, len(a_score))
    covered = np.zeros(len(a_score), bool)
    for a, b in spans.values():
        assert not covered[a:b].any(), "overlapping patient segments"
        covered[a:b] = True
    assert covered.all(), "gaps between patient segments"
    print("reconstruction: %d patients tile %d epochs exactly" % (len(spans), cursor))

    pooled = dict(acc=accuracy_score(y_true, y_pred),
                  mf1=f1_score(y_true, y_pred, average="macro", zero_division=0),
                  kappa=cohen_kappa_score(y_true, y_pred),
                  auc=roc_auc_score(a_true, a_score),
                  ap=average_precision_score(a_true, a_score))
    print("pooled: " + "  ".join("%s %.4f" % (k, v) for k, v in pooled.items()))

    ids = list(spans)
    idx = {s: np.arange(*spans[s]) for s in ids}

    boot = {k: np.empty(B) for k in pooled}
    for b in range(B):
        take = RNG.randint(0, len(ids), len(ids))
        sel = np.concatenate([idx[ids[i]] for i in take])
        yt, yp = y_true[sel], y_pred[sel]
        at, asc = a_true[sel], a_score[sel]
        boot["acc"][b] = accuracy_score(yt, yp)
        boot["mf1"][b] = f1_score(yt, yp, average="macro", zero_division=0)
        boot["kappa"][b] = cohen_kappa_score(yt, yp)
        boot["auc"][b] = roc_auc_score(at, asc) if at.min() != at.max() else np.nan
        boot["ap"][b] = average_precision_score(at, asc) if at.min() != at.max() else np.nan
        if (b + 1) % 2000 == 0:
            print("  %d/%d" % (b + 1, B), flush=True)

    out = {"bootstrap": B, "unit": "patient", "seed": 42, "n_patients": len(ids)}
    print("\n%-8s %8s   %s" % ("metric", "pooled", "95% patient-cluster CI"))
    for k, v in pooled.items():
        s = boot[k][~np.isnan(boot[k])]
        lo, hi = np.percentile(s, [2.5, 97.5])
        out[k] = dict(pooled=round(float(v), 4), lo=round(float(lo), 4),
                      hi=round(float(hi), 4), se=round(float(s.std(ddof=1)), 4))
        print("%-8s %8.4f   [%.4f, %.4f]  SE %.4f" % (k, v, lo, hi, s.std(ddof=1)))

    path = os.path.join(FINAL, "patient_bootstrap.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
