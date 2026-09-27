"""Items 6N, 6O, 10a: cohort bookkeeping on N = 99.

  * checks that the onset-epoch rule reproduces the cached training labels
    (data/mm_features, 96 of 99 on disk)
  * Table 1 stage shares and event prevalence on all 99
  * Fig. 1 quantities: per-patient burden, median, max, patients above 5 %
  * the true number of scored respiratory events (Flow Events rows), by type,
    against the 8,400 "maximal runs" of Table 8

  python cohort_stats.py        (CPU, ~1 min; reads the 100 xlsx files)
"""
import os
from collections import Counter

import numpy as np

import _common as C


def main():
    P, _ = C.probs()
    stage_counts = np.zeros(5, int)
    n_total = 0
    burden, check, per_patient, lens = {}, {}, {}, {}
    type_counts = Counter()
    n_events = 0
    for sid in C.SUBJECTS:
        y = C.stages(sid)
        n = len(y)
        lens[sid] = [n, int(len(P[sid]))]
        stage_counts += np.bincount(y, minlength=5)[:5]
        n_total += n
        lab = C.labels(sid, n, "onset")
        burden[sid] = float(100 * lab.mean())
        ev = C.events(sid)
        in_range = [e for e in ev if 0 <= e[0] // 30 < n]
        n_events += len(in_range)
        type_counts.update(e[2] for e in in_range)
        per_patient[sid] = dict(events=len(in_range), per_hour=len(in_range) / (n / 120.0),
                                positive_epochs=int(lab.sum()))
        f = os.path.join(C.DATA, "mm_features", sid + ".npz")
        if os.path.exists(f):
            cached = np.load(f)["apnea"].astype(int)
            check[sid] = int((cached[:n] != lab[:len(cached)]).sum()) if len(cached) == n else "length mismatch"

    b = np.array([burden[s] for s in C.SUBJECTS])
    allpos = sum(per_patient[s]["positive_epochs"] for s in C.SUBJECTS)
    mismatched = {k: v for k, v in check.items() if v != 0}
    res = {
        "n_patients": len(C.SUBJECTS),
        "n_epochs": int(n_total),
        "hours": n_total / 120.0,
        "stage_share_pct": dict(zip(["W", "N1", "N2", "N3", "R"],
                                    np.round(100 * stage_counts / n_total, 2).tolist())),
        "stage_counts": dict(zip(["W", "N1", "N2", "N3", "R"], stage_counts.tolist())),
        "label_rule": "onset epoch only (preprocessing/build_multimodal.py:read_apnea)",
        "label_check_vs_mm_features": {"n_checked": len(check),
                                       "n_identical": sum(1 for v in check.values() if v == 0),
                                       "mismatches": mismatched},
        "prevalence_pooled": allpos / n_total,
        "burden_pct": {"median": float(np.median(b)), "max": float(b.max()), "min": float(b.min()),
                       "above_5pct": int((b > 5).sum()), "at_or_below_5pct": int((b <= 5).sum()),
                       "zero": int((b == 0).sum())},
        "per_patient_burden_pct": burden,
        "scored_events": {"total": n_events, "by_type": dict(type_counts),
                          "per_hour_mean_over_patients": float(np.mean([per_patient[s]["per_hour"] for s in C.SUBJECTS])),
                          "per_hour_pooled": n_events / (n_total / 120.0),
                          "patients_with_any_event": int(sum(per_patient[s]["events"] > 0 for s in C.SUBJECTS))},
        "per_patient_events": per_patient,
        "length_check_stage_vs_probs": {k: v for k, v in lens.items() if v[0] != v[1]},
        "_note": "Table 8's '8,400 scored events' are maximal runs of positive epochs (run_event_level.py), not annotation rows.",
    }
    C.save("cohort_stats.json", res)
    print({k: res[k] for k in ("n_epochs", "stage_share_pct", "prevalence_pooled", "burden_pct")})
    print(res["scored_events"], res["label_check_vs_mm_features"]["n_identical"], len(mismatched))


if __name__ == "__main__":
    main()
