"""Item 10: event-level scoring against the annotated events themselves.

Table 8 (run_event_level.py) defines an "event" as a maximal run of positive
epochs, which merges neighbouring annotations: 8,400 runs against 15,207 annotated
Flow-Events rows. Here every annotation row is an event.

  * detected    : any epoch the annotation overlaps is flagged at the threshold
  * false alarm : a maximal run of flagged epochs that overlaps no annotation
  * rates per hour of recording and per hour of sleep (stage != W)
  * the annotated rate per hour of sleep is checked against the clinical AHI

Respiratory-effort-related arousals (RERA) and 'Body event' rows are reported
separately and excluded from the apnea/hypopnea count, as the AHI excludes them.

  python event_true.py        (CPU; seed-42 stored probabilities, all 99 patients)
"""
import numpy as np
from scipy.stats import spearmanr

import _common as C

AH = {"Hypopnea", "Obstructive Apnea", "Central Apnea", "Mixed Apnea",
      "Obstructive Hypopnea", "Central Hypopnea"}


def runs(x):
    x = np.concatenate([[0], x.astype(int), [0]])
    d = np.diff(x)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    P, _ = C.probs()
    A = C.ahi()
    st = {s: C.stages(s) for s in C.SUBJECTS}
    ev = {}
    for s in C.SUBJECTS:
        n = len(st[s])
        rows = []
        for a, b, name in C.events(s):
            k0, k1 = int(a // 30), int(b // 30)
            if k0 >= n or k1 < 0:
                continue
            rows.append((max(k0, 0), min(k1, n - 1), name))
        ev[s] = rows
    counts = {}
    for s in C.SUBJECTS:
        for _, _, nm in ev[s]:
            counts[nm] = counts.get(nm, 0) + 1
    hours = sum(len(st[s]) for s in C.SUBJECTS) / 120.0
    sleep_hours = sum((st[s] != 0).sum() for s in C.SUBJECTS) / 120.0
    n_ah = sum(1 for s in C.SUBJECTS for e in ev[s] if e[2] in AH)

    sweep = []
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        det = tot = fa = 0
        pred_rate, true_rate, ahis = [], [], []
        for s in C.SUBJECTS:
            f = P[s] >= th
            cover = np.zeros(len(f), bool)
            for k0, k1, nm in ev[s]:
                if nm not in AH:
                    continue
                tot += 1
                det += int(f[k0:k1 + 1].any())
                cover[k0:k1 + 1] = True
            fr = runs(f)
            fa += sum(1 for a, b in fr if not cover[a:b].any())
            sh = (st[s] != 0).sum() / 120.0
            pred_rate.append(len(fr) / max(sh, 1e-6))
            true_rate.append(sum(1 for e in ev[s] if e[2] in AH) / max(sh, 1e-6))
        sweep.append(dict(th=th, annotated_event_sensitivity=det / tot, false_alarms_per_hour=fa / hours,
                          predicted_runs_per_sleep_hour_mean=float(np.mean(pred_rate)),
                          annotated_per_sleep_hour_mean=float(np.mean(true_rate)),
                          rho_pred_rate_vs_ahi=float(spearmanr(pred_rate, [A[s] for s in C.SUBJECTS]).statistic)))
    true_rate = {s: sum(1 for e in ev[s] if e[2] in AH) / ((st[s] != 0).sum() / 120.0) for s in C.SUBJECTS}
    r = spearmanr([true_rate[s] for s in C.SUBJECTS], [A[s] for s in C.SUBJECTS])
    res = dict(n_patients=len(C.SUBJECTS), recording_hours=hours, sleep_hours=sleep_hours,
               annotation_rows_by_type=counts, apnea_hypopnea_events=n_ah,
               annotated_ah_per_sleep_hour_pooled=n_ah / sleep_hours,
               annotated_ah_per_sleep_hour_patient_mean=float(np.mean(list(true_rate.values()))),
               clinical_ahi_patient_mean=float(np.mean([A[s] for s in C.SUBJECTS])),
               rho_annotated_rate_vs_clinical_ahi=float(r.statistic), p=float(r.pvalue),
               threshold_sweep=sweep,
               note="detection = any overlapped epoch flagged; false alarm = flagged run touching no apnea/hypopnea annotation")
    C.save("event_true.json", res)
    for k in ("apnea_hypopnea_events", "annotated_ah_per_sleep_hour_pooled", "annotated_ah_per_sleep_hour_patient_mean",
              "clinical_ahi_patient_mean", "rho_annotated_rate_vs_clinical_ahi", "sleep_hours"):
        print(k, res[k])
    for x in sweep:
        print({k: round(v, 3) for k, v in x.items()})


if __name__ == "__main__":
    main()
