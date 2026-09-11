"""Regenerate every manuscript number that derives from the final model's predictions.

Six of the paper's eleven stale items are not separate experiments -- they are
different views of one set of pooled test predictions. Re-running the model once
per figure would be wasteful and, worse, would give each figure a slightly
different model because of GPU non-determinism. So all of them are computed here
from the single saved prediction set.

Covers:
  confusion matrix          row-normalised, 5x5
  per-class F1              Wake N1 N2 N3 REM
  operating characteristics ROC and precision-recall, and precision at the
                            sensitivity the referee asked about
  per-event-type AUC        needs event_labels.npz if present
  AHI correlation           per-patient burden vs clinical AHI (Spearman)
  staging by severity       accuracy within AASM AHI bands

The AHI mapping is positional (workbook row i -> SN(i+1)), verified exact for all
90 SN-labelled rows; the 10 AN-labelled rows occupy positions 11-20 and are
therefore SN11-SN20. This is the same mapping the published supplementary
notebook uses.

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/foundation/regen_derived.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             average_precision_score, roc_curve)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
XLSX = os.path.join(REPO, "data", "full100", "subject_description.xlsx")
STAGES = ["W", "N1", "N2", "N3", "R"]
SEV_CUTS = [5, 15, 30]
SEV_NAMES = ["Normal", "Mild", "Moderate", "Severe"]


def main():
    P = np.load(os.path.join(FINAL, "predictions_seed42.npz"))
    yt, yp = P["y_true"], P["y_pred"]
    at, asc = P["apnea_true"], P["apnea_score"]
    out = {}

    # ---- staging -----------------------------------------------------------
    out["staging"] = dict(
        acc=float(accuracy_score(yt, yp)),
        mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
        kappa=float(cohen_kappa_score(yt, yp)),
        n_epochs=int(len(yt)))
    cm = confusion_matrix(yt, yp, labels=range(5), normalize="true")
    out["confusion_row_normalised"] = [[round(float(v), 4) for v in row] for row in cm]
    pcf = f1_score(yt, yp, average=None, labels=range(5), zero_division=0)
    out["per_class_f1"] = {STAGES[i]: round(float(v), 4) for i, v in enumerate(pcf)}
    out["per_class_recall"] = {STAGES[i]: round(float(v), 4) for i, v in enumerate(cm.diagonal())}
    out["support"] = {STAGES[i]: int(v) for i, v in enumerate(np.bincount(yt, minlength=5))}

    # ---- respiratory operating characteristics -----------------------------
    out["respiratory"] = dict(auc=float(roc_auc_score(at, asc)),
                              ap=float(average_precision_score(at, asc)),
                              prevalence=float(at.mean()))
    prec, rec, _ = precision_recall_curve(at, asc)
    # The referee's specific concern: precision at high sensitivity. Reported at
    # several operating points rather than one, because a single point invites
    # the suspicion it was chosen after the fact.
    ops = {}
    for target in (0.90, 0.80, 0.70, 0.60):
        i = int(np.argmin(np.abs(rec - target)))
        ops["sens_%.2f" % target] = dict(sensitivity=round(float(rec[i]), 4),
                                         precision=round(float(prec[i]), 4))
    out["respiratory"]["operating_points"] = ops
    fpr, tpr, _ = roc_curve(at, asc)
    np.savez_compressed(os.path.join(FINAL, "curves_seed42.npz"),
                        fpr=fpr, tpr=tpr, precision=prec, recall=rec)

    # ---- per-patient burden vs clinical AHI --------------------------------
    ps = json.load(open(os.path.join(FINAL, "per_subject_seed42.json")))
    df = pd.read_excel(XLSX)
    ahi_col = pd.to_numeric(df["AHI_1_B"], errors="coerce")
    ahi = {i + 1: float(v) for i, v in enumerate(ahi_col) if not pd.isna(v)}

    burden, clin, sev_acc = [], [], {}
    for key, rec_ in ps.items():
        sid = int(key[2:])
        if sid not in ahi:
            continue
        b = float(np.mean(rec_["apnea"]))
        burden.append(b); clin.append(ahi[sid])
        band = SEV_NAMES[int(np.digitize(ahi[sid], SEV_CUTS))]
        sev_acc.setdefault(band, []).append(float(rec_["acc"]))
    rho, pval = spearmanr(burden, clin)
    out["ahi"] = dict(rho=float(rho), p=float(pval), n=len(burden))
    out["staging_by_severity"] = {
        k: dict(acc=round(float(np.mean(v)), 4), sd=round(float(np.std(v, ddof=1)), 4), n=len(v))
        for k, v in sorted(sev_acc.items(), key=lambda kv: SEV_NAMES.index(kv[0]))}

    # ---- per-event-type AUC ------------------------------------------------
    # Aligned PER SUBJECT, not by pooling. The earlier version concatenated all
    # 97 event-label subjects in sorted key order and compared the length to the
    # pooled prediction vector, which is in FOLD order over 99 subjects. The
    # lengths never matched, so every event type was silently skipped and the
    # table was reported as un-regenerable. per_subject_seed42.json carries each
    # patient's own apnea scores, which sidesteps the ordering entirely.
    ev = os.path.join(REPO, "MMNet_research", "MMNet_Submission", "all_codes",
                      "results", "npz", "event_labels.npz")
    if os.path.exists(ev):
        E = np.load(ev)
        cols = ["any", "hypopnea", "obstructive", "central", "mixed", "rera"]
        common = [s for s in ps if s in E.files and len(ps[s]["apnea"]) == len(E[s])]
        score = np.concatenate([np.asarray(ps[s]["apnea"], float) for s in common])
        lab = np.concatenate([np.asarray(E[s], int) for s in common])
        per_type = {}
        for i, name in enumerate(cols[1:], start=1):
            pos = lab[:, i] == 1
            # negatives are epochs with NO scored event, so each type is judged
            # against clean breathing rather than against the other event types
            neg = lab[:, 0] == 0
            m = pos | neg
            if pos.sum() < 20 or neg.sum() < 20:
                continue
            per_type[name] = dict(
                n_positive=int(pos.sum()),
                auc=round(float(roc_auc_score(pos[m].astype(int), score[m])), 4))
        out["per_event_type_auc"] = per_type
        out["per_event_type_note"] = (
            "%d subjects aligned; negatives are epochs with no scored event" % len(common))
    else:
        out["per_event_type_auc"] = "event_labels.npz not found -- not regenerated"

    dst = os.path.join(FINAL, "derived_seed42.json")
    json.dump(out, open(dst, "w"), indent=1)

    print("FINAL MODEL, derived from pooled predictions (seed 42, %d epochs)\n" % len(yt))
    print("staging   acc %.4f  mF1 %.4f  kappa %.4f"
          % (out["staging"]["acc"], out["staging"]["mf1"], out["staging"]["kappa"]))
    print("per-class F1:", out["per_class_f1"])
    print("\nrespiratory  AUC %.4f  AP %.4f  prevalence %.3f"
          % (out["respiratory"]["auc"], out["respiratory"]["ap"], out["respiratory"]["prevalence"]))
    for k, v in ops.items():
        print("   %-10s sensitivity %.3f -> precision %.3f" % (k, v["sensitivity"], v["precision"]))
    print("\nAHI  rho %.3f  p %.4g  n %d" % (out["ahi"]["rho"], out["ahi"]["p"], out["ahi"]["n"]))
    print("staging by severity:", {k: v["acc"] for k, v in out["staging_by_severity"].items()})
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
