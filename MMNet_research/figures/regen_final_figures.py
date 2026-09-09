"""Regenerate the prediction-derived figures for the final model.

Four of the manuscript's figures are not separate experiments -- they are
different views of one pooled prediction set. Re-running the model once per
figure would give each a slightly different model because of GPU
non-determinism, so all of them are drawn here from the single saved set that
`regen_derived.py` also produced its numbers from. The figures and the numbers
in the text therefore cannot disagree.

Regenerates, under figures/ with the filenames the .tex already references:

    fig_confusion.pdf   row-normalised confusion matrix
    fig_roc_pr.pdf      ROC and precision-recall, pooled
    fig_ahi.pdf         per-patient burden vs clinical AHI, by severity band

NOT regenerated here: fig_mm_perclass.pdf, which compares per-class F1 for the
neural-only against the full model. The ablation grid stored acc/mF1/kappa/AUC/AP
per fold but not per-class F1, so the neural-only half of that comparison does
not exist and would have to be re-run. Its \\stale marker stays until it is.

  KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/figures/regen_final_figures.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FINAL = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
XLSX = os.path.join(REPO, "data", "full100", "subject_description.xlsx")

STAGES = ["W", "N1", "N2", "N3", "R"]
SEV_CUTS = [5, 15, 30]
SEV_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
SEV_COLOURS = ["#4a72b0", "#5a8f66", "#c98a52", "#b8607a"]

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8})


def fig_confusion(P, out):
    yt, yp = P["y_true"], P["y_pred"]
    cm = confusion_matrix(yt, yp, labels=range(5), normalize="true")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "%.2f" % cm[i, j], ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(5)); ax.set_xticklabels(STAGES)
    ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
    ax.set_xlabel("predicted"); ax.set_ylabel("scored by technician")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return cm


def fig_roc_pr(P, C, out):
    at, asc = P["apnea_true"], P["apnea_score"]
    prev = float(at.mean())
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(at, asc); ap = average_precision_score(at, asc)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    a = axes[0]
    a.plot(C["fpr"], C["tpr"], color="#b8607a", lw=1.6)
    a.plot([0, 1], [0, 1], "--", color="0.6", lw=0.9)
    a.set_xlabel("false positive rate"); a.set_ylabel("true positive rate")
    a.set_title("(a) ROC   AUC = %.3f" % auc, fontsize=9)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    b = axes[1]
    b.plot(C["recall"], C["precision"], color="#4a72b0", lw=1.6)
    b.axhline(prev, ls="--", color="0.6", lw=0.9)
    b.text(0.98, prev + 0.015, "no skill (%.3f)" % prev, ha="right", fontsize=7, color="0.4")
    b.set_xlabel("recall (sensitivity)"); b.set_ylabel("precision")
    b.set_title("(b) Precision--recall   AP = %.3f" % ap, fontsize=9)
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return auc, ap, prev


def fig_ahi(out):
    ps = json.load(open(os.path.join(FINAL, "per_subject_seed42.json")))
    df = pd.read_excel(XLSX)
    ahi_col = pd.to_numeric(df["AHI_1_B"], errors="coerce")
    # positional mapping, workbook row i -> SN(i+1); verified exact for the 90
    # SN-labelled rows, and the 10 AN-labelled rows occupy positions 11-20
    ahi = {i + 1: float(v) for i, v in enumerate(ahi_col) if not pd.isna(v)}
    xs, ys, cs = [], [], []
    for key, rec in ps.items():
        sid = int(key[2:])
        if sid not in ahi:
            continue
        xs.append(ahi[sid]); ys.append(float(np.mean(rec["apnea"])))
        cs.append(int(np.digitize(ahi[sid], SEV_CUTS)))
    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)
    from scipy.stats import spearmanr
    rho, p = spearmanr(xs, ys)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for k, name in enumerate(SEV_NAMES):
        m = cs == k
        if m.any():
            ax.scatter(xs[m], ys[m], s=22, alpha=0.85, edgecolor="white",
                       linewidth=0.4, color=SEV_COLOURS[k], label=name)
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlabel("clinical AHI (events/h)")
    ax.set_ylabel("predicted event burden")
    ax.set_title(r"Spearman $\rho$ = %.3f, $n$ = %d" % (rho, len(xs)), fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return rho, p, len(xs)


def fig_ablation(out):
    """Grouped bars: what each modality's removal costs each head.

    Drawn from the 27-shard grid, not the seed-42 predictions, because the
    ablation is a retraining experiment rather than a view of one model.
    """
    import glob
    rows = {}
    for p in glob.glob(os.path.join(FINAL, "ablation_shards", "*.json")):
        for _, r in json.load(open(p)).items():
            rows.setdefault(r["condition"], []).append(r)
    fm = json.load(open(os.path.join(FINAL, "final_model.json")))
    base_a = float(np.mean(np.concatenate([fm[k]["acc"] for k in sorted(fm)])))
    base_u = float(np.nanmean(np.concatenate([fm[k]["auc"] for k in sorted(fm)])))
    order = ["-EEG", "-EOG", "-EMG", "-SpO2", "-pulse/HRV", "-ECG",
             "-airflow", "-effort", "-all cardio"]
    labels = ["EEG", "EOG", "EMG", "SpO$_2$", "pulse/HRV", "ECG",
              "airflow", "effort", "all cardio"]
    da, du = [], []
    for c in order:
        rs = sorted(rows[c], key=lambda x: x["seed"])
        da.append(base_a - float(np.mean(np.concatenate([r["acc"] for r in rs]))))
        du.append(base_u - float(np.nanmean(np.concatenate([r["auc"] for r in rs]))))
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - w / 2, da, w, color="#4a72b0", label="staging accuracy")
    ax.bar(x + w / 2, du, w, color="#b8607a", label="respiratory AUC")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("drop when removed")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return dict(zip(labels, zip(np.round(da, 4), np.round(du, 4))))


def fig_learning_curve(out):
    """Both heads against training-set size, mean +- SD over the ten folds."""
    import glob
    pts = []
    for f in sorted(glob.glob(os.path.join(FINAL, "lc_shards", "*.json"))):
        v = list(json.load(open(f)).values())
        pts.append((float(np.mean([x["n_train"] for x in v])),
                    np.array([x["acc"] for x in v]),
                    np.array([x["auc"] for x in v])))
    pts.sort(key=lambda t: t[0])
    n = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for vals, colour, lab in ((1, "#4a72b0", "staging accuracy"),
                              (2, "#b8607a", "respiratory AUC")):
        m = np.array([np.nanmean(p[vals]) for p in pts])
        s = np.array([np.nanstd(p[vals], ddof=1) for p in pts])
        ax.plot(n, m, "o-", color=colour, lw=1.6, ms=4, label=lab)
        ax.fill_between(n, m - s, m + s, color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("training patients")
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return [(int(p[0]), round(float(np.mean(p[1])), 4), round(float(np.nanmean(p[2])), 4))
            for p in pts]


def fig_perclass(out):
    """Per-class F1, neural-only against neural+cardio.

    Needs neural_only_pcf.json, which exists only because the ablation grid did
    not retain per-class F1 and the condition had to be re-run. Returns None if
    that run has not finished, so this script stays usable without it.
    """
    src = os.path.join(FINAL, "neural_only_pcf.json")
    if not os.path.exists(src):
        return None
    no = json.load(open(src))
    if not no:
        return None
    npcf = np.concatenate([no[k]["pcf"] for k in sorted(no)], axis=0).mean(0)
    fpcf = np.array([json.load(open(os.path.join(FINAL, "derived_seed42.json")))
                     ["per_class_f1"][s] for s in STAGES])
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(x - w / 2, npcf, w, color="#8a9099", label="neural only")
    ax.bar(x + w / 2, fpcf, w, color="#4a72b0", label="neural + cardiorespiratory")
    ax.set_xticks(x); ax.set_xticklabels(STAGES)
    ax.set_ylabel("per-class F1"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return {STAGES[i]: (round(float(npcf[i]), 4), round(float(fpcf[i]), 4)) for i in range(5)}


def main():
    P = np.load(os.path.join(FINAL, "predictions_seed42.npz"))
    C = np.load(os.path.join(FINAL, "curves_seed42.npz"))

    cm = fig_confusion(P, os.path.join(HERE, "fig_confusion.pdf"))
    print("fig_confusion.pdf   diagonal:",
          {STAGES[i]: round(float(cm[i, i]), 3) for i in range(5)})

    auc, ap, prev = fig_roc_pr(P, C, os.path.join(HERE, "fig_roc_pr.pdf"))
    print("fig_roc_pr.pdf      AUC %.4f  AP %.4f  prevalence %.4f" % (auc, ap, prev))

    rho, p, n = fig_ahi(os.path.join(HERE, "fig_ahi.pdf"))
    print("fig_ahi.pdf         rho %.3f  p %.3g  n %d" % (rho, p, n))

    d = fig_ablation(os.path.join(HERE, "fig_ablation.pdf"))
    print("fig_ablation.pdf    (acc drop, auc drop):")
    for k, v in d.items():
        print("      %-12s %+.4f  %+.4f" % (k, v[0], v[1]))

    lc = fig_learning_curve(os.path.join(HERE, "fig_learning_curve.pdf"))
    print("fig_learning_curve.pdf  n_train / acc / auc:", lc)

    pc = fig_perclass(os.path.join(HERE, "fig_mm_perclass.pdf"))
    if pc is None:
        print("\nSKIPPED fig_mm_perclass.pdf -- run foundation/run_neural_only_pcf.py first")
    else:
        print("fig_mm_perclass.pdf  (neural-only, full):", pc)


if __name__ == "__main__":
    main()
