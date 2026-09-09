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

    print("\nNOT regenerated: fig_mm_perclass.pdf -- the ablation grid did not "
          "store per-class F1, so the neural-only comparison does not exist.")


if __name__ == "__main__":
    main()
