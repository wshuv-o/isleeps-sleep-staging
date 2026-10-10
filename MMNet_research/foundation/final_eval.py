"""Evaluate a final MM-Net configuration from run_seed_ensemble.py: staging, respiratory events, AHI.

Per fold (the paper's ten folds), from the saved per-patient probabilities:
    staging      accuracy, macro-F1, kappa after the train-fold HMM (single seeds and 3-seed average)
    respiratory  AUC and AP of the per-epoch event probability over the fold's test epochs,
                 averaged over folds (as run_10fold reports them)
    AHI          Spearman rho between each test patient's mean event probability (burden) and the
                 clinical AHI (subject_description.xlsx, AHI_1_B; workbook row i -> SN(i+1)),
                 the same measure as regen_derived.py
References: the eight-channel MM-Net with frozen LaBraM (pflow_suite_B42.json / pflow_suite_B.json,
staging and respiratory per fold; per_subject_8ch_seed42.json for AHI) and the published
seven-channel model (per_subject_seed42.json, AHI).

    python final_eval.py --config base-nocw-cross-ftla-p8 [--name final_architecture]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, f1_score, roc_auc_score

import mmnet_core as C

RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs")
ENS = os.path.join(RUNS, "lesion", "seed_ensemble")
FINAL = os.path.join(RUNS, "final")
XLSX = os.path.join(REPO, "data", "full100", "subject_description.xlsx")
SEEDS = (42, 1, 7)


def clinical_ahi():
    col = pd.to_numeric(pd.read_excel(XLSX)["AHI_1_B"], errors="coerce")
    return {i + 1: float(v) for i, v in enumerate(col) if not pd.isna(v)}


def evaluate(cfg):
    out = {k: [] for k in ("acc", "mf1", "kappa", "auc", "ap", "acc_single", "auc_single")}
    burden, burden42 = {}, {}
    for fi in range(10):
        d = np.load(os.path.join(ENS, "%s_fold%d.npz" % (cfg, fi)))
        A, pi, te = d["A_log"], d["pi_log"], [int(s) for s in d["te"]]
        yt = np.concatenate([C.DATA[s][2] for s in te])
        at = np.concatenate([C.DATA[s][3] for s in te])
        sp = {s: np.mean([d["SN%d|%d" % (s, sd)] for sd in SEEDS], 0) for s in te}
        ap = {s: np.mean([d["A:SN%d|%d" % (s, sd)] for sd in SEEDS], 0) for s in te}
        yp = np.concatenate([C.hmm(A, pi, np.log(sp[s] + C.EPS)) for s in te])
        asc = np.concatenate([ap[s] for s in te])
        out["acc"].append(accuracy_score(yt, yp))
        out["mf1"].append(f1_score(yt, yp, average="macro", zero_division=0))
        out["kappa"].append(cohen_kappa_score(yt, yp))
        out["auc"].append(roc_auc_score(at, asc))
        out["ap"].append(average_precision_score(at, asc))
        accs, aucs = [], []
        for sd in SEEDS:
            ys = np.concatenate([C.hmm(A, pi, np.log(d["SN%d|%d" % (s, sd)] + C.EPS)) for s in te])
            accs.append(accuracy_score(yt, ys))
            aucs.append(roc_auc_score(at, np.concatenate([d["A:SN%d|%d" % (s, sd)] for s in te])))
        out["acc_single"].append(np.mean(accs))
        out["auc_single"].append(np.mean(aucs))
        for s in te:
            burden[s] = float(np.mean(ap[s]))
            burden42[s] = float(np.mean(d["A:SN%d|42" % s]))
    return {k: np.array(v) for k, v in out.items()}, burden, burden42


def rho(burden, ahi):
    ks = [s for s in burden if s in ahi]
    r, p = spearmanr([burden[s] for s in ks], [ahi[s] for s in ks])
    return float(r), float(p), len(ks)


def reference_8ch():
    b42 = json.load(open(os.path.join(FINAL, "pflow_suite_B42.json")))["full|42"]
    b = json.load(open(os.path.join(FINAL, "pflow_suite_B.json")))
    rows = [b42, b["full|1"], b["full|7"]]
    return {m: np.mean([r[m] for r in rows], 0) for m in ("acc", "mf1", "kappa", "auc", "ap")}


def burden_from_per_subject(fname):
    ps = json.load(open(os.path.join(FINAL, fname)))
    return {int(k[2:]): float(np.mean(v["apnea"])) for k, v in ps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="final_architecture")
    a = ap.parse_args()
    r, burden, burden42 = evaluate(a.config)
    ref = reference_8ch()
    ahi = clinical_ahi()
    lines = ["Final MM-Net configuration: %s (seeds 42, 1, 7; 10 test folds)" % a.config, ""]
    lines.append("%-34s %8s %8s %8s %8s %8s" % ("", "acc", "mf1", "kappa", "resp AUC", "resp AP"))
    lines.append("%-34s %8.4f %8.4f %8.4f %8.4f %8.4f" % ((a.config + ", 3-seed average",)
                 + tuple(r[m].mean() for m in ("acc", "mf1", "kappa", "auc", "ap"))))
    lines.append("%-34s %8.4f %8s %8s %8.4f" % (a.config + ", single model (seed mean)",
                 r["acc_single"].mean(), "", "", r["auc_single"].mean()))
    lines.append("%-34s %8.4f %8.4f %8.4f %8.4f %8.4f" % (("8-channel MM-Net, frozen LaBraM (seed mean)",)
                 + tuple(ref[m].mean() for m in ("acc", "mf1", "kappa", "auc", "ap"))))
    lines.append("")
    for lab, mine in (("single model", "acc_single"), ("3-seed", "acc")):
        d = r[mine] - ref["acc"]
        lines.append("staging %-12s vs 8-channel frozen: %+.4f, %d/10 folds up, Wilcoxon p=%.3f"
                     % (lab, d.mean(), (d > 0).sum(), wilcoxon(d).pvalue))
    for lab, mine in (("single model", "auc_single"), ("3-seed", "auc")):
        d = r[mine] - ref["auc"]
        lines.append("resp AUC %-11s vs 8-channel frozen: %+.4f, %d/10 folds up, Wilcoxon p=%.3f"
                     % (lab, d.mean(), (d > 0).sum(), wilcoxon(d).pvalue))
    lines.append("")
    for lab, b in (("final, 3-seed burden", burden), ("final, seed 42 burden", burden42),
                   ("8-channel frozen, seed 42", burden_from_per_subject("per_subject_8ch_seed42.json")),
                   ("published 7-channel, seed 42", burden_from_per_subject("per_subject_seed42.json"))):
        rr, p, n = rho(b, ahi)
        lines.append("AHI association, %-30s Spearman rho %.3f (p=%.2g, n=%d)" % (lab, rr, p, n))
    text = "\n".join(lines)
    out = os.path.join(RUNS, "lesion", a.name + ".txt")
    open(out, "w").write(text + "\n")
    json.dump({k: v.tolist() for k, v in r.items()}, open(out[:-4] + ".json", "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
