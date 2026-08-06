"""
analysis.py -- downstream analyses that reuse saved MM-Net predictions (no retraining):
  * per-event-type respiratory AUC (brief 4.3): hypopnea / obstructive / central
  * clinical validation vs AHI (brief 4.4): predicted per-patient burden vs clinical AHI
  * staging by SDB severity (brief 4.5)
  * paired Wilcoxon signed-rank tests over the 10 folds (brief 3.6)
All read results/revision/runs/<name>.json (per-subject apnea scores, per-fold metrics),
event_labels.npz and ahi.json. Every number traces to a real run.
"""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmnet_repro import REV, RUNS

TYPES = ["hypopnea", "obstructive", "central", "mixed", "rera"]


def load_run(name):
    return json.load(open(os.path.join(RUNS, f"{name}.json")))


def event_labels():
    z = np.load(os.path.join(REV, "event_labels.npz"))
    return {k: z[k] for k in z.files}   # SNk -> [n,6] any,hyp,obs,cen,mix,rera


def per_event_type_auc(run_name="headline_concat"):
    """pooled AUC of the model's apnea score against each event type (positives = that type,
    negatives = epochs with no respiratory event of any type)."""
    r = load_run(run_name); ev = event_labels()
    scores, lab = [], []
    for sk, d in r["per_subject"].items():
        if sk not in ev:
            continue
        s = np.asarray(d["apnea_scores"]); L = ev[sk][:len(s)]
        scores.append(s); lab.append(L)
    scores = np.concatenate(scores); lab = np.concatenate(lab)
    any_ev = lab[:, 0] == 1
    out = {}
    for ti, t in enumerate(["hypopnea", "obstructive", "central"], start=1):
        pos = lab[:, ti] == 1
        neg = ~any_ev                                   # clean negatives (no event at all)
        mask = pos | neg
        if pos.sum() >= 5 and neg.sum() >= 5:
            out[t] = dict(auc=float(roc_auc_score(pos[mask].astype(int), scores[mask])),
                          n_pos=int(pos.sum()))
    return out


def ahi_validation(run_name="headline_concat"):
    """predicted per-patient event burden (mean apnea score) vs clinical AHI (Spearman)."""
    r = load_run(run_name); ahi = json.load(open(os.path.join(REV, "ahi.json")))
    burden, clin, sev = [], [], []
    for sk, d in r["per_subject"].items():
        if sk not in ahi:
            continue
        burden.append(float(np.mean(d["apnea_scores"]))); clin.append(ahi[sk]["ahi"]); sev.append(ahi[sk]["severity"])
    rho, p = spearmanr(burden, clin)
    return dict(rho=float(rho), p=float(p), n=len(burden),
                burden=burden, ahi=clin, severity=sev)


def staging_by_severity(run_name="headline_concat"):
    """mean per-subject staging accuracy grouped by AHI severity class."""
    r = load_run(run_name); ahi = json.load(open(os.path.join(REV, "ahi.json")))
    groups = {}
    for sk, d in r["per_subject"].items():
        if sk not in ahi:
            continue
        groups.setdefault(ahi[sk]["severity"], []).append(d["acc"])
    return {s: dict(acc_mean=float(np.mean(v)), acc_std=float(np.std(v)), n=len(v))
            for s, v in groups.items()}


def wilcoxon_auc(run_a, run_b):
    """paired Wilcoxon signed-rank on the 10 per-fold apnea AUCs of two runs."""
    a = np.array([f["apnea_auc"] for f in load_run(run_a)["per_fold"]])
    b = np.array([f["apnea_auc"] for f in load_run(run_b)["per_fold"]])
    st, p = wilcoxon(a, b)
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), stat=float(st), p=float(p))


def wilcoxon_metric(run_a, run_b, metric="acc"):
    a = np.array([f[metric] for f in load_run(run_a)["per_fold"]])
    b = np.array([f[metric] for f in load_run(run_b)["per_fold"]])
    st, p = wilcoxon(a, b)
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), stat=float(st), p=float(p))


if __name__ == "__main__":
    print("per-event-type AUC:", json.dumps(per_event_type_auc(), indent=2))
    a = ahi_validation(); print(f"AHI Spearman rho={a['rho']:.3f} p={a['p']:.4g} (n={a['n']})")
    print("staging by severity:", json.dumps(staging_by_severity(), indent=2))
    print("Wilcoxon apnea AUC concat vs neural_only:", wilcoxon_auc("headline_concat", "neural_only"))
