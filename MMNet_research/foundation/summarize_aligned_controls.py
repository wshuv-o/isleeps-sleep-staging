"""Numbers for the manuscript tables from the aligned re-run (rerun_aligned_controls.py).

Every test is the two-sided Wilcoxon signed-rank test on the ten fold means (seeds averaged
within a fold), as in the manuscript; the ablation grid is Holm-corrected across its nine
conditions within each metric. Prints the joint-vs-single-task table, the full modality grid
(both outputs for every row, so nothing is read selectively) and the Huttunen et al. comparison,
and writes runs/final/aligned_controls_summary.json.

    python summarize_aligned_controls.py
"""
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
F = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
L = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
TAG = os.environ.get("MMNET_TAG", "_aligned")   # "" reads the September files


def fold_means(rows, metric):
    return np.nanmean(np.asarray([r[metric] for r in rows], float), axis=0)


def test(a, b):
    d = np.asarray(a) - np.asarray(b)
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    return dict(a=round(float(np.mean(a)), 4), b=round(float(np.mean(b)), 4),
                delta=round(float(np.mean(d)), 4), folds_up=int((d > 0).sum()), p=round(p, 4))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def main():
    out = {}
    joint = json.load(open(os.path.join(F, "final_model%s.json" % TAG)))
    jrows = [v for k, v in joint.items() if k.startswith("final|")]
    J = {m: fold_means(jrows, m) for m in ("acc", "kappa", "auc", "ap")}
    print("Joint frozen-encoder model (%s): acc %.4f  kappa %.4f  AUC %.4f  AP %.4f"
          % ((TAG or "September runs",) + tuple(np.mean(J[m]) for m in ("acc", "kappa", "auc", "ap"))))

    # ---- joint vs single-task ------------------------------------------------------------
    st = json.load(open(os.path.join(F, "single_task%s.json" % TAG)))
    rows = {t: [v for k, v in st.items() if k.startswith(t + "|")] for t in ("stage", "apnea")}
    jt = {}
    print("\nJoint vs single-task (single -> joint, Wilcoxon on fold means)")
    for task, metrics in (("stage", ("acc", "kappa")), ("apnea", ("auc", "ap"))):
        for m in metrics:
            r = test(J[m], fold_means(rows[task], m)); r = dict(single=r["b"], joint=r["a"],
                    delta=r["delta"], folds_joint_higher=r["folds_up"], p=r["p"])
            jt["%s|%s" % (task, m)] = r
            print("  %-6s %-6s single %.4f  joint %.4f  delta %+.4f  %d/10 folds  p=%.4f"
                  % (task, m, r["single"], r["joint"], r["delta"], r["folds_joint_higher"], r["p"]))
    out["joint_vs_single"] = jt

    # ---- modality-ablation grid ----------------------------------------------------------
    shards = {}
    for p in glob.glob(os.path.join(F, "ablation_shards%s" % TAG, "*.json")):
        for k, v in json.load(open(p)).items():
            shards.setdefault(v["condition"], []).append(v)
    conds = sorted(shards)
    grid = {}
    for m in ("acc", "auc"):
        res = {c: test(fold_means(shards[c], m), J[m]) for c in conds}
        adj = holm(np.array([res[c]["p"] for c in conds]))
        for c, a in zip(conds, adj):
            g = grid.setdefault(c, {"seeds": len(shards[c])})
            g[m] = dict(removed=res[c]["a"], full=res[c]["b"], delta=res[c]["delta"],
                        holm_p=round(float(a), 4))
    print("\nModality ablation (removed vs full; Holm across %d conditions)" % len(conds))
    print("  %-14s %5s  %-28s %-28s" % ("condition", "seeds", "staging acc (Holm p)", "respiratory AUC (Holm p)"))
    for c in conds:
        g = grid[c]
        print("  %-14s %5d  %.4f (%+.4f, p=%.3f)     %.4f (%+.4f, p=%.3f)"
              % (c, g["seeds"], g["acc"]["removed"], g["acc"]["delta"], g["acc"]["holm_p"],
                 g["auc"]["removed"], g["auc"]["delta"], g["auc"]["holm_p"]))
    out["ablation"] = grid

    # ---- Huttunen et al., five signals, against MM-Net ----------------------------------
    hp = os.path.join(F, "huttunen5%s.json" % TAG)
    if os.path.exists(hp):
        h = json.load(open(hp))
        hrows = [v for k, v in h.items() if k.startswith("seed|")]
        mm = json.load(open(os.path.join(L, "final_architecture_aligned.json")))
        hu = {}
        print("\nHuttunen et al. (five signals%s) vs MM-Net" % (", " + TAG.strip("_") if TAG else ""))
        for m, mm_key, label in (("acc", "acc_single", "staging acc, MM-Net single"),
                                 ("acc", "acc", "staging acc, MM-Net 3-seed"),
                                 ("auc", "auc_single", "resp AUC, MM-Net single"),
                                 ("auc", "auc", "resp AUC, MM-Net 3-seed")):
            r = test(np.asarray(mm[mm_key]), fold_means(hrows, m))
            hu[label] = dict(mmnet=r["a"], huttunen=r["b"], delta=r["delta"],
                             folds_mmnet_higher=r["folds_up"], p=r["p"])
            print("  %-28s MM-Net %.4f  Huttunen %.4f  delta %+.4f  %d/10  p=%.4f"
                  % (label, r["a"], r["b"], r["delta"], r["folds_up"], r["p"]))
        out["huttunen5"] = hu

    path = os.path.join(F, "aligned_controls_summary.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
