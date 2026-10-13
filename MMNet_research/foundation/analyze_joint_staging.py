"""Where does joint training change staging? Final MM-Net (joint) vs its staging-only control.

Uses the saved fold probabilities of run_seed_ensemble.py (joint) and run_final_single_task.py
(staging-only); nothing is trained. Staging is decoded exactly as in the paper (train-fold HMM,
three-seed ensemble). Comparisons, each paired over the ten fold means and over patients:
  1. epochs near a respiratory-event onset (within +-k epochs) vs the rest of the night
  2. patients grouped by annotated AHI (AASM bands)
  3. per-class F1 (W, N1, N2, N3, REM)

    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal \
    MMNET_PFLOW=aligned/pressure_flow python analyze_joint_staging.py
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

import run_seed_ensemble as SE

C = SE.C
JOINT = "base-nocw-cross-ftlae-p8-al"
STG = JOINT + "-stgonly"
SEEDS = [42, 1, 7]
K = (0, 1, 2)                       # half-width of the near-event window, in epochs


def decode(cfg, fi):
    d = np.load(os.path.join(SE.OUT, "%s_fold%d.npz" % (cfg, fi)))
    return {int(s): SE._decode({cfg: d}, d["A_log"], d["pi_log"], int(s), [(cfg, sd) for sd in SEEDS])
            for s in d["te"]}


def near(a, k):
    on = np.flatnonzero(a)
    m = np.zeros(len(a), bool)
    for t in on:
        m[max(0, t - k):t + k + 1] = True
    return m


def ahi(s):
    y, a = C.DATA[s][2], C.DATA[s][3]
    sleep = (y != 0).sum() * 30 / 3600.0
    return a[y != 0].sum() / max(sleep, 1e-6)


def main():
    rows = []                                   # per patient
    for fi in range(len(C.FOLDS)):
        pj, ps = decode(JOINT, fi), decode(STG, fi)
        for s in pj:
            y, a = C.DATA[s][2], C.DATA[s][3]
            r = {"sid": s, "fold": fi, "ahi": ahi(s), "n": len(y),
                 "acc_j": float((pj[s] == y).mean()), "acc_s": float((ps[s] == y).mean()),
                 "y": y, "pj": pj[s], "ps": ps[s]}
            for k in K:
                m = near(a, k)
                r["near%d" % k] = (int(m.sum()), float((pj[s][m] == y[m]).sum()), float((ps[s][m] == y[m]).sum()),
                                   int((~m).sum()), float((pj[s][~m] == y[~m]).sum()), float((ps[s][~m] == y[~m]).sum()))
            rows.append(r)

    out, lines = {}, ["Final MM-Net, joint vs staging-only (three-seed ensemble, HMM), 99 test patients"]

    def report(name, j, s, unit):
        j, s = np.asarray(j), np.asarray(s)
        ok = ~np.isnan(j) & ~np.isnan(s)
        j, s = j[ok], s[ok]
        p = wilcoxon(j, s).pvalue if len(j) > 1 and np.any(j != s) else float("nan")
        lines.append("  %-38s staging-only %.4f  joint %.4f  diff %+.4f  up %d/%d %s  Wilcoxon p=%.3f"
                     % (name, s.mean(), j.mean(), j.mean() - s.mean(), int((j > s).sum()), len(j), unit, p))
        out[name + " (" + unit + ")"] = {"single": float(s.mean()), "joint": float(j.mean()), "p": float(p),
                                         "up": int((j > s).sum()), "n": int(len(j))}

    lines.append("\n1. overall")
    for unit, key in (("folds", "fold"), ("patients", "sid")):
        g = _group(rows, key)
        report("overall accuracy", [np.mean([r["acc_j"] for r in v]) for v in g],
               [np.mean([r["acc_s"] for r in v]) for v in g], unit)

    lines.append("\n2. epochs near a respiratory-event onset vs the rest")
    for k in K:
        for unit, key in (("folds", "fold"), ("patients", "sid")):
            g = _group(rows, key)
            nj = [sum(r["near%d" % k][1] for r in v) / max(1, sum(r["near%d" % k][0] for r in v)) for v in g]
            ns = [sum(r["near%d" % k][2] for r in v) / max(1, sum(r["near%d" % k][0] for r in v)) for v in g]
            fj = [sum(r["near%d" % k][4] for r in v) / max(1, sum(r["near%d" % k][3] for r in v)) for v in g]
            fs = [sum(r["near%d" % k][5] for r in v) / max(1, sum(r["near%d" % k][3] for r in v)) for v in g]
            keep = [sum(r["near%d" % k][0] for r in v) > 0 for v in g]
            nj = [x if kp else np.nan for x, kp in zip(nj, keep)]
            ns = [x if kp else np.nan for x, kp in zip(ns, keep)]
            report("near event (+-%d ep) accuracy" % k, nj, ns, unit)
            report("away from events (+-%d ep) accuracy" % k, fj, fs, unit)

    lines.append("\n3. patients by annotated AHI (per-patient accuracy)")
    for lo, hi, nm in ((0, 5, "normal <5"), (5, 15, "mild 5-15"), (15, 30, "moderate 15-30"), (30, 1e9, "severe >=30")):
        v = [r for r in rows if lo <= r["ahi"] < hi]
        if len(v) > 1:
            report("AHI %s (n=%d)" % (nm, len(v)), [r["acc_j"] for r in v], [r["acc_s"] for r in v], "patients")

    lines.append("\n4. per-class F1 (fold-level)")
    for c, nm in enumerate(["W", "N1", "N2", "N3", "REM"]):
        fj, fs = [], []
        for v in _group(rows, "fold"):
            y = np.concatenate([r["y"] for r in v])
            fj.append(f1_score(y == c, np.concatenate([r["pj"] for r in v]) == c))
            fs.append(f1_score(y == c, np.concatenate([r["ps"] for r in v]) == c))
        report("F1 %s" % nm, fj, fs, "folds")

    text = "\n".join(lines)
    dst = os.path.join(SE.RL.OUT, "joint_staging_analysis")
    json.dump(out, open(dst + ".json", "w"), indent=1)
    open(dst + ".txt", "w").write(text + "\n")
    print(text)


def _group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r)
    return list(g.values())


if __name__ == "__main__":
    main()
