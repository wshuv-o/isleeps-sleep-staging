"""Dispersion and sample size for every results table in the manuscript.

Eleven of the twelve tables report bare point estimates: no standard deviation,
no n, no test statistic. The fold-values exist for nearly all of them, so the
numbers were available and simply were not carried through to the text. A cell
that reads 0.782 says less than one that reads 0.782 +- 0.038 over 30
patient-independent fold-values, and the difference is not presentation --- it
is whether the reader can tell a real gap from noise.

Emits every table's cells with mean, SD and n so the LaTeX can be filled in.

  KMP_DUPLICATE_LIB_OK=TRUE python table_dispersion.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(REPO, "MMNet_research", "results", "revision", "runs")
OUT = os.path.join(R, "final", "table_dispersion.json")
S = ["W", "N1", "N2", "N3", "R"]


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)), n=int(len(v)))


out = {}

# ---- headline / final model ------------------------------------------------
fm = json.load(open(os.path.join(R, "final", "final_model.json")))
out["final"] = {k: ms(np.concatenate([fm[s][k] for s in sorted(fm)]))
                for k in ("acc", "mf1", "kappa", "auc", "ap")}
pcf = np.concatenate([fm[s]["pcf"] for s in sorted(fm)], axis=0)
out["final_perclass"] = {S[i]: ms(pcf[:, i]) for i in range(5)}

# ---- neural-only, for the per-class table ----------------------------------
no = json.load(open(os.path.join(R, "final", "neural_only_pcf.json")))
npc = np.concatenate([no[k]["pcf"] for k in sorted(no)], axis=0)
out["neural_only_perclass"] = {S[i]: ms(npc[:, i]) for i in range(5)}

# ---- modality ablation -----------------------------------------------------
rows = {}
for p in glob.glob(os.path.join(R, "final", "ablation_shards", "*.json")):
    for _, r in json.load(open(p)).items():
        rows.setdefault(r["condition"], []).append(r)
out["ablation"] = {}
for c, rs in rows.items():
    rs = sorted(rs, key=lambda x: x["seed"])
    out["ablation"][c] = dict(
        acc=ms(np.concatenate([r["acc"] for r in rs])),
        auc=ms(np.concatenate([r["auc"] for r in rs])))

# ---- representation experiments -------------------------------------------
SH = os.path.join(R, "foundation", "sweep_shards")


def cfg(arm, cardio, mode, hidden, temporal):
    acc, auc = [], []
    for s in (42, 1, 7):
        tag = "%s__t%s__h%d__dr0.3__lr0.0003__wd0.0001__s%d" % (arm, temporal, hidden, s)
        if cardio != "features14":
            tag += "__c%s-%s" % (cardio, mode)
        p = os.path.join(SH, tag + ".json")
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        rec = d if "acc" in d else next(iter(d.values()))
        acc.append(rec["acc"]); auc.append(rec.get("auc", []))
    return dict(acc=ms(np.concatenate(acc)), auc=ms(np.concatenate(auc)))


out["cardio_repr"] = {}
for nm, c, m in (("features14", "features14", ""),
                 ("randproj_feat", "randproj_feat", "concat"),
                 ("randproj_raw", "randproj_raw", "concat"),
                 ("raw_cnn", "raw_cnn", "concat")):
    v = cfg("A+labram", c, m, 256, "lstm")
    if v:
        out["cardio_repr"][nm] = v

out["eeg_repr"] = {}
for nm, arm in (("features", "A"), ("frozen_alone", "labram"),
                ("features_plus_one", "A+labram"),
                ("features_plus_two", "A+pretrained+labram")):
    v = cfg(arm, "features14", "", 256, "none")
    if v:
        out["eeg_repr"][nm] = v

# ---- learning curve --------------------------------------------------------
out["curve"] = {}
for f in sorted(glob.glob(os.path.join(R, "final", "lc_shards", "*.json"))):
    v = list(json.load(open(f)).values())
    out["curve"]["%.2f" % v[0]["frac"]] = dict(
        n_train=int(np.mean([x["n_train"] for x in v])),
        acc=ms([x["acc"] for x in v]), auc=ms([x["auc"] for x in v]))

# ---- external validation: spread across recordings -------------------------
out["external"] = {}
for corpus in ("isruc", "sleepedf"):
    p = os.path.join(R, "final", "external_%s.json" % corpus)
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    pr = d.get("per_recording", {})
    if pr:
        out["external"][corpus] = dict(
            acc=ms([v["acc"] for v in pr.values()]),
            kappa=ms([v["kappa"] for v in pr.values()]),
            auc=ms([v["auc"] for v in pr.values() if v["auc"] is not None]))

json.dump(out, open(OUT, "w"), indent=1)


def show(title, d, keys):
    print("\n=== %s ===" % title)
    for k, v in d.items():
        if "mean" in v:                      # a flat ms() entry, not nested by metric
            print("  %-22s %.4f +- %.4f (n=%d)" % (k, v["mean"], v["sd"], v["n"]))
            continue
        cells = "  ".join("%s %.4f +- %.4f (n=%d)" % (kk, v[kk]["mean"], v[kk]["sd"], v[kk]["n"])
                          for kk in keys if isinstance(v.get(kk), dict))
        print("  %-22s %s" % (k, cells))


print("=== headline ===")
for k, v in out["final"].items():
    print("  %-6s %.4f +- %.4f  (n=%d)" % (k, v["mean"], v["sd"], v["n"]))
show("per-class F1, full model", out["final_perclass"], [])
show("per-class F1, neural only", out["neural_only_perclass"], [])
show("modality ablation", out["ablation"], ["acc", "auc"])
show("cardiorespiratory representation", out["cardio_repr"], ["auc", "acc"])
show("neural representation", out["eeg_repr"], ["acc", "auc"])
show("learning curve", out["curve"], ["acc", "auc"])
show("external, spread across recordings", out["external"], ["acc", "kappa", "auc"])
print("\nwrote", OUT)
