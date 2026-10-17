"""Staging by stroke characteristics, from saved predictions (nothing is trained).

Per patient: final MM-Net accuracy (three-seed ensemble, train-fold HMM) and the gain of the
sleep-adapted encoder over the frozen-encoder reference (base-nocw-cross, same fusion and class
weights; Table 6 'ref.'). Related to NIHSS, modified Rankin scale, Barthel index, lesion side and
time from stroke onset (cohort metadata, data/full100/subject_description.xlsx).

    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal \
    MMNET_PFLOW=aligned/pressure_flow python analyze_stroke_factors.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

import lesion
import run_seed_ensemble as SE

C = SE.C
FINAL, FROZEN = "base-nocw-cross-ftlae-p8-al", "base-nocw-cross"
SEEDS = [42, 1, 7]


def per_patient(cfg):
    out = {}
    for fi in range(len(C.FOLDS)):
        d = np.load(os.path.join(SE.OUT, "%s_fold%d.npz" % (cfg, fi)))
        for s in d["te"]:
            s = int(s)
            p = SE._decode({cfg: d}, d["A_log"], d["pi_log"], s, [(cfg, sd) for sd in SEEDS])
            out[s] = float((p == C.DATA[s][2]).mean())
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    fin, frz = per_patient(FINAL), per_patient(FROZEN)
    meta = pd.read_excel(os.path.join(SE.RL.REPO if hasattr(SE.RL, "REPO") else lesion.REPO,
                                      "data", "full100", "subject_description.xlsx"))
    side = lesion.lesion_table()
    rows = []
    for i, r in meta.iterrows():
        s = i + 1
        if s not in fin:
            continue
        rows.append({"sid": s, "acc": fin[s], "gain": fin[s] - frz[s], "side": side.get(s, (None,))[0],
                     "nihss": num(r["NIHSS_scale"]), "mrs": num(r["MRS_scale"]),
                     "barthel": num(r["Barthel_scale "]), "days": num(r["ictus_timeB.2.b"])})
    df = pd.DataFrame(rows)
    lines, res = ["Final MM-Net by stroke characteristics (%d patients)" % len(df)], {}

    for var in ("nihss", "mrs", "barthel", "days"):
        v = df.dropna(subset=[var])
        ra = spearmanr(v[var], v["acc"]); rg = spearmanr(v[var], v["gain"])
        lines.append("%-8s n=%2d  accuracy rho %+.3f (p=%.3f)   encoder gain rho %+.3f (p=%.3f)"
                     % (var, len(v), ra[0], ra[1], rg[0], rg[1]))
        res[var] = {"n": int(len(v)), "acc_rho": float(ra[0]), "acc_p": float(ra[1]),
                    "gain_rho": float(rg[0]), "gain_p": float(rg[1])}

    groups = {"NIHSS 0-4 (minor)": df[df.nihss <= 4], "NIHSS >=5 (moderate+)": df[df.nihss >= 5],
              "mRS 0-2 (independent)": df[df.mrs <= 2], "mRS 3-5 (dependent)": df[df.mrs >= 3]}
    lines.append("")
    for name, g in groups.items():
        lines.append("%-24s n=%2d  accuracy %.3f  frozen %.3f  encoder gain %+.4f"
                     % (name, len(g), g.acc.mean(), (g.acc - g.gain).mean(), g.gain.mean()))
        res[name] = {"n": int(len(g)), "acc": float(g.acc.mean()), "gain": float(g.gain.mean())}
    for a, b, key in (("NIHSS 0-4 (minor)", "NIHSS >=5 (moderate+)", "nihss_groups"),
                      ("mRS 0-2 (independent)", "mRS 3-5 (dependent)", "mrs_groups")):
        ga, gb = groups[a], groups[b]
        pa = mannwhitneyu(ga.acc, gb.acc).pvalue; pg = mannwhitneyu(ga.gain, gb.gain).pvalue
        lines.append("  %s vs %s: accuracy p=%.3f, encoder gain p=%.3f" % (a, b, pa, pg))
        res[key] = {"acc_p": float(pa), "gain_p": float(pg)}

    lines.append("")
    sg = {k: df[df.side == k] for k in ("L", "R", "B")}
    for k, g in sg.items():
        lines.append("lesion side %s  n=%2d  accuracy %.3f  encoder gain %+.4f" % (k, len(g), g.acc.mean(), g.gain.mean()))
    kw = kruskal(*[g.acc for g in sg.values() if len(g) > 1]); kg = kruskal(*[g.gain for g in sg.values() if len(g) > 1])
    lines.append("  across sides: accuracy Kruskal p=%.3f, encoder gain Kruskal p=%.3f" % (kw.pvalue, kg.pvalue))
    res["side"] = {k: {"n": int(len(g)), "acc": float(g.acc.mean()), "gain": float(g.gain.mean())} for k, g in sg.items()}
    res["side"]["acc_p"], res["side"]["gain_p"] = float(kw.pvalue), float(kg.pvalue)

    text = "\n".join(lines)
    dst = os.path.join(SE.RL.OUT, "stroke_factors")
    json.dump(res, open(dst + ".json", "w"), indent=1)
    open(dst + ".txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
