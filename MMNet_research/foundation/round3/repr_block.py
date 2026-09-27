"""Items 6H, 6I, 6J, 3: Table 11 recomputed from the sweep shards on ten fold-means.

Identifies the exact configuration behind every Table 11 row and repeats its tests
(Wilcoxon on the ten fold-means, Holm within block), and adds the rows the sweep ran
but the table omits (pretrained and randomly initialised MOMENT encoders, the
encoder four of five nested outer folds selected).

  python repr_block.py
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

import _common as C

SW = os.path.join(C.MM, "results", "revision", "runs", "foundation", "sweep_shards")


def load(stem):
    rows = []
    for s in (42, 1, 7):
        p = os.path.join(SW, stem.replace("{s}", str(s)))
        if not os.path.exists(p):
            return None
        rows.append(list(json.load(open(p)).values())[0])
    return rows


def summ(rows):
    A = np.array([r["acc"] for r in rows]); U = np.array([r["auc"] for r in rows])
    return dict(acc=float(A.mean()), acc_sd30=float(A.std(ddof=1)), auc=float(U.mean()),
                auc_sd30=float(U.std(ddof=1)), acc_fm=A.mean(0), auc_fm=U.mean(0))


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); out[i] = run
    return out


H = "__h256__dr0.3__lr0.0003__wd0.0001__s{s}"
cardio = {  # neural stream = A+labram (388-d), temporal = BiLSTM
    "14 engineered features": "A+labram__tlstm" + H + ".json",
    "random nonlinear expansion of the 14": "A+labram__tlstm" + H + "__crandproj_feat-concat.json",
    "random linear projection of raw": "A+labram__tlstm" + H + "__crandproj_raw-concat.json",
    "learned CNN (adopted)": "A+labram__tlstm" + H + "__craw_cnn-concat.json",
    "MOMENT, randomly initialised (not in paper)": "A+labram__tlstm" + H + "__cmoment_random-concat.json",
    "MOMENT, pretrained, concat (not in paper)": "A+labram__tlstm" + H + "__cmoment-concat.json",
    "MOMENT, pretrained, replace (not in paper)": "A+labram__tlstm" + H + "__cmoment-replace.json",
}
neural = {  # cardio stream = 14 engineered features, temporal = NONE (no BiLSTM)
    "188 engineered features": "A__tnone" + H + ".json",
    "frozen pretrained encoder alone": "labram__tnone" + H + ".json",
    "features + pretrained (adopted)": "A+labram__tnone" + H + ".json",
    "features + two pretrained encoders": "A+pretrained+labram__tnone" + H + ".json",
}
out = {}
for block, spec, metric in (("cardiorespiratory (neural = 188 + LaBraM, BiLSTM)", cardio, "auc"),
                            ("neural (cardio = 14 features, temporal = none)", neural, "acc")):
    S = {k: summ(load(v)) for k, v in spec.items() if load(v) is not None}
    base = list(S)[0]
    ps, names = [], []
    rows = {}
    for k, v in S.items():
        rows[k] = {kk: vv for kk, vv in v.items() if not kk.endswith("_fm")}
        if k == base:
            continue
        p = float(wilcoxon(v[metric + "_fm"], S[base][metric + "_fm"]).pvalue)
        rows[k]["delta_" + metric] = float(v[metric] - S[base][metric]); rows[k]["p"] = p
        ps.append(p); names.append(k)
    for k, h in zip(names, holm(ps)):
        rows[k]["holm_p_within_block"] = float(h)
    out[block] = rows
S2 = summ(load(neural["features + two pretrained encoders"])); S1 = summ(load(neural["features + pretrained (adopted)"]))
out["two_vs_one_encoder_acc"] = dict(delta=float(S2["acc"] - S1["acc"]), p=float(wilcoxon(S2["acc_fm"], S1["acc_fm"]).pvalue))
fm = json.load(open(os.path.join(C.FINAL, "final_model.json")))
out["final_model_run"] = dict(acc=float(np.mean([fm[k]["acc"] for k in fm])), auc=float(np.mean([fm[k]["auc"] for k in fm])),
                              note="final_model.json is a separate training run of the configuration in the 'learned CNN (adopted)' row")
C.save("repr_block.json", out)
for b, rows in out.items():
    print("==", b)
    if isinstance(rows, dict):
        for k, v in rows.items():
            print("  ", k, {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v)
