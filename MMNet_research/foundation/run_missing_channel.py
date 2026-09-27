"""Missing cardiorespiratory channels, handled explicitly.

Twelve of the 99 recordings lack one or more cardiorespiratory channels, which the
published model receives as zeros. The cardio tensor is z-scored per patient, so a
zero-filled channel looks exactly like a flat channel sitting at its mean, and the
network has no way to tell "absent" from "quiet". The referee asks for a
missing-channel indicator or channel dropout, and for respiratory performance
reported by subgroup rather than only pooled.

Arms, all on the final configuration (A+LaBraM, learned cardio CNN, BiLSTM, concat,
bypass, HMM), ten folds x three seeds, identical except for the cardio encoder:

  published  the encoder as published (re-run here, which also checks that the
             rebuilt caches reproduce the reported numbers)
  indicator  the seven per-recording availability flags are appended to the pooled
             convolutional features before the encoder's projection, so the model
             is told which channels are real
  dropout    indicator, plus channel dropout in training: each available channel is
             zeroed, and its flag cleared, with probability 0.15 per window, so the
             model learns to work from whatever subset a recording provides

For every arm, respiratory AUC and AP are reported pooled and separately for the
87 complete and 12 incomplete recordings, from the per-patient scores run_10fold
already returns.

  KMP_DUPLICATE_LIB_OK=TRUE python run_missing_channel.py [published indicator dropout]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402
import cardio_cnn               # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

MM = os.path.join(REPO, "data", "multimodal")
OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "missing_channel.json")
FINAL = dict(arm="A+labram", cardio="raw_cnn", cardio_mode="concat",
             temporal="lstm", hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
SEEDS = [42, 1, 7]
P_DROP = 0.15
N_CH, N_T = 7, 750                                # the published cardio tensor
N_SIG = N_CH * N_T                                  # raw samples per epoch

CVALID = {int(os.path.basename(f)[2:-4]): np.load(f)["cvalid"].astype(bool)
          for f in glob.glob(os.path.join(MM, "SN*.npz"))}

_raw_table = cardio_cnn.raw_cardio_table
_CardioCNN = cardio_cnn.CardioCNN


def table_with_flags(Cm, mm_dir):
    """The published table with the seven availability flags appended per epoch."""
    out = {}
    for sid, x in _raw_table(Cm, mm_dir).items():
        flags = np.tile(CVALID[sid].astype(np.float32), (len(x), 1))
        out[sid] = np.concatenate([x, flags], axis=1)
    return out


class FlaggedCardioCNN(_CardioCNN):
    """CardioCNN over [N, 7*750 + 7]: the flags join the pooled features before the
    projection, so the convolutional stack itself is the published one."""

    channel_dropout = 0.0

    def __init__(self, d=64, drop=0.3, width=48):
        super().__init__(d=d, drop=drop, width=width)
        self.head = nn.Sequential(nn.Linear(width * 4 + N_CH, d), nn.LayerNorm(d),
                                  nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        n = x.shape[0]
        z = x[:, :N_SIG].reshape(n, N_CH, N_T)
        f = x[:, N_SIG:]
        if self.training and self.channel_dropout > 0:
            keep = (torch.rand_like(f) >= self.channel_dropout).float()
            f = f * keep
            z = z * keep.unsqueeze(-1)
        h = self.net(z)
        h = torch.cat([h.mean(-1), h.amax(-1), f], dim=-1)
        return self.head(h)


class DropoutCardioCNN(FlaggedCardioCNN):
    channel_dropout = P_DROP


def patch(arm):
    """Install the arm's table and encoder; the config context reads both lazily."""
    if arm == "published":
        cardio_cnn.raw_cardio_table, cardio_cnn.CardioCNN = _raw_table, _CardioCNN
        cardio_cnn.N_T = N_T
        return
    cardio_cnn.raw_cardio_table = table_with_flags
    cardio_cnn.CardioCNN = FlaggedCardioCNN if arm == "indicator" else DropoutCardioCNN
    # arms.py sizes the cardio input as N_CH * N_T (7 x 751 = 7 x 750 + 7 flags); the
    # flagged encoder reshapes with its own constants, so only the width changes
    cardio_cnn.N_T = N_T + 1


def subgroup(per_subject, subs):
    ys, ps = [], []
    for s in subs:
        k = "SN%d" % s
        if k in per_subject:
            ys.append(C.DATA[s][3]); ps.append(np.asarray(per_subject[k]["apnea"]))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return dict(n_patients=len(ys), auc=float(roc_auc_score(y, p)),
                ap=float(average_precision_score(y, p)))


def main(arms, seeds=SEEDS):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in arms:
        for seed in seeds:
            key = "%s|%d" % (arm, seed)
            if key in res:
                print("[skip]", key); continue
            patch(arm)
            t0 = time.time()
            with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                              lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                              cardio_mode=FINAL["cardio_mode"]):
                keep = (arm == "published" and seed == 42)
                r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"], seed=seed,
                                 keep=keep)
                if keep:
                    # the pooled predictions the derived analyses need (Fig. 3, conformal),
                    # with the patient order run_10fold concatenates them in
                    order = [s for _, te in C.FOLDS for s in te]
                    P = r["pred"]
                    np.savez_compressed(OUT.replace("missing_channel.json",
                                                    "predictions_seed42_rerun.npz"),
                                        y_true=P[0], y_pred=P[1], apnea_true=P[2],
                                        apnea_score=P[3], subjects=np.array(order),
                                        n_epochs=np.array([len(C.DATA[s][2]) for s in order]))
                complete = [s for s in C.DATA if CVALID[s].all()]
                partial = [s for s in C.DATA if not CVALID[s].all()]
                sg = dict(complete=subgroup(r["per_subject"], complete),
                          incomplete=subgroup(r["per_subject"], partial))
            patch("published")
            res[key] = dict(arm=arm, seed=seed,
                            acc=[f["acc"] for f in r["per_fold"]],
                            mf1=[f["mf1"] for f in r["per_fold"]],
                            kappa=[f["kappa"] for f in r["per_fold"]],
                            auc=[f["auc"] for f in r["per_fold"]],
                            ap=[f["ap"] for f in r["per_fold"]],
                            subgroups=sg, minutes=(time.time() - t0) / 60)
            json.dump(res, open(OUT, "w"), indent=1)
            print("[ok] %-9s seed %-2d acc %.4f auc %.4f | complete AUC %.4f (n=%d) "
                  "incomplete AUC %.4f (n=%d)  [%.1f min]"
                  % (arm, seed, np.mean(res[key]["acc"]), np.nanmean(res[key]["auc"]),
                     sg["complete"]["auc"], sg["complete"]["n_patients"],
                     sg["incomplete"]["auc"], sg["incomplete"]["n_patients"],
                     res[key]["minutes"]), flush=True)

    summary = {}
    for arm in {v["arm"] for v in res.values() if isinstance(v, dict) and "arm" in v}:
        rows = [v for v in res.values() if isinstance(v, dict) and v.get("arm") == arm]
        s = {m: dict(mean=float(np.nanmean(np.concatenate([r[m] for r in rows]))),
                     sd=float(np.nanstd(np.concatenate([r[m] for r in rows]), ddof=1)))
             for m in ("acc", "mf1", "kappa", "auc", "ap")}
        for g in ("complete", "incomplete"):
            s[g] = {m: float(np.mean([r["subgroups"][g][m] for r in rows])) for m in ("auc", "ap")}
            s[g]["n_patients"] = rows[0]["subgroups"][g]["n_patients"]
        summary[arm] = s
    res["_summary"] = summary
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=["published", "indicator", "dropout"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    main(a.arms, a.seeds)
