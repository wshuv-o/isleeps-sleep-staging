"""Permutation importance for the final model -- the second, independent attribution.

Referee Finding 2 asked for an attribution that could have come out otherwise.
The answer in fig_attribution_agreement.py is to compute importance two ways
that share no machinery -- retrain without a modality, versus shuffle it at
inference in the trained model -- and show they agree. That figure is currently
built from `permutation_importance.json`, which was measured on the SUBMITTED
model: 188 engineered features and 14 engineered cardiorespiratory features.

The final model replaced both branches, so the permutation half of that
comparison has to be recomputed or the figure would correlate a new ablation
against an old attribution and call the result agreement.

Protocol copied from notebook 7 so the two are comparable:

  * three shuffle repeats per modality per fold, averaged
  * rows are permuted WITHIN each test recording, so per-recording z-scoring and
    class balance are preserved and only the temporal association is destroyed
  * the model is never retrained; this is inference-time attribution

Two things had to change for the final model, both of which would have produced
a plausible wrong number if left alone -- the same two traps as ablation_remap:

  * the EEG mask must cover the 200-d LaBraM embedding, which is built from the
    same four EEG derivations. Shuffling only the 112 engineered EEG columns
    would leave the embedding intact and understate EEG.
  * the cardio groups must address raw CHANNELS in the 7 x 750 tensor, not
    columns of a 14-d feature vector.

Both are taken from ablation_remap, so the two attributions carve the input the
same way and the agreement figure compares like with like.

  KMP_DUPLICATE_LIB_OK=TRUE python run_permutation_importance.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import ablation_remap           # noqa: E402
import sweep                    # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
FINAL = dict(arm="A+labram", cardio="raw_cnn", temporal="lstm",
             hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
N_REPEAT = 3


def groups(C, n_eeg_total):
    """{name: (stream, column indices)} for the final model's actual inputs."""
    g = {}
    eegm = ablation_remap.widened_eeg_masks(C, n_eeg_total)
    for name, key in (("EEG", "eeg"), ("EOG", "eog"), ("EMG", "emg")):
        g[name] = ("eeg", np.where(eegm[key])[0])
    raw = ablation_remap.raw_card_groups()
    for name, key in (("SpO2", "spo2"), ("pulse/HRV", "pulse_hrv"), ("ECG", "ecg"),
                      ("airflow", "airflow"), ("effort", "effort")):
        g[name] = ("card", np.asarray(raw[key]))
    return g


def evaluate(model, te, gname=None, cols=None, stream=None, rep=0, fi=0):
    """Pooled accuracy and respiratory AUC over the fold's test recordings."""
    ys, ps, ay, ap = [], [], [], []
    r = np.random.RandomState(1000 * fi + rep)
    for s in te:
        fe, fc, y, a = C.DATA[s]
        if gname is not None:
            perm = r.permutation(len(y))
            if stream == "eeg":
                fe = fe.copy(); fe[:, cols] = fe[perm][:, cols]
            else:
                fc = fc.copy(); fc[:, cols] = fc[perm][:, cols]
        sp, apn = C.infer_arrays(model, fe, fc, len(y))
        ys.append(y); ps.append(sp.argmax(1)); ay.append(a); ap.append(apn)
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    ay, ap = np.concatenate(ay), np.concatenate(ap)
    return (float(accuracy_score(ys, ps)),
            float(roc_auc_score(ay, ap)) if len(np.unique(ay)) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "permutation_importance_final.json")
    fold_rows = json.load(open(dst))["per_fold"] if os.path.exists(dst) else []
    done = {r["fold"] for r in fold_rows}

    t0 = time.time()
    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode="concat") as h:
        G = groups(C, h.dim)
        print("eeg %d-d, cardio %d-d" % (h.dim, h.n_card), flush=True)
        print("groups -> %s" % {k: len(v[1]) for k, v in G.items()}, flush=True)

        for fi, (tr_all, te) in enumerate(C.FOLDS):
            if fi in done:
                print("[skip] fold %d" % fi, flush=True); continue
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all); rng.shuffle(tr_all)
            nv = max(10, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]
            model = C.train_fold(tr, va, "concat", [], [], seed=a.seed,
                                 temporal=FINAL["temporal"])
            b_acc, b_auc = evaluate(model, te)
            row = {"fold": fi, "base_acc": b_acc, "base_auc": b_auc}
            for gname, (stream, cols) in G.items():
                accs, aucs = [], []
                for rep in range(N_REPEAT):
                    x, u = evaluate(model, te, gname, cols, stream, rep, fi)
                    accs.append(x); aucs.append(u)
                row[gname] = {"acc": float(np.mean(accs)), "auc": float(np.nanmean(aucs))}
            fold_rows.append(row)
            json.dump({"per_fold": fold_rows}, open(dst, "w"), indent=1)
            print("fold %d  base acc %.4f auc %.4f  [%.1f min]"
                  % (fi, b_acc, b_auc, (time.time() - t0) / 60), flush=True)

    base_acc = float(np.mean([r["base_acc"] for r in fold_rows]))
    base_auc = float(np.nanmean([r["base_auc"] for r in fold_rows]))
    imp = {}
    names = [k for k in fold_rows[0] if k not in ("fold", "base_acc", "base_auc")]
    for g in names:
        da = [r["base_acc"] - r[g]["acc"] for r in fold_rows]
        du = [r["base_auc"] - r[g]["auc"] for r in fold_rows]
        imp[g] = {"acc_drop": float(np.mean(da)), "acc_sd": float(np.std(da)),
                  "auc_drop": float(np.nanmean(du)), "auc_sd": float(np.nanstd(du))}
    json.dump({"baseline": {"acc": base_acc, "auc": base_auc},
               "importance": imp, "per_fold": fold_rows}, open(dst, "w"), indent=1)

    print("\nunpermuted baseline: acc %.4f  resp AUC %.4f\n" % (base_acc, base_auc))
    print("%-11s %11s %8s %11s %8s" % ("modality", "acc drop", "sd", "AUC drop", "sd"))
    print("-" * 54)
    for g in sorted(imp, key=lambda k: -imp[k]["auc_drop"]):
        v = imp[g]
        print("%-11s %11.4f %8.4f %11.4f %8.4f"
              % (g, v["acc_drop"], v["acc_sd"], v["auc_drop"], v["auc_sd"]))
    print("\nwrote %s  [%.1f min]" % (dst, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
