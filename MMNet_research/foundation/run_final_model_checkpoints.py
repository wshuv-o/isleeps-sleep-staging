"""Save the final model's per-fold weights, which the abstract says are released.

run_final_model.py never wrote any. train_fold keeps the best epoch's state in
memory, loads it into the model and returns, so the weights of the network behind
every reported number existed only for the lifetime of the process. The two .pt
files in results/models are the feature-only configuration at fold 0, not this
model.

Same configuration, same folds, same seed as run_final_model.py, so the released
weights are the ones that produced the reported figures rather than a fresh run
that merely resembles them. Each fold's checkpoint carries the fold index, the seed
and the fold's own scores, so a checkpoint can be matched against the row it
produced without trusting the filename.

  KMP_DUPLICATE_LIB_OK=TRUE python run_final_model_checkpoints.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "checkpoints")
FINAL = dict(arm="A+labram", cardio="raw_cnn", cardio_mode="concat",
             temporal="lstm", hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    manifest = []

    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode=FINAL["cardio_mode"]) as h:
        for fi, (tr_all, te) in enumerate(C.FOLDS):
            dst_probe = os.path.join(OUT, "mmnet_seed%d_fold%d.pt" % (SEED, fi))
            if os.path.exists(dst_probe) and os.path.getsize(dst_probe) > 1_000_000:
                d = torch.load(dst_probe, map_location="cpu", weights_only=False)
                manifest.append({"fold": fi, "file": os.path.basename(dst_probe),
                                 "acc": d["scores"]["acc"], "kappa": d["scores"]["kappa"],
                                 "auc": d["scores"]["auc"],
                                 "n_test_subjects": len(d["test_subjects"]),
                                 "mb": round(os.path.getsize(dst_probe) / 1048576, 2)})
                print("  fold %d  [already saved]" % fi, flush=True)
                del d
                continue
            rng = np.random.RandomState(100 + fi)
            tr_all = list(tr_all)
            rng.shuffle(tr_all)
            nv = max(10, len(tr_all) // 9)
            va, tr = tr_all[:nv], tr_all[nv:]

            model = C.train_fold(tr, va, "concat", [], [], seed=SEED,
                                 bypass=True, temporal=FINAL["temporal"])

            # score this fold so the checkpoint can be matched to its own row
            Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
            for s in tr:
                y = C.DATA[s][2]; pi[y[0]] += 1
                for x, z in zip(y[:-1], y[1:]):
                    Am[x, z] += 1
            A_log = np.log(Am / Am.sum(1, keepdims=True)); pi_log = np.log(pi / pi.sum())
            yt, ph, ay, ap = [], [], [], []
            for s in te:
                sp, apn = C.subj_infer(model, s, [], [])
                y = C.DATA[s][2]
                yt.append(y); ph.append(C.hmm(A_log, pi_log, np.log(sp + C.EPS)))
                ay.append(C.DATA[s][3]); ap.append(apn)
            yt, ph = np.concatenate(yt), np.concatenate(ph)
            ay, ap = np.concatenate(ay), np.concatenate(ap)
            acc = float(accuracy_score(yt, ph))
            kap = float(cohen_kappa_score(yt, ph))
            auc = float(roc_auc_score(ay, ap))

            n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
            dst = os.path.join(OUT, "mmnet_seed%d_fold%d.pt" % (SEED, fi))
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": dict(FINAL, fusion="concat", bypass=True,
                               n_eeg=h.dim, n_card=h.n_card, seed=SEED),
                "fold": fi,
                "test_subjects": sorted(int(s) for s in te),
                "scores": {"acc": acc, "kappa": kap, "auc": auc},
                "trainable_parameters": n_par,
            }, dst)
            manifest.append({"fold": fi, "file": os.path.basename(dst),
                             "acc": acc, "kappa": kap, "auc": auc,
                             "n_test_subjects": len(te),
                             "mb": round(os.path.getsize(dst) / 1048576, 2)})
            print("  fold %d  acc %.4f  kappa %.4f  auc %.4f  -> %s (%.1f MB)"
                  % (fi, acc, kap, auc, os.path.basename(dst), manifest[-1]["mb"]),
                  flush=True)

            # the windowed views are large and the loop otherwise holds ten folds
            # worth of them; fold 1 ran out of memory before this was here.
            del model, yt, ph, ay, ap, A_log, pi_log
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total = sum(m["mb"] for m in manifest)
    json.dump({"seed": SEED, "config": FINAL, "folds": manifest,
               "mean_acc": float(np.mean([m["acc"] for m in manifest])),
               "mean_auc": float(np.mean([m["auc"] for m in manifest])),
               "total_mb": round(total, 1)},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("\n%d checkpoints, %.0f MB total, mean acc %.4f  [%.1f min]"
          % (len(manifest), total, np.mean([m["acc"] for m in manifest]),
             (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
