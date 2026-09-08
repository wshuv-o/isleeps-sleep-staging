"""External validation of the final model on a held-out public corpus.

Train on all 99 iSLEEPS patients, evaluate on ISRUC or Sleep-EDF with no
adaptation of any kind. This is the protocol the manuscript already describes;
what changed is the model, so the numbers have to be regenerated.

Run this on a machine that HAS data/isruc_mm/ and data/sleep_edf_mm/. The 5080
box does not -- see EXTERNAL_VALIDATION_INSTRUCTIONS.md.

  KMP_DUPLICATE_LIB_OK=TRUE python run_external_validation.py \
      --corpus isruc --out results/revision/runs/final/external_isruc.json

Nothing here is fitted to the external corpus. The train/eval split is by corpus,
not by fold, so there is no tuning surface at all -- which is the point of the
table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import arms                     # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,        # noqa: E402
                             roc_auc_score, average_precision_score)

CORPORA = {"isruc": "isruc_mm", "sleep_edf": "sleep_edf_mm"}
FINAL = dict(arm="A+labram", cardio="raw_cnn", temporal="lstm",
             hidden=256, drop=0.3, lr=3e-4, wd=1e-4)


def load_external(name, labram_dir):
    """External corpus in the same shape mmnet_core.DATA uses.

    Each external .npz must provide Feeg (188), Fcard (14) and y; the LaBraM
    embedding for the same epochs comes from labram_dir. Everything is z-scored
    per recording, exactly as load_data does for iSLEEPS, so the external inputs
    reach the model on the same scale the training data did.
    """
    import glob
    base = os.path.join(REPO, "data", CORPORA[name])
    out = {}
    for f in sorted(glob.glob(os.path.join(base, "*.npz"))):
        rec = os.path.basename(f)[:-4]
        d = np.load(f)
        fe = np.nan_to_num(d["Feeg"]).astype(np.float32)
        fc = np.nan_to_num(d["Fcard"]).astype(np.float32)
        fe = (fe - fe.mean(0)) / (fe.std(0) + 1e-6)
        fc = (fc - fc.mean(0)) / (fc.std(0) + 1e-6)
        emb_path = os.path.join(labram_dir, rec + ".npz")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                "missing LaBraM embedding for %s -- run build_labram_cache.py first "
                "(see EXTERNAL_VALIDATION_INSTRUCTIONS.md step 3)" % rec)
        e = np.load(emb_path)["emb"].astype(np.float32)
        if len(e) != len(fe):
            raise ValueError("%s: %d embedded epochs vs %d feature rows" % (rec, len(e), len(fe)))
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)
        y = d["y"].astype(np.int64)
        a = d["apnea"].astype(np.int64) if "apnea" in d else np.zeros(len(y), np.int64)
        out[rec] = (np.concatenate([fe, e], axis=1), fc, y, a)
    if not out:
        raise FileNotFoundError("no recordings under %s" % base)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--labram-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    labram_dir = a.labram_dir or os.path.join(REPO, "data", "labram_ext", a.corpus)

    ext = load_external(a.corpus, labram_dir)
    print("%s: %d recordings, %d epochs"
          % (a.corpus, len(ext), sum(len(v[2]) for v in ext.values())), flush=True)

    t0 = time.time()
    # Train once on every iSLEEPS patient. No folds: the held-out set is the
    # external corpus, so a fold split would only shrink the training data.
    with arms.arm(C, FINAL["arm"], cardio=FINAL["cardio"]) as h:
        tr = list(C.SUBS)
        va = tr[:10]                      # small internal split, for early stopping only
        model = C.train_fold(tr, va, "concat", [], [], seed=a.seed,
                             bypass=True, temporal=FINAL["temporal"])
        saved = dict(C.DATA)
        try:
            C.DATA = {**saved, **ext}     # so subj_infer can address external records
            yt, yp, at, ascore = [], [], [], []
            for rec in ext:
                sp, apn = C.subj_infer(model, rec, [], [])
                yt.append(ext[rec][2]); yp.append(sp.argmax(1))
                at.append(ext[rec][3]); ascore.append(apn)
        finally:
            C.DATA = saved

    yt, yp = np.concatenate(yt), np.concatenate(yp)
    at, ascore = np.concatenate(at), np.concatenate(ascore)
    res = dict(corpus=a.corpus, seed=a.seed, n_recordings=len(ext), n_epochs=int(len(yt)),
               acc=float(accuracy_score(yt, yp)),
               mf1=float(f1_score(yt, yp, average="macro", zero_division=0)),
               kappa=float(cohen_kappa_score(yt, yp)),
               pcf=[float(v) for v in f1_score(yt, yp, average=None,
                                               labels=range(5), zero_division=0)],
               minutes=(time.time() - t0) / 60)
    if len(np.unique(at)) > 1:
        res["auc"] = float(roc_auc_score(at, ascore))
        res["ap"] = float(average_precision_score(at, ascore))
    else:
        res["auc"] = res["ap"] = None
        print("  (no respiratory annotations in this corpus)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\n%s: acc %.4f  mF1 %.4f  kappa %.4f  AUC %s  [%.1f min]"
          % (a.corpus, res["acc"], res["mf1"], res["kappa"],
             ("%.4f" % res["auc"]) if res["auc"] else "n/a", res["minutes"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
