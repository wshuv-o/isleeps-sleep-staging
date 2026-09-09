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
import ablation_remap           # noqa: E402
import arms                     # noqa: E402
import sweep                    # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,        # noqa: E402
                             roc_auc_score, average_precision_score)

CORPORA = {"isruc": "isruc_mm", "sleep_edf": "sleep_edf_mm"}
FINAL = dict(arm="A+labram", cardio="raw_cnn", temporal="lstm",
             hidden=256, drop=0.3, lr=3e-4, wd=1e-4)


def load_external(name, labram_dir, n_card):
    """External corpus in the same shape mmnet_core.DATA uses.

    Each external .npz must provide Feeg (188), Fcard and y; the LaBraM
    embedding for the same epochs comes from labram_dir. Everything is z-scored
    per recording, exactly as load_data does for iSLEEPS, so the external inputs
    reach the model on the same scale the training data did.

    n_card is the cardio width the ARM expects, which is no longer 14: the final
    model's branch takes the raw 7 x 750 tensor. Sleep-EDF has no SpO2, ECG,
    effort or pulse channel at all, so there is nothing to resample into that
    shape and the branch is fed zeros. Left unhandled this raised a reshape
    error inside CardioCNN; handled silently it would have been worse.
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
        if fc.shape[1] != n_card:
            if np.any(fc):
                raise ValueError(
                    "%s: cardio is %d-d but the arm expects %d-d, and the corpus "
                    "does carry cardio data -- it must be rebuilt in the arm's "
                    "representation rather than zeroed" % (rec, fc.shape[1], n_card))
            fc = np.zeros((len(fe), n_card), np.float32)
        else:
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
    # When the external corpus has no cardiorespiratory channels, the branch is
    # fed zeros at test time while the model was trained on real signal. That is
    # a train/test mismatch, not an ablation, and BatchNorm inside CardioCNN
    # makes it worth checking rather than assuming. This flag also zeroes the
    # branch during training, so the two agree, and the pair of runs brackets
    # the effect instead of leaving a referee to ask about it.
    ap.add_argument("--match-train-cardio", action="store_true",
                    help="zero the cardio branch during training too")
    a = ap.parse_args()
    labram_dir = a.labram_dir or os.path.join(REPO, "data", "labram_ext", a.corpus)

    t0 = time.time()
    # Train once on every iSLEEPS patient. No folds: the held-out set is the
    # external corpus, so a fold split would only shrink the training data.
    # sweep.config, not arms.arm: the arm alone leaves hidden=128 and lr=1e-3,
    # which is NOT the final model and would have been reported as if it were.
    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode="concat") as h:
        ext = load_external(a.corpus, labram_dir, h.n_card)
        print("%s: %d recordings, %d epochs  (eeg %d-d, cardio %d-d)"
              % (a.corpus, len(ext), sum(len(v[2]) for v in ext.values()),
                 h.dim, h.n_card), flush=True)
        tr = list(C.SUBS)
        va = tr[:10]                      # small internal split, for early stopping only
        drop = ["all"] if a.match_train_cardio else []
        restore = ablation_remap.apply(C, n_eeg_total=h.dim) if drop else None
        try:
            model = C.train_fold(tr, va, "concat", [], drop, seed=a.seed,
                                 bypass=True, temporal=FINAL["temporal"])
        finally:
            if restore:
                restore()
        # subj_infer here is not mmnet_core's function but a recompiled copy with
        # the width literals substituted (arms._recompile_with_width), and it was
        # compiled against ns = dict(C.__dict__) -- a SNAPSHOT. Rebinding C.DATA
        # therefore never reaches it, which is why the first attempt raised
        # KeyError: 'SC4001'. The snapshot holds the same dict OBJECT that
        # arms.arm installed, so mutating that object in place is visible.
        saved = dict(C.DATA)
        try:
            C.DATA.update(ext)
            yt, yp, at, ascore = [], [], [], []
            for rec in ext:
                sp, apn = C.subj_infer(model, rec, [], [])
                yt.append(ext[rec][2]); yp.append(sp.argmax(1))
                at.append(ext[rec][3]); ascore.append(apn)
        finally:
            C.DATA.clear(); C.DATA.update(saved)

    yt, yp = np.concatenate(yt), np.concatenate(yp)
    at, ascore = np.concatenate(at), np.concatenate(ascore)
    res = dict(corpus=a.corpus, seed=a.seed, n_recordings=len(ext), n_epochs=int(len(yt)),
               match_train_cardio=bool(a.match_train_cardio),
               eeg_dim=int(h.dim), card_dim=int(h.n_card),
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
