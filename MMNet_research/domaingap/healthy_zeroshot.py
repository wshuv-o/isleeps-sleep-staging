"""Run a HEALTHY-TRAINED sleep stager on iSLEEPS, zero-shot.

The manuscript claims deep models built for healthy sleep collapse on the injured
brain, and cites a domain-gap number. This produces that number independently:
take a stager whose weights were fitted ONLY on Sleep-EDF (healthy adults), apply
it to iSLEEPS without any adaptation or fine-tuning, and score it against the
clinical hypnograms.

Channel mapping is the one train_transfer.py documents for exactly this pairing:
  Sleep-EDF : Fpz-Cz, Pz-Oz, EOG horizontal, EMG submental
  iSLEEPS   : C4:M1,  O2:M1, E1:M2,          EMG      (processed7 columns 0,2,4,6)

Per-recording normalisation matches train_transfer.py's _norm_store: mean/std over
(epoch, time) per channel, so the input distribution is the one the checkpoint was
trained against rather than raw microvolts.
"""
import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             confusion_matrix)

REPO = r"d:\proc\isleeps-sleep-staging"
sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "train", "legacy_models"))
from staging_seq import StagingSeqNet          # noqa: E402

CKPT = os.path.join(REPO, "HAGNet_research", "model", "pretrained_sedf.pt")
P7 = os.path.join(REPO, "data", "processed7")
ISL_COLS = [0, 2, 4, 6]
L = 20
DUP = {28}
STAGES = ["W", "N1", "N2", "N3", "R"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    m = StagingSeqNet(in_ch=4).to(DEV)
    sd = torch.load(CKPT, map_location=DEV, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("checkpoint: %d tensors | missing=%d unexpected=%d"
          % (len(sd), len(missing), len(unexpected)))
    if missing:
        print("   missing   :", list(missing)[:6])
    if unexpected:
        print("   unexpected:", list(unexpected)[:6])
    m.eval()
    return m


@torch.no_grad()
def predict(model, x):
    """x [n,4,3000] normalised -> predicted labels [n]."""
    n = len(x)
    pad = (-n) % L
    if pad:
        x = np.concatenate([x, np.zeros((pad, x.shape[1], x.shape[2]), np.float32)])
    seq = x.reshape(-1, L, x.shape[1], x.shape[2])
    out = []
    for i in range(0, len(seq), 16):
        b = torch.tensor(seq[i:i + 16], dtype=torch.float32, device=DEV)
        out.append(model(b).float().cpu().numpy())
    logits = np.concatenate(out).reshape(-1, 5)[:n]
    return logits.argmax(1), logits


def main():
    model = load_model()
    files = sorted(glob.glob(os.path.join(P7, "SN*.npz")),
                   key=lambda p: int(os.path.basename(p)[2:-4]))
    yt_all, yp_all, per_subj = [], [], []
    for f in files:
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUP:
            continue
        d = np.load(f)
        x = d["x"][:, ISL_COLS, :].astype(np.float32)
        mu = x.mean((0, 2), keepdims=True)
        sd_ = x.std((0, 2), keepdims=True) + 1e-6
        x = (x - mu) / sd_                       # matches train_transfer._norm_store
        y = d["y"].astype(np.int64)
        p, _ = predict(model, x)
        per_subj.append((sid, float((p == y).mean())))
        yt_all.append(y); yp_all.append(p)
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)

    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average="macro", zero_division=0)
    kap = cohen_kappa_score(yt, yp)
    per = np.array([a for _, a in per_subj])
    print()
    print("HEALTHY-TRAINED (Sleep-EDF) MODEL, ZERO-SHOT ON iSLEEPS")
    print("  subjects            : %d" % len(per_subj))
    print("  epochs              : %d" % len(yt))
    print("  pooled accuracy     : %.4f" % acc)
    print("  macro-F1            : %.4f" % mf1)
    print("  Cohen kappa         : %.4f" % kap)
    print("  per-subject accuracy: %.4f +- %.4f   (min %.3f, max %.3f)"
          % (per.mean(), per.std(ddof=1), per.min(), per.max()))
    print("  majority-class rate : %.4f" % (np.bincount(yt, minlength=5).max() / len(yt)))
    print()
    f1s = f1_score(yt, yp, average=None, labels=range(5), zero_division=0)
    rec = confusion_matrix(yt, yp, labels=range(5), normalize="true").diagonal()
    print("  %-4s %8s %8s %10s" % ("stage", "F1", "recall", "support"))
    sup = np.bincount(yt, minlength=5)
    for i, s in enumerate(STAGES):
        print("  %-4s %8.3f %8.3f %10d" % (s, f1s[i], rec[i], sup[i]))
    print()
    print("  predicted-class distribution:",
          {STAGES[i]: round(float(v), 3) for i, v in
           enumerate(np.bincount(yp, minlength=5) / len(yp))})
    print("  true-class distribution     :",
          {STAGES[i]: round(float(v), 3) for i, v in enumerate(sup / len(yt))})


if __name__ == "__main__":
    main()
