"""The control: does the healthy-trained checkpoint work on HEALTHY sleep?

A stager scoring 0.308 on iSLEEPS is evidence of a domain gap only if the same
checkpoint, through the same preprocessing, scores properly on the data it was
trained for. Otherwise "the injured brain breaks the model" and "the input
pipeline is wrong" are indistinguishable.

Sleep-EDF is preprocessed here to match what the checkpoint expects, using the
channel pairing train_transfer.py documents:
  Fpz-Cz, Pz-Oz, EOG horizontal, EMG submental  ->  the 4 input channels

Sleep-EDF hypnograms are AASM-style with Sleep stage 3 and 4 scored separately
(R&K); both map to N3, as is standard for this corpus. '?' and 'Movement' are
dropped. Long leading/trailing wake blocks are trimmed to 30 min, the usual
convention -- without it Sleep-EDF is ~60% wake and accuracy is meaningless.
"""
import glob
import os
import sys

import numpy as np
import mne
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

REPO = r"d:\proc\isleeps-sleep-staging"
sys.path.insert(0, os.path.join(REPO, "HAGNet_research", "train", "legacy_models"))
from staging_seq import StagingSeqNet          # noqa: E402

RAW = os.path.join(REPO, "data", "sleep_edf_raw")
CKPT = os.path.join(REPO, "HAGNet_research", "model", "pretrained_sedf.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, SF, EPOCH = 20, 100, 30
STAGES = ["W", "N1", "N2", "N3", "R"]
mne.set_log_level("ERROR")

WANT = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
LMAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def load_recording(psg_path, hyp_path, trim_wake_min=30):
    raw = mne.io.read_raw_edf(psg_path, preload=True, stim_channel=None)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)
    missing = [c for c in WANT if c not in raw.ch_names]
    if missing:
        return None, None, "missing channels %s" % missing
    raw.pick(WANT)
    if int(raw.info["sfreq"]) != SF:
        raw.resample(SF)
    sig = raw.get_data() * 1e6                      # volts -> microvolts
    n_ep = sig.shape[1] // (SF * EPOCH)
    x = sig[:, :n_ep * SF * EPOCH].reshape(len(WANT), n_ep, SF * EPOCH).transpose(1, 0, 2)

    y = np.full(n_ep, -1, np.int64)
    for a in ann:
        lab = LMAP.get(a["description"])
        if lab is None:
            continue
        s = int(a["onset"] // EPOCH)
        e = int((a["onset"] + a["duration"]) // EPOCH)
        y[max(0, s):min(n_ep, e)] = lab
    keep = y >= 0
    x, y = x[keep], y[keep]
    if len(y) == 0:
        return None, None, "no scored epochs"

    # trim long wake tails, the standard convention for this corpus
    nz = np.flatnonzero(y != 0)
    if len(nz):
        pad = trim_wake_min * 2
        lo, hi = max(0, nz[0] - pad), min(len(y), nz[-1] + pad + 1)
        x, y = x[lo:hi], y[lo:hi]
    return x.astype(np.float32), y, None


@torch.no_grad()
def predict(model, x):
    mu = x.mean((0, 2), keepdims=True)
    sd = x.std((0, 2), keepdims=True) + 1e-6
    x = (x - mu) / sd
    n = len(x)
    pad = (-n) % L
    if pad:
        x = np.concatenate([x, np.zeros((pad, x.shape[1], x.shape[2]), np.float32)])
    seq = x.reshape(-1, L, x.shape[1], x.shape[2])
    out = []
    for i in range(0, len(seq), 16):
        b = torch.tensor(seq[i:i + 16], dtype=torch.float32, device=DEV)
        out.append(model(b).float().cpu().numpy())
    return np.concatenate(out).reshape(-1, 5)[:n].argmax(1)


def main():
    m = StagingSeqNet(in_ch=4).to(DEV)
    m.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True), strict=False)
    m.eval()

    psgs = sorted(glob.glob(os.path.join(RAW, "*-PSG.edf")))
    hyps = {os.path.basename(p)[:6]: p for p in glob.glob(os.path.join(RAW, "*-Hypnogram.edf"))}
    yt_all, yp_all, per = [], [], []
    for p in psgs:
        stem = os.path.basename(p)[:6]
        if stem not in hyps:
            continue
        x, y, err = load_recording(p, hyps[stem])
        if err:
            print("  skip %s: %s" % (stem, err)); continue
        pred = predict(m, x)
        per.append((stem, float((pred == y).mean()), len(y)))
        yt_all.append(y); yp_all.append(pred)
        print("  %-8s %5d epochs   acc %.4f" % (stem, len(y), per[-1][1]), flush=True)
    if not yt_all:
        print("no recordings scored"); return
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    accs = np.array([a for _, a, _ in per])
    print()
    print("CONTROL: same checkpoint, same preprocessing, on HEALTHY Sleep-EDF")
    print("  recordings          : %d" % len(per))
    print("  epochs              : %d" % len(yt))
    print("  pooled accuracy     : %.4f" % accuracy_score(yt, yp))
    print("  macro-F1            : %.4f" % f1_score(yt, yp, average="macro", zero_division=0))
    print("  Cohen kappa         : %.4f" % cohen_kappa_score(yt, yp))
    print("  per-recording acc   : %.4f +- %.4f" % (accs.mean(), accs.std(ddof=1)))
    print("  majority-class rate : %.4f" % (np.bincount(yt, minlength=5).max() / len(yt)))
    print("  predicted dist      :",
          {STAGES[i]: round(float(v), 3) for i, v in enumerate(np.bincount(yp, minlength=5) / len(yp))})
    print("  true dist           :",
          {STAGES[i]: round(float(v), 3) for i, v in enumerate(np.bincount(yt, minlength=5) / len(yt))})


if __name__ == "__main__":
    main()
