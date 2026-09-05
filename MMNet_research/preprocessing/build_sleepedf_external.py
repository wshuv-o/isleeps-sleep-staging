"""Sleep-EDF Expanded -> MM-Net feature arrays, for external validation.

MM-Net was trained on iSLEEPS with a seven-channel montage (4 EEG, 2 EOG, 1 EMG)
plus 14 cardiorespiratory features. Sleep-EDF Cassette provides EEG Fpz-Cz,
EEG Pz-Oz, EOG horizontal and EMG submental. This script maps what exists onto
the layout the trained feature extractor expects, so the model can be run
zero-shot with no dataset-specific tuning.

Channel mapping and its honest caveats
--------------------------------------
  c0 <- EEG Fpz-Cz      c1 <- EEG Pz-Oz      c2 <- EEG Fpz-Cz     c3 <- EEG Pz-Oz
  c4 <- EOG horizontal  c5 <- EOG horizontal c6 <- EMG submental

The two EEG derivations are duplicated to fill four slots rather than zero-padded,
because zeroed channels produce degenerate spectral features the model never saw in
training, whereas duplication preserves realistic feature distributions. This is
still a montage mismatch -- Sleep-EDF is frontal/occipital, iSLEEPS is
central/occipital -- and that mismatch is a confound on any performance drop,
which the accompanying notebook states rather than hides.

The 14 cardiorespiratory features are set to zero: Sleep-EDF has no SpO2, ECG,
effort or pulse channels. This is not a confound for the staging head, which our
ablation showed is unaffected by removing the entire cardiorespiratory stream
(Wilcoxon p = 0.91). The respiratory head is not evaluated here, because Sleep-EDF
carries no respiratory-event annotations.

Usage:  python build_sleepedf_external.py
"""
import glob
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mne  # noqa: E402
from features_v2 import extract_features_v2  # noqa: E402

mne.set_log_level("ERROR")

RAW = os.path.join(REPO, "data", "sleep_edf")
OUT = os.path.join(REPO, "data", "sleep_edf_mm")
SFREQ, WIN = 100, 30.0
SAMPLES = int(SFREQ * WIN)
CROP_WAKE_MIN = 30

WANT = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
# index into WANT for each of the seven channels the extractor expects
LAYOUT = [0, 1, 0, 1, 2, 2, 3]

STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
             "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def pair_files(raw_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "SC*-PSG.edf"))):
        h = glob.glob(os.path.join(raw_dir, os.path.basename(p)[:6] + "*-Hypnogram.edf"))
        if h:
            out.append((p, h[0]))
    return out


def process(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, include=WANT)
    missing = [c for c in WANT if c not in raw.ch_names]
    if missing:
        raise RuntimeError("missing channels: %s" % missing)
    raw.reorder_channels(WANT)
    raw.resample(SFREQ)
    raw.set_annotations(mne.read_annotations(hyp_path), emit_warning=False)

    events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP, chunk_duration=WIN)
    if len(events) == 0:
        raise RuntimeError("no stage events")
    ep = mne.Epochs(raw, events, event_id=None, tmin=0.0, tmax=WIN - 1.0 / SFREQ,
                    baseline=None, preload=True, on_missing="ignore")
    y = ep.events[:, 2].astype(np.int64)

    sleep = np.where(y != 0)[0]          # crop the long lights-on wake periods
    if len(sleep):
        k = int(CROP_WAKE_MIN * 60 / WIN)
        lo, hi = max(0, sleep[0] - k), min(len(y), sleep[-1] + k + 1)
        ep, y = ep[lo:hi], y[lo:hi]

    x4 = ep.get_data(copy=False).astype(np.float32) * 1e6      # [n,4,3000] uV
    if x4.shape[2] != SAMPLES:
        raise RuntimeError("epoch length %d != %d" % (x4.shape[2], SAMPLES))
    x7 = x4[:, LAYOUT, :]                                       # [n,7,3000]
    Feeg, _ = extract_features_v2(x7)                           # [n,188]
    Fcard = np.zeros((len(y), 14), np.float32)                  # absent in Sleep-EDF
    return np.nan_to_num(Feeg).astype(np.float32), Fcard, y


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = pair_files(RAW)
    print("found %d PSG/Hypnogram pairs in %s\n" % (len(pairs), RAW))
    tot, agg, ok = 0, np.zeros(5, np.int64), 0
    for psg, hyp in pairs:
        rec = os.path.basename(psg)[:6]
        dst = os.path.join(OUT, rec + ".npz")
        if os.path.exists(dst):
            print("[skip] %s" % rec)
            continue
        try:
            Feeg, Fcard, y = process(psg, hyp)
        except Exception as e:
            print("[FAIL] %s: %s: %s" % (rec, type(e).__name__, e))
            continue
        np.savez_compressed(dst, Feeg=Feeg, Fcard=Fcard, y=y)
        b = np.bincount(y, minlength=5)
        tot += len(y); agg += b; ok += 1
        print("[ok]   %s  %4d epochs  W%d N1:%d N2:%d N3:%d R%d"
              % (rec, len(y), b[0], b[1], b[2], b[3], b[4]))
    if tot:
        print("\n=== %d recordings, %d epochs ===" % (ok, tot))
        for i, nm in enumerate(["W", "N1", "N2", "N3", "R"]):
            print("  %-3s %6d (%4.1f%%)" % (nm, agg[i], 100 * agg[i] / tot))


if __name__ == "__main__":
    main()
