"""Sleep-EDF Expanded EDF + Hypnogram -> data/sleep_edf_proc/<rec>.npz, three channels.

The healthy-control scripts (run_healthy_baselines, run_utime, run_healthy_extra)
read this cache. It follows HAGNet_research/train/sleepedf/sleepedf_preprocess.py
exactly -- 100 Hz, 30 s epochs, stage 3+4 -> N3, wake cropped to 30 min either side
of sleep -- but keeps three derivations instead of one, so the two-channel baselines
can run on the same epochs:

  x[:, 0]  EEG Fpz-Cz      (channel 0, what every single-channel script reads)
  x[:, 1]  EEG Pz-Oz
  x[:, 2]  EOG horizontal

  python MMNet_research/domaingap/build_sleepedf_proc.py
"""
import glob
import os
import warnings

import mne
import numpy as np

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(REPO, "data", "sleep_edf")
OUT = os.path.join(REPO, "data", "sleep_edf_proc")
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
SFREQ, WIN, CROP_WAKE_MIN = 100, 30.0, 30
STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
             "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def process(psg, hyp):
    raw = mne.io.read_raw_edf(psg, preload=True, include=CHANNELS)
    raw.reorder_channels(CHANNELS)
    raw.resample(SFREQ)
    raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)
    events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP, chunk_duration=WIN)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0.0, tmax=WIN - 1.0 / SFREQ,
                        baseline=None, preload=True, on_missing="ignore")
    y = epochs.events[:, 2].astype(np.int64)
    sleep = np.where(y != 0)[0]
    if len(sleep):
        k = int(CROP_WAKE_MIN * 60 / WIN)
        lo, hi = max(0, sleep[0] - k), min(len(y), sleep[-1] + k + 1)
        epochs, y = epochs[lo:hi], y[lo:hi]
    x = epochs.get_data(copy=False).astype(np.float32) * 1e6     # [n, 3, 3000] uV
    assert x.shape[1:] == (3, 3000), x.shape
    return x, y


def main():
    os.makedirs(OUT, exist_ok=True)
    for psg in sorted(glob.glob(os.path.join(RAW, "SC*-PSG.edf"))):
        rec = os.path.basename(psg)[:6]
        hyp = glob.glob(os.path.join(RAW, rec + "*-Hypnogram.edf"))
        if not hyp:
            print("[skip] %s: no hypnogram" % rec); continue
        try:
            x, y = process(psg, hyp[0])
        except Exception as e:
            print("[FAIL] %s: %s: %s" % (rec, type(e).__name__, e)); continue
        np.savez_compressed(os.path.join(OUT, rec + ".npz"), x=x, y=y,
                            channels=np.array(CHANNELS), sfreq=SFREQ)
        b = np.bincount(y, minlength=5)
        print("[ok] %s: %4d ep  W%d N1:%d N2:%d N3:%d R%d" % (rec, len(y), *b), flush=True)


if __name__ == "__main__":
    main()
