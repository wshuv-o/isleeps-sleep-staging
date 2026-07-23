"""
sleepedf_preprocess.py — Sleep-EDF Expanded (sleep-cassette) EDF + Hypnogram -> npz.

Healthy-cohort SOURCE domain for the E2 / Pillar (i) domain-gap analysis. Harmonised
to a single central EEG channel (Fpz-Cz) to pair with iSLEEPS C4:M1, 100 Hz, 30 s
epochs, AASM labels (stage 3+4 -> N3). Wake is cropped to +/-30 min around sleep so
the huge lights-on wake periods don't dominate (standard Sleep-EDF practice).

Output per recording: data/sleep_edf_proc/<rec>.npz  with
  x [n, 1, 3000] float32 (uV), y [n] int64 (W0 N1 1 N2 2 N3 3 R4), channel, sfreq.

  python preprocess/sleepedf_preprocess.py
"""
import os
import glob
import warnings
import numpy as np
import mne

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sleep_edf")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sleep_edf_proc")
CHANNEL = "EEG Fpz-Cz"      # central-ish derivation, paired with iSLEEPS C4:M1
SFREQ = 100
WIN = 30.0
SAMPLES = int(WIN * SFREQ)
CROP_WAKE_MIN = 30

# Sleep-EDF hypnogram annotation -> our label ids (3 and 4 merge to N3 per AASM)
STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}


def pair_files(raw_dir):
    psg = sorted(glob.glob(os.path.join(raw_dir, "SC*-PSG.edf")))
    pairs = []
    for p in psg:
        pref = os.path.basename(p)[:6]
        hyps = glob.glob(os.path.join(raw_dir, f"{pref}*-Hypnogram.edf"))
        if hyps:
            pairs.append((p, hyps[0]))
    return pairs


def process(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, include=[CHANNEL])
    raw.resample(SFREQ)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)

    events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP, chunk_duration=WIN)
    if len(events) == 0:
        raise RuntimeError("no stage events")
    tmax = WIN - 1.0 / SFREQ
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0.0, tmax=tmax,
                        baseline=None, preload=True, on_missing="ignore")
    y = epochs.events[:, 2].astype(np.int64)

    # crop excess wake: keep CROP_WAKE_MIN around the sleep period
    sleep = np.where(y != 0)[0]
    if len(sleep) > 0:
        k = int(CROP_WAKE_MIN * 60 / WIN)
        lo = max(0, sleep[0] - k); hi = min(len(y), sleep[-1] + k + 1)
        epochs = epochs[lo:hi]; y = y[lo:hi]

    x = epochs.get_data(copy=False).astype(np.float32) * 1e6   # [n,1,3000] uV
    assert x.shape[1] == 1 and x.shape[2] == SAMPLES, x.shape
    return x, y


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = pair_files(RAW)
    print(f"found {len(pairs)} PSG/Hypnogram pairs")
    summ = []
    for psg, hyp in pairs:
        rec = os.path.basename(psg)[:6]
        try:
            x, y = process(psg, hyp)
        except Exception as e:
            print(f"[FAIL] {rec}: {type(e).__name__}: {e}"); continue
        np.savez_compressed(os.path.join(OUT, f"{rec}.npz"), x=x, y=y,
                            channel=CHANNEL, sfreq=SFREQ)
        b = np.bincount(y, minlength=5); summ.append((rec, len(y), b))
        print(f"[ok] {rec}: {len(y):4d} ep  W{b[0]} N1:{b[1]} N2:{b[2]} N3:{b[3]} R{b[4]}")
    if summ:
        tot = sum(n for _, n, _ in summ); agg = np.sum([b for _, _, b in summ], 0)
        print(f"\n=== {len(summ)} recordings, {tot} epochs ===")
        for i, nm in enumerate(["W", "N1", "N2", "N3", "R"]):
            print(f"  {nm:3s} {agg[i]:6d} ({100*agg[i]/tot:4.1f}%)")


if __name__ == "__main__":
    main()
