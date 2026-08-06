"""
sleepedf_full_preprocess.py — full Sleep-EDF cassette -> 4-channel npz for PRETRAINING.

Channels matched to iSLEEPS for transfer: [central EEG, occipital EEG, EOG, EMG]:
  Sleep-EDF: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental
  iSLEEPS  : C4:M1,      O2:M1,     E1:M2,          EMG
100 Hz, 30 s epochs, AASM labels (3+4->N3), wake cropped to +-30 min around sleep.
Output: data/sleep_edf_full_proc/<rec>.npz  (x [n,4,3000] float16, y).
"""
import os, glob, warnings
import numpy as np
import mne

warnings.simplefilter("ignore"); mne.set_log_level("ERROR")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "sleep_edf_full")
OUT = os.path.join(ROOT, "data", "sleep_edf_full_proc")
CH = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
SF, WIN, SAMP, CROP = 100, 30.0, 3000, 30
SMAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def process(psg, hyp):
    raw = mne.io.read_raw_edf(psg, preload=True, include=CH)
    raw.resample(SF); raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)
    ev, _ = mne.events_from_annotations(raw, event_id=SMAP, chunk_duration=WIN)
    if len(ev) == 0: raise RuntimeError("no events")
    ep = mne.Epochs(raw, ev, tmin=0.0, tmax=WIN - 1.0 / SF, baseline=None, preload=True, on_missing="ignore")
    y = ep.events[:, 2].astype(np.int64)
    sl = np.where(y != 0)[0]
    if len(sl):
        k = int(CROP * 60 / WIN); lo = max(0, sl[0] - k); hi = min(len(y), sl[-1] + k + 1)
        ep = ep[lo:hi]; y = y[lo:hi]
    x = ep.get_data(copy=False).astype(np.float32) * 1e6
    if x.shape[1] != 4: raise RuntimeError(f"got {x.shape[1]} ch")
    return x.astype(np.float16), y


def main():
    os.makedirs(OUT, exist_ok=True)
    psg = sorted(glob.glob(os.path.join(RAW, "SC*-PSG.edf")))
    ok = fail = 0
    for p in psg:
        pref = os.path.basename(p)[:6]; rec = os.path.basename(p)[:6]
        hyps = glob.glob(os.path.join(RAW, f"{pref}*-Hypnogram.edf"))
        if not hyps: continue
        outp = os.path.join(OUT, f"{rec}.npz")
        if os.path.exists(outp) and os.path.getsize(outp) > 1000: ok += 1; continue
        try:
            x, y = process(p, hyps[0])
        except Exception as e:
            print(f"[FAIL] {rec}: {type(e).__name__}: {e}"); fail += 1; continue
        np.savez_compressed(outp, x=x, y=y); ok += 1
        print(f"[ok] {rec}: {len(y)} ep")
    print(f"=== done ok={ok} fail={fail} -> {os.path.relpath(OUT)} ===")


if __name__ == "__main__":
    main()
