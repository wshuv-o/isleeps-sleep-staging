"""
build_npz.py — iSLEEPS EDF + annotation .xlsx  ->  per-subject .npz

Faithful single-pass reproduction of the official two-stage pipeline
(iSLEEPS_preprocess_v2: staging_preprocess.py -> numpy_subjects.py), but it
writes ONE .npz per subject directly instead of tens of thousands of
per-epoch files. Reuses the official `StagingPreprocess` class for the actual
signal handling (EDF read, annotation parse, resample, 30 s windowing).

Output per subject (data/processed/SN<k>.npz):
    x        : float32 [n_epochs, n_channels, 3000]   (uV, 100 Hz, 30 s)
    y        : int64   [n_epochs]                      (W0 N1 1 N2 2 N3 3 R4)
    channels : list[str]                               (fixed common montage)
    subject  : int
    sfreq    : 100

Common EEG montage present in ALL 40 subjects: C4:M1 C3:M2 O2:M1 O1:M2.
Frontal F3/F4 exist in only 28/40 subjects and are intentionally excluded so
every subject has identical channel dimensions for subject-independent CV.
"""
import os
import sys
import glob
import argparse
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mne  # noqa: E402
from staging_preprocess import StagingPreprocess  # noqa: E402
from channel_mapping import CHANNEL_MAPPING, LABEL_MAPPING  # noqa: E402

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")

COMMON_EEG = ["C4:M1", "C3:M2", "O2:M1", "O1:M2"]  # fixed order, all subjects
# Some subjects reference EEG to ear electrodes (A1/A2) rather than mastoids (M1/M2).
# A1/A2 and M1/M2 are near-identical locations, so harmonise the names (as the official
# numpy_subjects.py does) — otherwise ~36/60 of the Figshare cohort is silently dropped.
REF_RENAME = {"C4:A1": "C4:M1", "C3:A2": "C3:M2", "O2:A1": "O2:M1", "O1:A2": "O1:M2"}
WINDOW_SIZE = 30.0
SFREQ = 100
SAMPLES = int(WINDOW_SIZE * SFREQ)  # 3000


def process_subject(edf_path, ann_path, channels):
    ds = StagingPreprocess(
        edf_path, ann_path, CHANNEL_MAPPING, modality=["eeg"],
        window_size=WINDOW_SIZE, sfreq=SFREQ, preload=True,
    )
    epochs = StagingPreprocess.create_windows(
        ds.raw, ds.description, window_size=WINDOW_SIZE,
        window_stride=WINDOW_SIZE, label_mapping=LABEL_MAPPING,
        drop_last=True, drop_bad=True,
    )
    # harmonise A1/A2-referenced names to M1/M2 (only when target name is absent)
    rmap = {a: b for a, b in REF_RENAME.items() if a in epochs.ch_names and b not in epochs.ch_names}
    if rmap:
        epochs.rename_channels(rmap)
    # restrict to the fixed common montage, in a fixed order
    have = epochs.ch_names
    missing = [c for c in channels if c not in have]
    if missing:
        raise RuntimeError(f"missing channels {missing}; have {have}")
    epochs.pick(channels).reorder_channels(channels)

    x = epochs.get_data(copy=False).astype(np.float32) * 1e6  # -> uV, [n, ch, 3000]
    y = epochs.metadata["target"].to_numpy().astype(np.int64)
    assert x.shape[1] == len(channels) and x.shape[2] == SAMPLES, x.shape
    assert x.shape[0] == y.shape[0]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "data", "zenodo"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "processed"))
    ap.add_argument("--subjects", default="", help="comma list e.g. 1,2,3 (default all)")
    ap.add_argument("--channels", default=",".join(COMMON_EEG))
    args = ap.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    os.makedirs(args.out, exist_ok=True)

    edfs = sorted(glob.glob(os.path.join(args.raw, "*.edf")),
                  key=lambda p: int(os.path.basename(p)[2:-4]))
    if args.subjects:
        want = {int(s) for s in args.subjects.split(",")}
        edfs = [p for p in edfs if int(os.path.basename(p)[2:-4]) in want]

    summary = []
    for edf in edfs:
        sn = os.path.basename(edf)[:-4]
        sid = int(sn[2:])
        ann = os.path.join(args.raw, f"{sn}.xlsx")
        if not os.path.exists(ann):
            print(f"[skip] {sn}: no annotation file")
            continue
        try:
            x, y = process_subject(edf, ann, channels)
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {e}")
            continue
        out = os.path.join(args.out, f"{sn}.npz")
        np.savez_compressed(out, x=x, y=y, channels=np.array(channels),
                            subject=sid, sfreq=SFREQ)
        binc = np.bincount(y, minlength=5)
        summary.append((sn, len(y), binc))
        print(f"[ok] {sn}: {len(y):4d} epochs  "
              f"W{binc[0]} N1:{binc[1]} N2:{binc[2]} N3:{binc[3]} R{binc[4]}  "
              f"-> {os.path.relpath(out)}")

    if summary:
        tot = sum(n for _, n, _ in summary)
        agg = np.sum([b for _, _, b in summary], axis=0)
        print(f"\n=== {len(summary)} subjects, {tot} epochs total ===")
        names = ["W", "N1", "N2", "N3", "R"]
        for i, nm in enumerate(names):
            print(f"  {nm:3s} {agg[i]:6d}  ({100*agg[i]/tot:4.1f}%)")


if __name__ == "__main__":
    main()
