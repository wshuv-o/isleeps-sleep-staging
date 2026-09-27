"""Rebuild data/multimodal and data/mm_features for SN2, SN13 and SN17.

Figshare does not carry these three recordings; they come from the Zenodo release
(10.5281/zenodo.14873844), downloaded to data/zenodo/. build_multimodal.py and
extract_mm_features.py still resolve paths from before the repository was
restructured (MMNet_research/data/...), so this runs their own functions with the
paths pointed at the real data directory. Nothing in either module is changed.

Needs pyedflib (the testenv interpreter has it; the Python 3.14 one cannot build it):

  D:/EEG-TransNet/testenv/python.exe MMNet_research/preprocessing/rebuild_missing_subjects.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")
sys.path.insert(0, HERE)
SIDS = ["SN2", "SN13", "SN17"]

try:
    import build_multimodal as B      # noqa: E402
except ImportError:
    B = None
if B is not None:
    B.DS, B.P7, B.OUT = DATA, os.path.join(DATA, "processed7"), os.path.join(DATA, "multimodal")
MMDIR = os.path.join(DATA, "multimodal")


def locate(sid):
    edf = glob.glob(os.path.join(DATA, "zenodo", sid + ".edf"))
    xls = [p for p in glob.glob(os.path.join(DATA, "zenodo", sid + ".xlsx"))
           + glob.glob(os.path.join(DATA, "Dataset", "**", sid + ".xlsx"), recursive=True)
           if not os.path.basename(p).startswith("~$")]
    return edf[0], xls[0], os.path.join(DATA, "processed7", sid + ".npz")


def main(stage):
    if stage in ("multimodal", "all"):
        build_multimodal()
    if stage in ("features", "all"):
        build_features()


def build_multimodal():
    for sid in SIDS:
        dst = os.path.join(MMDIR, sid + ".npz")
        if os.path.exists(dst):
            print("[skip] multimodal", sid); continue
        edf, xlsx, p7 = locate(sid)
        res = B.process(sid, edf, xlsx, p7, verbose=True)
        np.savez_compressed(dst, **res)
        print("[ok] multimodal %s: %d epochs, cardio channels %s, apnea+ %.1f%%"
              % (sid, len(res["y"]), res["cvalid"].astype(int).tolist(),
                 100 * res["apnea"].mean()), flush=True)



def build_features():
    from features_v2 import extract_features_v2
    from cardio_features import cardio_feats
    out = os.path.join(DATA, "mm_features")
    for sid in SIDS:
        dst = os.path.join(out, sid + ".npz")
        if os.path.exists(dst):
            print("[skip] features", sid); continue
        d = np.load(os.path.join(MMDIR, sid + ".npz"), allow_pickle=True)
        Feeg, _ = extract_features_v2(d["eeg"].astype(np.float32), fs=100)
        Fcard = cardio_feats(d["card"].astype(np.float32))
        # the same fields and casts extract_mm_features.py writes
        np.savez_compressed(dst, Feeg=np.nan_to_num(Feeg).astype(np.float32),
                            Fcard=Fcard[:len(d["y"])], y=d["y"].astype(np.int64),
                            apnea=d["apnea"].astype(np.int64), cvalid=d["cvalid"])
        print("[ok] features %s: Feeg %s Fcard %s" % (sid, Feeg.shape, Fcard.shape), flush=True)


if __name__ == "__main__":
    # the multimodal stage needs pyedflib (testenv); the feature stage runs in the
    # interpreter that built the other 96 patients' features
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
