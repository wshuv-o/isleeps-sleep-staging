"""Item 8f: which Sleep-EDF Cassette recordings the healthy-sleep runs use.

Lists data/sleep_edf_proc (the recordings healthy_cv.json and the reimplementation
checks read) and reports the subjects and the nights present or absent.

  python sleepedf_inventory.py
"""
import glob
import os

import _common as C

recs = sorted(os.path.basename(p)[:6] for p in glob.glob(os.path.join(C.DATA, "sleep_edf_proc", "SC4*.npz")))
subj = sorted({int(r[3:5]) for r in recs})
missing = ["SC4%02d%d" % (s, n) for s in subj for n in (1, 2) if "SC4%02d%d" % (s, n) not in recs]
C.save("sleepedf_inventory.json", dict(n_recordings=len(recs), n_subjects=len(subj), subjects=subj,
                                       recordings=recs, absent_nights_within_subjects=missing,
                                       note="SC4132 does not exist in Sleep-EDF Expanded; SC4202 was not included"))
print(len(recs), len(subj), missing)
