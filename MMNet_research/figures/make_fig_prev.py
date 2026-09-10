"""Figure 1: sleep-disordered-breathing burden across the cohort.

Regenerated in the editorial style. Three changes beyond colour and type:

  * the in-image title is gone. It repeated the caption word for word, and a
    figure in a paper is already labelled.
  * the "median 13%" text over the plot is gone. The caption states the median,
    so the label was a second copy of a number the reader has just read.
  * the median line stays, in slate rather than black, because it is a reference
    marker rather than an annotation and the histogram is hard to read against
    without it.

  KMP_DUPLICATE_LIB_OK=TRUE python make_fig_prev.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import editorial_style as ES            # noqa: E402

plt = ES.apply()

REPO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPO)
FE = os.path.join(ROOT, "data", "mm_features")
OUT = os.path.join(HERE, "fig_mm_apnea_prev.pdf")
DUP = {28}

prev = []
for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")),
                key=lambda p: int(os.path.basename(p)[2:-4])):
    sid = int(os.path.basename(f)[2:-4])
    if sid in DUP:
        continue
    prev.append(100.0 * np.load(f)["apnea"].mean())
prev = np.array(prev)

fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.hist(prev, bins=20, color=ES.GREEN, edgecolor="white", linewidth=0.6)
ax.axvline(np.median(prev), ls="--", color=ES.SLATE, lw=1.0)
ax.set_xlabel("% of epochs with a scored respiratory event")
ax.set_ylabel("number of patients")
ax.grid(axis="x", visible=False)          # vertical rules fight the bars
ax.margins(x=0.01)
fig.savefig(OUT)
plt.close(fig)

print("patients: %d   median %.1f%%   max %.1f%%   above 5%%: %d"
      % (len(prev), np.median(prev), prev.max(), int((prev > 5).sum())))
print("wrote", OUT)
