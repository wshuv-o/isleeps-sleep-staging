"""Figure 5: what the respiratory read-out does at a clinically chosen threshold.

Panel (a) used to be an epoch-level ROC. That curve is real, and recomputing it
from the pooled predictions reproduces its AUC to four decimals, but it earns
little: a reader can picture a 0.78 ROC from the number alone, and nobody
operates a detector at a point chosen off an ROC.

It is replaced by the event-level operating characteristic, which is the axis a
clinician reads: how many scored respiratory events the model finds, against how
many false alarms per hour it raises. That is not in any table, and it shows
something the epoch-level view cannot.

The curve is non-monotonic. Dropping the threshold from 0.5 to 0.2 gains 31
points of event sensitivity while the false-alarm rate falls slightly, because
contiguous flagged epochs merge into fewer, longer runs and the event count drops
even as the flagged fraction rises. A detector tuned on epoch precision would
miss that entirely.

Panel (b), precision-recall, is unchanged. It is the evidence for describing the
read-out as screening-grade rather than scoring-grade.

  KMP_DUPLICATE_LIB_OK=TRUE python make_fig_rocpr.py
"""
import json
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
RUNS = os.path.join(REPO, "results", "revision", "runs")
OUT = os.path.join(HERE, "fig_roc_pr.pdf")

ev = json.load(open(os.path.join(RUNS, "event_level_metrics.json")))["threshold_sweep"]
ev = sorted(ev, key=lambda r: r["th"])
th = np.array([r["th"] for r in ev])
sens = np.array([r["event_sens"] for r in ev])
fa = np.array([r["fa_per_hour"] for r in ev])
flag = np.array([r["flagged_frac"] for r in ev])

cur = np.load(os.path.join(RUNS, "final", "curves_seed42.npz"))
prec, rec = cur["precision"], cur["recall"]
AP, PREVALENCE = 0.419, 0.157

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.9))

# ---- (a) event-level operating characteristic --------------------------
# marker area carries the flagged fraction, without which this panel would
# imply the low thresholds are strictly better; they are not, they flag most
# of the night
ax1.plot(fa, sens, "-", color=ES.BLUE, lw=1.4, zorder=2)
ax1.scatter(fa, sens, s=30 + 260 * flag, color=ES.BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.7, zorder=3)
for x, y, t, fr in zip(fa, sens, th, flag):
    ax1.annotate("%.1f  (%.0f%%)" % (t, 100 * fr), (x, y),
                 textcoords="offset points", xytext=(9, -2),
                 fontsize=7.5, color=ES.SLATE)
ax1.set_xlabel("false alarms per hour")
ax1.set_ylabel("event sensitivity")
ax1.set_title("(a) event-level operating characteristic")
ax1.set_ylim(0, 1)
ax1.set_xlim(1.0, 3.3)

# ---- (b) precision-recall, unchanged -----------------------------------
ax2.plot(rec, prec, color=ES.RED, lw=1.4, zorder=3)
ax2.axhline(PREVALENCE, ls="--", color=ES.SLATE, lw=0.9)
ax2.annotate("no skill (%.3f)" % PREVALENCE, (0.98, PREVALENCE + 0.02),
             ha="right", fontsize=8, color=ES.SLATE)
ax2.set_xlabel("recall (sensitivity)")
ax2.set_ylabel("precision")
ax2.set_title("(b) precision--recall   AP = %.3f" % AP)
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 1)

fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)

print("event sweep: %d thresholds, sensitivity %.3f to %.3f, "
      "false alarms %.2f to %.2f per hour"
      % (len(th), sens.min(), sens.max(), fa.min(), fa.max()))
print("wrote", OUT)
