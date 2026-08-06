"""Generate the multimodal paper's result figures from the saved CV json + caches."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
CLS = ["W", "N1", "N2", "N3", "R"]
summ = json.load(open(os.path.join(RES, "mm_final_cv.json")))

# ---- Fig 1: ablation ladder -- staging acc vs apnea AUC per variant ----
order = ["eeg_only", "concat", "cross", "cross_noflow"]
lab = {"eeg_only": "EEG only", "concat": "EEG+cardio\n(concat)", "cross": "EEG+cardio\n(cross-attn)",
       "cross_noflow": "EEG+cardio\n(no airflow)"}
stg = [summ[v]["acc_hmm"] for v in order]; stg_e = [summ[v]["acc_hmm_std"] for v in order]
auc = [summ[v]["apnea_auc"] for v in order]; auc_e = [summ[v]["apnea_auc_std"] for v in order]
x = np.arange(len(order))
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
ax2 = ax1.twinx()
b1 = ax1.bar(x - 0.2, stg, 0.4, yerr=stg_e, capsize=3, color="#4C72B0", label="Staging acc")
b2 = ax2.bar(x + 0.2, auc, 0.4, yerr=auc_e, capsize=3, color="#C44E52", label="Apnea AUC")
ax1.set_ylim(0.60, 0.78); ax2.set_ylim(0.60, 0.78)
ax1.set_ylabel("Staging accuracy (+HMM)", color="#4C72B0")
ax2.set_ylabel("Apnea detection AUC", color="#C44E52")
ax1.set_xticks(x); ax1.set_xticklabels([lab[v] for v in order], fontsize=7.5)
ax1.tick_params(axis="y", colors="#4C72B0"); ax2.tick_params(axis="y", colors="#C44E52")
ax2.grid(False)
ax1.set_title("Cardiorespiratory fusion: no staging gain, clear apnea gain", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_mm_ablation.pdf")); plt.close(fig)
print("wrote fig_mm_ablation.pdf")

# ---- Fig 2: per-class staging F1, multimodal vs EEG-only ----
pc_eeg = summ["eeg_only"]["pcf_hmm"]; pc_mm = summ["concat"]["pcf_hmm"]
fig, ax = plt.subplots(figsize=(4.6, 2.8))
xx = np.arange(5)
ax.bar(xx - 0.2, pc_eeg, 0.4, label="EEG only", color="#8899AA")
ax.bar(xx + 0.2, pc_mm, 0.4, label="EEG + cardio", color="#4C72B0")
ax.set_xticks(xx); ax.set_xticklabels(CLS); ax.set_ylabel("per-class F1 (+HMM)")
ax.set_ylim(0, 0.9); ax.legend(fontsize=8, loc="upper center", ncol=2)
ax.set_title("Staging per-class F1: cardio leaves staging unchanged", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_mm_perclass.pdf")); plt.close(fig)
print("wrote fig_mm_perclass.pdf")

# ---- Fig 3: apnea prevalence histogram (cohort SDB burden) ----
FE = os.path.join(ROOT, "data", "mm_features")
prev = []
for f in sorted(glob.glob(os.path.join(FE, "SN*.npz"))):
    if int(os.path.basename(f)[2:-4]) == 28:
        continue
    d = np.load(f); prev.append(100 * d["apnea"].mean())
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.hist(prev, bins=20, color="#55A868", edgecolor="white")
ax.axvline(np.median(prev), ls="--", c="k", lw=1)
ax.text(np.median(prev) + 1, ax.get_ylim()[1] * 0.85, f"median {np.median(prev):.0f}%", fontsize=8)
ax.set_xlabel("% of epochs with a scored respiratory event")
ax.set_ylabel("number of patients")
ax.set_title("Sleep-disordered-breathing burden across the cohort", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_mm_apnea_prev.pdf")); plt.close(fig)
print("wrote fig_mm_apnea_prev.pdf")
print("done")
