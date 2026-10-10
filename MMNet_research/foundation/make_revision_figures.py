"""Figures for results that no table reports, from saved predictions only.

    fig_rev_ahi           per-patient AHI, MM-Net estimate vs annotation: iSLEEPS (10-fold test) and
                          ISRUC-I (trained on iSLEEPS only, 7-channel final architecture)
    fig_rev_tsne          t-SNE of fold-0 test epochs: frozen LaBraM vs fine-tuned + aligned EEG/EOG encoder

Outputs: MMNet_research/paper/figures/revision/<name>.pdf and .png

    MMNET_FEATURES=aligned/mm_features MMNET_MULTIMODAL=aligned/multimodal python make_revision_figures.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MMNET_FEATURES", "aligned/mm_features")
os.environ.setdefault("MMNET_MULTIMODAL", "aligned/multimodal")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import mmnet_core as C

DATA = os.path.join(REPO, "data")
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
OUT = os.path.join(REPO, "MMNet_research", "paper", "figures", "revision")
STAGES = ["W", "N1", "N2", "N3", "REM"]
S = (42, 1, 7)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("wrote", name, flush=True)


def ahi_pairs():
    est = json.load(open(os.path.join(RUNS, "ahi_rescore.json")))["mmnet"]
    isl = [(float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%s.npz" % s))["ahi_ann"]), v) for s, v in est.items()]
    d = np.load(os.path.join(RUNS, "external", "isruc_I_final7_probs.npz"))
    k = json.load(open(os.path.join(RUNS, "external", "isruc_I_frozen.json")))["ahi_scale_k"]
    isr = []
    for f in sorted(glob.glob("H:/isleeps_ext/isruc/I/*.npz")):
        t = os.path.basename(f)[:-4]; z = np.load(f)
        y, nev = z["y"], z["n_events"]
        st = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["%s|%d" % (t, sd)] for sd in S], 0) + C.EPS))
        ap = np.mean([d["A:%s|%d" % (t, sd)] for sd in S], 0)
        isr.append((nev[y != 0].sum() / max((y != 0).sum() * 30 / 3600, 1e-6),
                    k * ap[st != 0].sum() / max((st != 0).sum() * 30 / 3600, 0.5)))
    return [("iSLEEPS (stroke), 10-fold test, n=%d" % len(isl), np.array(isl)),
            ("ISRUC-I, trained on iSLEEPS only, n=%d" % len(isr), np.array(isr))]


def ahi_scatter(pairs):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (title, p) in zip(axs, pairs):
        lim = max(p.max() * 1.05, 35)
        for lo, hi, c in ((0, 5, "#eafaf1"), (5, 15, "#fef9e7"), (15, 30, "#fdebd0"), (30, lim, "#fadbd8")):
            ax.axvspan(lo, hi, color=c, zorder=0)
        ax.plot([0, lim], [0, lim], "--", color="0.5", lw=0.8)
        ax.scatter(p[:, 0], p[:, 1], s=12, color="#2c3e50", alpha=0.8, zorder=2)
        r = spearmanr(p[:, 1], p[:, 0]).correlation
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Annotated AHI (events/h)"); ax.set_ylabel("MM-Net AHI (events/h)")
        ax.set_title("%s\nSpearman rho = %.2f, MAE = %.1f" % (title, r, np.abs(p[:, 1] - p[:, 0]).mean()), fontsize=8.5)
    for x0, lab in ((0.5, "normal"), (5.5, "mild"), (15.5, "moderate"), (30.5, "severe")):
        axs[0].text(x0, axs[0].get_ylim()[1] * 0.97, lab, fontsize=6.5, color="0.35", va="top")
    save(fig, "fig_rev_ahi")


# ----------------------------------------------------------------------------- 6
def tsne():
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    te = list(C.FOLDS[0][1])
    rng = np.random.RandomState(0)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    cols = np.array(["#f1c40f", "#7fb3d5", "#2471a3", "#1b2631", "#e74c3c"])
    for ax, (folder, title) in zip(axs, (("labram", "Frozen LaBraM"), ("labramfta_eog_f0", "Fine-tuned + aligned EEG/EOG encoder"))):
        E, Y = [], []
        for s in te:
            e = np.load(os.path.join(DATA, "cbramod_emb", folder, "SN%d.npz" % s))["emb"].astype(np.float32)
            E.append((e - e.mean(0)) / (e.std(0) + 1e-6)); Y.append(C.DATA[s][2])
        E, Y = np.concatenate(E), np.concatenate(Y)
        idx = rng.choice(len(E), min(4000, len(E)), replace=False)
        Z = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(E[idx])
        sil = silhouette_score(E[idx], Y[idx])
        for c in range(5):
            m = Y[idx] == c
            ax.scatter(Z[m, 0], Z[m, 1], s=2, color=cols[c], label=STAGES[c], alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s\nsilhouette by stage = %.3f" % (title, sil), fontsize=8.5)
    axs[1].legend(markerscale=5, frameon=False, fontsize=7, loc="lower right")
    fig.text(0.5, -0.02, "Fold-0 held-out patients (%d), 4000 random epochs, colour = scored stage" % len(te), ha="center", fontsize=8)
    save(fig, "fig_rev_tsne")


if __name__ == "__main__":
    which = sys.argv[1:] or ["ahi", "tsne"]
    for w in which:
        if w == "ahi":
            ahi_scatter(ahi_pairs())
        else:
            globals()[w]()
