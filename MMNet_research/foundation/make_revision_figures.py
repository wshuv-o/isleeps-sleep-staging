"""Figures for the revision results (fine-tuned encoder, AHI, ISRUC), from saved predictions only.

    fig_rev_components    single-model accuracy per fold, published MM-Net -> final, one step at a time
    fig_rev_baselines     every reimplemented baseline vs the final model, fold by fold
    fig_rev_ahi           per-patient AHI, MM-Net estimate vs annotation: iSLEEPS (10-fold test) and
                          ISRUC-I (trained on iSLEEPS only, 7-channel final architecture)
    fig_rev_bland_altman  AHI agreement for the same two panels
    fig_rev_confusion     row-normalised confusion, published vs final (3-seed, train-fold HMM, test folds)
    fig_rev_tsne          t-SNE of fold-0 test epochs: frozen LaBraM vs fine-tuned + aligned EEG/EOG encoder
    fig_rev_night         the test patient with the median staging accuracy: hypnogram + respiratory track

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
from compare_baselines import BASELINES, baseline_folds, mmnet_folds

DATA = os.path.join(REPO, "data")
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
ENS = os.path.join(RUNS, "seed_ensemble")
OUT = os.path.join(REPO, "MMNet_research", "paper", "figures", "revision")
FINAL = "base-nocw-cross-ftlae-p8-al"
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


def decode(cfg, fi, s, seeds=S):
    d = np.load(os.path.join(ENS, "%s_fold%d.npz" % (cfg, fi)))
    return C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in seeds], 0) + C.EPS))


# ----------------------------------------------------------------------------- 1
def components():
    steps = [("base", "Published\nMM-Net"), ("base-nocw", "No class\nweights"), ("base-nocw-cross", "Cross\nfusion"),
             ("base-nocw-cross-ftl", "+ fine-tuned\nLaBraM"), ("base-nocw-cross-ftla", "+ cross-fit\nalignment"),
             ("base-nocw-cross-ftlae", "+ EOG in\nencoder"), ("base-nocw-cross-ftlae-p8", "+ nasal\npressure"),
             (FINAL, "+ timeline\nfix (final)")]
    acc = np.array([mmnet_folds(c)[0] for c, _ in steps])          # [steps, 10]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(steps))
    for f in range(acc.shape[1]):
        ax.plot(x, acc[:, f], color="0.8", lw=0.7, zorder=1)
    ax.scatter(np.repeat(x, acc.shape[1]), acc.ravel(), s=9, color="0.55", zorder=2)
    ax.plot(x, acc.mean(1), "-o", color="#c0392b", lw=2, ms=5, zorder=3, label="mean of 10 folds")
    for i, m in enumerate(acc.mean(1)):
        ax.annotate("%.3f" % m, (i, m), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in steps], fontsize=7.5)
    ax.set_ylabel("Staging accuracy (single model)")
    ax.set_title("iSLEEPS, 10 patient-independent folds: from the published model to the final architecture", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_rev_components")


# ----------------------------------------------------------------------------- 2
def baselines():
    rows = []
    for name, f in BASELINES.items():
        try:
            v, _ = baseline_folds(f)
        except Exception:
            v = None
        if v is not None:
            rows.append((name, v))
    final = mmnet_folds(FINAL)[0]
    rows.sort(key=lambda r: r[1].mean())
    rows.append(("MM-Net (final)", final))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    jit = np.random.RandomState(0).uniform(-0.12, 0.12, 10)
    for i, (name, v) in enumerate(rows):
        fin = name == "MM-Net (final)"
        c = "#c0392b" if fin else "0.5"
        ax.scatter(v, i + jit, s=11, color=c, alpha=0.85, zorder=2, lw=0)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color="#c0392b" if fin else "k", lw=2, zorder=3)
        lab = "%.3f" % v.mean() if fin else "%.3f   final higher in %d/10 folds" % (v.mean(), int((v < final).sum()))
        ax.text(1.02, i, lab, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.5, color=c if fin else "k")
    ax.axvline(final.mean(), color="#c0392b", lw=0.8, ls=":", zorder=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.get_yticklabels()[-1].set_color("#c0392b")
    ax.set_xlabel("Staging accuracy: test folds (dots) and mean of 10 folds (bar)")
    ax.set_title("iSLEEPS, same patients and folds for every model", fontsize=9)
    save(fig, "fig_rev_baselines")


# ----------------------------------------------------------------------------- 3, 4
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


def bland_altman(pairs):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (title, p) in zip(axs, pairs):
        m, dlt = p.mean(1), p[:, 1] - p[:, 0]
        b, sd = dlt.mean(), dlt.std(ddof=1)
        ax.scatter(m, dlt, s=12, color="#2c3e50", alpha=0.8)
        for y, ls, lab in ((b, "-", "bias %.1f" % b), (b + 1.96 * sd, "--", "+1.96 SD %.1f" % (b + 1.96 * sd)),
                           (b - 1.96 * sd, "--", "-1.96 SD %.1f" % (b - 1.96 * sd))):
            ax.axhline(y, ls=ls, color="#c0392b", lw=0.9)
            ax.text(ax.get_xlim()[1] if False else m.max(), y, " " + lab, fontsize=7, va="bottom", ha="right", color="#c0392b")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("Mean of annotated and predicted AHI"); ax.set_title(title, fontsize=8.5)
    axs[0].set_ylabel("Predicted - annotated AHI")
    save(fig, "fig_rev_bland_altman")


# ----------------------------------------------------------------------------- 5
def confusion():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, (cfg, title) in zip(axs, (("base", "Published MM-Net"), (FINAL, "Final architecture"))):
        M = np.zeros((5, 5))
        for fi, (_, te) in enumerate(C.FOLDS):
            for s in te:
                yp = decode(cfg, fi, s)
                for a_, b_ in zip(C.DATA[s][2], yp):
                    M[a_, b_] += 1
        acc = np.trace(M) / M.sum()
        R = M / M.sum(1, keepdims=True)
        ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, "%.2f" % R[i, j], ha="center", va="center", fontsize=8, color="white" if R[i, j] > 0.55 else "black")
        ax.set_xticks(range(5)); ax.set_xticklabels(STAGES); ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
        ax.set_xlabel("Predicted")
        if cfg == "base":
            ax.set_ylabel("Scored")
        ax.set_title("%s (3-seed, acc %.3f)" % (title, acc), fontsize=9)
    save(fig, "fig_rev_confusion")


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


# ----------------------------------------------------------------------------- 7
def night():
    accs = []
    for fi, (_, te) in enumerate(C.FOLDS):
        for s in te:
            accs.append((float(np.mean(decode(FINAL, fi, s) == C.DATA[s][2])), fi, s))
    accs.sort()
    acc, fi, s = accs[len(accs) // 2]
    d = np.load(os.path.join(ENS, "%s_fold%d.npz" % (FINAL, fi)))
    yp = decode(FINAL, fi, s)
    ap = np.mean([d["A:SN%d|%d" % (s, sd)] for sd in S], 0)
    yt, at = C.DATA[s][2], C.DATA[s][3]
    order = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}                  # W top, then REM, N1, N2, N3
    t = np.arange(len(yt)) * 30 / 3600
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.8]})
    for ax, y, title in ((axs[0], yt, "Scored"), (axs[1], yp, "MM-Net (final)")):
        ax.step(t, [order[v] for v in y], where="post", color="#2c3e50", lw=0.9)
        ax.set_yticks(range(5)); ax.set_yticklabels(["N3", "N2", "N1", "REM", "W"], fontsize=7)
        ax.set_ylabel(title, fontsize=8)
    axs[2].fill_between(t, 0, at, step="post", color="#f5b7b1", label="scored event")
    axs[2].plot(t, ap, color="#c0392b", lw=0.8, label="predicted probability")
    axs[2].set_ylim(0, 1.05); axs[2].set_ylabel("Resp. event", fontsize=8); axs[2].set_xlabel("Hours from lights off")
    axs[2].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    axs[0].set_title("SN%d, held-out (fold %d): median staging accuracy of all 99 test patients (%.3f)" % (s, fi, acc), fontsize=9)
    save(fig, "fig_rev_night")


if __name__ == "__main__":
    which = sys.argv[1:] or ["components", "baselines", "ahi", "confusion", "tsne", "night"]
    for w in which:
        if w == "ahi":
            p = ahi_pairs(); ahi_scatter(p); bland_altman(p)
        else:
            globals()[w]()
