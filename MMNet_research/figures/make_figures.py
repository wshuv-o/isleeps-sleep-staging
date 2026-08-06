"""
make_figures.py -- real figures from saved MM-Net artifacts (no retraining). Every figure
also writes its source data (results/revision/figdata/) so it can be regenerated.

  fig_embedding_tsne   : PCA + t-SNE of the model's learned per-epoch embeddings,
                         coloured by sleep stage and by apnea (DREAM Fig.8 style, real).
  fig_confusion        : 5-class row-normalised confusion matrix from pooled test predictions.
  fig_ablation_heatmap : modality removed x {staging accuracy, respiratory AUC}.
  fig_ahi_scatter      : predicted per-patient burden vs clinical AHI, coloured by severity.
  fig_event_type_auc   : respiratory AUC by event type with/without the effort stream.

  KMP_DUPLICATE_LIB_OK=TRUE python revision/make_figures.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmnet_repro import REV, RUNS
import analysis as AN

FIG = os.path.join(REV, "figures"); FD = os.path.join(REV, "figdata")
os.makedirs(FIG, exist_ok=True); os.makedirs(FD, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25})
CLS = ["W", "N1", "N2", "N3", "R"]
STAGE_C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def fig_embedding_tsne(max_pts=8000, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    z = np.load(os.path.join(REV, "embeddings.npz"))
    h, stage, apnea = z["h"], z["stage"], z["apnea"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(h), size=min(max_pts, len(h)), replace=False)
    h, stage, apnea = h[idx], stage[idx], apnea[idx]
    pca = PCA(n_components=2).fit_transform(h)
    ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, max_iter=1000).fit_transform(
        PCA(n_components=30).fit_transform(h))
    np.savez_compressed(os.path.join(FD, "embedding_tsne.npz"), pca=pca, tsne=ts, stage=stage, apnea=apnea)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.6))
    for col, (emb, ttl) in enumerate([(pca, "PCA"), (ts, "t-SNE")]):
        # by stage
        for k in range(5):
            m = stage == k
            ax[0, col].scatter(emb[m, 0], emb[m, 1], s=4, c=STAGE_C[k], label=CLS[k], alpha=0.6, linewidths=0)
        ax[0, col].set_title(f"{ttl}: by sleep stage", fontsize=9.5)
        # by apnea
        for v, c, lab in [(0, "#4C72B0", "no event"), (1, "#C44E52", "apnea/hypopnea")]:
            m = apnea == v
            ax[1, col].scatter(emb[m, 0], emb[m, 1], s=4, c=c, label=lab, alpha=0.55, linewidths=0)
        ax[1, col].set_title(f"{ttl}: by respiratory event", fontsize=9.5)
    ax[0, 0].legend(markerscale=2.5, fontsize=7, loc="best", ncol=2)
    ax[1, 0].legend(markerscale=2.5, fontsize=7, loc="best")
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MM-Net learned per-epoch embeddings (held-out test epochs)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_embedding_tsne.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_embedding_tsne.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote fig_embedding_tsne")


def fig_confusion():
    from sklearn.metrics import confusion_matrix
    z = np.load(os.path.join(REV, "predictions.npz"))
    cm = confusion_matrix(z["y_true"], z["y_pred"], labels=range(5))
    cmn = cm / cm.sum(1, keepdims=True)
    np.savez_compressed(os.path.join(FD, "confusion.npz"), cm=cm, cm_norm=cmn)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLS); ax.set_yticks(range(5)); ax.set_yticklabels(CLS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
    ax.set_title("Staging confusion matrix (row-normalised)", fontsize=10)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_confusion.pdf"), bbox_inches="tight"); plt.close(fig)
    print("wrote fig_confusion")


def fig_ablation_heatmap():
    def g(name, k): return json.load(open(os.path.join(RUNS, f"{name}.json")))[k][0]
    rows = [("full (both streams)", "headline_concat"), ("- SpO2", "loo_spo2"),
            ("- effort", "loo_effort"), ("- pulse/HRV", "loo_pulse_hrv"), ("- ECG", "loo_ecg"),
            ("- airflow", "loo_airflow"), ("- EOG", "loo_eog"), ("- EMG", "loo_emg"),
            ("- all cardio", "neural_only")]
    rows = [(lab, n) for lab, n in rows if os.path.exists(os.path.join(RUNS, f"{n}.json"))]
    M = np.array([[g(n, "acc"), g(n, "apnea_auc")] for _, n in rows])
    np.savez_compressed(os.path.join(FD, "ablation_heatmap.npz"), matrix=M, rows=[r[0] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 4.4), gridspec_kw={"wspace": 0.5})
    for c, (ttl, cmap, lo, hi) in enumerate([("Staging acc", "Blues", 0.68, 0.73),
                                             ("Respiratory AUC", "Reds", 0.62, 0.72)]):
        im = ax[c].imshow(M[:, [c]], cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        for i in range(len(rows)):
            ax[c].text(0, i, f"{M[i,c]:.3f}", ha="center", va="center", fontsize=8)
        ax[c].set_xticks([0]); ax[c].set_xticklabels([ttl], fontsize=8.5)
        ax[c].set_yticks(range(len(rows))); ax[c].set_yticklabels([r[0] for r in rows], fontsize=8)
        ax[c].grid(False)
    fig.suptitle("Modality-ablation grid", fontsize=10.5)
    fig.savefig(os.path.join(FIG, "fig_ablation_heatmap.pdf"), bbox_inches="tight"); plt.close(fig)
    print("wrote fig_ablation_heatmap")


def fig_ahi_scatter():
    a = AN.ahi_validation()
    np.savez_compressed(os.path.join(FD, "ahi_scatter.npz"), burden=a["burden"], ahi=a["ahi"],
                        severity=a["severity"], rho=a["rho"], p=a["p"])
    sev_c = {"Normal": "#4C72B0", "Mild": "#55A868", "Moderate": "#DD8452", "Severe": "#C44E52"}
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for s in ["Normal", "Mild", "Moderate", "Severe"]:
        m = [i for i, x in enumerate(a["severity"]) if x == s]
        ax.scatter(np.array(a["ahi"])[m], np.array(a["burden"])[m], s=28, c=sev_c[s], label=s,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Clinical AHI (events/hour)"); ax.set_ylabel("Predicted event burden")
    ax.legend(fontsize=7.5, title=f"Spearman rho={a['rho']:.2f} (p={a['p']:.1e})", title_fontsize=7.5)
    ax.set_title("Predicted burden vs clinical AHI", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ahi_scatter.pdf"), bbox_inches="tight"); plt.close(fig)
    print(f"wrote fig_ahi_scatter (rho={a['rho']:.3f}, p={a['p']:.3g})")


def fig_event_type_auc():
    full = AN.per_event_type_auc("headline_concat")
    noeff = AN.per_event_type_auc("loo_effort") if os.path.exists(os.path.join(RUNS, "loo_effort.json")) else {}
    types = [t for t in ["hypopnea", "obstructive", "central"] if t in full]
    np.savez_compressed(os.path.join(FD, "event_type_auc.npz"),
                        types=types, full=[full[t]["auc"] for t in types],
                        no_effort=[noeff.get(t, {}).get("auc", np.nan) for t in types])
    x = np.arange(len(types)); fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.bar(x - 0.2, [full[t]["auc"] for t in types], 0.4, label="full model", color="#4C72B0")
    if noeff:
        ax.bar(x + 0.2, [noeff.get(t, {}).get("auc", np.nan) for t in types], 0.4,
               label="effort removed", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n={full[t]['n_pos']})" for t in types], fontsize=8)
    ax.set_ylabel("Detection AUC"); ax.set_ylim(0.5, 0.9); ax.legend(fontsize=8)
    ax.set_title("Respiratory detection by event type", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_event_type_auc.pdf"), bbox_inches="tight"); plt.close(fig)
    print("wrote fig_event_type_auc:", {t: round(full[t]["auc"], 3) for t in types})


if __name__ == "__main__":
    if os.path.exists(os.path.join(REV, "embeddings.npz")):
        fig_embedding_tsne(); fig_confusion()
    for fn in (fig_ablation_heatmap, fig_ahi_scatter, fig_event_type_auc):
        try: fn()
        except Exception as e: print(f"skip {fn.__name__}: {e}")
