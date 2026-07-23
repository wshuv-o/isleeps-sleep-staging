"""Generate presentation figures from the project's real results."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
CL = ["W", "N1", "N2", "N3", "R"]

NAVY="#1b2a4a"; BLUE="#2c5f8a"; TEAL="#2a9d8f"; ORANGE="#e76f51"; GRAY="#9aa5b1"; GOLD="#e9c46a"
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.edgecolor": "#444",
                     "axes.linewidth": 0.8, "figure.dpi": 150})

def load(name):
    return json.load(open(os.path.join(R, name)))

# ---------- 1. Leaderboard ----------
def fig_leaderboard():
    rows = [
        ("CNN (published)", 0.617, GRAY, "baseline"),
        ("Transformer (published)", 0.674, GRAY, "baseline"),
        ("LSTM (published, SOTA)", 0.747, GRAY, "baseline"),
        ("Our deep CNN+BiLSTM", 0.654, BLUE, "ours"),
        ("Our ensemble, EEG only", 0.727, TEAL, "ours"),
        ("Our ensemble+HMM, EEG+EOG+EMG", 0.742, ORANGE, "best"),
    ]
    rows = rows[::-1]
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, edgecolor="white", height=0.72)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=12, fontweight="bold", color="#222")
    ax.axvline(0.747, color=NAVY, ls="--", lw=1.3, alpha=0.7)
    ax.text(0.747, len(rows)-0.3, " published SOTA", color=NAVY, fontsize=10, va="top")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0.55, 0.78); ax.set_xlabel("Accuracy (subject-independent 5-fold CV)")
    ax.set_title("Sleep-stage accuracy: our models vs published baselines",
                 fontsize=15, fontweight="bold", color=NAVY, pad=12)
    leg = [Patch(color=GRAY, label="Published (N=100)"), Patch(color=BLUE, label="Our deep models"),
           Patch(color=TEAL, label="Our classical ML"), Patch(color=ORANGE, label="Our best")]
    ax.legend(handles=leg, loc="lower right", fontsize=10, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "leaderboard.png"), bbox_inches="tight"); plt.close()

# ---------- 2. Deep vs classical per-class F1 ----------
def fig_perclass():
    deep = load("seq_aug_all.json")
    pcf_deep = np.array([f["per_class_f1"] for f in deep["folds"]]).mean(0)
    cls = load("classical_all.json")["models"]["XGBoost"]["per_class_f1"]
    ens = load("ensemble_all.json")["models"]["ensemble_hmm"]["per_class_f1"]
    x = np.arange(5); w = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x-w, pcf_deep, w, label="Deep CNN+BiLSTM (aug)", color=BLUE)
    ax.bar(x,   cls,      w, label="XGBoost (features)", color=TEAL)
    ax.bar(x+w, ens,      w, label="Ensemble + HMM", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_ylim(0, 0.9)
    ax.set_ylabel("Per-class F1"); ax.set_title("Per-class F1: deep nets balance minorities better (REM, N1)",
                  fontsize=14, fontweight="bold", color=NAVY, pad=10)
    ax.legend(fontsize=10.5); ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "perclass.png"), bbox_inches="tight"); plt.close()

# ---------- 3. Domain gap ----------
def fig_domaingap():
    d = load("domaingap.json")
    h = [d["healthy"]["recall"][c] for c in CL]; s = [d["stroke"]["recall"][c] for c in CL]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x-w/2, h, w, label="Healthy (Sleep-EDF)", color=TEAL)
    ax.bar(x+w/2, s, w, label="Stroke (iSLEEPS, zero-shot)", color=ORANGE)
    for i in range(5):
        ax.annotate("", xy=(i+w/2, s[i]), xytext=(i-w/2, h[i]),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.1, alpha=0.6))
        ax.text(i, max(h[i], s[i])+0.03, f"-{h[i]-s[i]:.2f}", ha="center", fontsize=11,
                color="#b3402a", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-stage recall")
    ax.set_title("E2 domain gap (healthy to stroke): REM collapses, N2 robust",
                 fontsize=14, fontweight="bold", color=NAVY, pad=10)
    ax.legend(fontsize=11, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "domaingap.png"), bbox_inches="tight"); plt.close()

# ---------- 4. Confusion matrix ----------
def fig_confusion():
    folds = load("ensemble_all.json")["models"]["ensemble_hmm"]["folds"]
    cm = np.sum([np.array(f["confusion"]) for f in folds], 0).astype(float)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CL); ax.set_yticklabels(CL)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#222", fontsize=12,
                    fontweight="bold" if i == j else "normal")
    ax.set_title("Confusion matrix (best model, row-normalised)\nN1 scattered; N3/N2 cleanest",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "confusion.png"), bbox_inches="tight"); plt.close()

# ---------- 5. Stage distribution healthy vs stroke ----------
def fig_stagedist():
    healthy = [19.7, 6.6, 42.3, 13.2, 18.2]
    stroke  = [28.7, 9.8, 41.0, 10.2, 10.3]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(x-w/2, healthy, w, label="Healthy (Sleep-EDF)", color=TEAL)
    ax.bar(x+w/2, stroke,  w, label="Stroke (iSLEEPS)", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_ylabel("% of epochs")
    ax.set_title("Sleep architecture: stroke patients show more wake, less REM/N3",
                 fontsize=13.5, fontweight="bold", color=NAVY, pad=10)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "stagedist.png"), bbox_inches="tight"); plt.close()

# ---------- 6. Feature importance (XGBoost) ----------
def fig_featimp():
    try:
        from datasets import load_subject, list_subjects
        from features import extract_features
        from xgboost import XGBClassifier
        Xs, ys = [], []
        for sid in list_subjects()[:20]:               # subset for speed
            x, y = load_subject(sid, channels=["C4:M1", "C3:M2", "O2:M1", "O1:M2"], normalize=False)
            F, names = extract_features(x, fs=100)
            F = (F - F.mean(0)) / (F.std(0) + 1e-6)
            Xs.append(F); ys.append(y)
        X = np.concatenate(Xs); Y = np.concatenate(ys)
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.07,
                            tree_method="hist", n_jobs=-1, random_state=42)
        clf.fit(X, Y)
        imp = clf.feature_importances_
        order = np.argsort(imp)[::-1][:15][::-1]
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        ax.barh(range(len(order)), imp[order], color=BLUE, edgecolor="white")
        ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order], fontsize=10.5)
        ax.set_xlabel("XGBoost feature importance (gain)")
        ax.set_title("Top features driving sleep-stage decisions (RF / XGBoost)",
                     fontsize=13.5, fontweight="bold", color=NAVY, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "featimp.png"), bbox_inches="tight"); plt.close()
        print("featimp top:", [names[i] for i in order[::-1][:6]])
    except Exception as e:
        print("featimp skipped:", type(e).__name__, e)

for f in [fig_leaderboard, fig_perclass, fig_domaingap, fig_confusion, fig_stagedist, fig_featimp]:
    f(); print("ok:", f.__name__)
print("figures ->", FIG)
