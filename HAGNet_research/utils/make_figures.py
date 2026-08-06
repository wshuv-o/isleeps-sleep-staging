"""
make_figures.py -- regenerate all paper figures with a professional, hand-made look:
no titles baked into the image (captions live in LaTeX), a restrained coherent
palette, Arial, no chartjunk, vector PDF output. Plus a real-EEG teaser.
  d:/EEG-TransNet/testenv/python.exe make_figures.py
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import butter, filtfilt, hilbert
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repo root
for _p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, HERE)
from datasets import DUPLICATE_DROP  # noqa
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
FC = os.path.join(ROOT, "data", "featseq_cache")
P7 = os.path.join(ROOT, "data", "processed7")
os.makedirs(FIG, exist_ok=True)

# ---- house style -------------------------------------------------------------
INK, ACCENT = "#1b1b2f", "#c0392b"
BLUE, GREEN, TAN, GRAY = "#2c5f8a", "#3d7a5a", "#b3893f", "#9aa0a6"
STAGE_C = ["#3b4a6b", "#5b8bbf", "#c0392b", "#2e6f4e", "#c98a2b"]  # W,N1,N2,N3,R
rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": INK, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False, "legend.fontsize": 9,
})
CLS = ["W", "N1", "N2", "N3", "R"]


def subjects():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load_y():
    Y = {}
    for s in subjects():
        Y[s] = np.load(os.path.join(FC, f"SN{s}.npz"))["y"].astype(int)
    return Y


def stack(probs_file, Y):
    """concatenate a model's OOF probs and matching labels over subjects present."""
    d = np.load(os.path.join(RES, probs_file))
    yt, yp = [], []
    for k in d.files:
        s = int(k)
        if s not in Y:
            continue
        p = d[k]; y = Y[s][:len(p)]
        yt.append(y); yp.append(p.argmax(1))
    return np.concatenate(yt), np.concatenate(yp)


def per_subject(probs_file, Y):
    d = np.load(os.path.join(RES, probs_file)); acc, mf1 = [], []
    for k in d.files:
        s = int(k)
        if s not in Y:
            continue
        p = d[k]; y = Y[s][:len(p)]; pr = p.argmax(1)
        acc.append(accuracy_score(y, pr)); mf1.append(f1_score(y, pr, average="macro", zero_division=0))
    return np.array(acc), np.array(mf1)


# =========================================================== 1. TEASER (real EEG)
def fig_teaser():
    # pick a subject/epoch with a strong C4 sigma spindle in N2
    fs = 100
    def bp(x, lo, hi):
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band"); return filtfilt(b, a, x)
    # find a clean, isolated, artifact-free spindle
    best = None
    for s in subjects()[:30]:
        d = np.load(os.path.join(P7, f"SN{s}.npz"))
        x = d["x"].astype(float); y = d["y"].astype(int)
        for e in np.where(y == 2)[0][::4]:
            c4 = x[e, 0]
            if np.abs(c4).max() > 120:            # skip movement/artefact epochs
                continue
            env = np.abs(hilbert(bp(c4, 11, 16)))
            pk = env.argmax(); ratio = env.max() / (np.median(env) + 1e-6)
            dur = (env > 0.5 * env.max()).sum() / fs
            if 0.4 < dur < 1.3 and 5 * fs < pk < len(c4) - 5 * fs and ratio > 4:
                if best is None or ratio > best[0]:
                    best = (ratio, s, e)
    _, s, e = best
    d = np.load(os.path.join(P7, f"SN{s}.npz")); x = d["x"].astype(float)
    c4, c3 = x[e, 0], x[e, 1]
    f4, f3 = bp(c4, 11, 16), bp(c3, 11, 16)
    env4, env3 = np.abs(hilbert(f4)), np.abs(hilbert(f3))
    pk = env4.argmax(); w = int(0.85 * fs)         # ~1.7 s window shows the oscillation
    a0 = max(0, pk - w); a1 = min(len(c4), pk + w)
    t = (np.arange(a1 - a0) - (pk - a0)) / fs       # centred at spindle
    amp = max(np.abs(f4[a0:a1]).max(), np.abs(f3[a0:a1]).max()); OFF = amp * 1.35

    fig = plt.figure(figsize=(7.15, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.5)

    # (a) real sigma-band spindle on the homologous pair (clean oscillation + envelope)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, f4[a0:a1] + OFF, lw=0.9, color=ACCENT)
    ax.plot(t, env4[a0:a1] + OFF, lw=0.7, color=ACCENT, alpha=0.5)
    ax.plot(t, -env4[a0:a1] + OFF, lw=0.7, color=ACCENT, alpha=0.5)
    ax.plot(t, f3[a0:a1] - OFF, lw=0.9, color=BLUE)
    ax.plot(t, env3[a0:a1] - OFF, lw=0.7, color=BLUE, alpha=0.5)
    ax.plot(t, -env3[a0:a1] - OFF, lw=0.7, color=BLUE, alpha=0.5)
    ax.text(0.015, 0.97, "C4:M1", transform=ax.transAxes, va="top", fontsize=8.5, weight="bold", color=ACCENT)
    ax.text(0.015, 0.20, "C3:M2", transform=ax.transAxes, va="top", fontsize=8.5, weight="bold", color=BLUE)
    ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_ylim(-2.2 * OFF, 2.2 * OFF)
    ax.set_xlim(t[0], t[-1]); ax.spines["left"].set_visible(False)
    ax.set_title("(a) sigma spindle, homologous pair", fontsize=8.5, loc="left", pad=3)

    # (b) ipsilesional vs contralesional spindle power (real aggregate)
    li = json.load(open(os.path.join(RES, "lesion_ipsi.json")))["spindle_sigma_N2"]
    ax = fig.add_subplot(gs[0, 1])
    vals = [li["ipsi"], li["contra"]]
    ax.bar([0, 1], vals, width=0.6, color=[ACCENT, GRAY], edgecolor=INK, linewidth=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ipsi-\nlesional", "contra-\nlesional"], fontsize=8.5)
    ax.set_ylabel("sigma power (N2)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.annotate(f"$p={li['p']:.2f}$\n$n={li['n']}$", (0.5, max(vals) * 1.13),
                ha="center", fontsize=8, color=INK)
    ax.set_title("(b) hemispheric asymmetry", fontsize=8.5, loc="left", pad=3)

    # (c) the biomarker headline (reported statistic; per-subject NIHSS not re-plotted)
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.04, 0.12), 0.92, 0.76, transform=ax.transAxes,
                 fill=False, ec=INK, lw=1.0))
    ax.text(0.5, 0.74, "asymmetry index", ha="center", fontsize=9, color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.60, r"$\downarrow$", ha="center", fontsize=13, color=ACCENT, transform=ax.transAxes)
    ax.text(0.5, 0.45, "stroke severity", ha="center", fontsize=9, color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.28, r"Spearman $\rho=0.41$" + "\n" + r"$p=0.006,\ n=43$",
            ha="center", fontsize=8.5, color=ACCENT, transform=ax.transAxes, weight="bold")
    ax.set_title("(c) severity biomarker", fontsize=8.5, loc="left", pad=3)

    fig.savefig(os.path.join(FIG, "fig_teaser.pdf")); plt.close(fig)
    print("teaser: subject SN%d epoch %d" % (s, e))


# =========================================================== 2. PER-CLASS F1
def fig_perclass(Y):
    models = [("HAG-Net (prior)", "ensemble7_v2_probs.npz", ACCENT),
              ("Feature-seq BiLSTM", "featseq_probs.npz", BLUE),
              ("Graph/SSM deep", "kags_probs.npz", TAN)]
    data = []
    for name, f, c in models:
        if not os.path.exists(os.path.join(RES, f)):
            continue
        yt, yp = stack(f, Y)
        data.append((name, f1_score(yt, yp, average=None, labels=range(5), zero_division=0), c))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(5); w = 0.8 / len(data)
    for i, (name, f1, c) in enumerate(data):
        ax.bar(x + (i - (len(data) - 1) / 2) * w, f1, w, label=name, color=c, edgecolor=INK, linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CLS); ax.set_ylabel("F1 score"); ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e6e6e6", lw=0.6)
    fig.savefig(os.path.join(FIG, "fig_perclass.pdf")); plt.close(fig)


# =========================================================== 3. CONFUSION
def fig_confusion(Y):
    yt, yp = stack("ensemble7_v2_probs.npz", Y)
    cm = confusion_matrix(yt, yp, labels=range(5)).astype(float)
    cm /= cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLS); ax.set_yticklabels(CLS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, "fig_confusion.pdf")); plt.close(fig)


# =========================================================== 4. DOMAIN GAP
def fig_domaingap():
    dg = json.load(open(os.path.join(RES, "domaingap.json")))
    h = [dg["healthy"]["recall"][c] for c in CLS]; s = [dg["stroke"]["recall"][c] for c in CLS]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(5); w = 0.38
    ax.bar(x - w / 2, h, w, label="healthy (Sleep-EDF)", color=GRAY, edgecolor=INK, linewidth=0.5)
    ax.bar(x + w / 2, s, w, label="stroke (iSLEEPS)", color=ACCENT, edgecolor=INK, linewidth=0.5)
    for i in range(5):
        ax.annotate("", (i + w / 2, s[i]), (i - w / 2, h[i]),
                    arrowprops=dict(arrowstyle="->", color="#b0b0b0", lw=0.7))
    ax.set_xticks(x); ax.set_xticklabels(CLS); ax.set_ylabel("per-stage recall"); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e6e6e6", lw=0.6)
    fig.savefig(os.path.join(FIG, "fig_domaingap.pdf")); plt.close(fig)


# =========================================================== 5. PER-SUBJECT
def fig_persubject(Y):
    models = [("HAG-Net\n(prior)", "ensemble7_v2_probs.npz", ACCENT),
              ("Feature-seq\nBiLSTM", "featseq_probs.npz", BLUE),
              ("Graph/SSM\ndeep", "kags_probs.npz", TAN),
              ("Transfer", "transfer_probs.npz", GRAY)]
    accs, names, cols = [], [], []
    summary = {}
    for name, f, c in models:
        if not os.path.exists(os.path.join(RES, f)):
            continue
        a, m = per_subject(f, Y)
        accs.append(a); names.append(name); cols.append(c)
        summary[name.replace("\n", " ")] = dict(acc_mean=float(a.mean()), acc_std=float(a.std()),
                                                 acc_med=float(np.median(a)), acc_min=float(a.min()),
                                                 acc_max=float(a.max()), mf1_mean=float(m.mean()), n=int(len(a)))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bp = ax.boxplot(accs, positions=range(len(accs)), widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=1.2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(INK); patch.set_linewidth(0.7)
    rng = np.random.RandomState(0)
    for i, (a, c) in enumerate(zip(accs, cols)):
        ax.scatter(np.full_like(a, i) + rng.uniform(-0.14, 0.14, len(a)), a, s=6, color=c,
                   alpha=0.55, edgecolors="none", zorder=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("per-subject accuracy"); ax.set_ylim(0.3, 1.0)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e6e6e6", lw=0.6)
    fig.savefig(os.path.join(FIG, "fig_persubject.pdf")); plt.close(fig)
    json.dump(summary, open(os.path.join(RES, "persubject.json"), "w"), indent=2)
    print("per-subject summary:", json.dumps(summary, indent=1))


# =========================================================== 6. FEATURE IMPORTANCE
def fig_featimp(Y):
    from xgboost import XGBClassifier
    from features_v2 import extract_features_v2
    # real feature names (deterministic; run extractor on one small batch)
    xd = np.load(os.path.join(P7, f"SN{subjects()[0]}.npz"))["x"][:2].astype(float)
    _, NAMES = extract_features_v2(xd)
    nb = len(NAMES)
    # subsample rows for a fast, stable importance estimate
    subs = subjects(); X, yy = [], []
    for s in subs:
        d = np.load(os.path.join(FC, f"SN{s}.npz")); F = np.nan_to_num(d["F"]).astype(np.float32)
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        Fp = np.pad(F, ((3, 3), (0, 0)), mode="edge")
        Fc = np.concatenate([Fp[i:i + len(F)] for i in range(7)], axis=1)
        X.append(Fc); yy.append(d["y"].astype(int))
    X = np.concatenate(X); yy = np.concatenate(yy)
    rng = np.random.RandomState(0)
    if len(yy) > 40000:
        idx = rng.choice(len(yy), 40000, replace=False); X, yy = X[idx], yy[idx]
    clf = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, subsample=0.8,
                        tree_method="hist", n_jobs=-1, random_state=42)
    from sklearn.utils.class_weight import compute_sample_weight
    clf.fit(X, yy, sample_weight=compute_sample_weight("balanced", yy))
    imp = clf.feature_importances_
    base = imp[:nb * 7].reshape(7, nb).sum(0) if len(imp) >= nb * 7 else imp[:nb]
    order = np.argsort(base)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.barh(range(len(order)), base[order], color=BLUE, edgecolor=INK, linewidth=0.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[i] for i in order], fontsize=7.5)
    ax.set_xlabel("XGBoost importance (gain)")
    fig.savefig(os.path.join(FIG, "fig_featimp.pdf")); plt.close(fig)


if __name__ == "__main__":
    Y = load_y()
    print("subjects:", len(Y))
    fig_teaser()
    fig_perclass(Y)
    fig_confusion(Y)
    fig_domaingap()
    fig_persubject(Y)
    try:
        fig_featimp(Y)
        print("featimp ok")
    except Exception as e:
        print("featimp skipped:", type(e).__name__, e)
    print("done ->", FIG)
