"""build_starter_nb.py -- generate notebooks/starter_exploration.ipynb

A beginner-level exploration notebook: look at the signals, build simple features by
hand, try a few standard sklearn classifiers, and plot what happens. No deep learning.
Deliberately simpler and more visual than kaggle_experiments.ipynb, so it can be read,
run and explained without prior EEG background.
"""
import json, os

cells = []
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").split("\n")})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,
                           "outputs":[],"source":s.strip("\n").split("\n")})

md("""
# Getting started with the iSLEEPS sleep-staging data

The goal of this notebook is to understand the problem before trying to solve it:

1. What does the data actually look like?
2. What does an EEG epoch look like in each sleep stage?
3. Can simple, hand-built features separate the stages at all?
4. How do a few standard classifiers compare?

No deep learning here. Everything is `numpy`, `scipy` and `scikit-learn`.

**The one rule that matters:** a patient's epochs must never appear in both training and
test. Sleep epochs from the same night are highly similar, so splitting randomly would
let the model memorise the patient and report a score that is far too good.
""")

md("## 1. Load the data")
code('''
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

DATA = "/kaggle/input/isleeps-processed7/processed7"
if not os.path.isdir(DATA):
    DATA = "data/processed7"          # local fallback

FS = 100                               # samples per second
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
DUPLICATE = {28}                       # SN28 is an exact copy of SN15

files = sorted(glob.glob(os.path.join(DATA, "SN*.npz")))
sids  = [int(os.path.basename(f)[2:-4]) for f in files]
sids  = [s for s in sids if s not in DUPLICATE]
print(f"found {len(sids)} patients")
''')

code('''
# look at one patient first
d = np.load(os.path.join(DATA, f"SN{sids[0]}.npz"), allow_pickle=True)
x, y = d["x"], d["y"]

print("x shape:", x.shape, "  <- (epochs, channels, samples)")
print("y shape:", y.shape, "  <- one sleep stage per epoch")
print("channels:", list(d["channels"]))
print()
print(f"each epoch is {x.shape[2]} samples = {x.shape[2]/FS:.0f} seconds at {FS} Hz")
print(f"this patient slept for {len(y)*30/3600:.1f} hours")
''')

md("""
So each patient is a stack of 30-second windows. Seven channels: four EEG (brain), two
EOG (eye movement), one EMG (chin muscle). Every window has one label, 0 to 4.
""")

md("## 2. How common is each sleep stage?")
code('''
counts = np.bincount(y, minlength=5)
for name, c in zip(CLASS_NAMES, counts):
    bar = "#" * int(40 * c / counts.max())
    print(f"{name:4s} {c:5d}  {bar}")

plt.figure(figsize=(5,3))
plt.bar(CLASS_NAMES, counts, color="steelblue")
plt.ylabel("number of epochs"); plt.title(f"patient SN{sids[0]}")
plt.tight_layout(); plt.show()
''')

md("""
N2 dominates. This matters more than it looks: if 40% of epochs are N2, a lazy model that
always answers "N2" already scores 40% accuracy without learning anything. That is why we
will also look at **macro-F1**, which averages the score of each stage equally and so
punishes ignoring the rare ones.
""")

md("## 3. What does sleep look like over a night?")
code('''
plt.figure(figsize=(11,2.6))
plt.step(np.arange(len(y))*30/3600, y, where="post", lw=0.9, color="darkblue")
plt.yticks(range(5), CLASS_NAMES)
plt.xlabel("hours"); plt.ylabel("stage"); plt.title("Hypnogram (expert scoring)")
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
''')

md("""
This is a **hypnogram**. Two things to notice, because both become useful later:

- Stages come in long runs. The sleeper does not jump randomly between stages.
- Transitions are structured: you pass through N1 to get to N2, REM comes in cycles.

That means the *order* of epochs carries information. A model that classifies each epoch
in isolation is throwing that away.
""")

md("## 4. What does the raw signal look like in each stage?")
code('''
fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True)
t = np.arange(x.shape[2]) / FS
for k, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
    idx = np.where(y == k)[0]
    if len(idx) == 0:
        ax.text(0.5, 0.5, f"no {name} epochs", transform=ax.transAxes, ha="center"); continue
    ax.plot(t, x[idx[len(idx)//2], 0], lw=0.5, color="darkblue")   # channel 0 = C4:M1
    ax.set_ylabel(name, rotation=0, labelpad=22, va="center")
    ax.set_ylim(-150, 150)
axes[-1].set_xlabel("seconds")
fig.suptitle("One 30-second EEG epoch per stage (channel C4:M1)")
plt.tight_layout(); plt.show()
''')

md("""
Wake looks fast and low-amplitude. N3 looks slow and large. That difference is
**frequency content**, which is exactly what we can measure numerically.
""")

md("## 5. Turning a signal into numbers: frequency bands")
code('''
# The power spectrum tells us how much of the signal sits at each frequency.
bands = {"delta (0.5-4 Hz)": (0.5,4), "theta (4-8)": (4,8),
         "alpha (8-12)": (8,12), "sigma (12-16)": (12,16), "beta (16-30)": (16,30)}

plt.figure(figsize=(7,4))
for k, name in enumerate(CLASS_NAMES):
    idx = np.where(y == k)[0]
    if len(idx) == 0: continue
    f, P = welch(x[idx, 0].astype(float), fs=FS, nperseg=256)   # average over all epochs
    plt.semilogy(f, P.mean(0), label=name)
plt.xlim(0, 30); plt.xlabel("frequency (Hz)"); plt.ylabel("power")
plt.title("Average spectrum per stage"); plt.legend(); plt.tight_layout(); plt.show()
''')

md("""
The curves separate. N3 has much more low-frequency (delta) power; Wake has more fast
activity. So the power in each frequency band is a sensible feature to give a classifier.
""")

md("## 6. Build a simple feature vector")
code('''
def simple_features(epoch):
    """epoch: [7, 3000]  ->  a short list of numbers describing it"""
    out = []
    for ch in range(7):
        sig = epoch[ch].astype(float)
        f, P = welch(sig, fs=FS, nperseg=256)
        total = P[(f >= 0.5) & (f <= 30)].sum() + 1e-12
        for lo, hi in bands.values():
            power = P[(f >= lo) & (f < hi)].sum()
            out.append(power / total)          # relative power in this band
        out.append(sig.std())                  # how big is the signal
        out.append(np.mean(np.abs(np.diff(sig))))  # how fast does it wiggle
    return np.array(out, dtype=np.float32)

fv = simple_features(x[0])
print("features per epoch:", len(fv), "= 7 channels x (5 bands + 2 extras)")
''')

code('''
# extract features for a handful of patients (keep it small so it runs quickly)
N_PATIENTS = 12
use = sids[:N_PATIENTS]

F, Y, G = [], [], []                     # features, labels, patient id
for i, s in enumerate(use):
    dd = np.load(os.path.join(DATA, f"SN{s}.npz"), allow_pickle=True)
    xi, yi = dd["x"], dd["y"]
    F.append(np.stack([simple_features(e) for e in xi]))
    Y.append(yi); G.append(np.full(len(yi), s))
    print(f"  {i+1}/{len(use)} patients done", flush=True)

F = np.concatenate(F); Y = np.concatenate(Y); G = np.concatenate(G)
print("\\nfeature matrix:", F.shape, " labels:", Y.shape)
''')

md("## 7. Do the features actually separate the stages?")
code('''
# average delta and sigma power per stage - do they differ in the way we expect?
delta_idx, sigma_idx = 0, 3          # first channel: band order is delta, theta, alpha, sigma, beta
print(f"{'stage':6s} {'delta power':>13s} {'sigma power':>13s}")
for k, name in enumerate(CLASS_NAMES):
    m = Y == k
    if m.sum() == 0: continue
    print(f"{name:6s} {F[m, delta_idx].mean():13.3f} {F[m, sigma_idx].mean():13.3f}")

plt.figure(figsize=(6,4))
for k, name in enumerate(CLASS_NAMES):
    m = Y == k
    if m.sum() == 0: continue
    plt.scatter(F[m, delta_idx], F[m, sigma_idx], s=3, alpha=0.3, label=name)
plt.xlabel("relative delta power"); plt.ylabel("relative sigma power")
plt.title("Two features, coloured by stage"); plt.legend(markerscale=4)
plt.tight_layout(); plt.show()
''')

md("""
N3 sits high on delta, as expected from the spectra. The classes overlap a lot in just two
dimensions, which is why we hand all the features to a classifier instead of drawing lines
by hand.
""")

md("## 8. Split by patient, not by epoch")
code('''
from sklearn.model_selection import GroupShuffleSplit

# GroupShuffleSplit keeps all epochs of a patient on the same side of the split
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(F, Y, groups=G))

print("train patients:", sorted(set(G[train_idx])))
print("test  patients:", sorted(set(G[test_idx])))
print(f"\\ntrain epochs: {len(train_idx)}   test epochs: {len(test_idx)}")
assert not (set(G[train_idx]) & set(G[test_idx])), "patient leaked across the split!"
print("no patient appears on both sides - good")
''')

md("## 9. Try a few standard classifiers")
code('''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

scaler = StandardScaler().fit(F[train_idx])
Xtr, Xte = scaler.transform(F[train_idx]), scaler.transform(F[test_idx])
ytr, yte = Y[train_idx], Y[test_idx]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  n_jobs=-1, random_state=42),
}

scores = {}
for name, clf in models.items():
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, average="macro", zero_division=0)
    scores[name] = (acc, mf1, pred)
    print(f"{name:22s} accuracy={acc:.3f}   macro-F1={mf1:.3f}")
''')

md("""
Notice the gap between accuracy and macro-F1 for every model. That gap *is* the class
imbalance: the models do well on N2 and Wake and poorly on the rare stages.
""")

md("## 10. Look at the mistakes")
code('''
best = max(scores, key=lambda k: scores[k][1])
pred = scores[best][2]
print(f"best macro-F1: {best}\\n")
print(classification_report(yte, pred, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(yte, pred, labels=range(5)).astype(float)
cm = cm / cm.sum(1, keepdims=True)

fig, ax = plt.subplots(figsize=(5,4.2))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm[i,j] > 0.5 else "black", fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{best}")
plt.colorbar(im); plt.tight_layout(); plt.show()
''')

md("""
Read the rows: each row shows where the true stage actually went.

**N1 is the hard one.** It gets spread across Wake, N2 and REM. That makes sense — N1 is
the transition into sleep, so it genuinely looks like its neighbours. This is the single
biggest weakness in the whole task, and it stays a problem no matter which model is used.
""")

md("## 11. Which features mattered?")
code('''
rf = models["Random Forest"]
names = [f"ch{c}_{b.split()[0]}" for c in range(7) for b in list(bands) ]
names = []
for c in range(7):
    names += [f"ch{c}_{b.split()[0]}" for b in bands]
    names += [f"ch{c}_std", f"ch{c}_diff"]

order = np.argsort(rf.feature_importances_)[::-1][:15][::-1]
plt.figure(figsize=(6,4.5))
plt.barh(range(len(order)), rf.feature_importances_[order], color="steelblue")
plt.yticks(range(len(order)), [names[i] for i in order], fontsize=8)
plt.xlabel("importance"); plt.title("Top 15 features (Random Forest)")
plt.tight_layout(); plt.show()
''')

md("""
The channels that matter are a sanity check on the whole approach: EOG channels (4 and 5)
help find REM, the EMG channel (6) tracks muscle tone, and delta power on the EEG channels
picks out deep sleep. That matches how a human scorer reads a recording.
""")

md("""
## 12. What we learned, and what to try next

- The data is strongly imbalanced, so accuracy alone is misleading; macro-F1 is the honest
  metric.
- Simple band-power features already separate the stages reasonably well.
- Splitting by patient is essential. Splitting by epoch would inflate the score.
- N1 is by far the weakest class and gets confused with Wake, N2 and REM.
- Sleep has strong temporal structure that a per-epoch model completely ignores.

Natural next steps, in the order they are worth trying:

1. **Use neighbouring epochs.** Give the classifier the features of the epochs before and
   after, since stages come in long runs.
2. **Smooth the predictions over time**, so the output looks like a real hypnogram instead
   of flickering between stages.
3. **Stronger classifiers** — gradient boosting (XGBoost, LightGBM) usually beats a random
   forest on this kind of tabular feature.
4. **Better features** — explicitly detect sleep spindles and slow waves rather than only
   measuring average band power.

Those four steps are what `kaggle_experiments.ipynb` picks up.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python","version":"3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/starter_exploration.ipynb","w",encoding="utf-8"), indent=1)
print(f"wrote notebooks/starter_exploration.ipynb ({len(cells)} cells, "
      f"{sum(c['cell_type']=='code' for c in cells)} code)")
