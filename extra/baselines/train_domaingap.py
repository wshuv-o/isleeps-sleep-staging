"""
train_domaingap.py — E2 / Pillar (i): healthy -> stroke domain gap.

Train a stager on healthy Sleep-EDF (single central channel Fpz-Cz), then evaluate
on (a) held-out healthy Sleep-EDF and (b) zero-shot on stroke iSLEEPS (C4:M1). The
per-stage drop healthy->stroke quantifies the domain gap.

CONFOUND (reported, not hidden): Sleep-EDF Fpz-Cz vs iSLEEPS C4:M1 are different
derivations, so part of the drop is montage/hardware, not purely stroke pathology.
Per-subject feature z-normalisation reduces (not removes) this.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_domaingap.py
"""
import os
import sys
import glob
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from datasets import load_subject, list_subjects, CLASS_NAMES  # noqa
from features import extract_features  # noqa

ROOT = os.path.dirname(os.path.abspath(__file__))
SLEEP_PROC = os.path.join(ROOT, "data", "sleep_edf_proc")
RESULTS = os.path.join(ROOT, "results")
CONTEXT = 3


def add_ctx(F, k):
    if k <= 0:
        return F
    Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([Fp[i:i + len(F)] for i in range(2 * k + 1)], axis=1)


def feats_from(x, y):
    F, _ = extract_features(x, fs=100)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    return add_ctx(F, CONTEXT).astype(np.float32), y


def load_sleepedf():
    recs = sorted(glob.glob(os.path.join(SLEEP_PROC, "*.npz")))
    by_subj = {}
    for p in recs:
        rec = os.path.basename(p)[:-4]
        subj = rec[:5]                      # SC4ss (group both nights)
        d = np.load(p, allow_pickle=True)
        F, y = feats_from(d["x"].astype(np.float32), d["y"].astype(np.int64))
        by_subj.setdefault(subj, []).append((F, y))
    return by_subj


def load_isleeps():
    Fs, ys = [], []
    for sid in list_subjects():
        x, y = load_subject(sid, channels=["C4:M1"], normalize=False)   # single central channel
        F, y = feats_from(x, y)
        Fs.append(F); ys.append(y)
    return np.concatenate(Fs), np.concatenate(ys)


def per_stage(y, p):
    rec = recall_score(y, p, average=None, labels=list(range(5)), zero_division=0)
    return {CLASS_NAMES[i]: float(rec[i]) for i in range(5)}


def main():
    from lightgbm import LGBMClassifier
    os.makedirs(RESULTS, exist_ok=True)
    print("loading healthy (Sleep-EDF) features...")
    he = load_sleepedf()
    subs = sorted(he.keys())
    rng = np.random.RandomState(42); rng.shuffle(subs)
    n_test = max(4, len(subs) // 5)
    test_h, train_h = subs[:n_test], subs[n_test:]

    def stack(subset):
        Fs = [F for s in subset for F, _ in he[s]]
        ys = [y for s in subset for _, y in he[s]]
        return np.concatenate(Fs), np.concatenate(ys)

    Xtr, ytr = stack(train_h)
    Xhe, yhe = stack(test_h)
    print(f"healthy train {Xtr.shape} ({len(train_h)} subj) | healthy test {Xhe.shape} ({len(test_h)} subj)")
    print("loading stroke (iSLEEPS) features...")
    Xst, yst = load_isleeps()
    print(f"stroke (iSLEEPS) {Xst.shape}")

    from sklearn.utils.class_weight import compute_sample_weight
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                         class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
    clf.fit(Xtr, ytr)

    ph = clf.predict(Xhe); ps = clf.predict(Xst)
    acc_h = accuracy_score(yhe, ph); acc_s = accuracy_score(yst, ps)
    mf1_h = f1_score(yhe, ph, average="macro", zero_division=0)
    mf1_s = f1_score(yst, ps, average="macro", zero_division=0)
    rec_h = per_stage(yhe, ph); rec_s = per_stage(yst, ps)

    print("\n===== E2 domain gap (train healthy Sleep-EDF) =====")
    print(f"  healthy test : acc={acc_h:.3f}  macroF1={mf1_h:.3f}")
    print(f"  stroke (0shot): acc={acc_s:.3f}  macroF1={mf1_s:.3f}")
    print(f"  OVERALL DROP : acc -{acc_h-acc_s:.3f}  macroF1 -{mf1_h-mf1_s:.3f}")
    print(f"\n  per-stage RECALL   healthy -> stroke (drop)")
    for c in CLASS_NAMES:
        print(f"    {c:3s}  {rec_h[c]:.3f} -> {rec_s[c]:.3f}   (-{rec_h[c]-rec_s[c]:+.3f})")
    print("\n  Note: Fpz-Cz vs C4:M1 montage difference is a confound in this drop.")

    json.dump({"healthy": {"acc": acc_h, "macro_f1": mf1_h, "recall": rec_h},
               "stroke": {"acc": acc_s, "macro_f1": mf1_s, "recall": rec_s},
               "n_healthy_train_subj": len(train_h), "context": CONTEXT},
              open(os.path.join(RESULTS, "domaingap.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.relpath(os.path.join(RESULTS, 'domaingap.json'))}")


if __name__ == "__main__":
    main()
