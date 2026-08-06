"""
evaluate.py — collate results/*.json into an E1 comparison table vs the
published iSLEEPS baselines, plus per-stage F1 and mean confusion matrix.

  d:/EEG-TransNet/testenv/python.exe evaluate.py
"""
import os
import sys
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repo root
for _p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, HERE)


RESULTS = os.path.join(ROOT, "results")
CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]

# Published single-channel baselines (authors, full 100-subject set).
PUBLISHED = {"CNN (paper)": 0.6165, "Transformer (paper)": 0.6744, "LSTM (paper)": 0.7470}


def load_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "folds" not in d:
            continue
        runs[os.path.basename(p)[:-5]] = d
    return runs


def main():
    runs = load_runs()
    print("=== Published single-channel baselines (full 100-subj) ===")
    for k, v in PUBLISHED.items():
        print(f"  {k:22s} acc={v:.4f}")

    if not runs:
        print("\n(no local results yet)")
        return

    print("\n=== Our subject-independent CV runs (40-subj Zenodo subset, N=39) ===")
    print(f"{'run':16s} {'folds':5s} {'acc':>14s} {'macroF1':>14s} {'kappa':>14s}  "
          + "  ".join(f"{c:>5s}" for c in CLASS_NAMES))
    for name, d in runs.items():
        folds = d["folds"]
        acc = np.array([f["acc"] for f in folds])
        mf1 = np.array([f["macro_f1"] for f in folds])
        kap = np.array([f["kappa"] for f in folds])
        pcf = np.array([f["per_class_f1"] for f in folds]).mean(0)
        ch = "1ch" if len(d.get("channels", [])) == 1 else f"{len(d.get('channels', []))}ch"
        print(f"{name:16s} {len(folds):5d} {acc.mean():.3f}+-{acc.std():.3f}  "
              f"{mf1.mean():.3f}+-{mf1.std():.3f}  {kap.mean():.3f}+-{kap.std():.3f}  "
              + "  ".join(f"{v:5.3f}" for v in pcf) + f"   [{ch}]")
        # vs LSTM baseline
        delta = acc.mean() - PUBLISHED["LSTM (paper)"]
        print(f"{'':16s} {'':5s} vs LSTM 74.70%: {delta:+.3f} acc")

    # mean confusion matrix of the best run by macro-F1
    best = max(runs.items(), key=lambda kv: np.mean([f["macro_f1"] for f in kv[1]["folds"]]))
    cm = np.sum([np.array(f["confusion"]) for f in best[1]["folds"]], 0)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    print(f"\n=== Mean confusion (row-normalised) — best run: {best[0]} ===")
    print("        " + "  ".join(f"{c:>5s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:3s}  " + "  ".join(f"{cmn[i,j]:5.2f}" for j in range(5)))


if __name__ == "__main__":
    main()
