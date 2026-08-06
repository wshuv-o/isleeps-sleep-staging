# iSLEEPS — new-feature experiment on Kaggle

## What this tests, and why

Twelve levers have already been tried locally and **all returned nothing**:

| lever | result |
|---|---|
| class-prior correction | 0 (α=0 chosen in all 10 folds) |
| temperature calibration | +0.0009 macro-F1 (noise) |
| transition-matrix tuning | worse |
| duration-aware HSMM decoding | worse |
| stacking over 4 models | much worse (−0.027 macro-F1) |
| context width k=5 | won on a 3-fold probe, **lost at full scale** (−0.0026) |
| context width k=7 | worse |
| position-in-night | null |
| rolling ±10 statistics | worse |
| per-class decision bias | null (N1 +0.031, accuracy −0.018) |

They share one flaw: every one of them **re-slices the same 188 features**. The
per-class-bias result is the diagnostic — every N1 gained costs an N2/Wake
one-for-one, which means the features genuinely do not separate N1. It is an
*information* problem, not a threshold problem.

So this experiment adds **61 new features** (Sanei & Chambers, *EEG Signal
Processing*): nonlinear complexity (§2.7), wavelet time–frequency (§2.5.1),
K-complex/vertex transients (§6.5), and slow-vs-rapid eye movement — the last aimed
squarely at the N1↔REM confusion in our matrix.

## Baseline to beat

Identical protocol (10-fold subject-independent, XGBoost-CUDA, +HMM):

```
acc 0.7442 ± 0.0160 | macro-F1 0.6752 | kappa 0.6396 | N1 F1 0.321
```

Published state of the art on this dataset: **0.747 / 0.677 / 0.640**.

A result only counts if the delta exceeds the **±0.016** fold standard deviation.

## Setup

1. **Upload the raw signals** as a Kaggle Dataset named `isleeps-processed7`.
   From the repo root, with no extra disk needed:
   ```bash
   kaggle datasets create -p data/processed7 --dir-mode zip
   ```
   (or drag the `processed7` folder into the Kaggle "New Dataset" web uploader)

2. **Upload this bundle** as a Kaggle Dataset named `isleeps-bundle`
   (contains `featseq_cache/` = the existing 188 features, plus this code).

3. New Notebook → add both datasets → **Settings → Accelerator → GPU**.

## Run

```python
!cp /kaggle/input/isleeps-bundle/*.py .
!python run_kaggle.py --stage features          # ~25-30 min, 4 cores, pywt
!python run_kaggle.py --stage evaluate          # ~25 min on GPU
```

Paths are overridable if your dataset slugs differ:

```python
import os
os.environ["RAW_DIR"]  = "/kaggle/input/<your-raw-slug>/processed7"
os.environ["BASE_DIR"] = "/kaggle/input/<your-bundle-slug>/featseq_cache"
```

## Output

Three arms, identical folds/model/decoding so the only variable is the feature set:

- `base` — 188 existing features (must reproduce ≈0.7442 / 0.6752; if it doesn't,
  something is wrong with the setup, stop and check before trusting anything else)
- `new`  — the 61 new features alone
- `both` — 249 concatenated ← **the actual question**

Prints a `DELTA (both-base)` line and writes `kaggle_result.json`.

## How to read the result honestly

- **both − base > +0.016 on accuracy or macro-F1** → real gain, worth folding into
  the paper.
- **delta within ±0.016** → thirteenth null. Say so; do not report it as a win.
- **`new` alone scoring far below `base`** is expected and fine — these features are
  meant to be *complementary*, not a replacement.

Watch N1 F1 specifically. It is 0.321 and is the single largest recoverable deficit,
since macro-F1 weights it as one fifth of the score.

## Notes

- Lempel–Ziv complexity is deliberately excluded: the naive O(n²) implementation
  costs ~2.4 s per channel (~13 days for this dataset) and permutation entropy
  already covers the complexity axis at ~2 ms.
- Higuchi FD is computed on a 3× decimated signal; fractal dimension is
  scale-robust, so this does not change the estimate materially.
- `SN28` is a bit-identical duplicate of `SN15` and is dropped everywhere (N=99).
- Fold construction is byte-identical to `models/datasets.py::make_folds`, so
  Kaggle numbers are directly comparable to the local ones.
