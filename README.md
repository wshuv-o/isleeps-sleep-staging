# HAG-Net — Interpretable Sleep Staging in Subacute Ischemic Stroke

Sleep-stage classification on **iSLEEPS**, the first public polysomnography corpus of
subacute ischemic-stroke patients (99 usable subjects, 95,305 epochs, 5 AASM stages).

## Results

Subject-independent 10-fold cross-validation (patient-exclusive), N = 99:

| Model | Acc | Macro-F1 | Cohen's κ |
|---|---|---|---|
| **Ours (+HMM)** | **0.7464** ± 0.016 | 0.6753 | **0.6415** |
| Ours (no HMM) | 0.7375 | **0.6803** | 0.6342 |
| Published LSTM baseline | 0.747 | 0.677 | 0.640 |
| Published Transformer | 0.674 | 0.594 | 0.540 |
| Published CNN-ResNet18 | 0.617 | 0.544 | 0.480 |

We **match** the published state of the art on accuracy. Differences on macro-F1 and
κ are within the ±0.016 fold standard deviation and are reported as ties, not wins.

What the model adds beyond the headline number:

- **Calibrated uncertainty** — split-conformal prediction sets, empirical coverage
  0.900 at α = 0.1, expected calibration error 0.038
- **Lesion-severity biomarker** — inter-hemispheric spindle asymmetry correlates with
  NIHSS stroke severity (Spearman ρ = 0.41, p = 0.006, n = 43)
- **Statistical rigour** — subject-paired Wilcoxon with Bonferroni correction and
  Cohen's d; per-patient evaluation showing the tightest spread of any model tested
  (SD 0.091)
- **A documented 13-lever negative search** (see `extra/search/`) establishing that
  post-hoc decoding, calibration, class re-weighting and context width are exhausted
  on this feature set

## Repository layout

```
data/         datasets (git-ignored: clinical patient data, not redistributed)
model/        final model: HAG-Net architecture + its two training entry points
processing/   preprocessing (EDF -> npz) and the production feature extractors
utils/        folds/metrics, conformal calibration, significance tests, figures
extra/        everything else: baselines, failed architectures, the lever search,
              archived docs and figures
paper/        LaTeX manuscript and its figures
results/      per-experiment metrics (JSON)
```

## Pipeline

```bash
# 1. preprocess raw EDF -> data/processed7/SN*.npz  (7-channel montage)
python processing/build_npz_full.py

# 2. train the production ensemble (4 gradient boosters + HMM decoding)
python model/train_ensemble_full.py --v2 --folds 10 --context 3 --tag ensemble7_v2

# 3. train HAG-Net (asymmetry graph + selective SSM over the classical prior)
python model/train_hagnet.py --folds 10 --epochs 50

# 4. paper analyses
python utils/conformal.py        # conformal prediction sets + ECE
python utils/significance.py     # subject-paired Wilcoxon + Bonferroni + Cohen's d
python utils/make_figures.py     # all manuscript figures (vector PDF)
```

## Method

Each 30-second epoch of the seven-channel montage (C4:M1, C3:M2, O2:M1, O1:M2,
E1:M2, E2:M2, chin EMG) is mapped to 188 features: 161 spectral, Hjorth and
time-domain descriptors plus 27 physiological event features quantifying sleep
spindles, slow-wave activity, ocular movement and chin-muscle tone. Epochs are
concatenated with their ±3 neighbours and classified by a class-balanced soft-voting
ensemble of XGBoost, LightGBM, HistGradientBoosting and CatBoost. A hidden-Markov
model with Viterbi decoding then enforces the grammar of sleep.

HAG-Net adds a hemispheric-asymmetry graph module (montage graph with homologous-pair
asymmetry pooling over C4↔C3 and O2↔O1) and a bidirectional selective state-space
temporal decoder, combined with the classical prior through a residual gate anchored
at the prior at initialization.

## Honest notes

- On this 99-patient cohort the deep streams do **not** improve staging accuracy over
  the gradient-boosting prior. Their value is the asymmetry biomarker and the
  calibrated uncertainty. This is reported explicitly rather than hidden.
- N1 (F1 ≈ 0.32) is the limiting class. Per-class decision-bias tuning raises N1 by
  +0.031 but costs accuracy −0.018, i.e. gains are paid for one-for-one. N1 is an
  information problem in these features, not a threshold problem.
- Data are not redistributed here. iSLEEPS is available from its publication
  (doi:10.1038/s41597-026-06747-w); Sleep-EDF from PhysioNet.

## Requirements

See `requirements.txt`. Experiments ran on a single NVIDIA RTX 2060 (6 GB) with
GPU-accelerated boosting; `SN28` is a bit-identical duplicate of `SN15` and is
excluded everywhere, giving N = 99.
