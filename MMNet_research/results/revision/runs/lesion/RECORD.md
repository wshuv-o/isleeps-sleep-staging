# Experiments in this folder (Machine B, 2026-10-08 to 2026-10-10)

All on the paper's ten patient-independent folds, 99 patients. Within each fold, run_10fold's
train/validation split is kept; validation patients choose every variant, weight and decoder
setting, and test folds are only scored. Accuracy is test accuracy averaged over the ten folds.

## Lesion-aware inputs (run_lesion_aware.py, lesion_focus.py)

| Config | Accuracy (3 seeds) | vs base |
|---|---|---|
| base | 0.7369 | |
| mirror | 0.7384 | +0.0016, p=0.63 |
| mirror+asym | 0.7380 | +0.0012, p=1.00 |
| lesion | 0.7355 | -0.0013, p=0.70 |
| healthy-copy | 0.7365 | -0.0003, p=0.77 |
| damaged-drop | 0.7310 | -0.0059, p=0.027 |

summary.txt, lesion_focus_summary.txt.

## MM-Net training variants (run_seed_ensemble.py)

Per fold, seeds 42, 1, 7; probabilities in seed_ensemble/ (not in git, *.npz).

| Config | Single model (seed mean) | 3-seed average |
|---|---|---|
| base (published configuration) | 0.7380 | 0.7493 |
| base-nocw (uniform class weights) | 0.7442 | 0.7535 |
| base-nocw-cross (+ cross-attention fusion) | 0.7465 | 0.7552 |
| base-nocw-L40 (40-epoch window) | 0.7453 | 0.7516 |
| lesion | 0.7346 | 0.7425 |

seed_ensemble.txt.

## Fine-tuned LaBraM (finetune_labram.py)

LaBraM fine-tuned per fold on the training patients (EEG only, one 30 s epoch, no HMM):
frozen 0.6466 -> fine-tuned 0.7087, 10/10 folds, p=0.002 (labram_ft/fold<k>.json).

Inside MM-Net (base-nocw-cross), training patients' embeddings cross-fitted (3 parts per
fold, build_labramft_dirs.py); "aligned" maps each cross-fitted encoder's embedding into the
full-fold encoder's space with a ridge map fitted on training patients:

| Config | Single model | 3-seed average | Validation | vs frozen (3-seed) |
|---|---|---|---|---|
| frozen LaBraM | 0.7465 | 0.7552 | 0.7685 | |
| ftl (fine-tuned, not aligned) | 0.7515 | 0.7572 | 0.7738 | +0.0020, p=0.63 |
| ftla (aligned) | 0.7616 | 0.7689 | 0.7744 | +0.0138, p=0.020 |
| ftlo (stage probabilities only) | 0.7558 | 0.7624 | 0.7772 | +0.0072, p=0.020 |
| ftlap (aligned + probabilities) | 0.7595 | 0.7676 | 0.7792 | +0.0124, p=0.020 |

mmnet_ftl.json. Against the reimplemented baselines (compare_baselines.py), ftla single model:
+0.041 to +0.163, 10/10 folds, Holm p=0.016 for each of AttnSleep, LSTM (corpus), 1D-ResNet-SE-LSTM,
AT-BiLSTM, SleepTransformer, TinySleepNet, U-Time, Huttunen et al.

## Ensembles around MM-Net (run_boost_blend.py, run_decode_variants.py)

Blends of MM-Net probabilities, HistGradientBoosting stagers on the same features
(boost/, boost_ctx6/, boost_labram/) and the fine-tuned LaBraM stager, HMM-decoded:

| Pipeline (validation-chosen) | Validation | Test |
|---|---|---|
| base 3-seed + booster | 0.7696 | 0.7610 |
| 6 MM-Net + 2 boosters, alpha 10 | 0.7768 | 0.7654 |
| + fine-tuned LaBraM (seed 42), wx 0.35 | 0.7864 | 0.7788 |
| sys_final3: 6 MM-Net configs incl. ftla/ftlap/ftlo + 2 boosters + LaBraM (s42, s1) | 0.7901 | 0.7831 |

## In progress at the time of this record

Encoder fine-tuned with E1/E2 as F7/F8 (finetune_labram.py --eog), folds 0-3 test:
0.6785, 0.6993, 0.7067, 0.7598 (without EOG: 0.6812, 0.6747, 0.6730, 0.7662).
Pipeline: logs/eog_pipeline.sh on Machine B (outside the repository).
