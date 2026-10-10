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

## Encoder with EOG (finetune_labram.py --eog: E1/E2 fed to LaBraM as F7/F8)

Encoder alone (one epoch, no HMM), 10 folds: 0.7319 with EOG vs 0.7087 without (8/10 folds up).
Inside MM-Net (base-nocw-cross), cross-fitted and aligned:

| Config | Single model | 3-seed average | Validation |
|---|---|---|---|
| ftlae (aligned, encoder with EOG) | 0.7573 | 0.7628 | 0.7792 |
| ftlape (aligned + probabilities, encoder with EOG) | 0.7584 | 0.7643 | 0.7769 |

Ensemble with the EOG encoder added (sys_eog): validation 0.7927, test 0.7834.

## Final architecture (final_eval.py; final_architecture.txt)

Encoder variant chosen by validation accuracy among ftla, ftlap, ftlae, ftlape (ftlae, 0.7792),
trained with the eight cardiorespiratory channels (seven + nasal pressure): base-nocw-cross-ftlae-p8.

| | Staging acc | macro-F1 | kappa | Resp AUC | Resp AP |
|---|---|---|---|---|---|
| final, 3-seed average | 0.7601 | 0.6640 | 0.6529 | 0.8198 | 0.4729 |
| final, single model (seed mean) | 0.7569 | | | 0.8087 | |
| eight-channel MM-Net, frozen LaBraM (seed mean) | 0.7358 | 0.6683 | 0.6293 | 0.8099 | 0.4516 |

vs eight-channel frozen: staging +0.0211 single (9/10, p=0.006), +0.0243 3-seed (9/10, p=0.004);
respiratory AUC -0.0012 single (p=0.63), +0.0100 3-seed (9/10, p=0.006).

AHI association (Spearman, per-patient mean event probability vs AHI_1_B, n=99): final 3-seed
0.608, final seed 42 0.642, eight-channel frozen seed 42 0.642, published seven-channel seed 42 0.491.

Against the reimplemented baselines (single model): +0.037 to +0.158, 9-10/10 folds, Holm p=0.016
for each.
