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
| base (initial configuration: class weights, concat fusion, frozen LaBraM) | 0.7380 | 0.7493 |
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
0.608, final seed 42 0.642, eight-channel frozen seed 42 0.642, initial seven-channel configuration seed 42 0.491.

Against the reimplemented baselines (single model): +0.037 to +0.158, 9-10/10 folds, Holm p=0.016
for each.

## AHI: detector seeds, detector v2, combinations (ahi_rescore.py --det-seeds, ahi_final.py, ahi_combo.py)

Target: annotation AHI (data/aligned/resp_events), 99 test patients, stages predicted by
base-nocw-cross-ftlae-p8-al (3-seed, train-fold HMM). Detector counting rules (runs / split / peaks)
chosen per fold on validation patients. Detector v2 = run_event_ahi.py --aligned --eeg --tag v2
(seed 42; seeds 1 and 7 not run). MM-Net respiratory head on apnea/hypopnea onsets only
(-alahi): rho 0.745 vs 0.765 for the standard head.

| Estimator | rho | MAE | Severity acc | kappa | rho vs AHI_1_B |
|---|---|---|---|---|---|
| MM-Net (expected count, validation scale) | 0.765 | 10.4 | 0.54 | 0.45 | 0.633 |
| detector v1, seed 42 | 0.635 | 14.0 | 0.46 | 0.45 | 0.535 |
| detector v1, seeds 42+1 | 0.657 | 12.4 | 0.49 | 0.49 | 0.539 |
| detector v1, seeds 42+1+7 | 0.671 | 11.2 | 0.46 | 0.44 | 0.583 |
| detector v2, seed 42 | 0.652 | 11.5 | 0.46 | 0.41 | 0.544 |
| mean(MM-Net, v1 3 seeds) | 0.742 | 9.7 | 0.49 | 0.46 | 0.635 |
| mean(MM-Net, v2) | 0.745 | 9.7 | 0.51 | 0.44 | 0.625 |
| mean(MM-Net, v1 3 seeds, v2) | 0.744 | 9.5 | 0.48 | 0.46 | 0.629 |
| subset mean chosen per fold on validation | 0.666 | 11.0 | 0.45 | 0.40 | 0.566 |
| mean(MM-Net, v1 seed 42), rules from event_ahi_post_s42.json (ahi_rescore.txt) | 0.742 | 9.7 | 0.55 | 0.51 | 0.625 |

Detector v2 seed 42 alone: per-second AUC 0.809 (v1 0.789), event F1 0.248 (v1 0.253).

## ISRUC-Sleep, trained on all 99 iSLEEPS patients, applied unchanged (external_final.py; external/)

Data: subgroup II (8 subjects x 2 nights) and III (10 healthy subjects), rater-1 annotations.
EDF headers of all subgroup III files and of II_S1_1, II_S5_1 declare EEG/EOG at +-25 uV; their N3
peak-to-peak (9-23 uV) is ~10x below the other files (98-283 uV), so these channels are multiplied
by 10. Effort = Thorax + Abdomen; Pulse = heart rate from ECG R-peaks; nasal pressure = Flow2 (DC3).
Encoders fine-tuned on all 99 patients (11 validation), cross-fitted in 3 parts and aligned.
HMM from the training patients; 3 seeds. AHI reference: apneas + hypopneas in scored sleep per hour
of scored sleep. AHI estimate: expected count x k (k 0.459, iSLEEPS validation); "thr" = epochs with
p > 0.45 x 0.469 (both chosen on iSLEEPS validation, isruc_ahi_threshold.py).

| Config | Staging II+III (3-seed) | III | II | Resp AUC | AHI rho (expected) | AHI rho (thr) | AHI MAE (thr) |
|---|---|---|---|---|---|---|---|
| frozen (initial configuration: cw, concat, frozen LaBraM, 7 ch) | 0.7015 | 0.7156 | 0.6927 | 0.777 | 0.472 | 0.497 | 8.4 |
| final (nocw, cross, fine-tuned EEG+EOG aligned, 8 ch) | 0.6925 | 0.7128 | 0.6797 | 0.784 | 0.125 | 0.100 | 9.8 |
| final7 (as final, 7 ch, no nasal pressure) | 0.6921 | 0.7092 | 0.6814 | 0.786 | 0.445 | 0.458 | 13.0 |

Encoder alone (one epoch, no HMM): II 0.629, III 0.701. Previous ISRUC II run of the initial configuration
(external_isruc.json, no amplitude correction): 0.612.

## ISRUC II+III, trained and tested on ISRUC (isruc_cv.py; external/isruc_cv_IIIII_*.txt)

6 subject-independent folds (18 subjects, 26 recordings; 2 training subjects per fold for validation),
3 seeds, HMM from training recordings. final: encoder fine-tuned per fold on ISRUC training recordings,
cross-fitted (3 parts by subject) and aligned, 8 channels, no class weights, cross fusion.
AHI scale k from the 2 validation subjects per fold.

| Config | Staging II+III | III | II | mF1 | kappa | Resp AUC | AHI rho | AHI MAE |
|---|---|---|---|---|---|---|---|---|
| frozen (initial configuration) | 0.7156 | 0.7642 | 0.6852 | 0.6891 | 0.6317 | 0.757 | 0.061 | 8.0 |
| final | 0.7245 | 0.7787 | 0.6905 | 0.6921 | 0.6408 | 0.794 | 0.138 | 7.3 |

Per fold (3-seed acc), frozen / final: 0.671/0.658, 0.715/0.709, 0.760/0.775, 0.696/0.704, 0.750/0.778, 0.706/0.729.

## UCD sleep-apnoea database (PhysioNet ucddb), trained on all 99 iSLEEPS patients, applied unchanged

25 adults referred for suspected sleep-disordered breathing; annotated AHI 1.2-81.2. Recorded EEG is
C3-A2 and C4-A1 only (also fed as O1/O2); EDF headers declare 0..1 with no unit, so EEG/EOG/EMG are
scaled per recording to the iSLEEPS median whole-night standard deviation (no labels used). No nasal
pressure: 7-channel models. Stages: 0 W, 1 REM, 2 S1, 3 S2, 4-5 S3/S4 (N3). Events: APNEA-C/M/O,
HYP-C/M/O from <rec>_respevt.txt, onset in a scored sleep epoch per hour of scored sleep.

| Config | Staging (3-seed) | mF1 | kappa | Resp AUC | AP | AHI rho | MAE | Sev acc | Sev kappa |
|---|---|---|---|---|---|---|---|---|---|
| frozen encoder (initial configuration, 7 ch) | 0.6063 | 0.5261 | 0.4607 | 0.863 | 0.550 | 0.725 | 8.2 | 0.48 | 0.40 |
| fine-tuned encoder, no class weights, cross fusion, 7 ch (final7) | 0.6068 | 0.4949 | 0.4631 | 0.841 | 0.517 | 0.653 | 9.4 | 0.60 | 0.51 |

## ISRUC-I, trained and tested on ISRUC-I (isruc_cv.py --groups I --folds 10; external/isruc_cv_I_*.txt)

100 subjects, 10 subject-independent folds (80 train / 10 validation / 10 test subjects; validation
excluded from training -- fixed 2026-10-11, the earlier run with overlapping validation was discarded).
3 seeds, HMM from training recordings. Fine-tuned encoder: LaBraM-EOG fine-tuned per fold on the fold's
training subjects, cross-fitted in 3 parts by subject and ridge-aligned; 8 channels, no class weights,
cross fusion. AHI scale k from each fold's validation subjects.

| Config | Staging (3-seed) | single | mF1 | kappa | Resp AUC | AP | AHI rho | MAE | Sev acc | Sev kappa |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen encoder (initial configuration) | 0.7856 | 0.7820 | 0.7515 | 0.7220 | 0.872 | 0.447 | 0.523 | 6.8 | 0.43 | 0.26 |
| fine-tuned encoder (final) | 0.8033 | 0.8007 | 0.7647 | 0.7435 | 0.889 | 0.477 | 0.590 | 6.6 | 0.45 | 0.30 |

Per fold (3-seed acc), frozen: 0.748 0.828 0.766 0.814 0.802 0.753 0.796 0.779 0.796 0.775
                     final:  0.796 0.842 0.784 0.830 0.817 0.766 0.820 0.776 0.811 0.793
Final - frozen +0.0176, 9/10 folds higher, Wilcoxon p = 0.0039.
