# iSLEEPS multimodal — consolidated honest results

Cohort: 96 subjects (SN28 duplicate dropped), 89,532 epochs @30 s.
Stages: W 26.6% / N1 10.2% / N2 42.3% / N3 8.9% / R 12.1%.
Apnea: 16.0% of epochs positive; median 13%/subject (0–60.5%); 77/96 have ≥5% apnea epochs.
All metrics 10-fold subject-independent unless noted.

## 1. Sleep staging — every approach converges ~0.75
| approach | acc | mF1 | kappa | notes |
|---|---|---|---|---|
| Raw multimodal CNN (EEG+cardio, from scratch) | 0.6545 | 0.612 | 0.531 | single split, raw-signal wall |
| Deep feature-fusion BiLSTM (EEG+cardio), balanced CW | ~0.68 | ~0.64 | ~0.57 | 10-fold, pre-tuning |
| Deep feature-fusion, sqrt CW + HMM (tuned) | 0.713 | 0.637 | 0.599 | single split, tuned |
| **Boosting ensemble (EEG features)** | **0.7464** | **0.6753** | **0.6415** | our best stager |
| Published LSTM (EEG, 1 ch) | 0.747 | 0.677 | 0.640 | Maiti 2026 baseline |
| Ensemble + HAG-Net blend | 0.7576 | — | — | best single number |

**Finding:** staging is EEG-saturated at ~0.75 subject-independent. >80% needs a leaky
(epoch-level) split. Deep-on-features < gradient boosting on these 188 features.

## 2. Does the whole polysomnogram help STAGING? No.
| model | EEG-only staging | +cardio staging | Δ |
|---|---|---|---|
| Boosting (10-fold) | 0.7427 | 0.7459 | +0.3pp (within ±3.4pp) |
| Deep fusion (10-fold) | 0.6891 | 0.6787 (concat) / 0.6804 (cross) | ≈0 / slightly down |

Confirmed across 3 architectures + per-class (no Wake/REM gain). Cardio does not aid staging.

## 3. Does the whole polysomnogram help APNEA? Yes, modestly and non-circularly.
Deep joint model, improved apnea head (direct cardio path, MLP), 10-fold:
| variant | apnea AUC | apnea AP |
|---|---|---|
| eeg_only (cortical arousals only) | 0.661 | 0.250 |
| concat (+cardio) | 0.711 | 0.335 |
| cross (+cross-modal attention) | 0.708 | 0.336 |
| cross_noflow (airflow input deleted) | 0.696 | 0.317 |

**Findings:** (a) +5pp AUC from EEG-only→multimodal (0.66→0.71); (b) cross-modal attention
ties plain concatenation (attention is not the load-bearing novelty — reported honestly);
(c) removing the scored-airflow input keeps AUC 0.696 → the apnea signal is genuinely
SpO2-desaturation + effort + cardiac + EEG-arousal, not circular with the label source.

## 4. Staging-gap tuning sweep (single split, cross model)
| config (+HMM) | acc | mF1 |
|---|---|---|
| L=20 balanced CW | 0.697 | 0.626 |
| L=20 sqrt CW | 0.713 | 0.637 |
| L=20 no CW | 0.732 | 0.599 |
HMM smoothing +~1pp; sqrt CW is the acc/mF1 sweet spot; L=20 beats L=40.

## 5. Final locked joint model (10-fold) — mm_final_cv.py
Config: cross fusion + direct-cardio apnea MLP head + sqrt CW + L=20 + apnea_w=1.0 + HMM.
| variant | STAGING acc(+HMM) | mF1 | kappa | APNEA AUC | AP |
|---|---|---|---|---|---|
| eeg_only | 0.7206 ±.031 | 0.6349 | 0.6029 | 0.6551 ±.038 | 0.243 |
| concat (+cardio) | 0.7207 ±.034 | 0.6447 | 0.6058 | 0.6998 ±.037 | 0.333 |
| cross (+attention) | 0.7151 ±.035 | 0.6359 | 0.5996 | 0.6979 ±.042 | 0.324 |
| cross_noflow (no airflow) | 0.7207 ±.030 | 0.6492 | 0.6088 | 0.7069 ±.040 | 0.328 |
| *boosting ensemble (staging)* | *0.7464* | *0.6753* | *0.6415* | *—* | *—* |

**Final read:** the tuned deep joint model reaches **0.721 staging / 0.645 mF1 / 0.606 kappa**
(within ~2.5pp of boosting 0.746 and the published LSTM 0.747) AND **0.70 apnea AUC** from
one model. Cardio does not move staging (eeg_only=concat=0.721). Cardio lifts apnea +4.5pp
(0.655→0.700). Cross-modal attention TIES concatenation (0.698 vs 0.700) — attention is not
load-bearing, reported honestly. cross_noflow (airflow deleted) has the BEST apnea AUC (0.707)
and staging mF1 (0.649) — the apnea signal is non-circular (SpO2/effort/cardiac/arousal).

## Interpretation for the paper
The value of going multimodal on this cohort is a **second clinical capability (apnea
detection), not a staging gain** — the same shape as HAG-Net's honest thesis that depth
does not improve staging at 99 patients. The defensible contribution is a single
multimodal multi-task model: SOTA-level staging (via the strong prior) + joint apnea
detection from the cardiorespiratory channels, with a rigorous ablation and a
non-circularity check.
