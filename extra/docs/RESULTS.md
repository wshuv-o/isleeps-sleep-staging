# iSLEEPS — E1 results (subject-independent staging)

## >>> BEST MODEL (full 100-subject cohort, N=99 after duplicate drop) <<<

Subject-independent 5-fold CV, same patient-exclusive protocol as the published descriptor.

| Model | Channels | Acc | Macro-F1 | kappa | W / N1 / N2 / N3 / R |
|---|---|---|---|---|---|
| **Ensemble + HMM** | **7 (EEG+EOG+EMG)** | **0.742 ± .011** | 0.668 | **0.633** | .81/.31/.79/.69/.74 |
| Ensemble (no HMM) | 7 (EEG+EOG+EMG) | 0.734 ± .009 | **0.672** | 0.626 | .80/.35/.79/.69/.73 |
| Ensemble + HMM | 4 (EEG only) | 0.727 ± .020 | 0.644 | 0.612 | .79/.27/.79/.70/.68 |
| Deep CNN+BiLSTM + aug | 4 (EEG only) | 0.654 ± .029 | 0.610 | 0.549 | .77/.32/.70/.62/.65 |
| *Published LSTM (N=100)* | *single-ch* | *0.747* | *0.677* | *0.640* | |
| *Published Transformer* | | *0.674* | *0.594* | *0.540* | |
| *Published CNN-ResNet18* | | *0.617* | *0.544* | *0.480* | |

**We essentially MATCH the published state of the art: 0.742 vs 0.747 (a 0.5-pt gap, within the
fold noise +-.011), macro-F1 0.672 vs 0.677, kappa 0.633 vs 0.640** — with a simpler CLASSICAL
ensemble (XGB+LGB+HGB) on the standard sleep montage (4 EEG + 2 EOG + chin EMG), same
subject-independent protocol.

**Full progression:** 0.626 (deep, 39) -> 0.663 (ensemble+HMM, 39) -> 0.727 (full cohort, EEG only)
-> **0.742 (+ EOG/EMG)**. Adding EOG (eye movements) + EMG (atonia) lifted REM F1 .68 -> .73 and
N1 .31 -> .35 exactly as physiology predicts. The deep CNN+BiLSTM tops out ~0.65 here; the
feature ensemble is genuinely stronger on this data. HMM decoding adds ~+0.8 acc but slightly
over-smooths N1/REM (best macro-F1 is the ensemble WITHOUT HMM: 0.672).

Note: matched, not beaten. Chasing the last 0.5 pt is the crowded/incremental game; for the
proposal the model is the INSTRUMENT, and matching the published tool is a credible instrument.

---

## Pilot results (40-subject subset, N=39) - retained for the data-lever comparison

Our own models, trained from scratch on the 40-subject Zenodo subset (N=39 after
dropping the SN15/SN28 duplicate), 5-fold **subject-independent** CV. Channels:
C4:M1, C3:M2, O2:M1, O1:M2 @100 Hz, 30 s epochs.

## All runs

| Model | Selection | Acc | Macro-F1 | κ | W / N1 / N2 / N3 / R |
|---|---|---|---|---|---|
| CNN (per-epoch, 0.4M) | test-peek | 0.613 ± .036 | 0.565 | 0.485 | .70/.26/.66/.67/.53 |
| CNN+BiLSTM 25ep (1.1M) | test-peek | 0.645 ± .041 | 0.609 | 0.526 | .73/.31/.70/.68/.62 |
| CNN+BiLSTM 45ep (1.1M) | test-peek | 0.657 ± .038 | 0.620 | 0.540 | .73/.32/.72/.70/.64 |
| DeepSleepNet 2-stage (3.7M) | **honest val** | 0.615 ± .023 | 0.567 | 0.483 | .69/.22/.68/.68/.58 |
| **CNN+BiLSTM + augment (1.1M)** | **honest val** | **0.626 ± .021** | **0.578** | **0.500** | .74/.25/.66/.64/.60 |
| *Authors CNN (pub, N=100)* | — | *0.617* | — | — | |
| *Authors Transformer (pub, N=100)* | — | *0.674* | — | — | |
| *Authors LSTM (pub, N=100)* | — | *0.747* | — | — | |

## Findings (what the experiments establish)

1. **Pipeline validated.** Our per-epoch CNN (0.613) reproduces the published CNN (0.617).
2. **Temporal context helps** (+3-4 pts), most on minority/transition classes (REM, N1).
3. **We are DATA-limited, not capacity-limited.** A 3.7M-param dual-resolution
   DeepSleepNet with two-stage balanced pretraining (0.615 honest) does **no better** than
   the 0.4M CNN. More parameters do not help on 39 subjects.
4. **Augmentation > capacity.** EEG augmentation (amplitude/noise/channel-drop/time-mask)
   on the compact model gives the **best honest result (0.626, lowest variance ±0.021)** and
   pushes the best epoch late (overfitting delayed) — the correct lever for small data.
5. **The 0.657 was optimistic.** It used test-set epoch selection (peeking). Under honest
   validation-based selection, our ceiling is **~0.62-0.63 accuracy**.

## Classical ML (feature-based) — DIFFERENT PARADIGM, honest (single fit, no peeking)

Hand-crafted features (spectral band powers, spectral entropy/edge, Hjorth, time-domain)
+/- 3-epoch temporal context (644 features), 5-fold subject-independent CV.

| Model | Acc | Macro-F1 | κ | W / N1 / N2 / N3 / R |
|---|---|---|---|---|
| RandomForest | 0.655 ± .017 | 0.499 | 0.485 | .74/.02/.72/.56/.46 |
| ExtraTrees | 0.659 ± .014 | 0.506 | 0.497 | .74/.03/.72/.60/.44 |
| HistGB | 0.641 ± .023 | 0.555 | 0.498 | .73/.21/.72/.64/.48 |
| XGBoost | 0.654 ± .023 | **0.560** | **0.511** | .74/.19/.73/.64/.50 |
| **LightGBM** | **0.661 ± .022** | 0.549 | 0.511 | .74/.16/.74/.63/.48 |

**Classical beats deep on accuracy + kappa** (LightGBM 0.661 vs deep honest 0.626), confirming
the data-limited diagnosis. Deep keeps a small macro-F1 edge via better REM/N1. RF/ExtraTrees
get high accuracy by ignoring N1 (F1 ~.02); the boosters (XGB/HistGB) balance far better.
**Best overall = XGBoost** (acc 0.654, macro-F1 0.560, kappa 0.511) — competitive accuracy AND
minority handling. Run: `python train_classical.py --context 3`.

## Ensemble + HMM temporal smoothing (best accuracy)

Soft-vote of XGBoost+LightGBM+HistGB, then Viterbi decoding through a stage-transition
matrix estimated from training hypnograms. 5-fold subject-independent.

| Model | Acc | Macro-F1 | κ | W / N1 / N2 / N3 / R |
|---|---|---|---|---|
| Ensemble (XGB+LGB+HGB) | 0.658 ± .021 | 0.556 | 0.512 | .74/.18/.74/.64/.49 |
| **Ensemble + HMM** | **0.663 ± .025** | 0.550 | **0.513** | .74/.13/.74/.64/.49 |

HMM adds +0.5 acc but over-smooths transient N1 (F1 .18->.13) — the classic tradeoff.

## LEADERBOARD (honest, N=39)
| Rank | Model | Acc | Macro-F1 | κ |
|---|---|---|---|---|
| 1 (acc) | Ensemble + HMM | **0.663** | 0.550 | 0.513 |
| 2 | LightGBM | 0.661 | 0.549 | 0.511 |
| 3 | XGBoost | 0.654 | 0.560 | 0.511 |
| 4 | Deep CNN+BiLSTM (augmented) | 0.626 | **0.578** | 0.500 |
| 5 | Deep CNN per-epoch | 0.613 | 0.565 | 0.485 |

**Takeaways:** (a) best accuracy = ensemble+HMM (0.663), +3.7 pts over best deep — classical
paradigm wins on small data; (b) best macro-F1 = deep augmented (0.578) — neural nets handle
REM/N1 better; (c) all paradigms plateau ~0.63-0.66, far from published 0.747 -> the bottleneck
is data quantity (N=39 vs 100), not the algorithm.

## Honest ceiling vs published 74.70%
The ~12-point gap is now well-characterised and is **structural, not an optimisation failure**:
- **Half the data**: N=39 vs published N=100. Biggest single lever (also unblocks Pillar ii).
- **Protocol**: unconfirmed whether the authors' 74.70% is subject-independent; if it used
  epoch-level random splits it is inflated and not directly comparable to our rigorous number.
- **Genuine difficulty**: disrupted stroke sleep (4 subjects no-N3, 5 no-REM).

## Best-model confusion (row-normalised)
```
        W     N1     N2     N3     R
  W    .66   .22    .08    .01   .03      N1 is the scattered hard class (48%);
  N1   .20   .48    .20    .01   .11      bleeds into W / N2 / R.
  N2   .04   .16    .64    .10   .06      N3 cleanest (76%).
  N3   .01   .02    .20    .76   .00
  R    .05   .16    .13    .00   .66
```

## E2 — healthy->stroke domain gap (Pillar i)

Train LightGBM on healthy Sleep-EDF (Fpz-Cz), eval on held-out healthy + zero-shot on
stroke iSLEEPS (C4:M1).

| | Acc | Macro-F1 |
|---|---|---|
| Healthy (Sleep-EDF test) | 0.855 | 0.792 |
| Stroke (iSLEEPS 0-shot) | 0.585 | 0.434 |
| **Domain gap** | **-0.27** | **-0.36** |

Per-stage recall healthy->stroke: N2 .90->.86 (robust) | W .85->.58 | N1 .48->.10 |
N3 .84->.39 | **REM .90->.14 (collapse)**. Confirms RQ2: REM/N3/N1 degrade most, N2 robust.

**Confounds (honest):** Sleep-EDF Fpz-Cz (frontal) vs iSLEEPS C4:M1 (central) montage
mismatch. REM's collapse is heavily montage-confounded (REM EEG signature is frontal);
N3's drop is the cleanest stroke-pathology signal (reduced slow-wave sleep, well-captured
centrally). TODO control: train Sleep-EDF Pz-Oz -> test iSLEEPS O1:M2 (matched occipital)
to isolate montage from pathology. Healthy stage mix also differs (more REM/N3, less wake).

## Next levers (in priority order)
1. **Full 100-subject iSLEEPS** (iHUB-Data, manual) — the dominant lever + unblocks E3.
2. **Transfer learning from Sleep-EDF** — pretrain on a large healthy cohort, fine-tune on
   iSLEEPS. Brings external data to a data-limited problem AND *is* Pillar (i) / E2 (domain gap).
3. Squeeze the honest protocol: larger/CV-based val selection (the 6-subject selector is noisy).

## Repro (GPU env: d:/EEG-TransNet/testenv, set KMP_DUPLICATE_LIB_OK=TRUE)
```
python train.py          --all-folds --epochs 20 --batch 1024                 # CNN baseline
python train_seq.py      --all-folds --epochs 45 --batch 64                    # CNN+BiLSTM (test-peek)
python train_seq.py      --all-folds --epochs 45 --batch 64 --augment --val-subj 6   # honest + augment (best)
python train_deepsleep.py --all-folds --pre-epochs 10 --epochs 35             # DeepSleepNet 2-stage
python evaluate.py                                                             # comparison table
```
