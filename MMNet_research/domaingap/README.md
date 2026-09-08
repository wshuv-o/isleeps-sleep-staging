# The healthy-to-stroke domain gap, measured with its control

The manuscript claims deep models built for healthy sleep collapse on the injured
brain. This directory measures that independently, and — more importantly — runs
the control that makes the measurement interpretable.

## The problem with an uncontrolled number

A healthy-trained stager scoring 0.31 on iSLEEPS is evidence of a domain gap only
if the *same checkpoint through the same preprocessing* scores properly on
healthy sleep. Without that, "the injured brain breaks the model" and "our input
pipeline is wrong" produce identical numbers.

## What was run

`fetch_sleepedf.py`   Sleep-EDF Expanded (sleep-cassette) from PhysioNet.
`sedf_control.py`     The checkpoint on HEALTHY sleep — the control.
`healthy_zeroshot.py` The same checkpoint on iSLEEPS — zero-shot, no adaptation.

Model: `StagingSeqNet` (1.08 M params) with `pretrained_sedf.pt`, weights fitted
only on Sleep-EDF. Channel pairing is the one `train_transfer.py` documents:
Fpz-Cz, Pz-Oz, EOG horizontal, EMG submental ↔ iSLEEPS `processed7` columns
0, 2, 4, 6 (C4:M1, O2:M1, E1:M2, EMG). Per-recording normalisation matches
`train_transfer._norm_store`.

## Result

| | recordings | epochs | accuracy | macro-F1 | kappa |
|---|---|---|---|---|---|
| healthy (Sleep-EDF) | 32 | 33,837 | **0.8682** | 0.8440 | **0.8254** |
| stroke (iSLEEPS) | 99 | 92,560 | **0.3082** | 0.2315 | **0.0925** |

The control holds at 0.8682 (per-recording 0.8667 ± 0.0415), so the pipeline is
correct and the collapse is the cohort.

The failure mode is more informative than the headline. On healthy sleep the
predicted stage distribution tracks the truth. On stroke patients the model calls
**78.4% of epochs Wake** against a true 26.7%, and its N2 recall is **0.032** on
the stage that is 42% of the data. It does not degrade gracefully; it stops
recognising sleep architecture.

## The confound, stated plainly

Sleep-EDF uses Fpz-Cz / Pz-Oz; iSLEEPS uses C4:M1 / O2:M1. Part of the 56-point
drop is montage shift rather than pathology. Bounds worth reporting together:

* deep models trained ON iSLEEPS, matched montage, in-domain: 0.61–0.69
  → pathology alone costs roughly 0.19 against healthy-sleep performance
* healthy-trained, zero-shot, with montage shift: 0.3082
  → montage shift plus pathology costs 0.56

Reporting both is what makes the claim defensible rather than overstated.
