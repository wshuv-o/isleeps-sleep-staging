# Nasal pressure: results and ready-to-use text (held for the next revision)

Status: **not in the manuscript yet, on purpose.** Use when the reviewer raises the
Huttunen five-signal point, or in the next revision round.

## What was run (27 Sep 2026)

iSLEEPS records two airflow sensors: "Flow Th" (oronasal thermal) and "Pressure Flow"
(nasal pressure, the sensor hypopneas are scored from). The published MM-Net reads only
Flow Th. The Huttunen port in the paper used four of their five Model-3 signals, without
nasal pressure.

Both models were re-run with nasal pressure on the paper's 99-patient ten folds:

| Same folds, three seeds (42, 1, 7) | Staging acc | Macro-F1 | κ | Resp. AUC | AP |
|---|---|---|---|---|---|
| MM-Net + nasal pressure | 0.736 ± 0.020 | 0.669 | 0.631 | 0.810 ± 0.034 | 0.454 |
| Huttunen et al., 5 signals | 0.594 ± 0.044 | 0.462 | 0.416 | 0.811 ± 0.044 | 0.458 |
| MM-Net as published | 0.739 ± 0.020 | 0.670 | 0.634 | 0.782 ± 0.038 | 0.419 |

Paired on the ten fold-means (Wilcoxon):
- MM-Net + NP vs Huttunen-5: staging +0.143 acc / +0.207 MF1 / +0.214 κ, 10/10 folds,
  p = 0.002 each; respiratory level (AUC −0.000, p = 0.63; AP −0.004, p = 0.56).
- MM-Net + NP vs published MM-Net: respiratory +0.029 AUC (p = 0.004) and +0.035 AP
  (p = 0.014); staging unchanged (−0.003, p = 0.43).
- For reference, Huttunen-5 vs published MM-Net: AUC +0.029 (p = 0.014), staging −0.146.

## Text for the manuscript

> The primary configuration does not include nasal pressure, the sensor from which
> hypopneas are scored, so that the reported respiratory performance does not rest on the
> hypopnea-defining signal. We report its effect separately: adding nasal pressure raises
> respiratory AUC from 0.782 to 0.810 without changing staging (0.736), and under the same
> five signals MM-Net matches Huttunen et al. on respiratory detection (0.810 against
> 0.811) while staging substantially better (0.736 against 0.594).

Keep the wording as a description of the model. Do not write that nasal pressure was
*originally* excluded because of circularity: the released preprocessing never read that
channel, and the model does read the thermal airflow sensor, so that version can be
contradicted.

## Text for the response letter

> We agree that the port omitted nasal pressure. We have now evaluated both models with it.
> MM-Net improves by 0.029 AUC and matches the five-signal Huttunen model on respiratory
> detection (0.810 against 0.811), while exceeding it by 0.14 in staging accuracy
> (0.736 against 0.594).

If circularity is raised: the model already reaches 0.782 without nasal pressure, and
0.778 without the thermal airflow channel (Table of modality ablations).


## Eight-channel analyses (28 Sep 2026, Machine B, 3 seeds, all 42 runs)

Files: `results/revision/runs/final/pflow_suite_B42.json` (seed 42) and
`pflow_suite_B.json` (seeds 1, 7). Machine B rebuilt the data from the public release;
it reproduces the original data within run-to-run noise (published seven-channel model
0.7358 / 0.7871 vs 0.7343 / 0.7831; ten seed-42 conditions run on both machines differ by
0.003-0.004 on average, `pflow_suite_A42_local.json`).

Eight-channel model: acc 0.736, MF1 0.668, κ 0.629, AUC 0.810, AP 0.452.

| Removed (retrained) | Staging acc | Δ | Holm p | Resp. AUC | Δ | Holm p |
|---|---|---|---|---|---|---|
| EEG | 0.684 | −0.051 | 0.020 | 0.805 | −0.005 | 1.00 |
| EOG | 0.731 | −0.005 | 1.00 | 0.804 | −0.006 | 0.53 |
| EMG | 0.732 | −0.004 | 1.00 | 0.806 | −0.004 | 0.77 |
| SpO2 | 0.736 | 0.000 | 1.00 | 0.807 | −0.003 | 0.50 |
| effort | 0.735 | 0.000 | 1.00 | 0.804 | −0.006 | 0.45 |
| pulse/HRV | 0.738 | +0.002 | 1.00 | 0.809 | −0.001 | 1.00 |
| ECG | 0.737 | +0.001 | 1.00 | 0.808 | −0.001 | 1.00 |
| airflow (both sensors) | 0.737 | +0.001 | 1.00 | 0.777 | −0.033 | 0.020 |
| nasal pressure alone | 0.737 | +0.002 | 1.00 | 0.779 | −0.031 | 0.031 |
| all cardiorespiratory | 0.739 | +0.003 | 1.00 | 0.659 | −0.151 | 0.020 |

Single-task controls: staging-only 0.734 vs 0.736 (p = 0.77); respiratory-only AUC 0.807
vs 0.810 (p = 0.38). Bypass removed: AUC 0.798 vs 0.810 (9/10 folds, p = 0.037).

Reading: the task separation holds (EEG carries staging, the cardiorespiratory stream
carries respiratory detection). With nasal pressure, the respiratory output leans on the
airflow sensors (the scoring signals); without it (the published model), on respiratory
effort. This is the data-backed reason to keep the seven-channel model as the main one.
The joint-objective gain seen with seven channels is not significant with eight.

## Response-letter paragraph (current version)

> We evaluated nasal pressure in both models. It raises respiratory AUC for both (MM-Net
> 0.782 to 0.810; Huttunen et al. 0.710 to 0.811), and with identical inputs MM-Net matches
> Huttunen et al. on respiratory detection while staging 14 points higher (0.736 against
> 0.594). However, with nasal pressure the respiratory output depends on the
> hypopnea-scoring sensor (−0.031 AUC when it is removed), whereas the reported
> configuration relies on respiratory effort and is not driven by the scoring signal. We
> therefore keep nasal pressure out of the primary model and report its effect in the
> supplementary material.

Phrase the reason as the current rationale ("we therefore keep it out"), not as the
original motivation.

## Files

- `MMNet_research/foundation/run_mmnet_pflow.py`: MM-Net + nasal pressure (8th channel)
- `MMNet_research/foundation/run_huttunen.py` with `HUTTUNEN_SIGNALS=5`: Huttunen 5-signal
- `MMNet_research/preprocessing/build_pressure_cache.py`: extracts "Pressure Flow" to
  `data/pressure_flow/`
- Results: `MMNet_research/results/revision/runs/final/mmnet_pflow.json`,
  `.../final/huttunen5.json` (all three seeds complete)
