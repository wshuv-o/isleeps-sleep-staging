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

## Files

- `MMNet_research/foundation/run_mmnet_pflow.py`: MM-Net + nasal pressure (8th channel)
- `MMNet_research/foundation/run_huttunen.py` with `HUTTUNEN_SIGNALS=5`: Huttunen 5-signal
- `MMNet_research/preprocessing/build_pressure_cache.py`: extracts "Pressure Flow" to
  `data/pressure_flow/`
- Results: `MMNet_research/results/revision/runs/final/mmnet_pflow.json`,
  `.../final/huttunen5.json` (all three seeds complete)
