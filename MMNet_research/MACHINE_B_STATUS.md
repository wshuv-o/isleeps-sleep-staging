# Machine B: eight-channel suite, record of work

Machine B: RTX 3060 Ti (8 GB), 23.8 GB RAM, Windows 11, Python 3.11, torch 2.7.0+cu126.
Dates: 2026-09-27 to 2026-09-28.

## Runs completed

`run_pflow_suite.py`, all 14 conditions, seeds 42, 7 and 1. Files in
`MMNet_research/results/revision/runs/final/`:

| File | Contents |
|---|---|
| `pflow_suite_B42.json` | seed 42, 14 conditions |
| `pflow_suite_B.json` | seeds 7 and 1, 14 conditions each |
| `per_subject_8ch_seed42.json` | per-patient results, seed-42 `full` |

`full|1` and `full|7` in `pflow_suite_B.json` were carried over by the script from
`mmnet_pflow.json`. `predictions_8ch_seed42.npz` was written on Machine B and is not in
git (`*.npz` is ignored).

Runs were started one condition per process
(`run_pflow_suite.py --seeds S --only=<condition> --res ../results/revision/runs/final/<file>`).

## Results (accuracy / respiratory AUC, mean over the ten folds)

| Condition | Seed 42 | Seed 7 | Seed 1 |
|---|---|---|---|
| full | 0.7328 / 0.8128 | 0.7382 / 0.8028 | 0.7365 / 0.8140 |
| -EEG | 0.6806 / 0.8064 | 0.6826 / 0.8035 | 0.6902 / 0.8061 |
| -EOG | 0.7216 / 0.8064 | 0.7339 / 0.8030 | 0.7377 / 0.8023 |
| -EMG | 0.7319 / 0.8060 | 0.7285 / 0.8039 | 0.7345 / 0.8083 |
| -SpO2 | 0.7343 / 0.8085 | 0.7358 / 0.8070 | 0.7378 / 0.8052 |
| -pulse/HRV | 0.7347 / 0.8122 | 0.7384 / 0.8098 | 0.7418 / 0.8059 |
| -ECG | 0.7402 / 0.8131 | 0.7347 / 0.8022 | 0.7351 / 0.8101 |
| -airflow | 0.7381 / 0.7787 | 0.7388 / 0.7707 | 0.7348 / 0.7803 |
| -nasal pressure | 0.7332 / 0.7826 | 0.7361 / 0.7725 | 0.7431 / 0.7827 |
| -effort | 0.7370 / 0.8016 | 0.7336 / 0.8023 | 0.7359 / 0.8075 |
| -all cardio | 0.7363 / 0.6574 | 0.7384 / 0.6590 | 0.7422 / 0.6602 |
| stage-only | 0.7338 / 0.4606 | 0.7314 / 0.5742 | 0.7355 / 0.5046 |
| resp-only | 0.2092 / 0.8075 | 0.2123 / 0.8069 | 0.1709 / 0.8076 |
| no-bypass | 0.7307 / 0.7974 | 0.7345 / 0.7911 | 0.7403 / 0.8045 |

## Data

The data folders were built on Machine B from the public release: Figshare 29253068
(97 EDF recordings, annotation workbooks, `subject_description.xlsx`) and Zenodo
14873844 (SN2, SN13, SN17). All files were MD5-checked against the published
manifests. Build order: `build_npz_full.py` → `data/processed7`,
`build_multimodal.py` and `rebuild_missing_subjects.py multimodal` → `data/multimodal`,
`extract_mm_features.py` → `data/mm_features`,
`build_labram_cache.py --variant labram` → `data/cbramod_emb/labram`,
`build_pressure_cache.py` → `data/pressure_flow`. Each folder holds 100 files.

Path changes in `build_npz_full.py`, `build_multimodal.py` and
`extract_mm_features.py` (commit `f27e7d9`): data resolved under `data/` at the
repository root instead of `MMNet_research/data`, `data/Dataset` instead of
`data/full100`, and imports from `preprocessing/` instead of `processing/` and `extra/`.

Checks on the built data:

- Epoch counts equal the lengths in `per_subject_seed42.json` for all 99 patients in
  all five folders (92,560 epochs). SN15 and SN28 are bit-identical.
- `mmnet_core` loads 99 patients.
- `run_missing_channel.py published --seeds 42`, run with the committed
  `missing_channel.json` moved aside and then restored: acc 0.7358, AUC 0.7871
  (`final_model.json` `final|42`: 0.7343 / 0.7831).
- Eight-channel `full`, seed 42: acc 0.7328, AUC 0.8128 (`mmnet_pflow.json`
  `pflow|42`: 0.7344 / 0.8146).

## Commits

| Commit | Contents |
|---|---|
| `2fa707d` | `pflow_suite_B.json`, seed 7 |
| `8ccb9e6` | `pflow_suite_B42.json`, `per_subject_8ch_seed42.json` |
| `f27e7d9` | preprocessing path changes |
| this commit | `pflow_suite_B.json` with seed 1 added; this record |
