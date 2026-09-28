# Machine B status: eight-channel (nasal pressure) suite

Written 2026-09-28 by Machine B (RTX 3060 Ti, 23.8 GB RAM) for Machine A (RTX 2060).
Machine A was off, so Machine B took over its seed-42 share as well.

## Read this first, Machine A

- **Do not run the eight-channel suite for seed 42.** All 14 conditions are done and
  pushed, in `pflow_suite_B42.json`, not `pflow_suite.json`.
- **Seed 1 is running on Machine B now** (started 14:05, about 17:50 expected) and
  writes to `pflow_suite_B.json`. It is pushed when it finishes. Do not start it.
- **Before `git pull`:** if you saved any seed-42 suite results locally before the
  shutdown (`pflow_suite.json` entries with `|42`, `per_subject_8ch_seed42.json`,
  `predictions_8ch_seed42.npz`), move them aside first. The pull brings a complete
  `per_subject_8ch_seed42.json` and stops on an untracked file with the same name.
  `merge_pflow_suite.py` keeps the first copy of a key and reads `pflow_suite.json`
  first, so leftover local `|42` entries would override the complete set.

## Results

All in `MMNet_research/results/revision/runs/final/`.

| File | Contents | Commit |
|---|---|---|
| `pflow_suite_B42.json` | seed 42, all 14 conditions (incl. `full`) | `8ccb9e6` |
| `per_subject_8ch_seed42.json` | per-patient results, seed-42 `full` | `8ccb9e6` |
| `pflow_suite_B.json` | seed 7, all 14 conditions; seed 1 being added | `2fa707d` |

`full|1` and `full|7` come from the committed `mmnet_pflow.json` and are carried
over by the script, as the README describes. `predictions_8ch_seed42.npz` exists on
Machine B only, because `*.npz` is git-ignored.

To merge once seed 1 is pushed:

```bash
cd MMNet_research/foundation
python merge_pflow_suite.py pflow_suite_B.json pflow_suite_B42.json
```

### Nasal pressure, headline (seeds 42 / 7)

| Condition | Accuracy | Respiratory AUC |
|---|---|---|
| eight-channel `full` | 0.733 / 0.738 | 0.813 / 0.803 |
| `-nasal pressure` | 0.733 / 0.736 | 0.783 / 0.773 |
| `-airflow` (both sensors) | 0.738 / 0.739 | 0.779 / 0.771 |
| `-all cardio` | 0.736 / 0.738 | 0.657 / 0.659 |

Nasal pressure adds about 0.03 respiratory AUC and leaves staging unchanged.

## How the data was obtained on Machine B

The data folders could not be copied from Machine A, so they were rebuilt from the
public release: Figshare 29253068 (97 recordings, workbooks,
`subject_description.xlsx`) plus Zenodo 14873844 for SN2, SN13 and SN17. Every file
was MD5-checked against the manifest. Build order: `processed7`, `multimodal`
(plus `rebuild_missing_subjects.py` for the three Zenodo recordings), `mm_features`,
`cbramod_emb/labram`, `pressure_flow`.

Three preprocessing scripts still used paths from before the folder rename
(`MMNet_research/data` instead of `data/`). Their paths are fixed in this commit and
nothing else changed: `build_npz_full.py` (also reads `data/Dataset` instead of
`data/full100`), `build_multimodal.py` and `extract_mm_features.py` (also imports
from `preprocessing/` instead of the removed `processing/` and `extra/`).

Evidence that the rebuild matches the paper's data:

- Epoch counts match `per_subject_seed42.json` for all 99 patients in all five
  caches (92,560 epochs). SN15 and SN28 are bit-identical, as documented.
- `mmnet_core` loads 99 patients.
- The published seven-channel model, seed 42, retrained on the rebuilt data
  (`run_missing_channel.py published --seeds 42`, with the committed result file
  moved aside so it actually trained): acc 0.7358, AUC 0.7871, against the paper's
  0.7343 / 0.7831. The committed `missing_channel.json` was restored unchanged.
- Eight-channel `full`, seed 42: acc 0.7328, AUC 0.8128, against the committed
  `mmnet_pflow.json` 0.7344 / 0.8146.

## Two traps for whoever runs next

- `run_pflow_suite.py --res NAME` resolves `NAME` against the current directory, so
  the README's `--res pflow_suite_B.json` run from `foundation/` writes into
  `foundation/`. Pass `--res ../results/revision/runs/final/NAME`.
- One long `run_pflow_suite.py` process ran out of host RAM after 11 runs (memory
  builds up across runs). Machine B ran one condition per process instead:
  `--only=<condition>` (the `=` is needed for names starting with `-`), and the
  script's resume logic skips finished ones.

## Not done, and not needed for the nasal-pressure change

The baselines in the README's section 4 (DeepSleepNet, raw 14-channel CNN, 4-EEG CNN,
Bose SE-ResNet, healthy LSTM, respiratory raw-signal, missing-channel seeds 1/7) do
not use nasal pressure and were not re-run. The README lists permutation importance,
the learning curve and external validation as not yet adapted to eight channels;
those are open only if the paper reports them for the eight-channel model.
