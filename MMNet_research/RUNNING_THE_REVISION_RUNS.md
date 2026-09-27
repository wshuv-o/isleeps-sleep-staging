# Running the revision experiments on a second machine

The eight-channel model (the final model plus nasal pressure) and the remaining
baseline runs are split across two machines. Each machine writes its own results file;
merge at the end.

| Machine | Runs |
|---|---|
| Machine A (RTX 2060) | eight-channel suite, seed 42, all 14 conditions (`pflow_suite.json`) |
| **Machine B** | eight-channel suite, seeds 1 and 7 (`pflow_suite_B.json`), then the baseline list below |

## 1. Get the code

```bash
git clone https://github.com/wshuv-o/isleeps-sleep-staging.git
cd isleeps-sleep-staging
pip install -r requirements.txt      # torch with CUDA, numpy, scipy, scikit-learn
```

Python 3.10 or later. Linux or Windows both work; every script resolves paths from the
repository root.

## 2. Copy the data (not in git)

Copy these folders from machine A into `data/` at the repository root, keeping the names:

| Folder | Size | Needed for |
|---|---|---|
| `data/mm_features/` | 66 MB | every MM-Net run (100 files, SN1–SN100) |
| `data/multimodal/` | 4.1 GB | MM-Net raw cardio input, raw 14-ch CNN, respiratory baselines |
| `data/cbramod_emb/labram/` | 34 MB | MM-Net LaBraM embedding |
| `data/pressure_flow/` | 118 MB | nasal-pressure channel (eight-channel runs) |
| `data/processed7/` | 3.5 GB | 4-EEG CNN, DeepSleepNet, fold definitions |
| `data/sleep_edf_proc/` | 667 MB | only for the healthy-sleep LSTM run |

Check before starting: `data/mm_features` and `data/multimodal` must each hold 100 files.
With fewer, `mmnet_core` builds different folds from the paper's and the numbers are not
comparable.

```bash
python -c "import sys; sys.path.insert(0,'MMNet_research/model'); import mmnet_core as C; print(len(C.DATA))"   # must print 99
```

## 3. Machine B: eight-channel suite, seeds 1 and 7

```bash
cd MMNet_research/foundation
KMP_DUPLICATE_LIB_OK=TRUE python -u run_pflow_suite.py --seeds 1 7 --res pflow_suite_B.json
```

- 13 conditions per seed (the full model at seeds 1 and 7 is already done and is picked
  up automatically): 26 runs, about 50 min each on an RTX 2060, faster on a newer card.
- It resumes where it stopped; re-run the same command after any interruption.
- With at least 12 GB of GPU memory and 48 GB of RAM, run the two seeds as two processes,
  each with its own results file:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -u run_pflow_suite.py --seeds 1 --res pflow_suite_B1.json
KMP_DUPLICATE_LIB_OK=TRUE python -u run_pflow_suite.py --seeds 7 --res pflow_suite_B7.json
```

Run heavy jobs one per GPU otherwise: each MM-Net run holds about 6 GB of RAM.

## 4. Remaining baseline runs (either machine, one at a time)

All of these resume from their results files where noted.

```bash
cd MMNet_research/foundation
export KMP_DUPLICATE_LIB_OK=TRUE
python -u run_multichannel_10fold.py rawmm     --seeds 42 1 7   # raw 14-ch CNN, 10 folds (resumes)
python -u run_multichannel_10fold.py deepsleep --seeds 42 1 7   # DeepSleepNet, 4 EEG (resumes)
python -u run_multichannel_10fold.py cnn4ch    --seeds 42 1 7   # 4-EEG CNN, seeds 1 and 7 (resumes)
python -u run_resp_raw_baseline.py effort cardio7 --seeds 42 1 7   # raw-signal respiratory (resumes)
python -u run_missing_channel.py published indicator dropout --seeds 42 1 7   # (resumes)
python -u run_healthy_extra.py lstm                               # LSTM row, healthy Sleep-EDF
python -u run_bose_seresnet.py --seeds 1 7 --out ../results/revision/runs/final/bose_seresnet_s17.json
```

`run_bose_seresnet.py` does not resume and overwrites its output, which is why the extra
seeds go to their own file.

## 5. Bring the results back and merge

Copy machine B's `MMNet_research/results/revision/runs/final/pflow_suite_B*.json` (and any
baseline JSONs it produced) into the same folder on machine A, then:

```bash
cd MMNet_research/foundation
python merge_pflow_suite.py pflow_suite_B.json          # or pflow_suite_B1.json pflow_suite_B7.json
```

It merges into `pflow_suite.json` and prints the per-condition means over the seeds present.

## What the eight-channel suite runs

`run_pflow_suite.py`, conditions per seed, all on the paper's ten patient-independent folds:

| Condition | Meaning |
|---|---|
| `full` | eight-channel model; seed 42 also saves `predictions_8ch_seed42.npz` and `per_subject_8ch_seed42.json` |
| `-EEG`, `-EOG`, `-EMG` | neural modality removed |
| `-SpO2`, `-pulse/HRV`, `-ECG`, `-effort` | cardiorespiratory modality removed |
| `-airflow` | both airflow sensors removed (thermal and nasal pressure) |
| `-nasal pressure` | nasal pressure alone removed |
| `-all cardio` | whole cardiorespiratory stream removed |
| `stage-only`, `resp-only` | single-task controls (loss of one head only) |
| `no-bypass` | respiratory bypass removed |

Not yet adapted to eight channels: permutation importance, the learning curve and external
validation. See `MMNet_research/submission/NASAL_PRESSURE_NOTE.md` for the results so far
and the agreed wording.
