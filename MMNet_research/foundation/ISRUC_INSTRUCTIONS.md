# ISRUC external validation — run this on the machine that can reach the data

This is the one experiment the RTX 5080 session could not finish: the ISRUC host
was unreachable and every HuggingFace mirror we found ships six channels, not the
montage the feature extractor expects. **All the code is written and tested** on
Sleep-EDF, which uses the identical path. Once the data exists this takes about
ten minutes, most of it the single training run.

If ISRUC cannot be obtained, **dropping it is defensible** — Sleep-EDF already
provides an external corpus, and the manuscript states the montage caveat. Do not
delay submission for it.

---

## What you are producing

One number: zero-shot staging accuracy on ISRUC for a model trained on all 99
iSLEEPS patients, with no adaptation of any kind. For comparison, the same
protocol on Sleep-EDF gives **0.7940 acc / 0.7157 mF1 / 0.7174 kappa**.

ISRUC subgroup I (or III) carries respiratory annotations, so unlike Sleep-EDF
you may also get a respiratory AUC. The script emits it automatically when the
labels are present.

---

## Step 0 — environment

```bash
pip install pyedflib braindecode
pip install momentfm --no-deps        # only if you also rerun cardio embeddings
```

Pin `scipy==1.15.2`. On Windows, Smart App Control blocks unsigned scientific
wheels, and `pip install --force-reinstall` **makes it worse** by resetting the
file's reputation. Pin widely-deployed versions instead.

Always export `KMP_DUPLICATE_LIB_OK=TRUE` before anything that imports
`mmnet_core`.

You also need the iSLEEPS training data present (`data/processed7/`,
`data/multimodal/`, `data/cbramod_emb/labram/`), because the model is trained
here and only evaluated on ISRUC.

---

## Step 1 — get the data

ISRUC-SLEEP, subgroup I (100 subjects, one night each):
`https://sleeptight.isr.uc.pt/` → ISRUC-SLEEP → Subgroup I.

You need, per subject, the PSG `.rec`/`.edf` **and** the scorer's hypnogram
(`*_1.txt`). Put the extracted subject folders anywhere; the builder takes a path.

Required channels (the builder maps them; see step 2):
`C3-A2, C4-A1, O1-A2, O2-A1, LOC-A2, ROC-A1, X1 (chin EMG)`, plus
`SaO2, Flow, Thorax, Abdomen` for the cardiorespiratory branch.

---

## Step 2 — build the feature arrays

```bash
KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/preprocessing/build_isruc_external.py \
    --source /path/to/ISRUC/subgroupI \
    --out data/isruc_mm
```

Open `build_isruc_external.py` first and check the channel mapping block against
the names in your download — ISRUC's channel labels vary between subgroups and
between the `.rec` and `.edf` releases. The script raises on a missing channel
rather than silently zero-filling, which is what you want.

It must write, per recording, an `.npz` containing:

- `Feeg` — (n, 188) engineered neural features
- `Fcard` — (n, 14) engineered cardiorespiratory features, or all zeros if absent
- `y` — (n,) stage labels, mapped to `0=W 1=N1 2=N2 3=N3 4=R`
- `apnea` — (n,) 0/1 respiratory-event labels **if available**; omit the key otherwise
- `x` — (n, C, 3000) raw signal at 100 Hz, **required** for step 3

**If `x` is missing, step 3 cannot run.** Sleep-EDF stored it in a separate
`sleep_edf_proc/` directory; either is fine, you just have to point step 3 at
whichever holds the raw arrays.

Sanity check before continuing:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -c "
import numpy as np, glob
f = sorted(glob.glob('data/isruc_mm/*.npz'))
print(len(f), 'recordings')
d = np.load(f[0])
print({k: d[k].shape for k in d.files})
print('stages present:', np.unique(d['y']))
"
```

Expect 5 distinct stage values. If N3 is missing you have hit the R&K vs AASM
issue — ISRUC subgroup I is AASM-scored, but some mirrors ship S3/S4 separately.
Merge S3 and S4 into N3.

---

## Step 3 — build the frozen LaBraM embeddings

The final model's neural input is 388-d: the 188 engineered features **plus** a
200-d frozen LaBraM embedding. Without this step the model cannot run.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/foundation/build_labram_cache.py \
    --variant labram \
    --source data/isruc_raw \
    --out data/labram_ext/isruc \
    --eeg-idx 0 1 2 3 \
    --bs 64
```

`--source` is whichever directory holds the `x` arrays from step 2.

**`--eeg-idx` matters and is easy to get wrong.** It selects which four source
channels are fed to LaBraM as EEG, in the order `C4 C3 O2 O1`. LaBraM only
accepts canonical 10-20 names and never sees EOG or EMG.

- iSLEEPS stores `[C4:M1, C3:M2, O2:M1, O1:M2, E1:M2, E2:M2, EMG]`, so `0 1 2 3`.
- ISRUC typically stores `[C3-A2, C4-A1, O1-A2, O2-A1, LOC, ROC, X1]`. Note the
  **left/right order is swapped** relative to iSLEEPS. To match the training
  montage, pass `--eeg-idx 1 0 3 2`.
- Sleep-EDF has only two EEG derivations and needed `0 1 0 1` (duplication).

Getting this wrong does not raise an error — it silently feeds an EOG or EMG
trace to LaBraM as if it were EEG and quietly degrades the result. **Print your
channel list and check it by eye before running.**

This takes under a minute for 100 recordings.

---

## Step 4 — run the validation

```bash
KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/foundation/run_external_validation.py \
    --corpus isruc \
    --out MMNet_research/results/revision/runs/final/external_isruc.json
```

Trains once on all 99 iSLEEPS patients (~2 minutes) and evaluates on ISRUC. There
is no fold split and nothing is fitted to ISRUC, which is the point of the table.

**If ISRUC has no cardiorespiratory channels**, the branch is fed zeros and you
should also run the matched control, exactly as we did for Sleep-EDF:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/foundation/run_external_validation.py \
    --corpus isruc --match-train-cardio \
    --out MMNet_research/results/revision/runs/final/external_isruc_matched.json
```

Zeroing a BatchNorm'd CNN's input at test time while it trained on real signal is
a train/test mismatch, not an ablation. On Sleep-EDF the control came out
*lower* (0.7735 vs 0.7940), which is what showed the headline number was not a
mismatch artefact. Report the pair.

If ISRUC **does** carry cardio channels, build them into `Fcard` at the arm's
width instead — the loader now raises rather than silently zeroing a corpus that
actually has data.

---

## Step 5 — put it in the paper

The result goes in the external-validation table alongside Sleep-EDF. State the
same two caveats the Sleep-EDF row carries:

1. **Montage mismatch** is a confound on any performance drop, so the number is a
   lower bound on transfer, not a clean estimate of it.
2. **Scorer.** ISRUC ships two independent scorers; say which you used. Using
   scorer 1 is conventional. Inter-scorer agreement on ISRUC is roughly
   kappa 0.8, which is a ceiling on what any model can score against it.

---

## Known failure modes

| symptom | cause |
|---|---|
| `KeyError: '<recording id>'` in `subj_infer` | the arm's `subj_infer` is a recompiled copy whose globals are a snapshot; external records must be merged into `C.DATA` **in place**, not rebound. Already fixed in `run_external_validation.py` — do not reintroduce `C.DATA = {...}`. |
| `RuntimeError: replaced N occurrences of 188, expected M` | `mmnet_core.py` was edited; the width-substitution patch in `arms.py` needs rechecking. It fails loudly by design. |
| reshape error inside `CardioCNN` | `Fcard` width does not match the arm's 5250. See step 4. |
| `rc=3221226505` after the result printed | CUDA/OpenMP teardown crash *after* the output was written. Check the output file, not the return code. |
| all-Wake predictions | almost always a stage-label mapping error in step 2. Check `np.unique(y)` and the W/N1/N2/N3/R order. |
