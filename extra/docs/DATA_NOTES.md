# iSLEEPS — data acquisition & preprocessing notes

Status as of download + first preprocessing pass. This file records what was
downloaded, what the data actually looks like, and decisions that were open in
the README but are now settled.

## 1. What was downloaded

- Source: **Zenodo record 14873844** (CC BY 4.0), 40-subject open subset.
- Location: `data/zenodo/` — `SN1`–`SN40`, each `SN<k>.edf` + `SN<k>.xlsx`.
- Size: **7.34 GB**, 80 files, all byte-verified against the Zenodo manifest
  (`data/zenodo/manifest.tsv`). OK=80, BAD=0, MISSING=0.
- The full 100-subject set + clinical metadata is **iHUB-Data only** (see §5).

## 2. EDF signal survey (all 40 subjects)

- Format EDF+C, ~6–9 h/night (range 3.93–9.31 h, mean 7.83 h).
- **EEG channels present in ALL 40:** `C4:M1, C3:M2, O2:M1, O1:M2` (2 central + 2 occipital).
- Frontal `F4:M1, F3:M2` present in only **28/40** → excluded from the common montage.
- **Sampling rate is mixed across subjects: 128 Hz AND 256 Hz** → resampling mandatory.
- Annotation file has **24 sheets** (README said 17); staging label sheet = `Sleep profile`,
  data begins at row 8 (`[8:]`), columns `Time | Value`.

**Montage decision (was README TODO):** use the 4 common EEG channels; single-channel
default `C4:M1`. Frontal cannot be used without dropping 12 subjects.

## 3. Preprocessing

- `preprocess/build_npz.py` — single-pass reproduction of the official two-stage
  pipeline (`iSLEEPS_preprocess_v2`), reusing the official `StagingPreprocess` class.
- Resample → **100 Hz**, 30 s windows → **3000 samples**, scale → µV (×1e6).
- Label map: `Wake→0, N1→1, N2→2, N3→3, REM→4`; `A`/`Movement`/`N4`/unknown dropped.
- Output: one `data/processed/SN<k>.npz` per subject (NOT the official tens-of-thousands
  of per-epoch files), keys: `x [n,4,3000] float32`, `y [n] int64`, `channels`, `subject`, `sfreq`.
- Verified: all 40 files `(4,3000)`, no NaNs, valid labels.

### Stage totals (40 subjects, 37,183 epochs)

| Stage | Count | %    | (README full-100 %) |
|-------|-------|------|---------------------|
| W     | 10661 | 28.7 | 26.2 |
| N1    | 3647  | 9.8  | 9.9  |
| N2    | 15253 | 41.0 | 41.3 |
| N3    | 3775  | 10.2 | 8.7  |
| R     | 3847  | 10.3 | 11.8 |

Close match to the published balance → pipeline validated.

## 4. Data-quality flags (act on these before modelling)

- **DUPLICATE: SN15 == SN28.** Extracted EEG arrays are bit-identical
  (`np.array_equal(x15, x28) == True`) and labels identical, despite differing
  EDF/xlsx packaging. Same night under two IDs. **Drop one → effective N = 39.**
  Subject-independent CV must never place both in different folds.
- **Stages entirely absent in some subjects** (matters for leave-k-subjects-out CV):
  - No **N3**: SN11, SN18, SN25, SN39
  - No **REM**: SN15, SN19, SN23, SN28, SN37
- Per-subject distributions are highly variable (e.g. SN19 is 84% wake, SN34 has 4 N1
  epochs) — expected for disrupted stroke sleep; weight loss / report per-class.

## 5. Clinical metadata (TODO-0) — BLOCKED

- `subject_description.xlsx` (**63 columns, all 100 subjects**) is documented to contain
  demographics (Age, Sex, BMI…), clinical params (AHI, cardiovascular disease, blood sugar…),
  comorbidities/risk factors, and **lesion location + stroke characteristics**.
- It is **NOT in the Zenodo release** — only on **iHUB-Data** (free registration):
  https://india-data.org/dataset-details/0b801dfa-4e42-4ec6-9c56-c6892b907ed2
- Until downloaded, **Pillar (ii) / E3 (lesion-aware error analysis) is blocked.**
- Parser ready: `python metadata/parse_subject_description.py --xlsx data/subject_description.xlsx`
  — reports per-column missingness and flags hemisphere / location / NIHSS-mRS / onset-time fields.

## 6. Environment

Python 3.14.3; numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, openpyxl 3.1.5, mne 1.12.1,
joblib 1.5.3, tqdm 4.67.3. The official repo pins (numpy 1.24, pandas 1.5, mne 1.3.1)
do not build on 3.14 — see `requirements.txt`. `sklearn` not yet installed; `torch`
import currently broken (needs clean CUDA reinstall before training).
