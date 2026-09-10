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

## 5. Clinical metadata — AVAILABLE (this section was wrong)

- `subject_description.xlsx` (**63 columns, all 100 subjects**) contains demographics
  (Age, Sex, BMI…), clinical params (AHI, cardiovascular disease, blood sugar…),
  comorbidities/risk factors, and **lesion location + stroke characteristics**.
- It is **not** in the Zenodo release, but it **is** in the Figshare release, which is
  public, CC BY 4.0 and needs no registration:
  **Figshare article 29253068**, DOI `10.6084/m9.figshare.29253068.v2` —
  https://figshare.com/articles/dataset/iSLEEPS_Polysomnography_Dataset_for_Sleep_Analysis_in_Indian_Ischemic_Stroke_Patients/29253068
- That release is 198 files / 17.5 GB: **97 EDF recordings spanning SN1–SN100**, 100
  annotation workbooks, and `subject_description.xlsx`.
- **Figshare is missing SN2, SN13 and SN17**; `data/zenodo/` supplies exactly those three.
  Figshare (97) + Zenodo = the 100-subject cohort, which is why `build_npz_full.py` globs
  both `data/zenodo/` and `data/full100/` and takes the first source that has a subject.
- This section previously said the full cohort and the metadata were **iHUB-Data only** and
  that lesion-aware analysis was **BLOCKED**. That was wrong, and acting on it cost real
  time: a run was scoped to the 40-subject Zenodo subset on the belief that nothing larger
  could be downloaded. The iHUB-Data mirror
  (https://india-data.org/dataset-details/0b801dfa-4e42-4ec6-9c56-c6892b907ed2) exists but is
  not the only route and is not needed.
- Parser ready: `python metadata/parse_subject_description.py --xlsx data/subject_description.xlsx`
  — reports per-column missingness and flags hemisphere / location / NIHSS-mRS / onset-time fields.

## 6. Environment

Python 3.14.3; numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, openpyxl 3.1.5, mne 1.12.1,
joblib 1.5.3, tqdm 4.67.3. The official repo pins (numpy 1.24, pandas 1.5, mne 1.3.1)
do not build on 3.14 — see `requirements.txt`. `sklearn` not yet installed; `torch`
import currently broken (needs clean CUDA reinstall before training).

## 7. External validation corpora — provenance

The paper cites both corpora from their canonical references. This section records
how the bytes were actually obtained, which is not the same question and which
nothing else in this repository answered.

### Sleep-EDF Expanded

Fetched from PhysioNet by `MMNet_research/domaingap/fetch_sleepedf.py`, the official
route, no complications. 40 Sleep Cassette recordings; arrays built by
`preprocessing/build_sleepedf_external.py`.

### ISRUC-Sleep — obtained from a mirror, and why

**The official archive was unreachable.** Every download link on
`sleeptight.isr.uc.pt` points at `dataset.isr.uc.pt`. That host did not resolve on
**5 September 2026**, and still did not on **10 September 2026** — `nslookup` times
out and HTTP returns no connection at all (curl status `000`), while the
`sleeptight` landing page itself serves normally (`200`). So the dataset was
advertised but not downloadable.

MIT-BIH `slpdb` was considered as a substitute and rejected: it carries no EOG, EMG,
SpO2 or airflow, so the respiratory head cannot be evaluated on it at all.

**What was used instead.** A public MEGA mirror:

    https://mega.nz/folder/wVgH2ZAJ#TTxlduGNt4TwaR1eNnl7IQ

MEGA encrypts client-side and the folder key travels in the URL fragment, so no
ordinary downloader works. `preprocessing/mega_folder.py` is a minimal read-only
client for exactly this: list a public folder, download from it, nothing else. It is
in the repository so that `data/isruc/` can be rebuilt; it is an acquisition
workaround, not part of the method.

    python MMNet_research/preprocessing/mega_folder.py --url "<url above>" --list
    python MMNet_research/preprocessing/mega_folder.py --url "<url above>" --out data/isruc

Contents: ISRUC-Sleep subgroup II, 8 subjects recorded on two nights, 16 `.rec`
files plus `.xlsx` and `.txt` annotations, about 2.2 GB.

**Verify the mirror before trusting it.** A third-party copy is a weaker provenance
claim than the official archive, and while that archive is down there is nothing to
checksum against. There is, however, a free and decisive check: the results file
`results/revision/runs/external_validation_isruc.json`, produced when the corpus was
first used, records exactly what the data contained.

| session | recordings | epochs | respiratory-event prevalence |
|---|---|---|---|
| 1 | 8 | 7,122 | 0.10110 |
| 2 | 8 | 7,019 | 0.02408 |

Run `preprocessing/build_isruc_external.py` over the mirror and compare. If all
three quantities match per session, the mirror is content-identical to the copy the
submitted results came from, which is a stronger statement than any mirror could
otherwise support. **If they do not match, do not use it** — either drop ISRUC or
disclose the mirror in the paper itself.

**Two format traps**, both of which fail quietly rather than loudly:

- MNE refuses `.rec`, so the EDF headers are parsed by hand.
- Channel labels are inconsistent between recordings; channel identity has to be
  resolved from the EDF transducer field instead.

**What the paper should say.** Keep citing Khalighi *et al.* — that is the dataset's
provenance and it is correct regardless of which server served the files. The mirror
belongs here, not in the manuscript, *provided the epoch-count check above passes*.
