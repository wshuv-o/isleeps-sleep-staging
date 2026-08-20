# Evidence of Work

Supporting evidence that the work in this submission is genuinely ours, organised by the categories the brief lists.

## 1. Verification log — the claim-by-claim audit

`VERIFICATION_LOG.pdf` (included in this folder) records every headline claim in the manuscript, how it was checked, and whether it was **Confirmed**, **Corrected** or **Flagged**. It is the strongest single piece of evidence here, because it documents work that produced *negative* findings about our own paper — the kind of record that only exists if the checking actually happened.

Four claims came back **Corrected**:

| # | Claim as submitted | After checking |
|---|---|---|
| 4 | Abstract credited cross-modal attention with 0.721 / 0.700 | Those were the *concatenation* numbers; attention is 0.712 / 0.705 |
| 5 | Airflow removal left detection "marginally higher" | 0.704 vs 0.711 is within one fold SD — restated as "unchanged" |
| 7 | Cohort size stated as one number | Staging N=99, respiratory N=96; SN15/SN28 duplicate disclosed |
| 15 | AHI correlation presented as cohort-level | Naive name-join matched only **9 of 96**; corrected join gives ρ=0.315, p=0.0017, n=96 |

One item is **Flagged** and disclosed rather than fixed: SpO2 is recorded natively at 4 Hz and upsampled to the 25 Hz cardiorespiratory grid.

## 2. Experiment logs and result artifacts

All in the public repository, `github.com/wshuv-o/isleeps-sleep-staging`:

| Artifact | Path | What it evidences |
|---|---|---|
| Executed notebooks with saved outputs | `MMNet_Submission/all_codes/notebooks/` | Every figure and number traceable to a cell; notebook 1 re-runs end to end in 82.4 min |
| Per-fold result caches | `all_codes/results/engine_cache_per_fold/` | Raw ten-fold outputs for the headline, attention variant, SpO2 ablation and neural-only runs |
| Prediction artifacts | `all_codes/results/npz/predictions.npz` | Pooled `y_true` / `y_pred` / `apnea_score` over 89,532 epochs — a grader can recompute the headline directly |
| Summary CSVs | `all_codes/results/*.csv` | Benchmark, ablation grid, per-class F1, per-event-type AUC, severity, AHI |

## 3. Dataset access evidence

iSLEEPS is a public corpus. The raw distribution as downloaded — 97 EDF recordings, 103 annotation workbooks and `subject_description.xlsx` across five batch folders — is documented in `MMNet_research/docs/DATA_NOTES.md`, which also records the EDF channel survey across all subjects, the mixed 128 Hz / 256 Hz sampling rates that forced resampling, and the data-quality findings we made ourselves (the SN15/SN28 duplicate, and the subjects with no N3 or no REM).

## 4. Co-author repositories

The division of labour is visible across three separate commit histories:

- `github.com/wshuv-o/isleeps-sleep-staging` — main project, MM-Net, reproduction engine, manuscript
- `github.com/EsmeAbha/iSleep_experiments` — comparative experiment notebooks, per-fold results cache, reproducibility notes
- `github.com/Imtiaj-Sajin/Ischemic-Stroke-research-works` — preprocessing suite (`build_npz.py`, `channel_mapping.py`, `numpy_subjects.py`, `iSLEEPS_preprocessing.ipynb`), figure sources

## 5. Methodology-rewrite changelog

The module mapping table, the equations preserved verbatim, how the new component names were propagated, and the "ACTION REQUIRED FROM ME" list are in **Part 2 of `07_Guide_Compliance.pdf`**. The eight action items are all closed, and each maps to a numbered entry in the verification log.

## 6. Figure generation artefacts

`figures/README.md` (in the Overleaf project) records, per figure, the generation method and what was changed by hand — including the exact prompt used to generate the architecture diagram's draw.io XML and the five specific manual corrections made in draw.io afterwards. The `.drawio` sources are in the Overleaf project alongside the exported vectors.

## 7. AI session transcripts

> **To be added by the author.** Export the chat transcripts of the AI sessions used for drafting, for the referee report, and for the revision work, and place them in this folder. The brief asks for these explicitly.
