# Sleep staging and respiratory-event detection in subacute ischemic stroke

Two research tracks on **iSLEEPS**, the first public polysomnography corpus of subacute ischemic-stroke patients (100 subjects; 99 usable after duplicate removal).

| Track | Paper | Headline |
|---|---|---|
| [`MMNet_research/`](MMNet_research/) | *A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection* | Staging **0.739** acc / κ **0.634**; respiratory **0.782** AUC — one model, both outputs |
| [`HAGNet_research/`](HAGNet_research/) | *HAG-Net: Interpretable Sleep Staging in Subacute Ischemic Stroke* | Staging **0.746** acc / κ **0.642** (EEG-only, matches published SOTA) |

MM-Net is the current submission (**round-2 revision, under review — not published**).
HAG-Net is the earlier EEG-only track.

---

## ⚠ Before submitting — three things to decide

Training is finished. These are the only items that still need an author's
judgement, and two of them concern claims that the new results **contradict**.
Full detail and every number: **[`MMNet_research/SUBMISSION_HANDOFF.md`](MMNet_research/SUBMISSION_HANDOFF.md)**.

### 1. Two manuscript claims are now false, not merely stale

**C3 — the respiratory attribution.** The manuscript says SpO₂ carries the
read-out. Under the final model SpO₂ is the *most expendable* channel
(−0.002 AUC, p = 0.61) and **respiratory effort** is the anchor (−0.038,
Holm p < 0.001). The Results text and Table `tab:ablate` have been rewritten
accordingly; **the abstract, discussion and conclusion have not been checked
for surviving SpO₂ claims.** Search the source for `SpO` before submitting.

The replacement claim is stronger: the five individual cardiorespiratory
removals sum to −0.046 AUC while removing the stream as a whole costs −0.128,
a factor of **2.8**. The learned encoder distributes respiratory information
redundantly — each channel is dispensable because the others overlap it, while
the ensemble is essential. This is clinically favourable, since 12 of 100
recordings are missing at least one cardiorespiratory channel.

**C2 — the domain gap.** The manuscript says healthy-sleep deep models fail
here. That is one-directional and now beatable: a frozen public checkpoint
reaches parity with the 188 engineered features (TOST p = 0.008), so anyone can
run CBraMod and break the sentence. Replace with the measured asymmetry:

| direction | accuracy | κ |
|---|---|---|
| healthy-trained → **stroke** (3 models) | **0.313 ± 0.010** | 0.09–0.11 |
| **stroke-trained** → healthy (Sleep-EDF) | **0.794** | **0.717** |

> Transfer is strongly asymmetric. A model trained on pathological sleep
> generalises to healthy sleep; the reverse collapses.

### 2. Do not report the attribution agreement as one correlation

Referee Finding 2 asked for two independent attributions. Recomputed on the
final model they give **ρ = 0.833, p = 0.010 for staging** (the manuscript
reported 0.810) but **ρ = 0.381, p = 0.35 for respiration**. Pooling them to
0.691 is computable and misleading.

The respiratory correlation fails for a diagnosable reason: four of five cardio
channels have retraining effects indistinguishable from zero, so their rank
order is noise. **The respiratory panel should claim effort's dominance under
both methods** — ablation −0.038 and permutation −0.071, first by a wide margin
in both — not a rank correlation.

`figures/fig_attribution_agreement.py` still reads the **old** inputs and must be
repointed at `results/revision/runs/final/permutation_importance_final.json` and
`final/ablation_shards/` before the figure is regenerated.

### 3. Choose the multiplicity family, and declare it

EOG and EMG change verdict with the correction family:

| | EOG | EMG |
|---|---|---|
| Holm across all 9 conditions | 0.0795 **ns** | 0.1297 **ns** |
| Holm within family (3 neural → staging, 5 cardio → respiratory) | 0.0199 **sig** | 0.0199 **sig** |

The family-wise split is defensible — C4 is a modality-to-outcome mapping by
design, and nobody expects chin EMG to predict apnea — **but it is the more
favourable option, so declare it as pre-specified in the text.** If you would
rather not, report the pooled correction and describe EOG/EMG as directionally
consistent but not individually significant.

### Also outstanding

- **ISRUC is not run** — the data was never reachable here. Code is ready and
  tested via the identical Sleep-EDF path; see
  [`MMNet_research/foundation/ISRUC_INSTRUCTIONS.md`](MMNet_research/foundation/ISRUC_INSTRUCTIONS.md).
  Takes ~10 minutes once the data exists. **Dropping it is defensible** —
  Sleep-EDF already provides an external corpus. Do not delay submission for it.
- **Architecture figure**: swap in `figures/mm_architecture_v4.drawio` (or the
  ready-made `figures/fig_architecture_v4.pdf`). **Do not use v3** — it draws a
  model that was never trained.
- `\stale{}` red markers remain on figures whose underlying runs changed; remove
  each wrapper as its figure is regenerated.

---

## Environment

**Training (GPU) — used for all modelling**

| Component | Version |
|---|---|
| Python | 3.12.3 (conda) |
| PyTorch | 2.5.0 + CUDA 12.4 |
| NumPy / SciPy | 2.2.4 / 1.15.2 |
| scikit-learn | 1.6.1 |
| XGBoost / LightGBM | 3.3.0 / 4.6.0 |
| MNE | 1.12.1 |

**Hardware:** NVIDIA RTX 2060 (6 GB, compute capability 7.5), 16 GB system RAM, Windows 11.

> On Windows set `KMP_DUPLICATE_LIB_OK=TRUE` before training, or OpenMP raises a `libiomp5md.dll` double-initialisation error.

**Preprocessing** runs on system Python 3.14 (NumPy 2.4.4, pandas 3.0.2, MNE 1.12.1, openpyxl 3.1.5) — no GPU needed. PyTorch has no CUDA wheel for 3.14, which is why training uses the 3.12 environment.

Install: `pip install -r requirements.txt`

---

## Data

iSLEEPS is public but **not redistributed here** — `data/` is git-ignored, as it is clinical patient data.

1. Download the corpus (EDF recordings + annotation workbooks + `subject_description.xlsx`).
2. Place the recordings so that **both** raw layouts below are satisfied — the two build
   stages read different directories (see [`MMNet_research/docs/DATA_NOTES.md`](MMNet_research/docs/DATA_NOTES.md)):
   - `build_npz_full.py` globs `data/zenodo/` (the 40-subject open subset) and
     `data/full100/` (the full cohort, iHUB-Data only), or whatever `--raw` you pass;
   - `build_multimodal.py` globs `data/Dataset/` **recursively** for `*.edf` / `*.xlsx`,
     hard-coded with no flag to change it.

   The simplest arrangement is the recordings under `data/Dataset/` and `--raw data/Dataset`
   passed to `build_npz_full.py`, which satisfies both from one copy.
3. Build the arrays:

```bash
# 7-channel EEG/EOG/EMG -> data/processed7/
# (build_npz.py is the older 4-channel EEG-only variant and writes data/processed/ instead)
python MMNet_research/preprocessing/build_npz_full.py --raw data/Dataset --out data/processed7
python MMNet_research/preprocessing/build_multimodal.py     # + cardiorespiratory -> data/multimodal/
python MMNet_research/preprocessing/extract_mm_features.py  # -> data/mm_features/
```

> **Output paths.** All three scripts resolve their defaults relative to `MMNet_research/`,
> not the repository root, so run without arguments they write to `MMNet_research/data/...`
> while `mmnet_core` reads repo-root `data/`. Pass `--raw`/`--out` as above for
> `build_npz_full.py`; `build_multimodal.py` and `extract_mm_features.py` have no such flags
> (their `ROOT` is hard-coded), so either run them with `MMNet_research/data` symlinked to
> the repo-root `data/`, or move their output afterwards.

> **Extra dependency.** `build_multimodal.py` reads the cardiorespiratory channels with
> `pyedflib`, which is not in `requirements.txt`: `pip install pyedflib`.

This produces 100 subjects / 93,422 epochs at 100 Hz in 30-second epochs. Two known data facts are handled in code: **SN15 and SN28 are byte-identical** (same night, two IDs — collapsed to one patient and never split across folds), and several subjects lack N3 or REM entirely.

---

## Reproducing the paper

Everything in the MM-Net paper is produced by three executed notebooks, committed **with their outputs** so results are visible without re-running:

```
MMNet_research/MMNet_Submission/all_codes/notebooks/
├── 1_MM_Net_reproduction.ipynb    model, 10-fold training, ablation grid, baselines
├── 2_supplementary_analysis.ipynb retrain (seed 42), AHI, severity, clean figures
└── 3_figure_hypnogram.ipynb       whole-night qualitative figure (SN90)
```

Run top to bottom. Notebook 1 re-executes end to end in ~82 minutes on the hardware above.

### Where each paper result comes from

| Paper item | Value | Source |
|---|---|---|
| Headline staging | 0.722 acc / 0.651 mF1 / 0.611 κ | Notebook 1, training cell |
| Headline respiratory | 0.711 AUC / 0.337 AP | Notebook 1, respiratory head |
| Table V — staging benchmark | deep baselines 0.61–0.69 | Notebook 1 + `results/staging_benchmark.csv` |
| Table VI — modality ablation | 9 leave-one-out conditions | Notebook 1, ablation grid |
| Table VII — respiratory baselines | desat 0.596 / logreg 0.582 / gboost 0.670 | Notebook 1, `test/resp_baselines.py` |
| Per-event-type AUC | hypopnea 0.692, obstructive 0.763, central 0.840 | Notebook 1 |
| AHI association | ρ = 0.315, p = 0.0017, n = 96 | Notebook 2 |
| Staging by severity | 0.770 → 0.708 | Notebook 2 |
| Significance tests | respiratory p = 0.004; staging p = 0.91 | Notebook 1, Wilcoxon cells |
| Figures 4–10 | — | `MMNet_research/figures/` (see its README) |

Every number is produced by a live cell — none is read back from a cached JSON or CSV. The claim-by-claim audit is in [`VERIFICATION_LOG.md`](MMNet_research/MMNet_Submission/VERIFICATION_LOG.md), which records four corrections found during verification.

---

## Layout

```
MMNet_research/
├── model/          MM-Net + raw multimodal CNN
├── preprocessing/  EDF -> npz, neural + cardiorespiratory features
├── train/          10-fold reproduction engine, training entry points
├── test/           respiratory baselines, analyses
├── figures/        figure scripts, .drawio sources, figures/README.md
├── notebooks/      working copies of the executed notebooks
├── paper/          multimodal.tex, references.bib, compiled PDF
├── submission/     review report, response, contributions, compliance
└── MMNet_Submission/   self-contained bundle (code + notebooks + results)

HAGNet_research/    EEG-only track: model, preprocessing, train, paper
data/               git-ignored (clinical data)
```

## Citation and availability

iSLEEPS is publicly available; the accession is cited in the manuscript. The source cohort was collected under NIMHANS Institutional Ethics Committee approval — this work is a secondary analysis of that public release.
