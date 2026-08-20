# Sleep staging and respiratory-event detection in subacute ischemic stroke

Two research tracks on **iSLEEPS**, the first public polysomnography corpus of subacute ischemic-stroke patients (100 subjects; 99 usable after duplicate removal).

| Track | Paper | Headline |
|---|---|---|
| [`MMNet_research/`](MMNet_research/) | *A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection* | Staging **0.722** acc / κ **0.611**; respiratory **0.711** AUC — one model, both outputs |
| [`HAGNet_research/`](HAGNet_research/) | *HAG-Net: Interpretable Sleep Staging in Subacute Ischemic Stroke* | Staging **0.746** acc / κ **0.642** (EEG-only, matches published SOTA) |

MM-Net is the current submission. HAG-Net is the earlier EEG-only track.

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
2. Place it at `data/Dataset/`.
3. Build the arrays:

```bash
python MMNet_research/preprocessing/build_npz.py        # EEG/EOG/EMG -> data/processed7/
python MMNet_research/preprocessing/build_multimodal.py # + cardiorespiratory -> data/multimodal/
python MMNet_research/preprocessing/extract_mm_features.py  # -> data/mm_features/
```

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
