# MM-Net — Verify & Reproduce Submission

**Paper:** *A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and
Respiratory-Event Detection in Subacute Ischemic Stroke*
**Dataset:** iSLEEPS (first public polysomnography corpus of subacute ischemic stroke; 96 patients,
89,532 epochs used).

---

## What is in this folder

```
MMNet_Submission/
├── README.md                         ← this file
├── notebooks/                        ← run top-to-bottom; every number/figure is produced live
│   ├── 1_MM_Net_reproduction.ipynb   ← MAIN: model + 10-fold training + ablation grid + baselines
│   │                                    + per-event-type + AHI + Wilcoxon + t-SNE + confusion
│   ├── 2_supplementary_analysis.ipynb← retrains headline (identical), AHI (ρ=0.315), severity,
│   │                                    per-event-type, 7 clean figures
│   └── 3_figure_hypnogram.ipynb      ← whole-night qualitative figure (held-out SN90) + training curve
├── code/                             ← all MM-Net source (the notebooks INLINE this, so they are
│   │                                    self-contained; this is the standalone reference)
│   ├── mm_feature_net.py             ← the MM-Net model (two feature streams → fusion → BiLSTM → 2 heads)
│   ├── multimodal_net.py             ← raw-signal CNN baseline (the 0.655 "Raw multimodal CNN")
│   ├── mmnet_repro.py                ← 10-fold patient-independent reproduction engine
│   ├── features_v2.py                ← 188 EEG/EOG/EMG features (band power, spindles, slow waves, Hjorth)
│   ├── features.py                   ← base per-channel feature functions
│   ├── cardio_features.py            ← 14 cardiorespiratory features (SpO2, pulse/HRV, ECG, airflow, effort)
│   └── datasets.py                   ← fold assignment (seed 42) + duplicate handling (SN28)
├── paper/
│   ├── multimodal.pdf                ← corrected paper (matches the notebooks)
│   └── multimodal_source_overleaf.zip← LaTeX source (.tex + .bib + .bbl + figures) for Overleaf
├── VERIFICATION_LOG.md               ← Claim → Checked how → Confirmed / Corrected / Flagged
└── MM_Net_presentation.pptx          ← presentation deck (13 slides)
```

---

## How the notebooks map to the rubric

| Rubric item | Where |
|---|---|
| **1. Notebook reproducibility (30)** | All three notebooks; every number is produced by a live cell (no JSON/CSV loading of results). Headline staging **0.722 acc / κ 0.611**, respiratory **0.711 AUC / 0.337 AP** reproduce in notebook 1. |
| **2. Novel component (20)** | `code/mm_feature_net.py` — the two-stream fusion + BiLSTM + the **direct cardiorespiratory path** to the respiratory head; equations in the paper §Proposed Method. |
| **3. Ablation, run for real (15)** | Notebook 1, "modality-ablation grid" cells — leave-one-out over SpO2 / pulse-HRV / ECG / airflow / EOG / EMG / all-cardio, executed live (Table IV in the paper). |
| **4. Verification Log (15)** | `VERIFICATION_LOG.md` — section-by-section, includes the two **Corrected** claims and the **Flagged** SpO2 4→25 Hz upsampling. |
| **5. Live Q&A (20)** | `MM_Net_presentation.pptx`. |

---

## How to run

The notebooks are **provided already executed** (all cell outputs saved), so every reported number
and figure is visible without re-running.

To re-run from scratch you need the iSLEEPS data (not bundled — it is large and publicly released):

1. Place the per-subject feature files under `data/mm_features/SN*.npz` (produced by
   `features_v2.py` + `cardio_features.py` from the raw EDFs), and the raw EDFs under
   `data/Dataset/` (only notebook 3 needs the raw EDF, for the spectrogram).
2. Environment: Python 3, PyTorch, NumPy, scikit-learn, matplotlib, mne, scipy. A CUDA GPU is used
   if available (trained on a single NVIDIA RTX 2060; CPU also works, slower).
3. Open each notebook and **Run All**, in order 1 → 2 → 3.

The notebooks inline the model and training loop, so they do **not** import the `code/` folder —
`code/` is the standalone source for reference and reuse. Randomness is seeded to **42** for
deterministic runs.

---

## Honest notes (see VERIFICATION_LOG.md for the full list)

- **Data not included** — iSLEEPS is a public corpus and the feature/EDF files are large. The
  notebooks carry their outputs so the results are inspectable as submitted.
- **Two claims were corrected**, not kept: the abstract's "cross-modal attention" numbers were the
  *concatenation* variant's (concat is the reported headline; attention is an equivalent ablation),
  and airflow removal is "unchanged," not "marginally higher."
- **One item is flagged**: SpO2 is recorded natively at 4 Hz and upsampled to the 25 Hz
  cardiorespiratory grid.
- Every number in `paper/multimodal.pdf` traces to a cell in `notebooks/`.
