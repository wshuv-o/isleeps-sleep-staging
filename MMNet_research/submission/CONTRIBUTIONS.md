# Contributions

**Manuscript:** A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke

Roles follow the **CRediT** taxonomy (the contributor-role standard required by IEEE and Elsevier at submission), each followed by the specific artifacts the person produced.

## Repositories

| Author | Repository | Holds |
|---|---|---|
| Md Wahiduzzaman Suva | `github.com/wshuv-o/isleeps-sleep-staging` | Main project: MM-Net, reproduction engine, executed notebooks, manuscript |
| Esm-e Moula Chowdhury Abha | `github.com/EsmeAbha/iSleep_experiments` | Experiment notebooks, comparative model runs, results cache, reproducibility notes |
| Md Imtiaj Alam Sajin | `github.com/Imtiaj-Sajin/Ischemic-Stroke-research-works` | Preprocessing pipeline, channel mapping, feature extractors, figure sources |

---

## Md Wahiduzzaman Suva — Lead author

**CRediT roles:** Conceptualization · Methodology · Software · Investigation · Validation · Formal analysis · Writing – original draft · Project administration

Conceived the joint staging-and-respiratory formulation and led the study end to end. Designed and implemented the proposed MM-Net architecture — the two-stream encoder, the concatenative fusion, the BiLSTM temporal decoder, and the direct cardiorespiratory path to the respiratory head (`code/mm_feature_net.py`), together with the raw-signal multimodal CNN (`code/multimodal_net.py`). Wrote the ten-fold patient-independent reproduction engine and the fold-assignment logic enforcing patient independence and duplicate handling (`code/mmnet_repro.py`, `code/datasets.py`).

Ran the joint staging and respiratory experiments, the nine-condition modality-ablation grid, and the fold-wise Wilcoxon significance tests, and established the headline result. Led the verification pass that re-derived every reported number live — identifying and correcting the fusion-variant misattribution, the AHI join defect, and the abstract/table inconsistency.

Wrote **Sections I, V, VI, VII and VIII**, produced **Tables III–VI**, authored notebook `1_MM_Net_reproduction.ipynb`, assembled the manuscript, and prepared the response to review.

**Evidence:** `github.com/wshuv-o/isleeps-sleep-staging` — full commit history, executed notebooks with saved outputs, `VERIFICATION_LOG.md`.

## Esm-e Moula Chowdhury Abha — Comparative model training, architecture figure and literature

**CRediT roles:** Investigation · Software · Validation · Visualization · Writing – original draft

Trained and evaluated deep comparison models on the same patient-independent folds — DeepSleepNet, AttnSleep and the healthy-pretrained Sleep-EDF transfer model — contributing the baseline evidence for the paper's finding that architectures built for healthy sleep lose 15–20 accuracy points on this cohort. Assisted with hyperparameter tuning for the classical gradient-boosting comparisons, and maintained the per-fold results cache used to cross-check reported numbers.

**Produced the architecture diagram (Fig. 3)** through the required AI → XML → draw.io → vector pipeline: generated the initial `mxGraphModel` XML from a structured prompt, then refined it by hand in draw.io — re-laying out the two streams onto a shared baseline, re-routing the direct cardiorespiratory connection, correcting the tensor-dimension labels against `mm_feature_net.py`, and applying the manuscript palette — before exporting to vector PDF.

Wrote **Section II (Related Work)** and **Section III (Critical Gaps and Limitations)**, produced the recent-literature comparison in **Table VIII**, and compiled and verified the 49-reference bibliography.

**Evidence:** `github.com/EsmeAbha/iSleep_experiments` — experiment notebooks (`isleep-nb10.ipynb`, `isleep-final-version.ipynb`), per-fold results cache, `REPRODUCIBILITY.md`, CI workflow.

## Md Imtiaj Alam Sajin — Polysomnography preprocessing, respiratory baselines and result figures

**CRediT roles:** Data curation · Software · Investigation · Visualization

Built the polysomnography preprocessing pipeline: EDF parsing, channel mapping and harmonisation across the mixed 128 Hz / 256 Hz recordings, resampling to the common grid, 30-second epoching, and conversion to per-subject arrays (`build_npz.py`, `build_npz_full.py`, `channel_mapping.py`, `numpy_subjects.py`, `iSLEEPS_preprocessing.ipynb`). Implemented feature extractors including the cardiorespiratory set covering SpO2, pulse/HRV, ECG, airflow and effort (`features.py`, `features_v2.py`, `cardio_features.py`).

Trained and evaluated the respiratory reference detectors — the clinical desaturation rule, logistic regression and gradient boosting on the cardiorespiratory features (`test/resp_baselines.py`) — which calibrate the proposed model's detection result. Produced the **result figures (Figures 4–10)** and **Tables I, II and VII**, and wrote **Section IV (Dataset and Preprocessing)**.

**Evidence:** `github.com/Imtiaj-Sajin/Ischemic-Stroke-research-works` — `project/processing/` preprocessing suite and notebook, dataset documentation, figure sources.

## Md Iftekharul Mobin — Supervision (corresponding author)

**CRediT roles:** Supervision · Project administration · Conceptualization · Writing – review & editing · Resources

Supervised the project and defined its clinical framing — that a model for this cohort must be evaluated on clinical utility rather than on a staging leaderboard. Directed the scoping of the research questions and the choice of a patient-independent protocol. Reviewed and critically revised the manuscript across drafts, and directed the revision that made concatenation the stated headline model after the fusion-variant misattribution was identified. Provided institutional resources and computing access. Serves as corresponding author.

---

## Note on the commit history

Each author maintains their own public repository, listed above, and the division of labour described here is visible across those three histories. The main project repository shows a single committer because every training run was executed on the lead author's machine and GPU and pushed from that account; the preprocessing suite and the comparative experiment notebooks were developed in the co-authors' own repositories and are traceable there.
