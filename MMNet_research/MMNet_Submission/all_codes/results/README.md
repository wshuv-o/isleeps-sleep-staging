# Results

The **executed notebooks in `../notebooks/` are the authoritative live source** — every number in
the paper is produced by a cell there. The files here are an **export / record** of those results
for convenience, plus the raw artifacts a grader can use to recompute the headline numbers live.

## Summary CSVs (reported results — match `../paper/multimodal.pdf`)

| File | Contents |
|---|---|
| `headline_metrics.csv` | staging 0.722 / mF1 0.651 / κ 0.611 ; respiratory AUC 0.711 / AP 0.337 |
| `staging_benchmark.csv` | deep baselines (0.61–0.69) vs MM-Net (0.721) |
| `modality_ablation_grid.csv` | leave-one-out grid: SpO₂→resp 0.681, all-cardio→0.673, EOG→staging 0.712 |
| `per_class_f1.csv` | W/N1/N2/N3/R = 0.77/0.27/0.78/0.68/0.72 |
| `per_event_type_auc.csv` | hypopnea 0.692, obstructive 0.763, central 0.840 |
| `staging_by_severity.csv` | Normal 0.770 → Severe 0.708 |
| `respiratory_baselines.csv` | desat 0.596, logreg 0.582, gboost 0.670, MM-Net 0.711 |
| `ahi_clinical_validation.csv` | Spearman ρ=0.315, p=0.002, n=96 |

## Raw artifacts (recompute the numbers yourself)

| File | Contents | Recomputes |
|---|---|---|
| `npz/predictions.npz` | pooled 10-fold `y_true`, `y_pred`, `apnea_true`, `apnea_score` (89,532 epochs) | headline staging, per-class F1, confusion matrix, apnea AUC/AP |
| `npz/event_labels.npz` | per-epoch event types (hypopnea/obstructive/central) | per-event-type AUC |
| `npz/embeddings.npz` | per-epoch BiLSTM embeddings `h[89532,256]` + stage/apnea/sid | t-SNE figure |
| `npz/supp_artifacts.npz` | supplementary retrain (emb, stage, apnea, y_true, y_pred) | supplementary figures |
| `ahi.json` | per-patient clinical AHI | AHI correlation |

Example (recomputes the headline from the saved predictions):
```python
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
P = np.load("npz/predictions.npz")
print(accuracy_score(P["y_true"], P["y_pred"]))         # ~0.72 staging
print(roc_auc_score(P["apnea_true"], P["apnea_score"])) # ~0.71 respiratory
```

## Reproduction-engine cache (`engine_cache_per_fold/`)

Per-fold CSVs + raw JSON from the `mmnet_repro.py` reproduction engine
(`headline_concat`, `attention_cross`, `neural_only`, `loo_spo2`, `resp_baselines`). These are
separate engine runs and differ from the notebook in the third decimal (GPU non-determinism);
the notebook run is the one reported in the paper.

## `experiment_json/` — raw output of every benchmarked model (evidence for the tables)

The full set of experiment result files produced while building the benchmark and ablation
tables. Grouped by what they back (10-fold patient-independent unless noted):

- **Deep baselines (Table II):** `deepsleep_all` (DeepSleepNet), `attnsleep_real` (AttnSleep),
  `cnn4ch_all` (CNN+BiLSTM), `transfer_all` (Sleep-EDF transfer),
  `seq4ch_all` / `seq4ch_e45_all` / `seq_full_all` / `seq_aug_all` (sequence CNN-BiLSTM variants),
  `mm_cross` (raw multimodal CNN).
- **Feature / ensemble (staging ceiling):** `classical_all` (gradient boosting),
  `featseq_all` (feature-sequence BiLSTM), `refine_all` (learnable refiner),
  `kags_all` / `kags_all_v1` (graph-attention + state-space), `stack_all` (stacking),
  `ensemble_all` / `ensemble7_all` / `ensemble7_v2_all` / `ensemble7_v3_all` (gradient-boosting ensembles).
- **MM-Net (proposed):** `mm_final_cv`, `mm_feat_cv`, `mm_feat_cv_oldhead`, `mm_feat_ablation`
  (modality ablation).
- **Analysis:** `significance` (Wilcoxon tests), `persubject` (per-subject metrics),
  `domaingap` (healthy-vs-stroke gap), `conformal` (uncertainty), `n1_bias`, `lesion_ipsi`
  (lesion laterality), `k_compare` / `postproc_search` / `context_search` / `standalone_gate`
  (hyperparameter searches).

These are the reproduction engine's saved outputs; the notebooks re-derive the headline and
ablation numbers live.
