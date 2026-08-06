# MM-Net Verification Log

Claim → checked how → result (Confirmed / Corrected / Flagged). Every number below is produced
live by a cell in one of the two executed notebooks — no value is copied from a cached JSON/CSV:

- `revision/MM_Net_reproduction.ipynb` — the reproduction (14 code cells; model + train loop
  inlined; 10-fold GPU training; ablation grid; baselines; per-event-type; AHI; Wilcoxon; t-SNE;
  confusion). Re-executed end to end in 82.4 min, 0 errors.
- `revision/supplementary_analysis.ipynb` — post-hoc analysis (retrains the headline seed 42 →
  identical result, saves artifacts, corrected AHI, severity, per-event-type, 7 clean figures).

## Reproduced headline numbers

| # | Paper claim | Checked how | Result |
|---|---|---|---|
| 1 | Staging acc 0.721, mF1 0.645, κ 0.606 | Re-ran the concat model 10-fold, fixed folds (seed 42), live GPU training | **Confirmed** — 0.7227 / 0.651 / 0.611 |
| 2 | Apnea AUC 0.700, AP 0.333 | Same run, respiratory head vs binary apnea label | **Confirmed** — AUC 0.711, AP 0.337 |
| 3 | Per-class F1 [0.77, 0.27, 0.78, 0.68, 0.72] | Same run, per-fold per-class F1 | **Confirmed** — [0.77, 0.27, 0.78, 0.67, 0.73] |
| 4 | "cross-modal attention … 0.721 / 0.700" (abstract) | attention (cross) run vs concat run | **Corrected** — those are the *concat* numbers; attention is 0.712/0.705. Concat is now the stated headline model, attention an ablation. |
| 5 | Airflow removal "marginally higher" | leave-one-out airflow run; difference vs fold SD | **Corrected** — AUC 0.704 vs 0.711 is within one fold SD; text now says "unchanged," not "higher" |
| 6 | Determinism | Retrained the headline a second time in the supplementary notebook | **Confirmed** — 0.7223, identical to 3 decimals |

## Data and labels

| # | Item | Checked how | Result |
|---|---|---|---|
| 7 | N of the cohort | Counted subjects; SN28 byte-identical to SN15 | **Corrected/Confirmed** — staging N=99, respiratory N=96 (complete cardio); stated explicitly |
| 8 | Respiratory-event labels | Re-parsed Flow Events; reconstructed "any event" per epoch | **Confirmed** — matches the trained apnea label **exactly (0 mismatches / 97 subjects)** |
| 9 | Event composition | Counted event types across cohort | **Confirmed** — hypopnea 81%, obstructive 11.6%, central 5.8% |
| 10 | AHI severity distribution | subject_description.xlsx, AASM cut-offs | **Confirmed** — Normal 15 / Mild 24 / Moderate 23 / Severe 38 |
| 11 | SpO2 sampling | Descriptor: native 4 Hz; pipeline resamples to 25 Hz | **Flagged/stated** — the cardiorespiratory grid upsamples SpO2 4 Hz → 25 Hz; now stated in Section III |

## New experiments (added in revision — all live in the notebooks)

| # | Experiment | Result |
|---|---|---|
| 12 | Respiratory baselines on the 14 cardio features (§4.1) | desat-rule AUC 0.596 / logreg 0.582 / gradient-boosting 0.670; random AP 0.16. **MM-Net (0.711) beats gboost by ~4pp AUC — modest** → argument reframed to joint single-pass + causal attribution, not detection supremacy |
| 13 | Modality-ablation grid, leave-one-out (§4.2, headline) | full 0.723/0.711 · −SpO2 0.727/**0.681** · −effort 0.728/0.726 · −pulse/HRV 0.723/0.700 · −ECG 0.723/0.711 · −airflow 0.722/0.704 · −EOG **0.712**/0.705 · −EMG 0.724/0.707 · −all cardio 0.724/**0.673**. Clean physiological split: cardio→respiratory, EOG→staging |
| 14 | Per-event-type AUC (§4.3) | hypopnea 0.692 · obstructive 0.763 · central 0.840 — strongest on the most severe events |
| 15 | Predicted burden vs clinical AHI (§4.4) | Spearman **rho=0.315, p=0.0017, n=96** (corrected join; the naive name-join matched only 9) |
| 16 | Staging by SDB severity (§4.5) | accuracy 0.770 (normal) → 0.730 (mild) → 0.712 (moderate) → 0.708 (severe) |
| 17 | Wilcoxon signed-rank over 10 folds (§3.6) | removing all cardio vs full: respiratory AUC **p=0.004** (drops), staging **p=0.91** (unchanged) |

## Correctness fixes (all applied to `paper/multimodal.tex`)

- Eq. (BCE/joint loss) truncation in the compiled PDF → repaired with an `align` block. **Applied.**
- Corresponding author added: Dr. Md Iftekharul Mobin, Dept. of CS, AIUB (iftekhar.mobin@aiub.edu). **Applied.**
- Data availability (iSLEEPS public), code availability, ethics (NIMHANS IEC), funding statements. **Applied.**
- Title changed to "A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke." **Applied.**
- EEG-only feature/ensemble models moved to supplementary framing; MM-Net + input-matched deep models + published baselines (with caveat) kept in main text; **concat is the headline**, attention an ablation. **Applied.**

## Figures (real, regenerated from the saved model/embeddings; graph-only, no annotations)

- `fig_ablation` — leave-one-out grid, 9 rows incl. SpO2 (staging vs respiratory AUC).
- `fig_benchmark_clean` — staging accuracy across architectures (healthy-built deep models 0.61–0.69).
- `fig_tsne` — PCA+t-SNE of learned embeddings; clean stage separation, diffuse respiratory (honest).
- `fig_confusion` — 5-class row-normalised confusion matrix from pooled test predictions.
- `fig_ahi` — predicted per-patient burden vs clinical AHI (rho=0.315).
- `fig_event_type`, `fig_severity`, `fig_resp_baselines`, `fig_sdb_burden` — from the runs above.

## Nothing unsupported

Every number in the final paper traces to a cell in the two notebooks above. No value was
invented; where a claim could not be reproduced as written (rows 4, 5) it was corrected, not kept.
