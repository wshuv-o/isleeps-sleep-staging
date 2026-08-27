# Response to Reviewer

**Manuscript:** A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke
**Manuscript ID:** MMNet-2026-0713
**Authors:** Md Wahiduzzaman Suva, Esm-e Moula Chowdhury Abha, Md Imtiaj Alam Sajin, Md Iftekharul Mobin
**Round:** Response to Major Revision

Page references are to the revised manuscript (`02c_Paper_FRESH_final.pdf`). Every change is visible in the highlighted-changes copy (`02b`), produced by `latexdiff` against the submitted version (`02a`).

---

## Preface

We thank the reviewer for a report that engaged with our actual numbers rather than the narrative. The central criticism — that the empirical work was too thin to sustain the claims — was correct. The submitted version reported one ablation condition, no respiratory baseline, no significance test, no clinical validation, and 11 references.

The revision is substantial: the manuscript grew from 5 to 11 pages, Results expanded from 2 subsections to 10, tables from 3 to 9, and references from 11 to 49. **Every new number is produced by a live cell** in the executed notebooks; none is copied from a cached file. The reproduction notebook re-runs end to end in 82.4 minutes.

**We accept 10 findings in full and 2 in part** (Findings 1 and 9), with technical justification given below.

| # | Reviewer's concern | Disposition | Where changed |
|---|---|---|---|
| 1 | Non-circularity untested; null reported as increase | **Accepted in part** | §IV p.3; §VI-F p.6; §VI-G p.6; Abstract p.1 |
| 2 | Ablation is one condition | Accepted in full | §VI-F p.6; Table VI p.8 |
| 3 | No respiratory baseline | Accepted in full | §VI-H p.7; Table V p.7 |
| 4 | No statistical testing | Accepted in full | §VI-A p.5; §VI-F p.6 |
| 5 | No clinical validation | Accepted in full | §VI-I p.8; §VI-H p.7 |
| 6 | Cohort inconsistent; no data audit | Accepted in full | §IV p.3; §VI-A p.5 |
| 7 | Fusion described too vaguely | Accepted in full | §V-D p.5; Abstract p.1; §VI-D p.6 |
| 8 | Attribution asserted, not shown | Accepted in full | §VI-F p.6; §VII p.9 |
| 9 | References too few; citation attribution | **Accepted in part** | References p.10–11; Table VIII p.9 |
| 10 | No limitations section | Accepted in full | §VII-A p.9; §VII-B p.10 |
| 11 | No critical-gaps section | Accepted in full | §III p.3 |
| 12 | Declarations absent | Accepted in full | Front matter p.1; back matter p.10 |

---

## Finding 1 — Non-circularity claim untested; a null reported as an improvement

**Accepted in part.** We ran the experiment the reviewer asked for and corrected the wording. We differ only on the interpretation of the result.

**(a) The reviewer was right that the ablation did not test the claim.** AASM defines a hypopnea as reduced airflow *plus* a ≥3% desaturation or an arousal, so removing airflow while keeping SpO2 removes one definitional input and retains the other. We now state the AASM event definition explicitly in **§IV (p.3)** so the reader can see the input/label overlap, and we ran the ablation the reviewer specified — removing the entire cardiorespiratory stream, airflow and SpO2 together (**§VI-F, p.6; Table VI, p.8**):

| Condition | Staging acc | Respiratory AUC |
|---|---|---|
| Full model | 0.723 | **0.711** |
| − airflow only | 0.722 | 0.704 |
| − SpO2 only | 0.727 | **0.681** |
| − all cardiorespiratory | 0.724 | **0.673** |

**(b) The wording is corrected.** "Marginally higher" is gone. The text now states the airflow difference is **unchanged**, because 0.704 against 0.711 is within one fold standard deviation.

**Where we respectfully differ.** We do not conclude the read-out is circular, and we have not withdrawn the physiological argument — we have restated it as what the data support. With every cardiorespiratory channel removed, detection falls to 0.673 but does not collapse to chance; the residual signal is the cortical arousal following an obstructed breath, visible in EEG and EOG. Circularity would mean the model recovers the label from a channel that mechanically encodes it. What the grid shows is a graded dependence in which no single channel is necessary and the largest contributor is the one physiology predicts. **§VI-G (p.6)** now says the read-out "degrades gracefully and in a physiologically ordered way," and no longer uses the airflow ablation as a non-circularity proof.

---

## Finding 2 — The ablation consists of one condition

**Accepted in full.** The single airflow row is replaced by a complete leave-one-out grid over every channel group, executed live on the same folds (**§VI-F, p.6; Table VI, p.8; Fig. 8**). Nine conditions: full model, −SpO2, −effort, −pulse/HRV, −ECG, −airflow, −EOG, −EMG, −all cardiorespiratory. Both staging accuracy and respiratory AUC are reported for every row. **No cell is left pending or unreported.**

---

## Finding 3 — The respiratory result has no baseline

**Accepted in full.** Three reference points added on identical folds (**§VI-H, p.7; Table V, p.7**):

| Detector | Respiratory AUC |
|---|---|
| Clinical desaturation rule | 0.596 |
| Logistic regression (14 cardio features) | 0.582 |
| Gradient boosting (same features) | 0.670 |
| **Proposed model** | **0.711** |

We also accepted the consequence the reviewer anticipated. The margin over gradient boosting is ~4 points of AUC, which is modest, and the text says so rather than presenting the model as a decisive winner. The contribution is reframed around the **joint single-pass formulation** — one compact model producing both clinical outputs with causal modality attribution — not detection supremacy.

---

## Finding 4 — No statistical testing

**Accepted in full.** Every results table now carries fold-wise mean ± SD (**§VI-A, p.5**), and we ran the paired test the reviewer specified across the ten folds (**§VI-F, p.6**):

| Comparison | Metric | Wilcoxon p | Conclusion |
|---|---|---|---|
| Full vs − all cardiorespiratory | Respiratory AUC | **0.004** | Significant drop |
| Full vs − all cardiorespiratory | Staging accuracy | **0.91** | Unchanged |

This pair is now central rather than incidental: it is the cleanest statement of the paper's finding. Directional language has been removed wherever a difference does not clear the noise.

---

## Finding 5 — Clinical utility claimed without clinical validation

**Accepted in full.** Three analyses added.

**AHI association (§VI-I, p.8; Fig. 9).** Predicted per-patient event burden against the clinically scored index: **Spearman ρ = 0.315, p = 0.0017, n = 96**, with all three quantities stated. We report this as a modest association, not validation, and the section is titled accordingly. The discussion states plainly that the model tracks severity at group level and is not a substitute for a scored AHI.

**Severity stratification (§VI-I, p.8).** Staging accuracy by AASM band: 0.770 normal → 0.730 mild → 0.712 moderate → 0.708 severe.

**Per-event-type detection (§VI-H, p.7).** Hypopnea 0.692 (81.0% prevalence), obstructive apnea 0.763 (11.6%), central apnea 0.840 (5.8%). The pattern was invisible in the pooled number: detection is strongest on the most severe events, and hypopneas dominate the pooled label and drag the aggregate toward the hardest class. The reviewer's point about central events after stroke is addressed in the discussion.

---

## Finding 6 — Cohort inconsistent; no data-quality audit

**Accepted in full.** **§IV (p.3)** now states each cohort size separately — **staging N = 99, respiratory N = 96** (patients with complete cardiorespiratory channels), **AHI analysis n = 96** — which resolves the 99/96 discrepancy the reviewer caught.

The audit is reported. Two records, SN15 and SN28, are **byte-identical** in both signal arrays and labels — the same night released under two identifiers. We verified this by array equality across the full recording, and the two are treated as one patient and assigned to the same fold, so patient independence holds; the fold logic is in `code/datasets.py`. We also report the subjects lacking stages entirely (four with no N3, five with no REM), which matters for per-class metrics under patient-independent splits.

---

## Finding 7 — Fusion described too vaguely to reimplement

**Accepted in full.** The reviewer correctly identified that we described a mechanism without saying which one produced the headline, while reporting that the choice did not matter.

**Concatenation is now named as the headline model** throughout — abstract (**p.1**), contributions (**p.1**), method (**§V-D, p.5**), results (**§VI-D, p.6**) and conclusion. Cross-modal attention is reported as an ablation with its own honest numbers (0.712 / 0.705). The fusion operation is given explicitly in equations in **§V-D**, and the keyword list was updated to drop "cross-modal attention," which no longer describes the headline architecture.

We took the reviewer's suggestion to state the negative architectural result plainly: attention does not beat concatenation here, and we say so.

---

## Finding 8 — Physiological attribution asserted rather than shown

**Accepted in full, and this reframing improved the paper.** We had treated the ablation grid as a robustness check; the reviewer correctly identified it as the actual interpretability result.

**§VI-F (p.6)** now presents the grid as a falsifiable prediction: removing cardiorespiratory channels should cost respiratory performance and not staging, and removing ocular channels should do the reverse. Both hold — cardiorespiratory removal costs respiratory AUC (0.711 → 0.673, p = 0.004) and costs staging nothing (p = 0.91), while EOG removal costs staging (0.723 → 0.712) and costs detection little. That is a prediction that could have come out otherwise and did not. The claim is stated in **§VII (p.9)** as attribution rather than assertion.

---

## Finding 9 — References too few; citation attribution

**Accepted in part.**

**Accepted.** The reference list grew from **11 to 49** entries, weighted to recent work (27 of the 49, or 55%, are 2023 or later), including the EEG foundation-model literature the reviewer noted was missing. We added the comparison table the reviewer asked for (**Table VIII, p.9**), setting the accuracy recent methods report on healthy cohorts against their measured accuracy on this stroke cohort, and we benchmarked the standard single-channel architectures on our own folds (**Table IV, p.6**).

**Where we differ.** The reviewer objected that our first-ness claim rested only on the dataset citation. We have narrowed the claim rather than removed it: the text now states that no prior method addresses *joint* staging and respiratory-event detection **on this corpus**, and supports it with the benchmark table showing that every published method evaluated here produces a staging output only. We think that is a defensible scoped claim supported by our own evidence rather than by a citation. We have also softened the sleep-disordered-breathing/recovery sentence in **§I** so the strength of the association matches what the cited studies report.

---

## Finding 10 — No limitations section

**Accepted in full.** **§VII-A Limitations (p.9)** added: single centre; N = 96 for the respiratory task; epoch-level rather than event-level scoring; staging at the cohort ceiling so the model does not raise accuracy over strong classical pipelines; and the feature-based design stated as a deliberate trade for data efficiency on a small cohort. We also disclose that SpO2 is recorded at 4 Hz and upsampled, so its effective temporal precision is 250 ms.

**§VII-B Future Work (p.10)** added: event-level metrics (sensitivity per event, false alarms per hour), finer breathing-waveform modelling, and linking the multimodal representation to stroke severity and recovery.

---

## Finding 11 — Critical-gaps section missing

**Accepted in full.** **§III Critical Gaps and Limitations of Prior Work (p.3)** added as a standalone section, not folded into Related Work. Each contribution in **§I (p.1)** now answers a named gap identified there.

---

## Finding 12 — Mandatory declarations absent

**Accepted in full.** All five added:

- **Corresponding author:** Dr. Md Iftekharul Mobin, Dept. of CSE, AIUB (p.1).
- **Data availability:** iSLEEPS is public; accession cited in §IV.
- **Code availability:** repository URL given, with executed notebooks, model source and extracted features.
- **Ethics:** source cohort collected under NIMHANS Institutional Ethics Committee approval; this work is a secondary analysis of that public release.
- **Funding:** declared.

The joint-loss equation, which was clipped at the column boundary in the submitted PDF, is retypeset in an `align` environment and now renders in full (**§V-F, p.5**).

---

## Closing

The revision added five experiments that did not exist in the submitted version (Findings 2, 3, 4, 5 and the per-event-type breakdown), corrected one overstated result (Finding 1), disclosed three methodological facts that were previously silent (Finding 6 and the SpO2 upsampling), and added three required sections (Findings 10, 11, 12).

Several of these changes make the paper's claims *smaller*: the respiratory margin over gradient boosting is modest, the AHI association is weak, and the cardiorespiratory channels do not improve staging at all. We think the paper is stronger for stating them accurately, and we thank the reviewer for insisting on it.
