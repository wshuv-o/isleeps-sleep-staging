# Response to Reviewer

**Manuscript:** A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke
**Manuscript ID:** MMNet-2026-0713
**Authors:** Md Wahiduzzaman Suva, Esm-e Moula Chowdhury Abha, Md Imtiaj Alam Sajin, Md Iftekharul Mobin
**Round:** Response to Major Revision

---

## Preface

We thank the reviewer for a report that engaged with the actual numbers rather than the narrative. Four of the findings (1, 2, 3 and 4) identified genuine errors in the submitted version, and one of them — Finding 1 — was an error we would not have caught ourselves, because the misattribution was consistent across the abstract and the discussion.

Every change described below was made by re-running the experiment, not by editing text. Each revised number is produced by a live cell in one of the two executed notebooks in the repository, and every claim below is cross-referenced to the entry in `VERIFICATION_LOG.md` that records how it was checked. **No number in the revised manuscript is copied from a cached file.** The reproduction notebook was re-executed end to end in 82.4 minutes with zero errors.

We accept eleven findings in full, and two in part (Findings 2 and 13), with our technical reasoning given below.

**Summary of disposition**

| Finding | Disposition | Where changed |
|---|---|---|
| 1 — headline attributed to wrong model | Accepted in full | Abstract, §V-D, §VI-D, Keywords, Conclusion |
| 2 — non-circularity not established | **Accepted in part** | §VI-F, §VI-G, Abstract |
| 3 — abstract/table disagreement | Accepted in full | Abstract, Table V, all figure captions |
| 4 — AHI join drops the cohort | Accepted in full | §VI-I, Fig. 9 |
| 5 — no statistical testing | Accepted in full | §VI-A, §VI-F, Table VI |
| 6 — cohort size / duplicate | Accepted in full | §IV, §VI-A |
| 7 — SpO2 upsampling undisclosed | Accepted in full | §IV, Table II |
| 8 — interpretability not demonstrated | Accepted in full | §VI-F, §VII, Fig. 10 |
| 9 — respiratory baselines absent | Accepted in full | §VI-H, Table VII |
| 10 — per-event-type pooled | Accepted in full | §VI-H, Table VII(b) |
| 11 — equation truncated | Accepted in full | §V-F, Eq. (9) |
| 12 — declarations missing | Accepted in full | Front matter, Back matter |
| 13 — reference attribution | **Accepted in part** | Bibliography, §VI-D |

---

## Finding 1 — The abstract attributes the headline result to the wrong model

> **Reviewer:** "The abstract states that the model *fuses the two streams with cross-modal attention* and reports the headline as 0.721/0.700. Those are the **concatenation** run. The cross-modal-attention run is 0.712/0.705. The named novel component is not the component behind the reported result."

**This was correct, and it was the most serious error in the paper.** We re-ran both fusion variants under identical folds and seeds to confirm the reviewer's reading before changing anything. The reviewer's diagnosis was exact: the abstract described attention, the number came from concatenation.

**What we changed.** We made concatenation the stated headline model throughout, and demoted cross-modal attention to an ablation reported with its own honest number. The change was propagated to every location where the headline is named: the abstract, the contributions list, the method section, the results discussion, the keyword list, and the conclusion. The keyword "cross-modal attention" was removed, since it no longer describes the headline architecture.

**Before / after (the most serious finding):**

> **Before (submitted version, Abstract):**
> "The model encodes the neural and the cardiorespiratory signals in two parallel streams and **fuses them with cross-modal attention**, follows the night with a bidirectional recurrent decoder… The single model stages sleep at **0.721** accuracy… and detects respiratory events at **0.700** area under the ROC curve."

> **After (revised, Abstract):**
> "The model encodes the neural and the cardiorespiratory signals in two parallel streams, **fuses them**, follows the night with a bidirectional recurrent decoder, and drives a staging head and a respiratory head, **with a direct connection that carries the cardiorespiratory signal to the respiratory head** so oxygen desaturation reaches the decision at full strength… The single model stages sleep at **0.722** accuracy with Cohen's κ of **0.611** … and it detects respiratory events at **0.711** area under the ROC curve."

The revised abstract now describes the mechanism that actually produced the number — the direct cardiorespiratory path to the respiratory head — rather than a component that did not. Cross-modal attention is reported in §VI-D as an ablation at 0.712 / 0.705.

*Verification:* `VERIFICATION_LOG.md` entry 4 (Corrected). Reproduced live in `1_MM_Net_reproduction.ipynb`.

---

## Finding 2 — The airflow-removal argument does not establish non-circularity

> **Reviewer:** "A hypopnea requires a reduction in airflow accompanied by either a ≥3% oxygen desaturation or an arousal. Removing airflow while retaining SpO2 does not remove the label's definitional inputs… either run the ablation that actually tests the claim — remove airflow **and** SpO2 together — or withdraw the claim."

**We accept this in part.** The reviewer is right on the substance: the AASM definition makes desaturation co-definitional with the event, so the airflow-only ablation does not establish what we claimed. We were wrong to present it as a non-circularity proof.

**What we changed — and we chose the reviewer's option (a), not (b).** We ran the ablation the reviewer asked for. The leave-one-out grid now includes a **remove-all-cardiorespiratory** condition, which strips airflow, SpO2, effort, pulse/HRV and ECG simultaneously, leaving only EEG, EOG and EMG:

| Condition | Staging acc | Respiratory AUC |
|---|---|---|
| Full model | 0.723 | **0.711** |
| − SpO2 | 0.727 | **0.681** |
| − airflow | 0.722 | 0.704 |
| − all cardiorespiratory | 0.724 | **0.673** |

This is a more informative result than the one we originally claimed. With every cardiorespiratory channel removed, respiratory AUC falls to 0.673 but does not collapse to chance — the residual signal is the cortical arousal that follows an obstructed breath, which is visible in EEG and EOG. And removing SpO2 alone (0.681) is more damaging than removing airflow alone (0.704), exactly as the reviewer predicted.

**Where we respectfully differ** is on the framing. We do not think the correct conclusion is that the read-out is circular. Circularity would mean the model recovers the label from a channel that mechanically encodes it; what the grid shows is a graded dependence in which no single channel is necessary and the largest single contributor is the one physiology predicts. We have therefore restated the claim descriptively rather than withdrawing it: the manuscript now says the respiratory read-out **degrades gracefully and in a physiologically ordered way** under channel removal, and it no longer uses the airflow ablation as a non-circularity proof.

We have added the AASM event definition to §IV as requested, so the reader can judge the input/label overlap directly.

*Verification:* `VERIFICATION_LOG.md` entries 5 (Corrected) and 13 (new experiment). The phrase "marginally higher" was itself corrected to "unchanged," since 0.704 vs 0.711 is within one fold standard deviation.

---

## Finding 3 — Headline numbers in the abstract disagree with the benchmark table

> **Reviewer:** "The abstract reports 0.722 / 0.651 / 0.611. Table V reports 0.721 / 0.645 / 0.606. All three figures disagree… Table V lists neural-only and neural+cardio at identical accuracy, yet boldface is applied to the cardio row as though it won."

**Accepted in full.** This was a propagation failure on our side: after re-running the model we updated the abstract but left the benchmark table holding the pre-rerun values.

**What we changed.** We removed every hardcoded metric from the manuscript body. All headline numbers are now LaTeX macros defined once at the top of the source and bound to the values produced by the reproduction notebook, so a single re-run updates the abstract, the tables, the captions and the conclusion together and they cannot drift apart again. Table V, the confusion-matrix caption, the per-class figure and the conclusion now read from the same macros as the abstract: **0.722 / 0.651 / 0.611**.

On the boldface: the reviewer is right that the two configurations tie on accuracy. The table now bolds only macro-F1 and κ, where the difference is real, and the text states explicitly that adding cardiorespiratory channels **does not improve staging accuracy** — which is itself one of the paper's findings, and was being obscured by the formatting.

*Verification:* `VERIFICATION_LOG.md` entry 1 (Confirmed at 0.7227 / 0.651 / 0.611) and entry 6 (determinism: a second independent retrain gave 0.7223, identical to three decimals).

---

## Finding 4 — The AHI correlation is computed over a join that drops most of the cohort

> **Reviewer:** "The patient join is performed on a name/ID field that matches only a small subset… the manuscript does not report *n*, the *p*-value, or the confidence interval."

**Accepted in full, and the reviewer under-stated the severity.** On investigation the naive name-join matched **9 patients**, not a "small subset" — the reported correlation was computed over 9 of 96 and presented as cohort-level validation. This was the most embarrassing error in the submission.

**What we changed.** We rebuilt the join on the numeric subject identifier rather than the name string and recomputed over every patient with complete cardiorespiratory data. The corrected result is **Spearman ρ = 0.315, p = 0.0017, n = 96**, now reported with all three quantities in the text and in the Fig. 9 caption.

We also accepted the reviewer's point about the subsection title. ρ = 0.315 is a modest association, not a validation, and the section is now titled to say so. The discussion states plainly that the model tracks clinical severity at the group level and is not a substitute for a scored AHI.

*Verification:* `VERIFICATION_LOG.md` entry 15 (Corrected).

---

## Finding 5 — No statistical testing supports any comparative claim

> **Reviewer:** "Not one of these comparisons is accompanied by a significance test… run a paired test over folds (Wilcoxon signed-rank is appropriate for n=10) for the two claims the paper depends on."

**Accepted in full.** We ran exactly the test requested, paired across the ten folds:

| Comparison | Metric | p (Wilcoxon signed-rank) | Conclusion |
|---|---|---|---|
| Full vs − all cardiorespiratory | Respiratory AUC | **p = 0.004** | Significant drop |
| Full vs − all cardiorespiratory | Staging accuracy | **p = 0.91** | Unchanged |

This pair of results is now central to the paper rather than incidental, because together they are the cleanest statement of the finding: the cardiorespiratory channels matter significantly for breathing and not at all for staging. Every comparison and ablation table now carries fold-wise mean ± SD, and directional language has been removed wherever the difference does not clear the noise.

*Verification:* `VERIFICATION_LOG.md` entry 17.

---

## Finding 6 — Cohort size is stated inconsistently and a duplicate record is undisclosed

> **Reviewer:** "The abstract states 99 patients… the manuscript does not state these three *n* values separately… the iSLEEPS distribution contains two subject records whose signal arrays and labels are byte-identical (SN15 and SN28)."

**Accepted in full.** The reviewer independently identified a data-quality issue we had handled in code but never disclosed in the manuscript, which is a fair criticism — an undisclosed fix is not a fix from the reader's point of view.

**What we changed.** §IV now states the three cohort sizes separately: **staging N = 99, respiratory N = 96** (patients with complete cardiorespiratory channels), **AHI analysis n = 96**. §IV also documents the duplicate: SN15 and SN28 were found to be byte-identical in both signal arrays and labels, we verified this with an array-equality check across the full recording, and we confirm the two records are treated as a single patient and assigned to the same fold, so patient-independence holds. The fold-assignment logic that enforces this is in `code/datasets.py` in the released repository.

*Verification:* `VERIFICATION_LOG.md` entry 7.

---

## Finding 7 — Undisclosed signal resampling in the cardiorespiratory stream

> **Reviewer:** "The manuscript does not state that SpO2 is recorded natively at 4 Hz and is upsampled… Upsampling creates no new information."

**Accepted in full.** §IV now gives a channel table with the native sampling rate of every signal, the common 25 Hz grid, and an explicit statement that SpO2 is recorded at 4 Hz and upsampled by linear interpolation.

We have also added the caveat the reviewer implies: because SpO2 is the single most important channel for the respiratory head (Finding 2's grid) and also the lowest-resolution one, its effective temporal precision is 250 ms, and the model cannot localise a desaturation more finely than that. This is stated as a limitation rather than buried in preprocessing.

*Verification:* `VERIFICATION_LOG.md` entry 11 (Flagged and stated).

---

## Finding 8 — "Physiologically interpretable" is claimed but not demonstrated

> **Reviewer:** "A t-SNE showing that sleep stages separate demonstrates that the representation is discriminative, not that it is physiologically interpretable… the ablation grid is in fact the paper's genuine interpretability result."

**Accepted in full, and this reframing improved the paper.** We had treated the ablation grid as a robustness check and the t-SNE as the interpretability contribution; the reviewer correctly inverted that.

**What we changed.** The modality-ablation grid is now presented as the paper's interpretability result and stated as a falsifiable attribution claim: *each modality contributes where its physiology lives* — removing cardiorespiratory channels costs respiratory AUC (0.711 → 0.673, p = 0.004) and costs staging nothing (p = 0.91), while removing EOG costs staging (0.723 → 0.712) and costs respiratory detection little. That is a prediction that could have come out otherwise, and did not.

The t-SNE is retained but re-scoped. Its caption now states what it does and does not show, including the honest observation that stage structure separates cleanly while respiratory events remain diffuse — which is consistent with events being brief and sparse relative to 30-second epochs.

*Verification:* `VERIFICATION_LOG.md` entry 13; figure regenerated from saved embeddings.

---

## Finding 9 — Respiratory baselines are absent, so the respiratory result is uncalibrated

> **Reviewer:** "Without a floor, an AUC is not evidence of a contribution… add at least three reference points computed on the same folds."

**Accepted in full.** We added the three baselines requested, all trained and evaluated on the identical folds:

| Detector | Respiratory AUC |
|---|---|
| Clinical desaturation rule | 0.596 |
| Logistic regression on the 14 cardiorespiratory features | 0.582 |
| Gradient boosting on the same features | 0.670 |
| **Proposed model** | **0.711** |

**We also accepted the consequence the reviewer anticipated.** The margin over gradient boosting is ~4 points of AUC, which is modest, and we say so in the text rather than presenting the model as a decisive winner on detection. The contribution has been reframed accordingly: the claim is now the **joint single-pass formulation** — one compact model producing both clinical outputs, with causal modality attribution — and not detection supremacy. This reframing follows the reviewer's own suggested wording.

*Verification:* `VERIFICATION_LOG.md` entry 12.

---

## Finding 10 — Per-event-type performance is pooled

> **Reviewer:** "Report AUC separately per event type, with the prevalence of each."

**Accepted in full.** Added, with prevalence:

| Event type | Prevalence | AUC |
|---|---|---|
| Hypopnea | 81.0% | 0.692 |
| Obstructive apnea | 11.6% | 0.763 |
| Central apnea | 5.8% | **0.840** |

The pattern is clinically meaningful and was invisible in the pooled number: detection is strongest on the most severe events and weakest on hypopneas, which dominate the pooled label and therefore drag the aggregate AUC toward the hardest class. The reviewer's point about central events after stroke is addressed in the discussion — central apnea is both the best-detected type here and the one with specific post-stroke significance.

*Verification:* `VERIFICATION_LOG.md` entry 14.

---

## Finding 11 — A displayed equation is truncated in the compiled PDF

**Accepted in full.** The joint-loss equation overflowed the column and clipped the weighting term. It is retypeset in an `align` environment, and we confirmed in the compiled PDF that it now renders inside the column with the weighting term visible.

*Verification:* `VERIFICATION_LOG.md`, Correctness fixes (Applied).

---

## Finding 12 — Mandatory declarations and corresponding author are missing

**Accepted in full.** All five added:

- **Corresponding author:** Dr. Md Iftekharul Mobin, Department of Computer Science, AIUB (iftekhar.mobin@aiub.edu).
- **Data availability:** iSLEEPS is publicly available; the record and accession are cited in §IV.
- **Code availability:** repository URL given in the manuscript, containing the executed notebooks, the model source and the extracted features.
- **Ethics:** the source cohort was collected under approval from the NIMHANS Institutional Ethics Committee; our work is a secondary analysis of that public release.
- **Funding:** declared.

---

## Finding 13 — Reference list: currency is good, attribution needs checking

> **Reviewer:** "Several 2026 entries need verification that they are published rather than preprints… the claim that a cited concurrent work supports 'depth alone does not help on this cohort' should be checked against what that work actually concludes."

**Accepted in part.**

**Accepted:** we re-checked every post-2024 entry against its published record and corrected venue and year where an entry was still cited as a preprint. The list stands at 49 references with 27 from 2023 or later, which we believe is appropriate currency for this area.

**Where we differ:** we have retained the concurrent-work citation in §VI-D, but narrowed the sentence it supports. The cited work reports the same directional finding on the same cohort — that deep architectures do not outperform feature-based approaches here — and we think the citation is apt. However, the reviewer is right that our sentence generalised beyond what that work claims, so the text now attributes to it only the specific observation it makes, and our own broader statement is supported by our own benchmark table rather than by the citation.

---

## Closing

The revision changed four results (Findings 1, 3, 4 and the airflow wording in 2), added five experiments that did not exist in the submitted version (Findings 2, 5, 9, 10 and the significance tests), and disclosed three methodological facts that were previously silent (Findings 6, 7 and the duplicate handling). Two claims were withdrawn or narrowed rather than defended.

We are aware that several of these changes make the paper's claims *smaller* — the respiratory margin is modest, the AHI association is weak, and cardiorespiratory channels do not help staging at all. We think the paper is stronger for stating them accurately, and we thank the reviewer for insisting on it.
