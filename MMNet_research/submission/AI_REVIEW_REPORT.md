# Referee Report

**Manuscript:** A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke
**Manuscript ID:** MMNet-2026-0713 (first submission)
**Reviewer:** AI referee (large language model, acting as journal referee)
**Date of report:** 13 July 2026

---

## Recommendation: **MAJOR REVISION**

I read the full manuscript: the abstract, all nine sections, the eight tables and the eleven figures, the modality-ablation grid, and the reference list (49 entries). I checked the headline staging and respiratory numbers quoted in the abstract against those printed in the benchmark table, the confusion matrix, the per-class F1 figure and the ablation grid; I traced the cross-modal-attention claim in the abstract back to the run it is supposed to describe; I checked the airflow-removal argument against the AASM definition of the events being detected; and I checked the reference list for currency, completeness and for whether the cited works support the statements attached to them.

The work is genuinely worth publishing. The clinical problem is real and under-served, the cohort is the first public polysomnography corpus in subacute ischemic stroke, the patient-independent ten-fold protocol is correct, and the central empirical observation — that deep architectures built for healthy sleep lose 15–20 accuracy points on this cohort while a compact, physiologically-grounded model does not — is an honest and useful contribution. The joint staging-plus-respiratory formulation is a sensible design.

However, I cannot recommend acceptance in the present form. The abstract attributes the headline result to a component that did not produce it; the respiratory claim rests on an ablation that does not establish what the text says it establishes; the clinical-validation section reports a correlation computed over a join that silently drops most of the cohort; and one of the paper's own load-bearing claims is contradicted by its own table. These are correctness problems, not presentation problems. Each is fixable without new data.

---

## Overall Assessment

| Area | Score /10 | Main concern |
|---|---|---|
| Title | 8 | Accurate and specific, but "Physiologically Interpretable" is asserted before it is earned |
| Novelty | 6 | Architecture is a sensible recombination of known parts; novelty rests on the application and the negative result, which the framing under-sells |
| Methodology | 6 | Sound protocol, but the fusion actually used is not the fusion the abstract advertises |
| Dataset rigor | 5 | Cohort size is stated inconsistently (99 vs 96); the SN15/SN28 duplicate and the SpO2 upsampling are not disclosed |
| Experimental validity | 6 | Ten-fold patient-independent is correct, but no significance testing accompanies the headline comparisons |
| Ablation | 7 | The leave-one-out grid is the strongest part of the paper; it is under-analysed and its SpO2 row is not discussed |
| Explainability | 5 | t-SNE and attention weights are shown but not tied to any physiological claim that could be falsified |
| Claims | 3 | Several claims exceed what the experiments support; one contradicts the paper's own table |
| References | 7 | 49 refs, well-weighted to 2023–2026, but at least one is mis-attributed and several 2026 entries need verification |
| Writing | 8 | Clear, well-organised, unusually readable for the genre |
| **Overall** | **5.5** | Publishable after the claim/evidence mismatches are repaired |

---

## Numbered Findings

### Finding 1 — The abstract attributes the headline result to the wrong model (most serious)

**Location:** Abstract, lines 4–8; Section VI-A; Table V.

The abstract states that the model *"fuses the two streams with cross-modal attention"* and reports the headline as **0.721 staging accuracy / 0.700 respiratory AUC**. The experiment section, however, reports two fusion variants: concatenation and cross-modal attention. The numbers 0.721/0.700 are the **concatenation** run. The cross-modal-attention run is **0.712 / 0.705** — lower staging, marginally higher AUC.

The abstract therefore credits the headline to a mechanism that did not produce it. This is the single most serious problem in the manuscript: the named novel component is not the component behind the reported result. A reader — or a downstream citer — would take away that cross-modal attention delivers 0.721, which it does not.

**Why it is a problem:** it misattributes causation to the paper's own claimed novelty, and it makes the headline unreproducible for anyone who implements the described architecture.

**Required action:** decide which model is the headline and report it consistently. If concatenation is the headline, say so in the abstract, in the contributions, and in the conclusion, and demote cross-modal attention to an ablation with its own honest number. If attention is the headline, report 0.712/0.705 as the headline. Do not mix them. Update the keyword list accordingly.

---

### Finding 2 — The airflow-removal argument does not establish non-circularity

**Location:** Section VI-D; abstract, final third.

The manuscript argues the respiratory read-out is not circular because *"when we remove the airflow channel from which the events were scored, detection holds at 0.704."*

Respiratory events under the AASM scoring rules are **not** defined by airflow alone. A hypopnea requires a reduction in airflow **accompanied by either a ≥3% oxygen desaturation or an arousal**. Removing the airflow channel while retaining SpO2 therefore does not remove the label's definitional inputs — it removes one of two, and retains the other. The ablation as performed cannot support the conclusion drawn from it.

Worse, the manuscript's own ablation grid shows that removing SpO2 is the single most damaging intervention (**AUC 0.711 → 0.681**), which is consistent with the model leaning on precisely the channel that co-defines the label.

**Why it is a problem:** the non-circularity claim is one of two pillars of the respiratory contribution, and the experiment offered does not test it.

**Required action:** either (a) run the ablation that actually tests the claim — remove airflow **and** SpO2 together, and report what survives on effort, ECG and EEG arousal alone; or (b) withdraw the non-circularity claim and restate the result descriptively. State explicitly, in Section III, the AASM definition of the events being predicted, so the reader can judge the overlap between inputs and labels.

---

### Finding 3 — Headline numbers in the abstract disagree with the benchmark table

**Location:** Abstract vs Table V (`tab:bench`).

The abstract reports staging **0.722 / macro-F1 0.651 / κ 0.611**. Table V reports, for the same model (MM-Net, neural + cardio), **0.721 / 0.645 / 0.606**. All three figures disagree.

Additionally, Table V lists MM-Net (neural only) and MM-Net (neural + cardio) at **identical** accuracy (0.721 / 0.721), yet boldface is applied to the cardio row as though it won. The ablation grid elsewhere reports neural-only staging at 0.724, a third value.

**Why it is a problem:** a referee cannot tell which number is the result. Three internally inconsistent values for a single headline metric is the most common cause of post-publication correction.

**Required action:** recompute from a single stored artifact and propagate one value everywhere — abstract, table, confusion-matrix caption, per-class figure, conclusion. Bold only genuine wins; where two configurations tie, say they tie.

---

### Finding 4 — The AHI correlation is computed over a join that drops most of the cohort

**Location:** Section VI-F, "Clinical Validation against AHI and Severity."

The reported Spearman correlation between predicted event burden and clinical AHI is presented as cohort-level clinical validation. On inspection the patient join is performed on a name/ID field that matches only a small subset of the cohort — the effective *n* is far below the stated cohort size, and the manuscript does not report *n*, the *p*-value, or the confidence interval.

**Why it is a problem:** a correlation over a silently truncated subset is not clinical validation, and without *n* and *p* the reader cannot assess it at all.

**Required action:** repair the join, recompute over the full cohort with complete cardiorespiratory data, and report ρ, *n*, and *p* explicitly. If the corrected correlation is weak, report it as weak — a modest, honest correlation is publishable; an unstated one is not. Retitle the subsection: a correlation of this magnitude is "association with clinical severity," not "clinical validation."

---

### Finding 5 — No statistical testing supports any comparative claim

**Location:** Sections VI-A through VI-E; all comparison tables.

The manuscript compares the proposed model against six baselines and reports nine ablation conditions, and draws directional conclusions throughout ("above every deep baseline," "holds," "marginally higher"). Not one of these comparisons is accompanied by a significance test, a confidence interval, or a fold-wise standard deviation.

Several of the differences being interpreted are small — the airflow ablation moves the AUC by 0.007, and the two fusion variants differ by 0.009 in accuracy. With ten folds these are very plausibly within fold-to-fold noise, and the manuscript gives the reader no way to know.

**Why it is a problem:** the paper's conclusions are directional claims about small differences, which is exactly the regime where per-fold variance decides whether a claim survives.

**Required action:** report mean ± SD across the ten folds for every number in the comparison and ablation tables, and run a paired test over folds (Wilcoxon signed-rank is appropriate for n=10) for the two claims the paper depends on: full model vs no-cardio for the respiratory head, and full model vs no-cardio for staging. State the *p*-values in-text. Where a difference does not reach significance, describe it as unchanged.

---

### Finding 6 — Cohort size is stated inconsistently and a duplicate record is undisclosed

**Location:** Abstract ("99 patients"); Section III; Section VI-F.

The abstract states 99 patients. The respiratory experiments appear to run on a smaller subset (those with complete cardiorespiratory channels), and the AHI analysis on smaller still. The manuscript does not state these three *n* values separately.

Separately, the iSLEEPS distribution contains two subject records whose signal arrays and labels are byte-identical (SN15 and SN28 — the same night released under two identifiers). The manuscript does not mention this. If both records entered the cross-validation and landed in different folds, the patient-independence of the protocol is broken for that patient.

**Why it is a problem:** undisclosed duplicates are a data-leakage risk that invalidates a patient-independent protocol; unstated per-experiment *n* prevents replication.

**Required action:** state *n* explicitly for staging, for the respiratory task, and for the AHI analysis. State how the duplicate was detected and handled, and confirm in Section III that the two records never occupy different folds.

---

### Finding 7 — Undisclosed signal resampling in the cardiorespiratory stream

**Location:** Section III, Preprocessing.

The cardiorespiratory channels are placed on a common sampling grid, but the manuscript does not state that SpO2 is recorded natively at 4 Hz and is upsampled to reach that grid. Upsampling creates no new information; a reader assessing whether the SpO2 stream can support the reported respiratory performance needs to know its true resolution.

**Why it is a problem:** it is a material preprocessing detail on the single most important channel for the paper's respiratory claim (per the ablation grid).

**Required action:** state the native sampling rate of every channel in a table, state the common grid, and state which channels are upsampled and by what method.

---

### Finding 8 — "Physiologically interpretable" is claimed but not demonstrated

**Location:** Title; Section VII (Discussion); Figure 10 (t-SNE).

The title claims physiological interpretability. The evidence offered is a t-SNE of learned embeddings and a set of attention weights. A t-SNE showing that sleep stages separate demonstrates that the representation is discriminative, not that it is physiologically interpretable — any accurate model would produce that plot.

**Why it is a problem:** interpretability is claimed in the title, which sets an expectation the analysis does not meet.

**Required action:** the ablation grid is in fact the paper's genuine interpretability result — it shows a clean physiological dissociation (cardiorespiratory channels drive the respiratory head, EOG drives staging). Promote that argument explicitly, state it as an attribution claim, and either tie the t-SNE to a specific falsifiable physiological statement or move it to supplementary material.

---

### Finding 9 — Respiratory baselines are absent, so the respiratory result is uncalibrated

**Location:** Section VI-E.

The respiratory AUC is reported against no baseline. The reader cannot tell whether it reflects the proposed architecture or is obtainable from the cardiorespiratory features with any off-the-shelf classifier. Given that respiratory events are, by construction, strongly signalled by desaturation, a simple desaturation rule may perform respectably.

**Why it is a problem:** without a floor, an AUC is not evidence of a contribution.

**Required action:** add at least three reference points computed on the same folds — a clinical desaturation rule, a logistic regression on the cardiorespiratory features, and a gradient-boosting model on the same features — and report the proposed model's margin over the strongest. If the margin is modest, say so and reframe the contribution around the joint single-pass formulation rather than detection supremacy.

---

### Finding 10 — Per-event-type performance is pooled, hiding the clinically relevant breakdown

**Location:** Section VI-E.

Respiratory events are pooled into a single binary label. Hypopneas, obstructive apneas and central apneas differ in physiology and in clinical consequence, and central events in particular carry specific significance after stroke.

**Required action:** report AUC separately per event type, with the prevalence of each in the cohort, and discuss which the model detects best.

---

### Finding 11 — A displayed equation is truncated in the compiled PDF

**Location:** Section IV-F, the joint loss.

The joint-loss equation runs past the column boundary and is clipped in the compiled PDF; the weighting term is not readable. The loss function of a multi-task model is not an optional detail.

**Required action:** typeset the equation in a breaking environment (`align`/`split`) and confirm it renders inside the column in the compiled output.

---

### Finding 12 — Mandatory declarations and corresponding author are missing

**Location:** Front matter and back matter.

There is no corresponding author designation, no data-availability statement, no code-availability statement, no ethics statement identifying the approving body for the source cohort, and no funding declaration. The abstract states "Code and features are released" without a location.

**Required action:** add all five. The ethics statement must name the committee that approved the original data collection. The code statement must give a resolvable URL.

---

### Finding 13 — Reference list: currency is good, attribution needs checking

**Location:** Bibliography (49 entries).

The distribution is appropriate — 27 of 49 entries are 2023 or later, which suits a fast-moving area. Two concerns. First, several 2026 entries need verification that they are published rather than preprints, and should be cited with their final venue. Second, the claim in Section VI-A that a cited concurrent work supports "depth alone does not help on this cohort" should be checked against what that work actually concludes — the sentence attaches a strong claim to a citation and the attribution should be exact.

**Required action:** verify every post-2024 entry against its published record; correct venue and year where they are preprints. Re-read each citation attached to a claim in Sections V–VII and confirm the cited work states what the sentence says it states.

---

## Summary for the Editor

The underlying study is sound and the dataset contribution is real. My concerns are concentrated in the gap between what the experiments show and what the text claims — specifically Findings 1, 2, 3 and 4, each of which is a correctness issue rather than a matter of presentation. All are addressable with analyses the authors can run on data they already hold. I would be glad to see a revised version.
