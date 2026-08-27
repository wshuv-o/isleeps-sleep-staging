# Referee Report

**Manuscript:** A Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke
**Manuscript ID:** MMNet-2026-0713 (first submission)
**Reviewing model:** Claude Opus 5
**Date of report:** 13 July 2026

---

## Recommendation: **MAJOR REVISION**

I read the submitted manuscript in full: the abstract and index terms on page 1, all seven sections (I Introduction, II Related Work, III The Polysomnogram We Model, IV the model, V Results, VI Discussion, VII Conclusion), the four figures, the three tables, the single airflow-removal ablation reported in Table III, and the complete reference list of **11 entries**. I checked the headline numbers quoted in the abstract (0.721 accuracy, κ 0.606, 0.700 AUC, 0.707 airflow-removed) against those printed in Section V-B and Section V-D; I checked the ablation against what the text claims it proves; I checked the respiratory result for a baseline to compare against; and I checked the reference list for coverage and currency.

The core idea is good and the cohort is valuable. iSLEEPS is the first public polysomnography corpus of subacute ischemic stroke, the ten-fold patient-independent protocol is the correct evaluation design, and framing staging and respiratory-event detection as one joint problem is sensible and clinically motivated. The writing is clear and unusually readable.

But this is a thin paper in its present state. The entire results section runs to two subsections and reports one ablation condition. There is no baseline for the respiratory task, no significance testing anywhere, no clinical validation against any external measure, no limitations section, and a reference list of 11 entries for a field this active. Most seriously, the paper's central non-circularity claim rests on an ablation that does not test it, and states a difference as an improvement when it is within noise. These are correctness problems, not presentation problems.

---

## Overall Assessment

| Area | Score /10 | Main concern |
|---|---|---|
| Title | 7 | Accurate but generic; does not signal what distinguishes the contribution |
| Novelty | 6 | Two-stream fusion plus recurrent decoder is a sound recombination of known parts; novelty rests on the application, which the framing under-sells |
| Methodology | 6 | Protocol is correct, but the fusion mechanism is described too vaguely to reimplement |
| Dataset rigor | 5 | Cohort size stated once as 99 and once as 96 with no explanation; no data-quality audit reported |
| Experimental validity | 3 | No baselines, no significance tests, no variance reported anywhere |
| Ablation | 2 | A single condition — airflow removal — is the entire ablation |
| Explainability | 4 | Physiological attribution is asserted in prose, not demonstrated by any experiment |
| Clinical claims | 3 | Clinical utility claimed with no validation against any clinical measure |
| References | 3 | 11 entries is far too few; several key comparators uncited |
| Writing | 7 | Clear and well organised; some claims stated more strongly than the evidence |
| **Overall** | **4** | Promising study, materially incomplete as submitted |

---

## Numbered Findings

### Finding 1 — The non-circularity claim is not tested by the ablation offered, and a null result is reported as an improvement (most serious)

**Location:** Section V-D, "The respiratory read-out reads physiology, not the label"; Table III; abstract, lines 10–13.

The manuscript writes:

> "Because the events were scored from airflow, a model that keyed on the airflow channel would be right for a trivial reason. We remove the two airflow features and test again. Detection is not merely preserved, it is **marginally higher**, at **0.707** AUC (Table III). The model therefore does not lean on the labeling signal."

Two separate problems.

**(a) The ablation does not test the claim.** Respiratory events under AASM rules are not defined by airflow alone: a hypopnea requires reduced airflow *accompanied by* a ≥3% oxygen desaturation or an arousal. Removing airflow while retaining SpO2 removes one definitional input and keeps the other. The experiment therefore cannot establish that the read-out is independent of the labelling signal.

**(b) A difference within noise is reported as an increase.** 0.707 against a baseline of 0.700 is one quarter of a typical ten-fold standard deviation. Calling it "marginally higher" and concluding "the model therefore does not lean on the labeling signal" reads a direction into noise, and the manuscript reports no fold-wise variance that would let a reader judge.

**Required action:** run the ablation that actually tests the claim — remove airflow **and** SpO2 together, and report what survives on effort, cardiac and cortical-arousal information alone. State the AASM event definition in the data section so the reader can see the input/label overlap. Replace "marginally higher" with a statement of no significant difference, supported by a test.

---

### Finding 2 — The ablation consists of one condition

**Location:** Section V-D; Table III.

The paper argues throughout that "each modality contributes where its physiology lives" and that the modality split is "cleanly attributable." The only ablation performed is the removal of airflow features. No other channel is ablated. There is no condition removing SpO2, effort, ECG, pulse variability, EOG or EMG, and no condition removing the cardiorespiratory stream as a whole.

A one-row ablation cannot support a claim about how seven signal types divide their labour.

**Required action:** produce a complete leave-one-out modality grid over every channel group, reporting both staging accuracy and respiratory AUC for each condition, so the attribution claim is demonstrated rather than asserted. The table must be complete — no cells left unreported.

---

### Finding 3 — The respiratory result has no baseline, so its magnitude cannot be judged

**Location:** Section V-B; Table II.

The respiratory AUC of 0.700 is reported against nothing. There is no comparison with a simple clinical rule, a linear model, or a standard classifier on the same features and folds. Since respiratory events are strongly signalled by oxygen desaturation, a threshold rule on SpO2 may well perform respectably, and the reader cannot tell how much of 0.700 is attributable to the proposed architecture.

**Required action:** add at least three reference points computed on identical folds — a clinical desaturation rule, a logistic regression on the cardiorespiratory features, and a gradient-boosting model on the same features — and state the proposed model's margin over the strongest. If that margin is modest, say so and reframe the contribution around the joint single-pass formulation rather than detection performance.

---

### Finding 4 — No statistical testing supports any comparative claim

**Location:** Sections V-B, V-D; Tables II and III; Discussion.

The manuscript draws directional conclusions throughout — the fusion variants "give the same numbers," detection is "marginally higher" without airflow, the cardiorespiratory channels "leave staging unchanged" — but reports no significance test, no confidence interval, and no fold-wise standard deviation anywhere in the paper. With ten folds and differences of 0.007 AUC, the reader has no basis to distinguish signal from fold-to-fold variation.

**Required action:** report mean ± SD across folds for every number in every results table, and run a paired test across folds (Wilcoxon signed-rank is appropriate at n=10) for the two claims the paper depends on: full model versus no-cardiorespiratory for the respiratory head, and the same comparison for staging. State the p-values in the text.

---

### Finding 5 — Clinical utility is claimed but never validated against a clinical measure

**Location:** Abstract; Section VI Discussion.

The Discussion states the model "flags the breathing events that fragment their sleep and shadow their recovery," and the abstract offers "a second clinical capability that an electroencephalogram alone cannot provide." No analysis in the paper connects the model's output to any clinical variable. There is no comparison against the scored apnea–hypopnea index, no stratification by severity, and no breakdown by event type — even though hypopneas, obstructive apneas and central apneas differ in physiology and in post-stroke significance.

**Required action:** validate the per-patient predicted event burden against the clinically scored AHI, reporting the correlation with its *n* and *p*. Report staging performance stratified by AASM severity band. Report detection AUC separately per event type with the prevalence of each. If the association is modest, report it as modest.

---

### Finding 6 — Cohort size is stated inconsistently and no data-quality audit is reported

**Location:** Abstract ("99 patients"); Section I ("77 of 96 patients"); Section III.

The abstract says 99 patients; the introduction says 96. Neither figure is explained, and the manuscript never states how many patients enter each experiment. A public corpus of this size also warrants an integrity check that is not reported: duplicated recordings, subjects missing entire sleep stages, and per-subject label distributions all bear on a patient-independent protocol.

**Required action:** state *n* explicitly for each experiment and reconcile 99 against 96. Report a data-quality audit: verify that no recording is duplicated under two identifiers, and if one is, state how it was handled in the fold assignment. Report which subjects lack stages entirely.

---

### Finding 7 — The fusion mechanism is described too vaguely to reimplement

**Location:** Abstract; Section IV-B, "Two-stream encoding and cross-modal fusion"; Section V-C.

The abstract says the model "lets the two streams exchange information." Section V-C then reports that "the attention fusion and a plain concatenation give the same numbers." So the paper describes a fusion mechanism without specifying which one produced the reported results, while simultaneously reporting that the choice does not matter.

This leaves the reader unable to reimplement the headline, and it undercuts the architectural contribution: if concatenation matches attention, the attention mechanism is not doing the work the narrative implies.

**Required action:** name one configuration as the headline model and report its numbers consistently in the abstract, contributions, results and conclusion. Report the other as an ablation with its own figures. Give the fusion operation explicitly in equations. If concatenation is sufficient, say so plainly — a negative architectural result is worth reporting.

---

### Finding 8 — Physiological attribution is asserted rather than shown

**Location:** Section V-C; Figure 4; Discussion.

The manuscript states that "the cortical channels carry sleep staging, and the cardiorespiratory channels tell it when the patient stops breathing," and calls the split "cleanly attributable." The supporting evidence is Figure 4, showing per-class F1 is unchanged when cardiorespiratory channels are added. That an addition changes nothing is weak evidence for a claim about where information lives; it is consistent with the channels being uninformative for staging, which is a different statement.

**Required action:** support the attribution with the full ablation grid from Finding 2, stated as a falsifiable prediction — removing cardiorespiratory channels should cost respiratory performance and not staging, and removing ocular channels should do the reverse. Report whether that prediction holds.

---

### Finding 9 — The reference list is far too short and omits the obvious comparators

**Location:** References (11 entries).

Eleven references is inadequate for a paper positioned against the sleep-staging literature. Several architectures a reader would expect to see benchmarked or discussed are absent, and there is no engagement with recent EEG foundation models, which are the most active current line of work in this area. Without them the novelty claim cannot be assessed.

I also checked each citation against the statement it is attached to. The entries themselves are real and correctly attributed — [2] DeepSleepNet, [3] SeqSleepNet, [4] AttnSleep and [5] XSleepNet are cited for what they in fact report, and [1] correctly supports the AASM scoring rules. Two attribution problems remain. First, the claim in Section I that sleep-disordered breathing "is tied to worse recovery" carries two citations doing heavy lifting for a strong clinical assertion, and the strength of that link in the cited work should be stated rather than implied. Second, the novelty claim — that no prior method addresses joint staging and respiratory detection on this cohort — is supported only by the dataset citation [9], which establishes that the corpus exists, not that no one has done this. A claim of first-ness needs a stated search, not a dataset reference.

**Required action:** expand the reference list substantially, weighted toward 2023–2026 work, and add a comparison table setting the accuracy recent methods report on healthy cohorts against their performance on this stroke cohort. Benchmark at least the standard single-channel architectures on the same folds.

---

### Finding 10 — No limitations section

**Location:** Sections VI–VII.

The Discussion moves directly to the Conclusion with no statement of limitations. Single-centre data, small cohort, epoch-level rather than event-level scoring, and the feature-based design all constrain the conclusions and none is acknowledged.

**Required action:** add an explicit limitations section, and a future-work section naming concrete next steps.

---

### Finding 11 — A critical-gaps section is missing from the structure

**Location:** Section II Related Work.

Related Work surveys prior methods but never states what specifically is unaddressed and why this paper's design follows from those gaps. The motivation is therefore implicit, and the contribution list in Section I is not tied to identified deficiencies in the literature.

**Required action:** add a dedicated section stating the critical gaps and limitations of prior work, and make each contribution answer a named gap.

---

### Finding 12 — Mandatory declarations are absent

**Location:** Front and back matter.

There is no corresponding-author designation, no data-availability statement, no code-availability statement, no ethics statement naming the body that approved the original data collection, and no funding declaration. The abstract states "Code and features are released" without giving a location.

**Required action:** add all five. The ethics statement must name the approving committee for the source cohort, and the code statement must give a resolvable URL.

---

## Summary for the Editor

The study rests on a genuinely valuable cohort and a sensible joint formulation, and the writing is clear. My concerns are that the empirical work supporting it is currently too thin to sustain the claims: one ablation condition, no respiratory baseline, no statistical testing, no clinical validation, and 11 references. Findings 1 through 5 are the ones that must be addressed before this can be assessed properly; all are achievable with data the authors already hold. I would be glad to see a revised version.
