# MM-Net Manuscript — Revision Brief

Working brief for Claude Code. Covers the revision of *Multimodal Polysomnography for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke* (MM-Net, iSLEEPS corpus).

**Target venue:** IEEE JBHI (first choice) or TNSRE. IEEE two-column format, currently in place.

---

## 0. Read this first

The current manuscript draft is an **internal working document**, not a submission draft. It contains every model run during development, including EEG-only configurations that were built to answer a different question. Do not treat the existing tables as final scope. Section 2 below defines what stays.

**Hard rule: never invent a number.** If a table cell, figure, or claim requires a result that has not been produced yet, insert `[TODO: run X]` and say so in your summary. Do not estimate, interpolate, or carry a number over from a similar configuration. Every number in this paper must trace to a run.

---

## 1. What the paper claims (revised positioning)

The paper does **not** claim state-of-the-art sleep staging. It should never be framed that way.

The thesis, in one paragraph:

> Sleep staging on this stroke cohort saturates regardless of architecture. A single compact model reading the full polysomnogram produces a competitive stage *and* a respiratory-event read-out from one forward pass, at 0.86M parameters. The respiratory read-out is grounded in physiology rather than in the labeling signal: it survives deletion of the airflow channel from which events were scored, and a modality-ablation grid shows which physiological channels carry the decision and that they are the ones a clinician reads.

Every framing decision follows from this. When a sentence competes on staging accuracy, rewrite it to compete on the second output or on physiological attribution.

---

## 2. Table scope — what stays, what moves, what gets a caveat

The main-text benchmark is restricted to **configurations comparable in input** to MM-Net.

**Keep in main text:**
- MM-Net (neural only) and MM-Net (neural + cardio) — the core comparison
- Deep models reading multi-channel input under our protocol (raw multimodal CNN 14ch, sequence CNN-BiLSTM 7ch)
- The published single-channel baselines from [13], **with an explicit input-configuration caveat** (see below)

**Move to supplementary:** the 7-channel EEG-only feature and ensemble models (gradient-boosting ensemble, stacking ensemble, graph attention + SSM, learnable refiner, feature-sequence BiLSTM, single gradient boosting). These answer a different question — how far EEG-only staging goes on this cohort — and are not input-matched to a 14-channel multimodal model. Add one sentence in the main text pointing to the supplementary table so nothing looks hidden.

**On the published LSTM (0.747):** retain it, and add a caveat sentence. It is in the abstract of the iSLEEPS data descriptor, so any reviewer familiar with the corpus knows the number. Removing it silently would be noticed. The caveat should state: reported value from [13], single-channel raw EEG, 100 subjects, different fold assignment, included as the published reference point on this corpus and not as a same-input comparison. Framing it this way is stronger than either omitting it or leaving it bare in the table.

Bolding a competitor's number where it legitimately wins is a credibility signal. Do not remove bolding to make MM-Net look better.

---

## 3. Critical fixes (do these first — they are correctness issues)

1. **Concat vs. attention mismatch.** The abstract describes cross-modal attention and reports 0.721 / 0.700, but Table IV shows those are the **concatenation** numbers (attention: 0.715 / 0.698). Resolve by making concatenation the stated headline model and demoting attention to an ablation — this is consistent with the paper's own finding that the gain is the modality, not the fusion mechanism. Update the abstract, the title if it implies attention, Figure 2, and all body text accordingly.

2. **N = 99 vs 96.** Abstract says 99 patients; Table I says "77 of 96 patients above 5%." State once, explicitly, in Section III: staging uses N=99, respiratory tasks use N=96 (those with complete cardiorespiratory channels). Verify every reported number against the correct N.

3. **Figure 5 caption/content mismatch.** Caption reads "Staging accuracy against model size"; the figure is a row-normalised confusion matrix. Fix whichever is wrong.

4. **Equation 11 is truncated** in the compiled PDF (`(1−at) log(1−y | {z`). Repair the LaTeX source.

5. **Missing statements.** Add: data availability (Zenodo DOI 10.5281/zenodo.14873844 and the india-data.org portal), code availability with an actual URL, ethics statement citing the NIMHANS IEC approval documented in the descriptor, funding declaration.

6. **Statistical testing.** No paired tests currently. Add Wilcoxon signed-rank over the 10 paired folds for the central comparison (with vs. without cardiorespiratory stream, 0.700 vs 0.655 AUC) and for every ablation contrast in the new grid. Report p-values in the table.

7. **SpO2 sampling.** The descriptor records SpO2 at 4 Hz natively; the manuscript resamples cardiorespiratory channels to 25 Hz. State the upsampling explicitly in Section III.

8. **Authorship.** No corresponding author is marked and no senior author appears. Resolve before submission.

---

## 4. Experiments to run

### 4.1 Respiratory baselines — highest priority

The paper currently benchmarks fifteen architectures on staging and **zero** on the respiratory task, which is the task the contribution rests on. "The only model that also detects respiratory events" is currently true only relative to a table of staging models.

Run on the same folds, same 14 cardiorespiratory features, N=96:
- **Desaturation threshold rule** — clinical rule of thumb, threshold on desaturation depth
- **Logistic regression** on the 14 cardiorespiratory features
- **Gradient boosting** on the 14 cardiorespiratory features

Report AUC and AP for each, plus the AP of a random classifier at 16% prevalence so the 0.333 AP is interpretable. If MM-Net does not beat gradient boosting on cardio-only features, we need to know that now and reframe around the joint single-pass argument rather than detection quality.

### 4.2 Modality ablation grid — the new headline result

Two passes over the same folds.

**Leave-one-out**, dropping each in turn and measuring both staging accuracy and respiratory AUC:
SpO2 · effort (thoracic/abdominal/summed/asynchrony) · pulse-HRV · ECG · airflow (already have) · EOG · EMG · full cardiorespiratory stream (already have)

**Cumulative build-up** on the respiratory head: SpO2 alone → +effort → +HRV → +airflow. Leave-one-out understates a modality when two channels carry redundant information; desaturation and thoraco-abdominal asynchrony both mark an obstructive event, so dropping either alone may barely move the metric while dropping both collapses it. Run both, report both.

Expected pattern to test (not to assume): cardiorespiratory ablations move respiratory AUC and leave staging flat; neural ablations do the reverse. This is the paper's "clean physiological line" claim, currently asserted on the strength of a single ablation.

### 4.3 Per-event-type breakdown

From the descriptor, event composition is: **hypopnea 80.8%, obstructive apnea 11.6%, central apnea 5.6%, mixed 0.8%.** The current binary label collapses all of these.

This matters physiologically: central apnea has *absent* effort, obstructive has *increased* effort against a closed airway, so the asynchrony feature should behave in opposite directions for the two. Report AUC per event type, and cross this with the effort ablation from 4.2. If dropping effort hurts obstructive detection and does not hurt (or helps) central detection, that asymmetry is a genuine physiological finding and probably the most persuasive result available.

### 4.4 Clinical validation against AHI

Aggregate per-epoch respiratory predictions into an estimated per-patient event burden and correlate against the clinical AHI in `subject_description.xlsx`. Also report severity-class agreement (Normal / Mild / Moderate / Severe — cohort distribution is 15% / 24% / 23% / 38%).

No clinician stages apnea by epoch-level AUC. This converts a modest 0.700 into a clinically legible result and requires no retraining.

### 4.5 Severity stratification of staging

Break staging accuracy down by AHI severity class. "Does staging degrade as sleep-disordered breathing worsens?" is a real question in this cohort and the metadata is already available.

### 4.6 Improved airflow features (robustness check on the key claim)

Current airflow features are amplitude and line-length only. An apnea is a sustained *reduction* in airflow over ≥10 s, which a 30-second mean amplitude captures poorly. This weakens the airflow-removal argument: the result may show the airflow features were uninformative rather than that the model reads deep physiology.

Add a proper airflow feature — minimum 10-second windowed amplitude relative to a preceding 2-minute baseline — and re-run the removal test. Report whether the conclusion survives a fair airflow representation.

Also: in the current Table IV, removing airflow *improves* every metric (AUC 0.700→0.707, macro-F1 0.645→0.649, N1 F1 0.27→0.29), all within fold SD (~0.04). Rewrite the text to say "unchanged," not "marginally higher." Overclaiming here invites the obvious retort.

---

## 5. Figures

Replace the current figure set. Each figure must be an argument, not decoration.

1. **Modality ablation heatmap** — rows = modality removed, columns = staging accuracy / respiratory AUC. This is the headline figure and replaces current Fig. 4.
2. **Event-locked average traces** — all scored respiratory events aligned at onset, ±60 s window, showing mean SpO2, effort, asynchrony, heart rate, and the model's predicted probability. If the prediction rises with desaturation and falls with recovery, this single panel demonstrates physiological grounding more convincingly than any saliency map.
3. **Per-event-type detection** — AUC for hypopnea / obstructive / central, with the effort-ablation overlaid.
4. **Predicted burden vs. clinical AHI** — scatter with correlation, points coloured by severity class.
5. **Worked example night** — one patient: true and predicted hypnogram, scored events, predicted respiratory probability, SpO2 trace, on a shared time axis.
6. **Raw signal exemplars** — a 30-second window during an obstructive event vs. a matched normal-breathing window, all 14 channels.

Keep the cohort SDB-burden histogram (current Fig. 1) and the architecture diagram (current Fig. 2, updated for the concat decision).

---

## 6. Framing and the reference PDF

A second PDF is supplied as a **style reference only**: *DREAM: A novel explainable neural network for detecting sleep apnea using single-lead ECG signals* (Akter et al., published in Biomedical Signal Processing and Control, PII S1746809425018026).

### Take from it

- **Clinical anchoring in the opening.** It leads with disease burden and patient impact, not with benchmark numbers. Our introduction should do the same — sleep-disordered breathing after stroke, link to functional recovery.
- **Efficiency as a co-claim.** It pairs accuracy with parameter count and inference cost. We have a genuine version of this: 0.86M parameters producing two clinical outputs.
- **Figure density.** It carries a large number of figures and reviewers respond to that. Section 5 above matches or exceeds it.
- **A second contribution axis that isn't a leaderboard.** This is the structural lesson. It succeeded partly because "explainable + efficient" is not a dimension a reviewer can beat with a five-minute literature search.

### Do NOT take from it

- **Its evaluation design.** It filters "ambiguous samples" out of the dataset before the train/test split, using asymmetric criteria per class, and splits 80/10/10 on pooled segments rather than by subject. Both inflate its numbers substantially. Our protocol is ten-fold **patient-independent** cross-validation and stays that way. Never introduce any filtering, resampling, or normalisation fitted outside the training fold.
- **Its accuracy claims as a comparison target.** Do not cite its 98.79%/99.93% as a benchmark we should approach. Different dataset, different task, and the numbers are not leak-free.
- **Unverifiable saliency as the interpretability story.** Grad-CAM over a scalogram has no ground truth — nobody can check whether the highlighted region is anatomically correct. Our interpretability claims are falsifiable against annotated physiology, which is a stronger position. See below.
- **Its proofreading.** Its Table 6 contains a row identical to the proposed model's across all six columns, contradicted by the surrounding prose. This is exactly the class of error listed in Section 3 — sweep our tables and captions carefully.

### On the "explainable" wording

The instinct to use interpretability framing is right, but adopt **"physiologically interpretable"** rather than "explainable AI." Reasons:

- Our attribution comes from causal channel ablation, not from a post-hoc saliency heuristic. Removing a channel and measuring the effect is a stronger test than a heatmap.
- The two-stream architecture with the respiratory-head bypass gives structural attribution directly, without needing a saliency method.
- We have event-level ground truth to align predictions against (figure 2 in Section 5).

"Explainable AI" in a title now attracts XAI-methods reviewers who will ask for faithfulness metrics and SHAP comparisons. "Physiologically interpretable" promises exactly what we can deliver and is defensible on the evidence we will have.

Suggested title direction: *A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke*.

---

## 7. Dataset facts (from the descriptor — use these, don't re-derive)

Source: Maiti et al., *Polysomnography Dataset for Sleep Analysis in Ischemic Stroke Patients*, Scientific Data 13:421 (2026), DOI 10.1038/s41597-026-06747-w. Repo: github.com/suvadeepmaiti/iSLEEPS

- 100 patients, NIMHANS Bengaluru, Sept 2018–Dec 2021, all within one month of ischemic stroke onset
- Mean age 50.5 ± 12.0; 77 male, 23 female
- SOMNOscreen plus (SOMNOmedics); 6 EEG, 2 EOG, 3 EMG, plus airflow, thoracoabdominal effort, SpO2, snore, ECG, body position
- Native sampling: 128 Hz EEG/EOG/respiratory, 256 Hz EMG/ECG/snore, **4 Hz SpO2**
- Hardware filters: EEG/EOG/EMG 0.2–35 Hz with 50 Hz notch; ECG 0.3–70 Hz; respiratory 0.1–15 Hz
- Scored to AASM 2017 by independent researchers, supervised by a sleep neurologist
- 95,305 total annotations; stage shares W 26.20%, N1 9.94%, N2 41.27%, N3 8.74%, R 11.78%
- 15,462 respiratory events: hypopnea 80.8%, obstructive apnea 11.6%, central apnea 5.6%, mixed 0.8%, plus RERA and body events
- AHI severity: Normal 15%, Mild 24%, Moderate 23%, Severe 38%
- Clinical metadata in `subject_description.xlsx`: age, sex, medication, stroke-specific factors
- Published baselines used raw single-channel EEG (C4:M1 or C4:A1) **and** EOG (E1:M2 or EOG1:A2), 80/10/10 patient-wise with 10-fold CV: LSTM 74.70%, Transformer 67.44%, CNN 61.65%
- Ethics: NIMHANS IEC No. NIMHANS/34th IEC (BS&NS DIV.)/2022, dated 05.02.2022

Note our own audit finding, which stays in the paper as a methodological strength: SN28 is byte-identical to SN15 and is excluded, giving N=99.

---

## 8. Working conventions

- Run every new experiment on the **existing ten fold assignments**. Do not regenerate folds.
- Log every run with its configuration so each manuscript number traces to a specific run.
- Report fold standard deviations alongside every headline metric, as the current Table IV does.
- When a result contradicts the framing in Section 1, flag it rather than working around it. The framing follows the results, not the reverse.
- Run the ablations in Section 4 **before** rewriting the discussion. If dropping SpO2 barely moves the AUC, the interpretability story changes shape and the discussion should be written to what actually happened.
