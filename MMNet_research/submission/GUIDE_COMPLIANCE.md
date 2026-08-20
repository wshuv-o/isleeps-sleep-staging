# How We Followed the Two Guides

This document works through both guides in Google Drive → *Computer Science Applications and Advancements (MScCS)* → `topics and novelty`, item by item, and states honestly where each requirement is **Present**, **Weak**, or **Missing** in the manuscript.

Section numbers refer to the compiled manuscript (`paper/multimodal.pdf`, IEEEtran two-column).

---

# Part 1 — `research_paper_instructions.md`

## Submission checklist, item by item

| # | Requirement | Status | Where it is |
|---|---|---|---|
| 1 | Professional journal-style formatting (IEEE two-column) | **Present** | `\documentclass[journal]{IEEEtran}`. Target venue IEEE JBHI — see `TARGET_JOURNAL.md` |
| 2 | Strong research motivation | **Present** | §I, opening: sleep-disordered breathing after stroke, and why healthy-sleep models fail here |
| 3 | Comprehensive comparative literature review | **Present** | §II, four themed subsections: single-channel deep staging, multi-channel/foundation models, SDB detection, sleep and breathing after stroke |
| 4 | **Critical gaps and limitations section (mandatory)** | **Present** | §III, a standalone section — not folded into Related Work |
| 5 | Clearly stated core contributions (bulleted) | **Present** | §I, bulleted `itemize` list (manuscript lines 135–154) |
| 6 | High-quality pipeline overview figure | **Present** | Fig. 3, vector PDF from the draw.io pipeline — see `figures/README.md` |
| 7 | Professional architecture diagram | **Present** | Fig. 3, generated as draw.io XML then hand-refined; `.drawio` sources committed |
| 8 | Customised architecture with explained novel component | **Present** | §V-C to §V-E. The novel element is the direct cardiorespiratory path to the respiratory head, so desaturation reaches the decision undiluted |
| 9 | Mathematical formulation of key components | **Present** | §V-A (problem formulation), §V-D (fusion), §V-F (joint loss, Eq. 9, retypeset in `align` after the equation was found truncated) |
| 10 | Full dataset details and working availability links | **Present** | §IV: iSLEEPS, 99 patients, channel table, sampling rates, AASM label definitions, public accession cited |
| 11 | GitHub repository / code-availability statement | **Present** | Code-availability declaration in the back matter with repository URL |
| 12 | Complete experimental setup (hardware + hyperparameters) | **Present** | §VI-B and Table III: optimiser, schedule, epochs, batch size, class-balanced loss, GPU |
| 13 | Appropriate, justified evaluation metrics | **Present** | §VI-A: accuracy, macro-F1, Cohen's κ for staging; AUC and average precision for respiratory events — AP justified by class imbalance |
| 14 | Comparison table against recent literature | **Present** | Table V (input-matched baselines) and Table VIII (recent single-channel methods with the accuracy each reports) |
| 15 | **Ablation study (mandatory)** | **Present** | §VI-F, Table VI: nine-condition leave-one-out modality grid, executed live, with Wilcoxon tests |
| 16 | Explainability analysis | **Present** | §VI-F and §VII: the ablation grid as physiological attribution, plus the t-SNE in Fig. 10 |
| 17 | Results and discussion with interpretation | **Present** | §VI (results) and §VII (discussion) |
| 18 | Honest limitations section | **Present** | §VII-A `\subsection{Limitations}`. Single centre, n=96 for the respiratory task, epoch-level rather than event-level scoring, staging at the cohort ceiling, and the feature-based (not end-to-end) design stated as a deliberate trade |
| 19 | Concrete future work | **Present** | §VII-B `\subsection{Future Work}`. Event-level metrics (sensitivity per event, false alarms per hour), finer breathing-waveform modelling, and linking the multimodal representation to stroke severity and recovery |
| 20 | Conclusion | **Present** | §VIII |
| 21 | 40–60 IEEE references, mostly 2023–2026 | **Present** | 49 references; 27 (55%) from 2023 or later — 8×2023, 7×2024, 6×2025, 6×2026 |

**Summary: 21 Present, 0 Weak, 0 Missing.**

## Note on items 18 and 19

At first pass these two were **Weak**, not for lack of content but for placement: the limitations and the forward-looking discussion existed as an unnumbered bold run-in paragraph and a following paragraph inside §VII, while the guide asks for them as identifiable sections. Both were promoted to numbered subsections (§VII-A and §VII-B) with the text preserved as written, and the manuscript recompiled. No new claims were introduced.

## Additional guide requirements beyond the checklist

**Tables required by the guide** (literature comparison, dataset statistics, experimental setup, hyperparameters, ablation, final comparison, computational complexity) — all **Present**: Table I (stage distribution), Table II (channels/sampling), Table III (training configuration), Table V (staging benchmark), Table VI (ablation grid), Table VII (respiratory baselines and per-event-type), Table VIII (recent literature), plus the model-size and capability table in §VI-J.

**Figure quality rules** — all figures are vector PDF, no screenshots or raster captures. Provenance for every figure, including the generation prompt and what was hand-changed, is in `figures/README.md`.

**Integrity rule** — the guide states that AI-generated text submitted without verification is misconduct. Our verification pass is recorded in `MMNet_Submission/VERIFICATION_LOG.md`: every headline number was re-derived by a live cell, and the pass found and corrected four real errors, including a headline misattributed to the wrong fusion variant and an AHI correlation computed over 9 patients instead of 96.

---

# Part 2 — `methodology_rewrite_prompt_template.md`

> **Note for the author:** the template file itself is not in this repository — it lives in the shared Drive folder. The account below describes the rewrite we actually performed and is accurate to what is in the manuscript and the verification log. Please confirm the module names against your copy of the template before submitting, and replace this note.

## Module mapping table

The rewrite renamed the architecture's components so that the name of each module states its physiological function rather than its tensor operation:

| Original (draft) name | Rewritten name | Why |
|---|---|---|
| Encoder A / branch 1 | **Neural stream** | Consumes EEG, EOG, EMG — the 188 neural features |
| Encoder B / branch 2 | **Cardiorespiratory stream** | Consumes SpO2, effort, airflow, ECG, pulse — the 14 cardio features |
| Fusion block | **Concatenative fusion** | Renamed once the ablation showed concatenation, not attention, produced the headline |
| Skip connection | **Direct cardiorespiratory path** | It is the paper's novel component; the name now says what it carries and why |
| Sequence model | **Temporal decoder (BiLSTM)** | Distinguishes night-level context modelling from per-epoch encoding |
| Head 1 / Head 2 | **Staging head / Respiratory head** | Names the clinical output, not the tensor shape |

## What changed in the Methodology

§V was restructured to follow the guide's required order: problem formulation, then feature representations, then the two encoders, then fusion, then the temporal decoder and heads, then training and decoding. The novel component (the direct cardiorespiratory path) was given its own explanation stating the physiological reason it exists — that desaturation must reach the respiratory decision without being diluted through fusion with the much larger neural feature vector.

## Equations preserved verbatim

The mathematical content was **not** rewritten — only the symbol names were made consistent with the new module names. Preserved verbatim: the problem formulation, the per-epoch feature-extraction definitions, the fusion operation, the BiLSTM recurrence, and the two output heads. The joint-loss equation was preserved in content but retypeset in an `align` environment, because the original overflowed the column and clipped its weighting term in the compiled PDF.

## Propagation of the new names

The renaming was propagated everywhere the old names appeared, which is where the rewrite caught a genuine error. Updated: the abstract, the bulleted contributions in §I, the critical-gaps section §III, the results discussion in §VI, the ablation table and its caption in §VI-F, the keyword list, and the conclusion.

The propagation exposed that the abstract described *cross-modal attention* while reporting the *concatenation* run's numbers (0.721 / 0.700). Concatenation is now the stated headline throughout and cross-modal attention is reported honestly as an ablation at 0.712 / 0.705. This is recorded as entry 4 in `VERIFICATION_LOG.md` and is the before/after example in `RESPONSE_TO_REVIEW.md`.

## "ACTION REQUIRED FROM ME" list

The rewrite produced the following items that could not be resolved by editing text and required running something or supplying information:

1. **Re-run both fusion variants** under identical folds to establish which produced the headline. → Done; concatenation confirmed.
2. **Repair the AHI patient join**, which matched 9 of 96 patients. → Done; corrected to ρ = 0.315, p = 0.0017, n = 96.
3. **Add respiratory baselines**, absent from the draft. → Done; desaturation rule 0.596, logistic regression 0.582, gradient boosting 0.670.
4. **Add significance testing** over the ten folds. → Done; Wilcoxon p = 0.004 (respiratory), p = 0.91 (staging).
5. **Disclose the SN15/SN28 duplicate** and state per-experiment *n*. → Done in §IV.
6. **State the SpO2 native sampling rate** and the upsampling step. → Done in §IV.
7. **Supply the corresponding author and the declarations** (data, code, ethics, funding). → Done.
8. **Confirm the ethics approving body** for the source cohort. → NIMHANS Institutional Ethics Committee.
