# MM-Net BSPC Revision: Implementation Report

Manuscript: "MM-Net: A Multimodal Multi-Task Network for Sleep Staging and Sleep Apnea Assessment After Ischemic Stroke"
Target journal: Biomedical Signal Processing and Control (Elsevier CAS single-column template, cas-sc)
Source of truth: the current 26-page IEEE Access source (multimodal_access.tex, Overleaf copy) and the result files under results/revision.

## 1. Deliverables

| Item | File | Status |
|---|---|---|
| (a) Revised manuscript source | manuscript/mmnet_bspc.tex with cas-sc.cls, cas-common.sty, model1-num-names.bst, references.bib, figures/, thumbnails/, author photographs | Delivered |
| (b) Compiled PDF | mmnet_bspc.pdf (32 pages: 28 pages of manuscript and references, 4 pages of biographies) | Delivered |
| (c) Architecture XML | fig_architecture.drawio, byte-identical to the input (MD5 f89fd66892e4f1201eaa582fd10f1945) | BLOCKED (Section 3); no edit made |
| (d) Language audit | MM-Net_Complete_Language_Audit.docx (176 logged occurrences) | Delivered |
| (e) Implementation report | this file (.md and .docx) | Delivered |
| (f) Issues for approval | Section 9 of this file | Delivered |
| Overleaf package | MMNet_BSPC_overleaf.zip (contents of manuscript/) | Delivered |

## 2. CHANGE 001A: learning curve removed (complete)

- Removed the subsection "Data Efficiency and the Limits of Cohort Size" (Experiments), its figure environment (label fig:curve, file fig_learning_curve.pdf, formerly Figure 7) and its caption. This subsection existed only to present and interpret that figure; it had 36 source lines.
- Removed "learning curve," from the list of controlled analyses in Experiments > Protocol and Metrics. This was the only other reference to the experiment.
- No other text cited fig:curve. A search of the revised source for "learning curve", "learning-curve", "fig:curve" and "Data Efficiency" returns nothing.
- Renumbering is label-driven. The t-SNE figure (fig:tsne, formerly Figure 8) is now Figure 7. Verified figure sequence in the compiled PDF:
  Fig. 1 apnea prevalence (p. 5), Fig. 2 architecture (p. 7), Fig. 3 hypnogram (p. 15), Fig. 4 confusion matrix (p. 16), Fig. 5 ROC/PR (p. 19), Fig. 6 AHI (p. 21), Fig. 7 t-SNE (p. 24).
- fig_learning_curve.pdf is not included in the delivered figures folder.

## 3. CHANGE 001B: architecture XML (BLOCKED, not marked complete)

### 3.1 Inventory
- The editable source is MMNet_research/figures/fig_architecture.drawio: uncompressed draw.io XML with 138 mxCells (109 vertices, 27 edges, 2 root cells).
- fig_architecture.pdf, the file included in the manuscript, carries no embedded XML. The .drawio file is the only editable source.

### 3.2 Final model, validated against the code and the manuscript
Validated against the configuration base-nocw-cross-ftlae-p8-al, Table 3 of the manuscript (training configuration) and foundation/cardio_cnn.py:
- LaBraM-Base fine-tuned per fold on 4 EEG and 2 EOG channels. Training embeddings are cross-fitted (K = 3) and ridge-aligned (λ = 0.01).
- 188 engineered descriptors plus the 200-d embedding form a 388-d input to the two-layer feature MLP (to 128).
- CardioCNN over 8 × 750 (seven channels plus nasal pressure).
- Cross-attention fusion with 4 heads.
- BiLSTM with L = 20 and 512-d output.
- Staging head with HMM Viterbi decoding; respiratory head reading [h_t; c_t].
- Uniform staging class weights; respiratory positive weight π = n−/n+; λ = 1.
- AHI = k · Σ onset scores over decoded sleep / sleep hours.

### 3.3 Component-level audit (before = after; no byte changed)

| Cells | Component (existing text) | Final model | Verdict |
|---|---|---|---|
| 137, 138, 139, 4, 147 | Neural input block: "Expert physiology, f_eeg ∈ ℝ188"; "band power, spindle, Hjorth"; legend "EEG" | 188 descriptors | Correct |
| 5 | "Frozen LaBraM ❄ z_fnd ∈ ℝ200 · 4 EEG channels, no gradient" | Sleep-adapted (fine-tuned) LaBraM, 4 EEG + 2 EOG, cross-fitted and aligned, 200-d | MISMATCH; requires a label change (conflict C1) |
| 6, 17, 144 | concatenation node "C" joining descriptors and embedding | 188 + 200 = 388 | Correct |
| 27–50 | FeatMLP: 388 → Linear W₁ 128 → LN+GELU → Linear W₂ 128 → LN+GELU → e (128); "EEG encoder φeeg · FeatMLP 388 → 128" | Same (Eq. 5) | Correct |
| 94, 96, 97, 148, 149, 150, 95 | Cardiorespiratory input: "x_car ∈ ℝ7×750 per-subject z-score"; "Raw cardiorespiratory tensor"; legend "PSG", "raw", "7×750" | 8 × 750 (nasal pressure added) | MISMATCH in cells 96 and 150; requires label changes (conflict C2) |
| 51–92 | CardioCNN layers: Conv 25 s2 → 48×375, Pool 4 → 48×93, Conv 15 → 96×93, Pool 4 → 96×23, Conv 7 → 96×23, mean ⊕ max 192, Linear → 64 | Same layer stack (cardio_cnn.py) | Correct; feature-map shapes do not depend on the input channel count |
| 93 | "Cardio encoder φcar CardioCNN: 7×750 → c ∈ ℝ64 · 155,232 p" | 8×750; 156,432 parameters (155,232 is the 7-channel count; the first convolution adds 48 × 25 = 1,200 weights) | MISMATCH; requires a label change (conflict C2) |
| 7, 10, 11, 19, 20, 21, 135, 200–211 | Fusion: "Multimodal fusion"; "192"; "concat [e ; c]"; "Linear W_f"; "128"; "GELU + Dropout" | Cross-attention fusion, 4 heads | MISMATCH; the block draws the concatenation fusion of the frozen-encoder configuration; requires changing existing labels (conflict C3) |
| 13, 14, 12 | "BiLSTM 2 layers, bidirectional context L = 20 epochs (10 min) h_t ∈ ℝ512" | Same | Correct |
| 15, 22 | "Staging head 512 → 5 + HMM Viterbi decode" | Same | Correct |
| 16, 23, 24, 25 | "Respiratory head [h_t ; c_t] · 576 → 256 → 1 direct cardio bypass"; "direct c_t" | Same | Correct |
| 26 | "Joint objective: L = CE√w(ŷstg, ystg) + λ BCEpw(ŷapn, yapn), λ = 1" | Staging cross-entropy with uniform weights (no √w); BCEpw and λ = 1 correct | MISMATCH; requires a label change (conflict C4) |
| none | (absent) EOG input to the encoder | 2 EOG channels feed the encoder | MISSING component (addition A1) |
| none | (absent) cross-fitting and ridge alignment of training embeddings | Part of the trained pathway | MISSING component (addition A2) |
| none | (absent) AHI output | AHI from respiratory scores over decoded sleep (named in the caption) | MISSING component (addition A3) |
| 151–162 | Legend swatches and words (input, conv, pool, linear, norm, output) | n/a | Correct |

### 3.4 Why the task is blocked
Conflicts C1 to C4 can only be corrected by changing existing text elements, which the instruction forbids. Adding A1 to A3 alone would place new components next to labels that still describe a frozen, 4-channel-input, 7-channel, concatenation-fusion, √w-weighted model. The figure would then contradict itself. Per the instruction, I stopped and made no edit. The delivered .drawio is identical to the input. The figure caption in the manuscript already describes the final model, so the drawing and its caption currently disagree.

### 3.5 Minimal edits proposed for approval (not applied)
- C1, cell 5: "Sleep-adapted LaBraM z ∈ ℝ200 · 4 EEG + 2 EOG, fine-tuned per fold" (drop "Frozen", "❄" and "no gradient").
- C2, cells 96 and 150: "7×750" → "8×750". Cell 93: "7×750" → "8×750" and "155,232 p" → "156,432 p".
- C3, cells 201–208: relabel the fusion stack as cross-attention with 4 heads (for example, cell 202 "cross-attention [e ; c], 4 heads"), keeping the same boxes and positions.
- C4, cell 26: "CE√w" → "CE".
- A1: one input arrow labelled "EOG ×2" into cell 5.
- A2: one small annotation box under cell 5: "training embeddings: cross-fitted (K = 3), ridge-aligned".
- A3: one output box after cell 16: "AHI = k Σ onset scores / sleep h".

## 4. CHANGE 002: abstract (complete; two flags)
- The approved base text was inserted verbatim. The only edits are LaTeX encodings: "76.3\%", "Cohen's $\kappa = 0.656$" and the en dash in "apnea--hypopnea". No readable word was changed.
- Length: 194 words, under the 250-word limit used for BSPC. The current Guide for Authors could not be retrieved (HTTP 403), so this limit is unverified (Section 9).
- Every number was checked against the result files:
  - 99 patients.
  - 76.3% staging accuracy and κ 0.656 (3-seed mean, ten patient-independent folds).
  - Respiratory AUC 0.825.
  - ISRUC-I: 80.3% accuracy and AUC 0.889 (ten-fold CV).
  - "outperforming the evaluated baselines under matched protocols" holds: all 13 retrained staging baselines and all respiratory baselines (desaturation rule, logistic regression, gradient boosting, raw-effort CNN, raw 7-channel CNN, the last at 0.733) score below MM-Net under the same folds.
  - External respiratory discrimination: AUC 0.833 on ISRUC-I (transfer) and 0.841 on UCD.
- Flagged and not applied: two "while" sentences (audit LA-001, LA-002). The closing sentence is retained as approved. It does not attribute the controlled analyses to a specific configuration; an optional qualifier is offered in Section 9.

## 5. CHANGE 003A: contributions (complete, verbatim)
- The lead-in sentence and the four-item list in the Introduction were replaced by the approved text, including "The principal contributions of this study are threefold:".
- Markup added: \begin{enumerate} and \item only. Character-level comparison with the instruction file: identical.
- No scientific contradiction was found. Contribution 3 mentions "controlled signal interventions". These were run in the frozen-encoder configuration, and the body states this wherever they are reported. The locked text does not attribute them to the final model, so it is not a contradiction; it is listed for information in Section 9.
- The language audit did not modify this passage. Its one "while retaining" is logged as Unchanged.

## 6. CHANGE 003B: Conclusion (complete; two flags)
- The approved draft replaced the previous Conclusion. The only change is a LaTeX en dash in "training--inference".
- Claim check:
  - "improving sleep-stage discrimination": supported by Table 13. Against the frozen-encoder reference row (0.755, same fusion and class weights), the cross-fitted and ridge-aligned sleep-adapted encoder reaches 0.769 3-seed accuracy (p = 0.020); the final MM-Net reaches 0.763.
  - "supporting concurrent respiratory-event detection": supported (AUC 0.825).
  - "experiments across independent sleep cohorts": supported (ISRUC-I/II/III, UCD, Sleep-EDF).
  - "controlled interventions reveal distinct roles …": the modality ablation, joint-objective, airflow and permutation analyses were all run in the frozen-encoder configuration. The sentence is not qualified to that setting. FLAGGED (LA-175) with a minimal fix; not applied.
- "Together, …, while providing a foundation …" is a generic concluding formula. FLAGGED (LA-176) with a proposal; not applied.
- The old Conclusion's numerical summary is no longer present. AHI remains fully reported in Results (Section 5.12, Tables 10–11, Fig. 6), in the external validation (Tables 14–15) and in the Discussion.

## 7. CHANGE 004: language audit (complete)
- All 691 sentences of the revised source were read: abstract, body, figure and table captions, table notes, statements and Conclusion.
- 176 occurrences are logged in MM-Net_Complete_Language_Audit.docx:

| Status | Count |
|---|---|
| Corrected (applied) | 122 |
| Flagged (approved text, proposal only) | 4 |
| Unchanged (genuine comparison, factual superlative or locked text) | 50 |

| Category | Corrected | Flagged | Unchanged |
|---|---|---|---|
| Conversational or informal expression | 69 | 0 | 0 |
| 'while X, Y' / 'whereas' construction | 13 | 2 | 35 |
| Factual or configuration-labelling inaccuracy | 12 | 1 | 0 |
| Inflated or unsupported claim | 11 | 0 | 0 |
| 'not X but Y' / 'X, not Y' construction | 5 | 0 | 8 |
| Evaluative or promotional term | 0 | 0 | 7 |
| Anthropomorphic wording | 3 | 0 | 0 |
| Unnecessary methodological restatement | 3 | 0 | 0 |
| 'not only/merely … but' construction | 2 | 0 | 0 |
| 'rather than X, Y' construction | 2 | 0 | 0 |
| Generic or vague concluding sentence | 1 | 1 | 0 |
| 'more than just' / 'beyond simply' construction | 1 | 0 | 0 |

- Seven sentences were deleted as restatements or conversational fragments: LA-027, LA-049, LA-065, LA-083, LA-085, LA-126, LA-173. Each is logged with its full original text.
- Configuration labelling: where a sentence attributed a frozen-encoder result to "the reported model" or to the model in general, the configuration is now named. This covers sentences in Selection Inside the Training Data, Physiological Validity, Fusion and the Discussion opening (rows marked "Factual or configuration-labelling inaccuracy").
- Numbers: a numeral-by-numeral comparison of the body before and after the audit shows no number changed. The only additions are the existing iSLEEPS values 0.306–0.312, quoted to correct "as it does on iSLEEPS" (LA-143), and template values.
- Not touched by the audit: the abstract base text, the locked contributions, the Conclusion draft, all numbers, citations, caveats and unfavourable results.

## 8. Structural fix and CAS conversion

### 8.1 Structural fix
- The paragraph "The learned cardiorespiratory encoder" sat inside the subsection "Sleep-Adapted EEG Encoder". I moved it, unchanged, to the end of "Physiological Feature Representations", directly after the 14 cardiorespiratory descriptors it replaces. "Sleep-Adapted EEG Encoder" now contains only the Adaptation, Cross-fitting and Alignment paragraphs.

### 8.2 Template conversion (body text carried over unchanged)
- Class: ieeeaccess → cas-sc [a4paper, fleqn].
- Citations: cite → natbib [numbers, sort&compress]. Bibliography style: IEEEtran → model1-num-names (numbered).
- Front matter: IEEE \history, \doi, \author/\authorrefmark, \address, \corresp and \titlepgskip were replaced by CAS \shorttitle, \shortauthors, \title, \author/\ead, \affiliation and \cormark/\cortext.
  - Same seven authors in the same order, four affiliations, and two corresponding authors (M. I. Mobin, J. Uddin).
  - The three student e-mails were assigned in the order listed in the IEEE address line (Section 9).
- Keywords: comma list → \sep list (same 9 keywords).
- Removed IEEE-only commands: \bstctlcite (and its IEEEexample:BSTcontrol key), \PARstart drop cap, \EOD.
- Biographies: 7 IEEEbiography environments → CAS \bio … \endbio. Text and photographs unchanged.
- Floats:
  - All 7 figures and 16 tables are single-column and pinned with [H] (float package).
  - The cas-sc figure environment drops a bare [H]/[t] (its key parser returns an empty position), so every figure was deferred to the end of the document. The environment is redefined in the preamble with identical styling, passing pos=H to float.sty.
  - Verified: each figure and table appears in its source position, in the section where it is discussed (page map in Section 2).
- Tables: five \resizebox{\columnwidth} became \adjustbox{max width=\textwidth} (shrink only). The two tabular* tables that overflowed the single column (Tables 5 and 14) became tabular inside adjustbox. Cell content is unchanged.
- Figure widths were adjusted to the single-column page:
  - Formerly single-column figures: 0.55–0.72 \textwidth.
  - Architecture figure: 1.1\textwidth → \textwidth; its existing height of 7 cm is kept.

### 8.3 Build verification
- latexmk: 0 undefined references, 0 undefined citations, 0 "??" in the PDF.
- The only overfull boxes are inside the CAS title block (template) and one of 1.7 pt.
- 53 cited keys, all present in references.bib. No citation was dropped except the IEEE BST-control key.
- Every \ref has a matching \label.

## 9. Issues for user approval
1. **CHANGE 001B (BLOCKED).** Approve the label edits C1–C4 and the additions A1–A3 (Section 3.5), or confirm the drawing should remain as is. Until approved, Figure 2 shows the frozen-encoder/7-channel/concatenation configuration while its caption describes the final model.
2. **Abstract LA-001:** replace ", while respiratory disturbances introduce" with " and respiratory disturbances introduce".
3. **Abstract LA-002:** replace ", while cross-fitting and embedding alignment address representation mismatch" with "; cross-fitting and embedding alignment address the representation mismatch".
4. **Abstract closing sentence (optional):** "Controlled analyses in a frozen-encoder configuration identify distinct contributions …" (+4 words; total 198). This qualifies the configuration of the controlled analyses.
5. **Conclusion LA-175 (evidence qualification; recommended):** "Second, controlled interventions, performed in the frozen-encoder configuration of the joint framework, reveal distinct roles for neural and cardiorespiratory information, showing that …".
6. **Conclusion LA-176:** "These results establish a computational framework … after ischemic stroke and provide a foundation for further investigation …" (removes "Together," and "while providing").
7. **Contribution 3 (locked, information only):** "controlled signal interventions" were performed in the frozen-encoder configuration. The text makes no wrong attribution and is left verbatim.
8. **BSPC Guide for Authors could not be retrieved** (HTTP 403). These items are therefore unverified and need your check:
   - (a) Abstract limit of 250 words: the abstract has 194.
   - (b) Highlights: Elsevier journals normally require 3–5 bullets of at most 85 characters each, supplied as a separate file. None were added, because adding them is outside the approved scope. Candidate text for your approval:
     - "MM-Net jointly stages sleep and detects respiratory events in post-stroke PSG" (77)
     - "Cross-fitting and alignment integrate a sleep-adapted EEG encoder" (65)
     - "iSLEEPS: 76.3% staging accuracy (κ 0.656) and 0.825 respiratory-event AUC" (73)
     - "ISRUC-I: 80.3% staging accuracy and 0.889 respiratory-event AUC" (64)
     - "Ablations show distinct neural and cardiorespiratory roles in the two tasks" (75)
   - (c) Keyword limit: the manuscript has 9; many Elsevier journals allow at most 6–7.
   - (d) Author biographies: kept as CAS \bio blocks; BSPC may not publish biographies.
   - (e) Required declarations not present in the source and not written by me: Declaration of competing interest, CRediT authorship contribution statement, and the declaration on generative AI use in the writing process.
9. **Author e-mails:** the three student addresses were mapped to Suva, Abha and Sajin in the order the IEEE address line lists them. Please confirm. The CAS title page also prints an empty "ORCID(s):" line; supply ORCIDs or approve its removal.
10. **Position of one sentence (LA-101):** the reworded sentence on attention versus concatenation fusion still sits at the end of "Recordings With an Incomplete Montage", where it was in the original. Approve moving it to Proposed Method > Fusion, or keep it.
11. **Layout:** pinning floats with [H] leaves white space at the foot of pages 20 and 24, where a large figure follows a table. Approve [H] (exact placement) or allow [!htbp] for those two figures.
