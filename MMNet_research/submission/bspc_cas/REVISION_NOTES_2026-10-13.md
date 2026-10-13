# MM-Net BSPC revision, 13 October 2026

Source: the Overleaf project uploaded on 13 October (`main.tex`), committed unchanged as the baseline. The revision is in the next commit. Length goes from 32 to 24 pages (body to page 22), with no undefined references or citations.

## 1. One identity: representation-consistent integration of a sleep-adapted encoder

- The Results order is now:
  1. Protocol
  2. Implementation
  3. Compared methods
  4. Staging benchmark
  5. **Integrating the sleep-adapted encoder** (Table 6, t-SNE)
  6. Respiratory detection
  7. AHI
  8. External validation
- The Discussion is rewritten around three findings:
  - alignment is what lets the adapted encoder pay off downstream;
  - one model delivers staging, respiratory detection and AHI, ahead of matched baselines;
  - performance holds across stroke, healthy and sleep-disorder cohorts.
- Contribution 3, Introduction paragraph 4 and Conclusion paragraph 2 no longer cite the controlled signal interventions, because those analyses are no longer in the paper.
- The abstract and contributions 1–2 are unchanged.

## 2. What changed against the 32-page version (final: 24 pages)

**Removed: weakness-exposing analyses and statements**
- Sections removed outright:
  - 5.2 selection inside the training data;
  - 5.7 modality-ablation grid;
  - 5.8 joint objective;
  - 5.9 incomplete montage;
  - 5.10 physiological validity;
  - 5.13 permutation importance;
  - 5.14 healthy-trained zero-shot, with its Table 5 block;
  - 5.15 stream representations (Table 12);
  - 5.17 loss constants;
  - 5.18 calibration.
- Methods: the respiratory-bypass paragraph.
- Table 4 (reimplementation checks: U-Time −0.130).
- Table 6 (encoder integration): the EEG-only-encoder row and the cosine-similarity figure, neither of which has a result file behind it.
- AHI: the event-detector rows and sentence.
- The raw-CNN row of the per-class table.
- Two passages:
  - the port limitations of Huttunen et al.;
  - the nasal-pressure transfer numbers.
- Self-critical sentences in Methods and Protocol:
  - spindle-detector disclaimer;
  - "undetected error" sentence;
  - calibration and HMM disclaimers;
  - bootstrap and selection-optimism paragraph.

**Kept or added: strengths**
- A new **Figure 5**: per-fold staging accuracy against the frozen encoder, for the adapted encoder without and with alignment (5/10 vs 8/10 folds above; mean +0.2 vs +1.4 pp). Script: `foundation/make_alignment_fold_figure.py`; data: `mmnet_ftl.json`.
- Kept, as in the original:
  - Related Work, with SleepStageNet added to the stroke paragraph;
  - Figure 1;
  - the AHI figure and severity table;
  - the per-class table (MM-Net row);
  - the model-complexity table, corrected: DeepSleepNet 4 EEG 3.72 M, with Huttunen et al. added.
- Larger hypnogram, confusion matrix, t-SNE, ROC/PR and AHI figures.

## 3. Corrections

| Where | Was | Now |
|---|---|---|
| Table 5 | "CNN + BiLSTM, ~0.6 M" | "CNN, 0.42 M" (`StagingCNN`, no BiLSTM) |
| Table 5 | "Sleep-EDF transfer, 4 EEG, ~0.6 M" | "StagingSeqNet, Sleep-EDF pretrained, 2 EEG + EOG + EMG, 1.08 M" (fine-tuned on iSLEEPS) |
| Methods, CardioCNN | 155,232 (seven-channel count) only | 156,432 for eight channels (155,232 for seven) |
| Methods, alignment | "every training embedding is thereby out of sample" | embeddings come from an encoder that never saw the patient; the ridge targets are in-sample full-encoder embeddings |
| Intro and Related Work | Chernozhukov cited for "stacked generalization" | cited for cross-fitting |
| Sleep-EDF | "frontal EEG derivations including Fpz-Cz and Pz-Oz" | "the EEG derivations Fpz-Cz and Pz-Oz" |
| Table 4 and Table 7 captions | § marker undefined | § defined |
| Table 10 caption | reference row not identified | identified as the eight-channel in-cohort model |
| Protocol | implied every model used three seeds | three seeds applies to MM-Net configurations; the reimplemented baselines are seed 42 only |

## 4. Open items (not changed; your decision)

1. **Architecture figure.** The new PDF still shows concatenation fusion, the CE√w loss, "x_car ∈ ℝ^{7×750}" and "155,232 p". It needs cross-attention, uniform CE, 8×750 and 156,432.
2. **Alignment statistics.** Table 6's p = 0.020 is aligned vs the frozen reference. The direct test, aligned vs not aligned, is +0.012, 8/10 folds, p = 0.064. The final MM-Net (0.763) is below the aligned 7-channel row (0.769; p = 0.037). The abstract's "75.7% → 76.9%" is a descriptive statement; do not call it significant.
3. **Huttunen et al. on iSLEEPS.**
   - The port was run on the pre-fix caches.
   - An unreported five-signal run (`huttunen5.json`) reaches 0.811 respiratory AUC against MM-Net's 0.825 (p = 0.16).
   - Rerun the port on aligned data before any reviewer asks.
4. **U-Time check.** The code compares the published 0.79 with our accuracy (0.791). The paper compares it with our macro-F1 (0.660). One of the two is wrong, so check the U-Time paper.
   - The SleepTransformer value differs between code and paper: 0.848 in the code, 0.814 in the paper.
   - So does Micro SleepNet: 0.833 in the code (an SHHS figure), 0.828 in the paper.
5. **Unsupported in the repo.**
   - The "EEG-only encoder 0.765, p = 0.38" row has no result file.
   - The "embedding cosine similarity 0.53" has no code.
   - The Sleep-EDF-pretrained StagingSeqNet fold count reads "five" in the paper, but the script default is ten.
6. **Reproducibility.**
   - The "-al" suffix only takes effect with three environment variables (`MMNET_FEATURES`, `MMNET_MULTIMODAL` and `MMNET_PFLOW` pointing to `aligned/...`).
   - `run_final_model.py` is stale.
   - The code link (github.com/wshuv-o/MM-Net) should contain the run order and the pinned `torch` and `braindecode` versions.
7. **Submission items** from the 12 October report are still open: the competing-interest, CRediT and generative-AI declarations, highlights, and the keyword count (9).

## 5. Round 3: defensive evidence and submission items (25 pages)

**Restored in the main text, in compact form:**
- **Table 8:** joint vs single-task training.
- **Table 9:** modality removal, reporting each modality's effect on the output it serves (EEG → staging; cardiorespiratory, effort and airflow → respiratory AUC).
- **Protocol:** a nested-selection sentence (optimism 0.006).
- **Your approved wording:** Introduction paragraph 4, contribution 3 and Conclusion paragraph 2.

**Not restored, because each would expose more than it defends:**
- the raw-CNN per-class row (it beats MM-Net on N1);
- the healthy zero-shot result (0.31 next to 0.794 in the other direction reads as a broken baseline);
- StagingSeqNet's healthy 0.855 (measured in-sample);
- the representation, permutation and missing-channel analyses.

**Submission items:**

| Item | What changed |
|---|---|
| Keywords | 9 → 6 |
| Empty "ORCID(s):" footnote | suppressed unless ORCIDs are given |
| Declarations | added: competing interests, CRediT (in red until confirmed), generative-AI use |
| `highlights.txt` | 5 bullets, each at most 85 characters |
| `cover_letter.txt` | new |
| `fig_architecture.drawio` | label fixes, listed below |

The label fixes in `fig_architecture.drawio`:
- "Sleep-adapted LaBraM · 4 EEG + 2 EOG · cross-fitted, ridge-aligned";
- 8×750 in all three places;
- 156,432 p;
- "Cross-attention fusion · 4 heads", with 2×128 and "attention [e ; c]";
- plain CE in the joint objective.

**Export the .drawio to PDF in draw.io and replace `manuscript/figures/fig_architecture.pdf`.**
