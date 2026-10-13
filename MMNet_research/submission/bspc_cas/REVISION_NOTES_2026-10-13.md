# MM-Net BSPC revision, 13 October 2026

Source: the Overleaf project uploaded on 13 October (`main.tex`), committed unchanged as the baseline. The revision is in the next commit. Length goes from 32 to 20 pages, with no undefined references or citations.

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

## 2. Removed from the main text

These analyses ran on the frozen-encoder configuration before the cardiorespiratory epoch-alignment fix (results dated 18–28 September; fix committed 10 October):
- modality-ablation grid;
- joint-objective controls;
- respiratory bypass;
- incomplete-montage analysis;
- airflow "validity" analysis;
- permutation importance;
- stream representations;
- loss constants;
- calibration and conformal prediction;
- nested selection.

They can come back once they are rerun on the aligned caches. The modality-ablation grid is the one worth rerunning first.

Also removed for length or because of errors:
- the healthy zero-shot block and its section;
- the model-complexity table (its DeepSleepNet entry contradicted Table 5);
- Figure 1 (event burden);
- the AHI scatter figure and severity table (their values are kept in the text);
- the per-class table (its values are kept in the text);
- the LSTM gate equations;
- several defensive digressions in Methods.

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
