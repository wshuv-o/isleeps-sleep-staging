# Module Mapping — Hi-TGNet → MM-Net

Produced per **Part B** of `methodology_rewrite_prompt_template.md`: re-theme each of the reference paper's named modules into an equivalent that is faithful to our own architecture and data modality. Nothing here is invented — every module below already exists in `mm_feature_net.py`; what changes is that each is *named* and its role stated, rather than described generically.

**Modality note.** Hi-TGNet operates on 2-D MRI slices; MM-Net operates on 1-D physiological time series summarised as per-epoch features. The mapping therefore re-themes the *function* of each block, not its geometry — a dilated convolution widening a spatial receptive field maps to a recurrent decoder widening a *temporal* one.

---

## Mapping table

| # | Hi-TGNet module | Function in the reference | **MM-Net equivalent** | Function here |
|---|---|---|---|---|
| 1 | Adaptive Spiral Block | Parallel dilated branches widen the effective receptive field across scales | **Nocturnal Context Decoder (NCD)** | Bidirectional recurrence widens the receptive field from a 30 s epoch to a 20-epoch (10-minute) window, so a stage is read in the context of the night around it |
| 2 | Hierarchical Tumor Guidance | Attention steers representation toward tumour-relevant regions | **Direct Cardiorespiratory Pathway (DCP)** | Routes the cardiorespiratory stream to the respiratory head without passing through fusion, so oxygen desaturation reaches the decision undiluted by a 13× larger neural vector |
| 3 | Uncertainty-aware dual-head predictor | Two heads with confidence scaling on ambiguous cases | **Uncertainty-Aware Dual Read-out (UDR)** | Staging and respiratory heads from one forward pass, with split-conformal prediction sets giving distribution-free coverage and an explicit abstention |
| — | *(no counterpart)* | — | **Dual-Physiology Encoder (DPE)** | Two parallel encoders keep the neural (188-D) and cardiorespiratory (14-D) representations separate before fusion, which is what makes the modality ablation causally interpretable |

---

## Why each name is earned

**NCD** — the BiLSTM is not decoration. Removing temporal context is the difference between per-epoch classification and hypnogram-shaped output; the HMM decoding layered on top is what makes stage transitions physiologically plausible rather than epoch-independent.

**DCP** — this is the paper's actual novel component and the one the ablation isolates. Permutation importance now shows SpO2 costs 0.031 respiratory AUC and −0.001 staging accuracy: the pathway carries exactly what it was built to carry.

**UDR** — the *only* module for which we have stronger evidence than the reference paper. Hi-TGNet reports an uncertainty-aware head; we report calibrated coverage (0.894 at a 0.90 target) and show that abstention separates hard epochs from easy ones (86.9% vs 59.1% accuracy), with N1 — the stage human scorers agree on least — receiving a singleton set only 19% of the time.

**DPE** — worth naming because it is load-bearing for the interpretability claim. A single fused encoder would make modality ablation uninterpretable; keeping the streams separate is what licenses the attribution argument.

---

## Propagation required if adopted

Per Part E of the template, adopting these names means updating: the abstract, the bulleted contributions in §I, the methodology subsection headings in §V, the ablation table caption, the architecture figure labels (`mm_architecture_v2.drawio`), and the conclusion. The equations themselves are unchanged — only symbol names would be aligned.

---

## Open decision: the model's own name

Hi-TGNet takes its acronym from its guiding module (Hierarchical Tumor Guidance → Hi-TGNet). "MM-Net" is generic by comparison and does no work for the reader. Options that follow the same pattern:

| Candidate | Derivation | Reads as |
|---|---|---|
| **DCP-Net** | Direct Cardiorespiratory Pathway | Names the novel component, exactly parallel to Hi-TGNet |
| **PhysioRoute-Net** | Physiologically routed fusion | Names the principle rather than the block |
| **CardioCue-Net** | Cardiorespiratory cues driving the second read-out | Clinical, memorable, slightly softer on novelty |

**This is an authorial decision and I have not made it.** `DCP-Net` is the closest structural parallel to the reference paper, but renaming the model touches every document in the submission, so it should be a deliberate choice rather than a default.
