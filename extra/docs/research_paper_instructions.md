# Research Paper Writing Assignment — Instruction Guide (MScCS)

**Course:** Computer Science Applications and Advancements (MScCS)
**Submission Deadline:** 21 July 2026 (11:59 PM)
**Reference paper:** *Hi-TGNet: Hierarchical Tumor Guidance Network for Explainable Brain Tumor Classification* (provided in Google Drive → `topics and novelty` → `Sample quality journal.pdf`)

---

## How to Use This Guide

This is a rewrite of the original assignment brief. Instead of listing requirements in the abstract, every section below shows **what the requirement means** and **how the sample Hi-TGNet paper actually satisfied it**. Read the sample paper once end-to-end first, then use this document as a checklist while you write your own manuscript.

Your goal is not to imitate the brain-tumour topic. Pick your own problem. The goal is to match the *structure, technical depth, and presentation standard* the sample paper demonstrates.

---

## The Standard You Are Aiming For

Your paper must read like a submission to a reputable Q1/Q2 journal, not a classroom report. Concretely, that means:

- A genuinely novel method (a new architecture or module), not a summary of existing work.
- A methodology section deep enough that a competent reader could reimplement your model.
- Mathematical formulation of your key components.
- Proper experimental evaluation, including in-domain testing **and** external validation on a second dataset.
- A mandatory ablation study proving each component earns its place.
- 40–60 references, mostly 2023–2026, in IEEE style.
- Publication-quality figures (vector diagrams, not screenshots) and clean tables.

**Do not** copy from review papers, paste raw ChatGPT output, use hand-drawn or blurry figures, or submit weak referencing. AI-generated text submitted without verification and refinement is treated as academic misconduct.

---

## Required Sections (in order)

The sample paper follows this exact skeleton. Yours should too.

### 1. Title
Modern, specific, and method-forward. Name the technique and the application.

- **Good pattern:** *"Hi-TGNet: Hierarchical Tumor Guidance Network for Explainable Brain Tumor Classification"* — it names the model, the mechanism (hierarchical guidance), a differentiator (explainable), and the task.
- **Weak pattern:** *"CNN for Brain Tumor Detection"* — generic, no novelty signalled.

### 2. Abstract (one paragraph, ~200–250 words)
Cover, in order: the problem, the gap in existing methods, your proposed method and its named components, the datasets, the headline results with numbers, and the takeaway.

- *How the sample did it:* it states the accuracy–efficiency trade-off problem, names its three components (Adaptive Spiral Block, Hierarchical Tumor Guidance module, uncertainty-aware dual head), lists datasets (BraTS 2020, ND-5, BRISC 2025, Mendeley), and reports concrete numbers (97.32% in-domain, 98.70% and 93.54% external, 3.98M parameters). Copy this level of specificity — quantify.

### 3. Introduction
Include: background and why the problem matters, current challenges, why existing methods fall short, your objectives, and a bulleted list of contributions.

- *How the sample did it:* it opens with epidemiological stakes, narrows to the specific classification task, surveys the CNN/ViT/hybrid landscape at a high level, names the concrete failure modes (heavy models, background confusion, weak multi-scale handling, poor interpretability), then closes with a clean bulleted contributions list. **End your introduction with that same bulleted contributions list.**

### 4. Related Work
Recent literature (mostly 2023–2026), from strong venues. **Do not** write one summary after another. Group papers thematically and compare — strengths, weaknesses, and the trend each represents.

- *How the sample did it:* it organises related work into named sub-themes ("Early Stages of Automated Detection", "CNNs, Transformers, and Hybrid Architectures", "Ensemble, Federated Models, and XAI", "Attention Mechanisms and Specialized Convolutions"), and inside each it contrasts approaches ("heavy CNNs matched hybrid accuracy but with smaller inference cost", "Swin-Large hit 99.37% but at heavy compute"). Use thematic sub-headings and comparative sentences, not a citation list.

### 5. Critical Gaps and Limitations of Previous Studies *(mandatory)*
State plainly why prior methods are insufficient, and make sure **your contributions answer these exact gaps.**

- Typical gaps: heavy architectures, poor explainability, no uncertainty estimation, limited/single-dataset evaluation, poor generalisation, no external validation, high compute cost, weak feature extraction, missing ablation.
- *How the sample did it:* it isolates the black-box problem, dataset imbalance/overlap, and the MRI-protocol generalisation gap — then its method (lightweight design, guidance-based interpretability, uncertainty head, external validation on two datasets) maps one-to-one onto those gaps. **Make that mapping obvious in your paper.**

### 6. Core Contributions
Bullet points. Each bullet = one concrete novelty, phrased as a component or capability, not a vague claim.

- *How the sample did it:* five tight bullets, each naming a component and its function (e.g. "An Adaptive Spiral Block that captures multi-scale context using parallel depthwise branches with learnable dilation"). Avoid "we improved accuracy" — name the *mechanism* that produced the improvement.

---

## Proposed Methodology — the heart of the paper

Spend the most effort here. Break it into the following subsections.

### A. Pipeline Overview
One end-to-end figure showing the full dataflow (input → preprocessing → augmentation → feature extraction → your novel module(s) → prediction), plus a paragraph walking through it.

- *How the sample did it:* Figure 1 shows the whole pipeline (pretraining data path, encoder backbone, fusion/head, external validation path), and a paragraph maps the forward pass compactly, even giving the tensor notation `{f1,...,f4} = E(I)`. Include a written walkthrough, not just the diagram.

### B. Proposed Architecture
A professional architecture diagram: high resolution, clear labels, consistent colour scheme, readable fonts, modular blocks.

- **Build it in** PowerPoint, draw.io, Figma, Illustrator, Visio, or Lucidchart.
- **Never** use blurry screenshots, low-res images, or hand-drawn figures.
- *How the sample did it:* Figures 1–4 are clean vector block diagrams with consistent colour coding per module and labelled tensor shapes. Match that polish.

### C. Custom Model Design *(this is where novelty lives)*
Do **not** just plug in an off-the-shelf backbone. Explain what you changed and why.

- Options: a new attention block, adaptive/multi-scale feature fusion, dynamic convolution, a lightweight decoder, a GNN or transformer module, residual enhancement, context aggregation, cross-scale attention.
- *How the sample did it:* it introduces three genuinely custom pieces — the Adaptive Spiral Block (parallel dilated depthwise branches with *learnable, interpolated* dilation), the Hierarchical Tumor Guidance module (produces both feature reweighting and a spatial guidance map), and an uncertainty-aware dual head. Each is explained on its own with its purpose. You need at least one clearly novel component explained at this depth.

### D. Mathematical Formulation
Formalise your key components with equations. Number every equation and refer to it in the text.

- Candidates: loss function, attention formulation, feature-fusion operation, pooling, normalisation, similarity/contrastive terms.
- *How the sample did it:* it formalises the forward pass (Eq. 1), the spiral branch convolutions (Eq. 4), the level-weight softmax (Eq. 5), cross-scale projection (Eq. 6), dual-pooling descriptor (Eq. 7), SE gating (Eq. 9), GeM pooling (Eq. 10), and the full label-smoothed focal + uncertainty + guidance loss (Eqs. 11–13). You don't need this many, but your *novel* components must be written mathematically.

---

## Experimental Reporting

### Dataset Description
Give complete information: name, source, download link, sample count, class count, image size, class distribution, train/val/test split, preprocessing, and augmentation.

- *How the sample did it:* it fully specifies BraTS 2020 (369 subjects, 4 modalities, 90/10 split), ND-5 (17,888 images, 4 classes, 70:30 train split with reserved test), plus two external sets (BRISC 2025 with 6,000 images, Mendeley with 12,064), each with a working link in the Data Availability section. **Provide real, working dataset links.**

### GitHub Repository / Code Availability
State the repo link, language, framework, Python and PyTorch/TensorFlow versions, and key libraries. If code is not public, write exactly: *"Source code is not publicly available."*

- *How the sample did it:* it links a public GitHub repo in the Data Availability section.

### Experimental Setup
Report the hardware and every training hyperparameter: GPU, CPU, RAM, OS, Python version, framework, optimizer, batch size, learning rate, epochs, scheduler, and loss.

- *How the sample did it:* it names the P100 GPU and gives a full hyperparameter table (Table 2): AdamW, weight decay 1e-4, cosine scheduler, per-stage epochs, batch sizes, learning rates, and loss configuration. **Put your hyperparameters in a table.**

### Evaluation Metrics
Choose metrics that fit your task and justify them briefly.

- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, Cohen's Kappa, Sensitivity, Specificity.
- Segmentation: Dice, IoU.
- *How the sample did it:* macro-averaged Precision/Recall/F1, macro AUC (one-vs-rest), plus per-class tables and confusion matrices for every evaluated dataset.

### Comparison with Existing Studies
A comparison table against recent papers. Suggested columns: Method, Dataset, Accuracy, F1, Parameters, Year.

- *How the sample did it:* Table 5 compares four recent baselines against the proposed model on both accuracy *and* parameter count, then discusses the accuracy–efficiency trade-off in prose. Reporting parameters alongside accuracy is what makes the comparison credible — do the same if efficiency is part of your claim.

### Ablation Study *(mandatory)*
Show the contribution of each component by adding them one at a time and reporting the metric at each step (baseline → +module A → +module B → … → full model).

- *How the sample did it:* the paper frames its whole design around isolable components and stable evaluation, and explicitly argues (via the related-work discussion of ablation) that each part must justify itself. Present your ablation as a table with a monotone build-up and comment on which module gave the biggest lift.

### Explainability Analysis *(if applicable)*
If your task supports it, include Grad-CAM / Grad-CAM++, saliency maps, attention maps, SHAP, or LIME, and explain why the model attends to meaningful regions.

- *How the sample did it:* Figure 8 shows Grad-CAM++, vanilla saliency, and guided saliency side by side across tumour-positive and tumour-negative samples, arguing that cross-method agreement on anatomically plausible regions supports trust. If explainability is a stated contribution, this section is not optional for you.

---

## Discussion and Closing Sections

### Results and Discussion
Explain *why* your model performs better, its advantages and weaknesses, computational cost/efficiency, and practical or clinical/industrial significance. Interpret your numbers — don't just restate the tables.

### Limitations
Every strong paper states its own limitations honestly (e.g. dataset size, 2D-only inputs, domain shift, no multimodal signal, scalability).

- *How the sample did it:* it concedes its 2D single-slice limitation and lack of full 3D context, then ties that directly to planned future work.

### Future Work
Short, concrete next steps (larger datasets, 3D/2.5D inputs, transformers, GNNs, self-supervised or federated learning, real-time/edge deployment, calibration).

### Conclusion
Summarise the problem, method, key findings, contributions, and future directions in one tight paragraph. No new claims.

---

## References
- **IEEE style.**
- **40–60 references**, the majority from **2023–2026**.
- Prefer Q1/Q2 journals; recent conference papers are fine; minimise website citations.
- *How the sample did it:* ~49 numbered IEEE references, heavily weighted to 2023–2025 journal and arXiv sources.

---

## Figures and Tables — Presentation Rules

**Figures** must be high-resolution, professionally designed, clearly labelled, readable, and stylistically consistent. No screenshots, no pixelated images, no low-quality internet images.

**Tables** — include professional tables for: literature comparison, dataset statistics, experimental setup, hyperparameters, ablation, final comparison, and computational complexity.

Number every figure and table and reference each one in the body text.

---

## Writing Quality
Logical organisation, scientific tone, technical depth, originality, critical thinking, correct grammar, consistent formatting. Always paraphrase prior work in your own words and cite the source. No plagiarism.

---

## Submission Checklist

- [ ] Professional journal-style formatting (IEEE two-column template)
- [ ] Strong research motivation
- [ ] Comprehensive, comparative literature review
- [ ] Critical gaps and limitations section (mandatory)
- [ ] Clearly stated core contributions (bulleted)
- [ ] High-quality pipeline overview figure
- [ ] Professional architecture diagram
- [ ] Customised (not off-the-shelf) architecture with an explained novel component
- [ ] Mathematical formulation of key components
- [ ] Full dataset details and working availability links
- [ ] GitHub repository link or explicit code-availability statement
- [ ] Complete experimental setup (hardware + hyperparameters)
- [ ] Appropriate, justified evaluation metrics
- [ ] Comparison table against recent literature
- [ ] Ablation study (mandatory)
- [ ] Explainability analysis (if applicable)
- [ ] Results and discussion with interpretation
- [ ] Honest limitations section
- [ ] Concrete future work
- [ ] Conclusion
- [ ] 40–60 IEEE references, mostly 2023–2026
- [ ] High-quality numbered figures and tables, all referenced in text
- [ ] Original writing, verified and refined (no unedited AI text)

---

## Submission Instructions
- Submit as **PDF**.
- Use a standard **IEEE two-column** journal/conference template unless told otherwise.
- Number and reference every figure, table, and equation in the text.
- Include page numbers and keep formatting consistent throughout.
- The work must be your own original work. Plagiarism or unverified AI-generated content is academic misconduct.

**Deadline: 21 July 2026 (11:59 PM).** Late submissions may incur penalties per course policy.

---

## Final Note
The point of this assignment is to practise producing a *publishable* manuscript. Study the Hi-TGNet paper for structure, technical depth, figure quality, and writing style — then apply that same standard to your own problem. Prioritise originality, critical analysis, methodological rigour, and clean presentation throughout.
