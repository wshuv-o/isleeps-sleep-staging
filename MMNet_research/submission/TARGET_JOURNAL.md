# Target Journal Declaration

**Target:** *IEEE Journal of Biomedical and Health Informatics* (JBHI)
**Second choice:** *IEEE Transactions on Neural Systems and Rehabilitation Engineering* (TNSRE)
**Manuscript format:** IEEE two-column journal format (`\documentclass[journal]{IEEEtran}`), which is the format JBHI requires for submission.

---

## Why JBHI fits this paper

**Scope match.** JBHI publishes informatics and machine-learning methods applied to biomedical and health data, with an explicit emphasis on work that connects computational method to clinical use. This manuscript is exactly that: a multimodal model that reads the full clinical polysomnogram and produces two clinically actionable outputs — the sleep stage of every 30-second epoch and a per-epoch respiratory-event label — evaluated on a patient cohort rather than a healthy convenience sample.

**Clinical population, not a benchmark dataset.** The cohort is iSLEEPS, the first public polysomnography corpus of subacute ischemic stroke (99 patients). The paper's central finding is clinical rather than architectural: deep architectures developed on healthy sleep lose 15–20 accuracy points on this population, while a compact, physiologically-grounded model does not. JBHI's readership — clinical informatics researchers and biomedical engineers — is the audience for whom that finding changes practice. A general machine-learning venue would treat it as a negative result; JBHI treats it as a deployment constraint.

**Multi-signal physiological data is core JBHI territory.** The model consumes EEG, EOG, EMG, ECG, airflow, thoracic and abdominal effort, and SpO2. JBHI regularly publishes multimodal physiological-signal work, including sleep staging and sleep-disordered-breathing detection, so both the signal modality and the task have established precedent in the journal.

**Clinical validation is expected, not optional.** The manuscript reports association with clinically scored AHI (Spearman ρ = 0.315, p = 0.0017, n = 96), staging accuracy stratified by AASM severity band, and per-event-type detection (hypopnea 0.692, obstructive 0.763, central 0.840). These clinical-facing analyses are the kind JBHI reviewers ask for and many methods venues consider out of scope.

**Interpretability framed as physiological attribution.** The modality-ablation grid demonstrates a clean dissociation — cardiorespiratory channels drive the respiratory head (AUC 0.711 → 0.673 on removal, p = 0.004) while contributing nothing to staging (p = 0.91). JBHI's clinical readership requires that a model's behaviour be explainable in physiological terms rather than by saliency heatmaps alone.

---

## Why not the alternatives

**TNSRE** (second choice) is a strong fit for the neural-signal processing, but its centre of gravity is neural engineering, rehabilitation and neural interfaces. The respiratory and cardiovascular half of this paper sits outside that emphasis, and the joint staging-plus-breathing formulation is the contribution we most want reviewed.

**Elsevier's *Computers in Biology and Medicine*** is a reasonable scope match and was considered. We prefer JBHI because its reviewer pool is more specialised in physiological time-series and sleep medicine, and because the stroke-cohort framing is more likely to be evaluated on clinical merit there.

**General ML venues** (NeurIPS/ICLR-style) are not appropriate. The architecture is a considered recombination of established components rather than a novel learning algorithm, and the paper's value is the clinical finding and the cohort — which such venues systematically undervalue.

---

## Format compliance

The manuscript is prepared in IEEEtran two-column journal style, matching JBHI's author requirements: structured abstract, IEEE keyword block, IEEE numeric citation style, two-column floats, and the corresponding-author and declaration blocks (data availability, code availability, ethics, funding) required at submission.
