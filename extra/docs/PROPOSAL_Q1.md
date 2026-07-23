# Q1 Proposal — Lesion-Resolved Sleep-Microstructure Disruption in Ischemic Stroke

> Reframed from "build a better stroke sleep stager" (incremental) to "use automated
> sleep analysis as an instrument to map how focal injury reorganizes sleep" (a clinical-
> neuroscience finding). Grounded in a full structured literature review: see
> **[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)** (themes, citations, critical synthesis).

> **Two facts the review corrected/sharpened (June 2026):**
> 1. The published iSLEEPS LSTM **74.70%** is a *legitimate subject-independent* benchmark
>    (patient-exclusive 10-fold; macro-F1 67.68, kappa 0.64) — NOT a leaky split. So our
>    pilot gap is a **data-quantity** effect (39 vs 100 subjects), now addressed with the full
>    cohort; we adopt the same patient-exclusive protocol for comparability.
> 2. **85% of iSLEEPS subjects are apneic** (Mild 24% / Mod 23% / Severe 38%). Sleep-disordered
>    breathing fragments architecture and is the **dominant confound** — AHI must be a covariate
>    in every analysis. The mechanism backbone (TRN/thalamocortical spindle generation; slow-
>    wave-driven plasticity) gives the lesion hypotheses a principled, non-incremental basis.

## 1. Crowdedness map (evidence-based)

| # | Sub-area | Verdict | Key evidence |
|---|---|---|---|
| 1 | Deep-learning sleep staging (healthy) | **CROWDED / saturated** | SOTA ~85%+, multiple 2024-25 reviews, "approaching a practical ceiling"; dozens of CNN/LSTM/Transformer variants. |
| 2 | Automatic staging in **stroke** specifically | **EMERGING (sparse)** | Almost only the iSLEEPS descriptor (Maiti 2026); general clinical staging exists, stroke-focused work does not. |
| 3 | Healthy to pathological **domain gap / transfer** in staging | **EMERGING** | Transfer with channel+population mismatch actively studied (Frontiers Physiol 2023), but "performance improvement is limited"; not stroke-specific, never framed as a biomarker. |
| 4 | Post-stroke sleep-EEG **microstructure** (spindles / slow waves / asymmetry) | **ESTABLISHED but small-N and CONTESTED** | Disturbed NREM-oscillation laterality post-stroke (Sleep 2023, PMC10187327); ipsilesional spindle-amplitude reduction; slow waves promote recovery (J Neurosci 2020). BUT contradictions: "unilateral thalamic stroke does NOT decrease ipsilateral spindles" (PubMed 10811377); "only left-sided/posterior spindle reduction, spindles only mildly laterally affected" (Sci Rep 2018). N=12-23, HD-EEG, manual detection. |
| 5 | **Lesion-symptom mapping** linked to sleep/EEG | **OPEN (literature-stated gap)** | Thalamic-spindle study explicitly: "Future studies should prioritize lesion-symptom mapping to clarify differential effects of infarct topography on sleep parameters." Done only at tiny N, manual methods. |
| 6 | EEG **sleep biomarkers of severity/outcome** (NIHSS, mRS) | **EMERGING** | SWA(NREM/wake) ratio correlates with NIHSS (r=-0.47); spindle density predicts motor recovery; qEEG brain-symmetry index prognostic; Denis 2024 (J Sleep Res). All classical metrics, not ML-derived, not lesion-resolved. |
| 7 | **Model-as-instrument** (classifier error / representational shift as a biomarker) | **OPEN** | Interpretable staging exists (SleepTransformer, WaveSleepNet) but for trust/clinical adoption, NOT to quantify pathology. No one uses a stager's error/uncertainty as a stroke biomarker. |
| 8 | **iSLEEPS-specific** downstream work | **OPEN (brand new)** | Only the 2026 descriptor + CNN/LSTM/Transformer baselines; no downstream studies. |

**Read-out:** the *crowded* part is exactly what we were optimizing all week (rows 1-3, "which model scores higher"). The *open* part is rows 4-8: linking sleep disruption to lesion location, severity, and outcome at scale, with the model as instrument.

## 2. The open, Q1-defensible niche

The genuinely open space is the **intersection of rows 4, 5, 6, 7, 8**:

> **A large-cohort, lesion-resolved characterization of how ischemic stroke reorganizes
> sleep microstructure — adjudicating the contested ipsilesional-asymmetry question and
> linking automated sleep-EEG signatures to infarct topography and stroke severity — on
> the largest stroke PSG cohort (iSLEEPS, N=100), with automated staging + spindle/slow-
> wave quantification as the enabling instrument (not the contribution).**

Why it survives a "so what / this is crowded" review:
- The neuroscience findings (row 4) are real but from **N=12-23 HD-EEG studies with directly contradictory results** -> a 100-patient cohort can **adjudicate** them.
- The literature **explicitly asks** for lesion-symptom mapping of sleep (row 5).
- Biomarker work (row 6) uses classical metrics, never lesion-resolved at scale, never via a model-instrument.
- iSLEEPS (row 8) is new and uniquely enables it (lesion location + NIHSS/mRS + full PSG, N=100).
- Sleep staging itself (row 1, crowded) is **explicitly demoted to the instrument**, not the claim.

## 3. The proposal

**Working title:** *Lesion-Resolved Disruption of Sleep Microstructure in Ischemic Stroke:
A Large-Cohort EEG Study Linking Sleep Architecture, Spindles and Slow Waves to Infarct
Topography and Severity.*

**Central thesis:** Focal ischemic injury reorganizes sleep microstructure in a spatially
specific, severity-graded manner. Using a validated automated pipeline as an instrument,
we map this at unprecedented scale and test whether disruption (a) is lateralized
(ipsilesional), (b) depends on lesion topography, and (c) scales with NIHSS / tracks
functional status.

### Research questions & hypotheses (fundamental, grounded)

- **RQ1 (lateralization — adjudicate a contested result).** Within-patient, is sleep-
  microstructure disruption (spindle density/power, slow-wave activity, stage-specific
  metrics) greater on the **ipsilesional** vs contralesional derivation? Does it depend on
  hemisphere/territory?  *H1: ipsilesional reduction in spindle power and SWA, modulated by
  territory — resolving PMC10187327 vs PubMed 10811377 at scale.*
- **RQ2 (topography — fill the stated gap).** Does lesion **location/territory** (cortical
  vs subcortical/thalamic; vascular territory) predict **which** sleep features and stages
  are most disrupted?  *H2: subcortical/thalamic -> spindle and N2/N3 loss; cortical -> slow-
  wave/SWA changes.*
- **RQ3 (severity / biomarker).** Do sleep-microstructure signatures — and the automated
  stager's per-patient error/uncertainty (the "model-surprise" marker) — scale with
  **NIHSS** and relate to **mRS/Barthel**?  *H3: greater disruption and greater model-surprise
  ↔ higher NIHSS / worse functional status.*
- **RQ4 (the instrument — demoted to feasibility).** Can a subject-independent automated
  pipeline reliably quantify these signatures under altered stroke physiology (handling
  imbalance, label noise)?  Pilot already supports this (validated vs published baselines).

### Methodology (what makes it Q1, not incremental)

- **Cohort:** iSLEEPS N=100 (largest stroke PSG cohort) — already downloaded.
- **Instrument:** validated subject-independent stager (our pilot) **plus automated spindle
  and slow-wave detection** (established detectors, e.g. A7 / YASA-style) to quantify
  *microstructure*, not just stage labels.
- **Strong design — within-subject ipsilesional vs contralesional contrast** (controls for
  inter-subject variability; the single biggest methodological strength).
- **Lesion-resolved stratification:** hemisphere/side (~60% populated), vascular territory
  (~56%), cortical vs subcortical.
- **Clinical correlation:** NIHSS (99%), mRS/Barthel (99%) — *verify admission vs follow-up
  timing before any outcome claim*.
- **Controls & rigor (non-negotiable for Q1):**
  - Healthy reference (Sleep-EDF) for the normal asymmetry baseline, **montage-harmonized**
    and confound-controlled (the Fpz-Cz vs C4:M1 issue from the pilot).
  - **Mixed-effects models** (patient as random effect), effect sizes + CIs, multiple-
    comparison correction — not bare p-values.
  - **AHI / sleep-disordered breathing as a covariate** (SDB confounds sleep architecture;
    must adjust — 100% populated).
  - Honest missingness handling + sensitivity analyses.
- **Interpretability/attribution** (Grad-CAM / attention) to tie model decisions back to
  spindle/slow-wave structure, linking statistics to mechanism.

### What clears Q1 vs reads incremental
- Headline is a **clinical-neuroscience finding** (lesion -> sleep map; a candidate
  biomarker), never an accuracy number.
- **Adjudicates a contested question** (lateralization) at 4-8x the sample of prior work.
- **Fills a literature-stated gap** (lesion-symptom mapping of sleep).
- **Rigorous within-subject design + mixed-effects stats + confound control.**
- **Honest limitations** stated up front (below).

### Target Q1 journals (matched to framing)
- Clinical / translational (best fit for the biomarker + neuroscience framing):
  **Sleep**, **Journal of Sleep Research**, **NeuroImage: Clinical**, (**Stroke** if very clinical).
- Engineering / signal-processing (methods + biomarker hybrid):
  **IEEE J-BHI**, **Computers in Biology and Medicine**, **Biomedical Signal Processing and Control**.

### Honest ceiling / risks (state these in the paper)
- Lesion-location granularity is moderate (territory ~56%, fine region ~22%) -> likely
  resolve **hemisphere + broad territory only**; fine topography is exploratory.
- **Cross-sectional, single-night, single-scorer, N=100** -> associations not causation;
  "biomarker" = cross-sectional association, not a validated longitudinal predictor.
- Effect sizes will be modest -> frame as a **rigorous, scaled, hypothesis-generating**
  characterization, the largest of its kind.
- Verify mRS/Barthel timing (admission vs discharge/follow-up) before any outcome claim.

## 4. Key references (from the survey)
- Maiti et al. iSLEEPS dataset. Nature Scientific Data (2026).
- Disturbed laterality of NREM sleep oscillations post-stroke (pilot). PMC10187327 (2023).
- Unilateral thalamic stroke does not decrease ipsilateral spindles. PubMed 10811377.
- Individual spindle detection in thalamic stroke. Sci Rep (2018), PMC6294746.
- Slow waves promote sleep-dependent plasticity and recovery after stroke. J Neurosci (2020), PMC7643301.
- Sleep health parameters and functional recovery after stroke. Denis et al., J Sleep Res (2024).
- Sleep-EEG changes in acute hemispheric stroke correlate with volume/outcome. PubMed 11311681.
