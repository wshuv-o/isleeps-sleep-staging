# Referee Report — Second Round

**Manuscript:** A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke
**Manuscript ID:** MMNet-2026-0713.R1
**Reviewing model:** Claude Opus 5
**Date of report:** 5 September 2026
**Previous recommendation:** Major Revision (Overall 4/10)

---

## Recommendation: **MINOR REVISION**

I re-read the manuscript in full against my first report and against the marked-up copy the authors supplied. I checked each of my twelve findings for whether the change claimed in the response letter is actually present in the paper; I re-derived the headline numbers from the released prediction artifact; I checked the new ablation grid for completeness; I checked the new baselines, significance tests and clinical analyses against the tables that report them; and I re-read the expanded reference list for currency and for whether each citation supports the sentence it is attached to.

**The revision is substantial and honest.** The manuscript has gone from 5 pages to 12, Results from 2 subsections to 10, tables from 3 to 9, references from 11 to 49, and the ablation from a single condition to a complete nine-condition leave-one-out grid. Every one of my twelve findings has been addressed, and — more to the point — several were addressed in the direction that *weakens* the paper's claims rather than strengthening them. The authors now report that their respiratory margin over gradient boosting is modest, that their AHI association is weak, and that cardiorespiratory channels do not improve staging at all. That is the behaviour of authors reporting what they found rather than what they wanted.

Two responses deserve specific credit. On Finding 1, the authors did not simply delete the claim I objected to; they ran the ablation I asked for (removing airflow *and* SpO2 together), reported that detection falls to 0.673 without collapsing, and restated the claim as graded dependence rather than non-circularity. Their disagreement with my framing is technically reasoned and I accept it. On Finding 3, they added the baselines and then followed the evidence to a reframing of their own contribution, which is the right response to an uncomfortable result.

I am also satisfied by something I did not ask for: the new Figure 7 reports both the pooled AUC (0.706) and the cross-validated per-fold mean (0.711 ± 0.034), and explains why the two estimators differ. Most authors would have quietly reported whichever was higher.

The remaining concerns are matters of degree and framing, not correctness. None requires new experiments.

---

## Overall Assessment

| Area | Score /10 | Change | Main concern |
|---|---|---|---|
| Title | 7 | — | Accurate; still does not signal the paper's most interesting finding |
| Novelty | 7 | +1 | Architecture remains a sound recombination; the novelty is the cohort and the negative result, now better framed |
| Methodology | 8 | +2 | Fusion is now explicit and reimplementable; protocol correct throughout |
| Dataset rigor | 8 | +3 | Per-experiment *n* stated, duplicate disclosed and handled, channel table complete |
| Experimental validity | 8 | +5 | Baselines, fold-wise SD, paired tests, and operating characteristics all present |
| Ablation | 9 | +7 | Complete nine-condition grid; no cell pending; both metrics per row |
| Explainability | 7 | +3 | Ablation-as-attribution is genuinely falsifiable; the t-SNE still adds little |
| Clinical claims | 7 | +4 | Claims now match the evidence; the evidence itself remains modest |
| References | 8 | +5 | 49 entries, 55% from 2023 onward, attributions checked |
| Writing | 8 | +1 | Clear; the new sections integrate well |
| **Overall** | **8** | **+4** | Publishable after minor revision |

---

## Verification of the previous findings

| # | Previous finding | Status |
|---|---|---|
| 1 | Non-circularity untested; null reported as increase | **Resolved** — full cardiorespiratory ablation added (0.673); "marginally higher" removed; AASM definition stated §IV |
| 2 | Ablation was one condition | **Resolved** — nine-condition grid, Table VI |
| 3 | No respiratory baseline | **Resolved** — desaturation rule 0.596, logistic 0.582, boosting 0.670, Table V |
| 4 | No statistical testing | **Resolved** — Wilcoxon p = 0.004 / p = 0.91; fold SD throughout |
| 5 | No clinical validation | **Resolved** — AHI ρ = 0.315 (p = 0.0017, n = 96), severity bands, per-event-type |
| 6 | Cohort inconsistent; no audit | **Resolved** — N = 99 / 96 / 96 stated; SN15–SN28 duplicate disclosed and fold-locked |
| 7 | Fusion too vague | **Resolved** — concatenation named as headline; attention reported as ablation |
| 8 | Attribution asserted | **Resolved** — stated as a falsifiable prediction and tested |
| 9 | References too few | **Resolved** — 11 → 49; comparison table added |
| 10 | No limitations | **Resolved** — §VII-A and §VII-B |
| 11 | No critical-gaps section | **Resolved** — §III |
| 12 | Declarations absent | **Resolved** — all five present |

---

## Remaining Findings

### Finding 1 — The title still undersells the paper's most defensible contribution

**Location:** Title; Abstract.

The paper's most interesting and most transferable result is not that a multimodal model works — it is that architectures developed on healthy sleep lose 15–20 accuracy points on this cohort while a physiologically grounded feature model does not. That finding is what a reader in clinical informatics will cite. The current title describes the artifact rather than the finding, and "Physiologically Interpretable" is the weakest of its claims.

**Required action:** consider a title that carries the finding. This is a suggestion, not a condition of acceptance.

---

### Finding 2 — The t-SNE panel still does not earn its place

**Location:** §VII; Fig. 9.

The authors accepted my previous point and promoted the ablation grid to the interpretability argument, which was the right move. The t-SNE was retained with a more honest caption, but it still shows only that the representation separates sleep stages — which any accurate stager would show — and that respiratory events are diffuse. Neither statement is falsifiable.

**Required action:** move Fig. 9 to supplementary material, or replace it with something that could have come out otherwise — for example a per-feature permutation importance restricted to the respiratory head, which would test whether the channels the ablation implicates are the ones the model actually weights.

---

### Finding 3 — Epoch-level scoring remains the ceiling on the respiratory claim, and the operating points show it

**Location:** §VI-H (p.7); Fig. 7(b) (p.8); §VII-A (p.10).

The new precision-recall panel is welcome and is correctly described. It also makes plain something the AUC concealed: at 0.80 sensitivity the model holds 0.22 precision, meaning roughly four false positives per true event. The authors acknowledge this as a screening-grade signal and list event-level scoring as future work, which is the honest position.

My concern is that the abstract still leads with the AUC. A reader who stops at the abstract will form a more favourable impression of the respiratory capability than the precision-recall panel supports.

**Required action:** add one clause to the abstract noting that detection is epoch-level and screening-grade. The limitations section says this; the abstract should not be more optimistic than the limitations.

---

### Finding 4 — The AHI association is now honestly reported but thinly interpreted

**Location:** §VI-I (p.9); Fig. 8 (p.9).

ρ = 0.315 over 96 patients is real (p = 0.0017) but explains under 10% of variance, and the authors say so. What is missing is an interpretation of *why* it is weak. Candidate explanations — that mean per-epoch probability is a poor estimator of an hourly event rate, that epoch-level labels cannot recover event counts, that AHI itself has substantial night-to-night variability — are each testable and each would tell the reader something.

**Required action:** add two or three sentences interpreting the weak association rather than only reporting it.

---

### Finding 5 — The released per-fold cache does not match the reported run

**Location:** Supplementary artifacts (`results/engine_cache_per_fold/`).

This is a reproducibility-package issue, not a manuscript error. The per-fold CSVs in the released cache give a respiratory AUC of 0.7066 and macro-F1 of 0.6459, whereas the reproduction notebook — which is the authoritative source and which I was able to follow — prints 0.7111 and 0.6510. The cache appears to be from an earlier run and was not regenerated.

A reader who loads the cache rather than running the notebook will conclude the paper's numbers do not reproduce, which would be an unfair impression of work that does in fact reproduce.

**Required action:** regenerate the per-fold cache from the reported run, or label it explicitly as a superseded earlier run in the results README.

---

## Summary for the Editor

This is a thorough and unusually honest revision. The authors ran every experiment I asked for, reported results that reduced their own claims, disagreed with me twice on clearly stated technical grounds — and were right both times. The empirical work now supports what the paper says.

My five remaining points are minor. Finding 5 is a packaging inconsistency that should be fixed because it misrepresents the authors' own reproducibility; Findings 1–4 are matters of framing and interpretation that the authors can address in a short revision without new experiments. I recommend acceptance after minor revision.
