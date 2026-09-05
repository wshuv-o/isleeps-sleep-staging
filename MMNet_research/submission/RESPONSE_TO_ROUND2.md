# Response to Reviewer — Second Round

**Manuscript:** A Physiologically Interpretable Multimodal Model for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic Stroke
**Manuscript ID:** MMNet-2026-0713.R2
**Round:** Response to Minor Revision (previous: Overall 8/10)

Page references are to the revised manuscript, now 13 pages. Every number below is produced by a live cell in a committed notebook; none is copied from a cached file.

---

## Preface

We thank the reviewer for a second reading that was as specific as the first. Four of the five remaining findings are addressed with new experiments rather than with text; the fifth we have declined, with our reasoning given.

We also ran three experiments the reviewer did not ask for, because the second report's standard of evidence made their absence conspicuous. **Two of the three produced results that weaken claims we had been making**, and we report them as they came.

| # | Finding | Disposition | Where |
|---|---|---|---|
| 1 | Title undersells the contribution | **Declined**, with reasoning | — |
| 2 | t-SNE does not earn its place | Accepted — replaced with permutation importance | §VI-K, p.9 |
| 3 | Abstract more optimistic than limitations | Accepted in full | Abstract p.1; §VI-H p.7; §VII-A p.12 |
| 4 | AHI association thinly interpreted | Accepted in full | §VI-I, p.9 |
| 5 | Released per-fold cache disagrees with the run | Accepted in full | `results/revision/runs/` |
| — | *(unprompted)* seed stability | Added | §VI-A, p.5 |
| — | *(unprompted)* calibration and conformal abstention | Added | §VI-M, p.10 |
| — | *(unprompted)* data-efficiency learning curve | Added | §VI-L, p.10, Fig. 10 |

---

## Finding 1 — The title still undersells the contribution

**Declined.** The reviewer suggests a title carrying the finding that healthy-built architectures lose 15–20 points on this cohort, and marks it explicitly as a suggestion rather than a condition.

We agree that is the most transferable result. We have kept the current title because it names what the paper *is* — a joint staging and respiratory model for this population — whereas a title asserting the comparative finding would foreground a result established on re-implementations of other groups' architectures rather than on their released weights, which is a caveat we state in the text and would not want to compress into a title. We would revisit this at the editor's direction.

---

## Finding 2 — The t-SNE does not earn its place

**Accepted.** The reviewer asked for something falsifiable, and specifically suggested permutation importance restricted to the respiratory head, to test whether the channels the ablation implicates are the ones the model actually weights.

We ran it (**§VI-K, p.9**). Each modality's features are permuted across epochs at inference — the channel remains present but is no longer aligned to the epoch it describes — averaged over three shuffles and all ten fold models.

| Modality | Δ staging acc | Δ respiratory AUC |
|---|---|---|
| EEG | **0.225** | 0.038 |
| EOG | 0.045 | 0.016 |
| EMG | 0.007 | 0.006 |
| SpO2 | **−0.001** | **0.031** |
| pulse/HRV | −0.002 | 0.013 |
| ECG | −0.001 | 0.002 |
| airflow | −0.001 | 0.006 |
| effort | −0.004 | −0.001 |

The dissociation is clean: EEG carries staging and barely touches breathing; SpO2 carries breathing and costs staging nothing.

**The agreement test is the point.** Across the seven modalities common to both analyses, the rank correlation between permutation drop and the retraining-ablation drop is **ρ = 0.893 (p = 0.007)** for respiratory AUC. Two methods that could have disagreed — one applied to retrained models, one to fixed ones — order the modalities identically. That is the falsifiable evidence the reviewer asked for.

The t-SNE is retained but demoted: the attribution argument now rests on the ablation and permutation results, not on the embedding projection.

---

## Finding 3 — The abstract is more optimistic than the limitations

**Accepted in full, and the reviewer's instinct was better founded than we realised.**

The abstract now states the respiratory read-out is *"an epoch-level, screening-grade read-out rather than event-by-event scoring"* (**p.1**).

We then went further and measured what event-level scoring would actually give (**§VI-H, p.7**). The result is worse than we would have guessed:

| Threshold | Event sensitivity | False alarms/hour | % epochs flagged |
|---|---|---|---|
| 0.20 | 0.857 | 2.22 | 67.8 |
| 0.50 | 0.549 | 2.61 | 33.1 |
| 0.70 | 0.270 | 1.41 | 13.0 |

And the model recovers the wrong *number* of events: a predicted **4.5 events/hour against a scored 11.1**, with a per-patient rank correlation of only **ρ = 0.142 (p = 0.17, n = 96)** — not significant, and far weaker than the ρ = 0.315 obtained from the aggregated epoch probability. Consecutive events separated by a few epochs merge into one flagged run, so an epoch-level detector systematically under-counts.

This is now stated in the results and in the limitations. It explains why we report a burden rather than an index, and it converts "event-level scoring is future work" from a formality into a quantified gap.

---

## Finding 4 — The AHI association is honestly reported but thinly interpreted

**Accepted in full.** §VI-I (p.9) now gives three mechanisms that cap the correlation, each a property of the formulation rather than of the model: mean per-epoch probability compresses the event-count variation the index is built from; the clinical index is normalised by total sleep time whereas our burden averages over all scored epochs including wake, and the denominators diverge most in the patients with the most fragmented sleep; and single-night AHI carries substantial night-to-night variability, which caps any correlation against it.

The event-level analysis above independently supports the first mechanism.

---

## Finding 5 — The released per-fold cache does not match the reported run

**Accepted in full — and the reviewer was right that it misrepresented our own reproducibility.**

We retrained the headline configuration and regenerated the cache from that run. The reproduction matches the published values **to four decimal places on all five metrics**:

| Metric | This run | Published | Difference |
|---|---|---|---|
| staging accuracy | 0.7227 ± 0.039 | 0.7227 | −0.0000 |
| macro-F1 | 0.6510 ± 0.038 | 0.6510 | −0.0000 |
| Cohen's κ | 0.6106 ± 0.053 | 0.6106 | −0.0000 |
| respiratory AUC | 0.7111 ± 0.034 | 0.7111 | 0.0000 |
| respiratory AP | 0.3367 ± 0.083 | 0.3367 | −0.0000 |

`results/revision/runs/headline_concat_per_fold.csv` now derives from this run.

---

## Unprompted additions

### Seed stability (§VI-A, p.5)

A single-seed headline invites the question of whether the result is a property of the model or of the initialisation. We retrained under five seeds on the same folds. Staging accuracy varies by **0.003 across seeds against 0.032 across folds**; respiratory AUC by **0.004 against 0.035**. Uncertainty is dominated by fold assignment — a property of a 96-patient cohort — not by the optimiser's starting point.

**We also state a fact that does not flatter us:** seed 42, which we report throughout, has the *highest* respiratory AUC of the five. The seed mean is 0.705. The manuscript now tells the reader to treat 0.711 as the optimistic end of a 0.701–0.711 interval.

### Calibration and abstention (§VI-M, p.10)

**The model is overconfident.** Expected calibration error is 0.076 and the gap runs one way in every bin — epochs scored above 0.9 are correct 88% of the time. Stated confidences are ordered, not probabilities, and we now say so.

Ranking is nonetheless sound, which is what abstention needs. Split-conformal prediction, calibrated on each fold's validation patients, gives empirical coverage tracking the target (0.894 at a 0.90 target). At α = 0.10 the model returns a single stage for 45.8% of epochs and is right on **86.9%** of those, against **59.1%** elsewhere — the abstention selects genuinely hard epochs rather than hedging. It concentrates where a human scorer would expect: N1 receives a singleton set only 19% of the time, the lowest of any stage.

### Data efficiency (§VI-L, p.10, Fig. 10)

We had been describing this cohort as data-limited. **The learning curve only partly supports that, and we have narrowed the claim accordingly.**

| Training patients | Staging accuracy | Respiratory AUC |
|---|---|---|
| 19 | 0.6929 | 0.6556 |
| 38 | 0.7135 | 0.6852 |
| 57 | 0.7213 | 0.6957 |
| 76 | 0.7227 | 0.7111 |

Staging's marginal gains are +0.021, +0.008, **+0.001** — the final quartile buys 4% of one fold standard deviation, which is flat within noise. Respiratory AUC's final step is +0.015, close to half its fold standard deviation, and still climbing.

**Staging has reached this cohort's ceiling; respiratory detection has not.** More patients of this kind would improve breathing detection and would not improve staging, so the staging ceiling is a property of the labels and the injured brain rather than of training-set size. This is a narrower and better-supported claim than the one we made before.

---

## Closing

Of the three experiments we added unprompted, one strengthened the paper (seed stability), one produced a mixed result we had to write against ourselves (calibration: the model is overconfident), and one required us to retract a framing we had used since the first submission (data efficiency: staging is not data-limited). The event-level analysis requested under Finding 3 produced the weakest number in the paper.

We think the manuscript is more useful for containing them.
