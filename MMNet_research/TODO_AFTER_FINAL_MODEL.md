# What to update once the final model is fixed

Working checklist from the RTX 5080 session (7-8 Sep 2026). Everything on branch
`foundation-model-experiment`; `master` is untouched at `9db9d3d`.

Items are ordered by consequence, not by effort. **Blocking** items make a
manuscript claim false or unsupported if left undone.

Terminology: MM-Net is **under review (round-2 revision), not published**.
"Current manuscript" below means the configuration the submitted paper reports.

---

## 0. RESOLVED — the cardio branch, settled 8 Sep

The control ran. The gain is **not** a foundation-model result, and the
decomposition is now complete, three seeds x ten folds per rung, every step
significant after Holm:

| cardio representation | resp AUC | step | p |
|---|---|---|---|
| 14 engineered features | 0.6977 | — | — |
| random nonlinear expansion of those 14 | 0.7153 | +0.0176 | 0.0002 |
| random linear projection of raw signal | 0.7312 | +0.0159 | 0.0087 |
| learned CNN, 0.155 M, trained | 0.7816 | +0.0504 | <0.00001 |
| random deep transformer, 341 M, **untrained** | 0.7947 | +0.0131 | 0.0155 |
| pretrained MOMENT, 341 M | **0.8058** | +0.0110 | 0.0009 |

**No architectural claim.** A CNN designed for cardiorespiratory physiology and
trained end to end is *significantly worse* than the same architecture with
random weights. Do not write "we designed the encoder"; the data refuses it.

**What to claim instead**, supported on both branches:

1. *The engineered cardiorespiratory features are the bottleneck.* A trained CNN
   beats them by 0.0839, the pretrained encoder by 0.1081, both p < 0.00001.
2. *16% of that gap needs no new information at all* — a random nonlinear
   expansion of the same 14 features recovers it, so the manuscript's FeatMLP leaves
   nonlinear structure unextracted in features already computed. Cheapest fix
   available.
3. *On 99 patients a representation must be imported, not learned.* Three
   independent instances: fine-tuning CBraMod loses to freezing it; a trained CNN
   loses to an untrained deep network; frozen and random rich representations win
   on both branches.

Unresolved and still a judgement call: MOMENT alone matches MOMENT plus the
engineered features (0.8058 vs 0.8078), so the interpretable configuration and
the performant one are different models. Decide whether the paper leads with
0.71 AUC and validated SpO2 attribution, or 0.81 with a 1024-d embedding, or
presents both as a frontier.

## 1. Blocking — manuscript claims that are now false or unsupported

- [ ] **Rewrite C2.** As written it says healthy-sleep deep models fail here. A
      public checkpoint, frozen, reaches parity with the 188 features
      (TOST p = 0.008). Anyone can now run CBraMod and break the claim. Replace
      with the sharper and better-supported version: *training from scratch*
      fails on 99 patients; pretraining transfers; the two prior sources are
      complementary. `paper/multimodal_access.tex` lines ~180-184.
- [ ] **Update the headline numbers** if the final model changes: current
      manuscript 0.7275 acc / 0.6536 mF1 / 0.7070 AUC -> best configuration found
      0.7485 / 0.6620, and the cardio branch pending the control above.
- [ ] **Add C5** for representational complementarity. Two independently
      pretrained encoders (CBraMod, LaBraM) each add significantly to the
      engineered features; a third stream adds nothing. This is the novelty the
      round-2 referee said the architecture itself does not supply.
- [ ] **Strengthen the domain-gap paragraph** with the three-model benchmark:
      healthy 0.834 +- 0.028 -> stroke 0.313 +- 0.010, kappa 0.73-0.82 -> 0.09-0.11,
      all three models labelling 65-78% of stroke epochs Wake. Report BOTH bounds
      so the montage confound cannot be used against it: pathology alone ~0.19
      from in-domain models, montage shift plus pathology 0.56.
      Source: `MMNet_research/domaingap/`.

## 2. Blocking — referee round 2

- [x] **Finding 5** — per-fold cache mismatch. Fixed in `f0e36d8`; the reported
      run's per-fold CSV now ships in the submission bundle and the superseded
      cache is labelled.
- [ ] **Finding 2** — replace the t-SNE. The figure exists
      (`fig_attribution_agreement.pdf`, two independent attributions agreeing at
      rho = 0.810, p = 0.0149). Still needs the swap at
      `paper/multimodal_access.tex:994` and a rewritten caption.
- [x] **Finding 3** — abstract now states the read-out is epoch-level and
      screening-grade.
- [x] **Finding 4** — AHI weakness interpreted with three mechanisms.
- [ ] **Finding 1** — title. Author decision; referee called it a suggestion,
      not a condition.

## 3. Figures

- [ ] Swap `figures/fig_architecture.pdf` (line 354) for **v4**. Editable source
      is `figures/mm_architecture_v4.drawio`; export PDF from draw.io, or use the
      matplotlib `fig_architecture_v4.pdf` the builder also emits.
      **Do not use v3** -- it draws a model we never ran (two interchangeable SSL
      encoders, engineered cardio features, a dashed BiLSTM annotated with a
      trade-off measured on a superseded config, and an EEG encoder parameter
      count that was only the first Linear). v4's counts are asserted against the
      live modules at build time.
- [ ] Swap Fig. 9 for `fig_attribution_agreement.pdf` (see Finding 2).
- [ ] Regenerate any figure whose underlying run changed.

## 4. Repository hygiene

- [ ] **`requirements.txt` is missing three packages now required**: `pyedflib`
      (build_multimodal), `momentfm` (cardio embeddings; install with
      `--no-deps`, its pinned huggingface_hub will not build on 3.12), and
      `braindecode` (LaBraM). Pin `scipy==1.15.2` — see below.
- [ ] **Cherry-pick `34d4152` to master.** It corrects three README data-build
      errors (wrong script for the 7-channel montage, a raw directory nothing
      reads, and the `MMNet_research/data` vs repo-root `data` mismatch). It is
      unrelated to this experiment and applies to anyone rebuilding arrays.
- [x] Smart App Control documented in the 5080 handoff — pin widely-deployed
      versions, never `--force-reinstall` to clear an import block.
- [x] DATA_NOTES section 5 corrected: the full cohort and metadata are on
      Figshare (article 29253068, CC BY 4.0), not iHUB-Data-only.
- [ ] Decide whether to push the branch, and whether the foundation work lands as
      part of this paper or a follow-up.

## 5. Results to commit once final

- [ ] `runs/foundation/` — arm JSONs, per-fold CSVs, sweep shards worth keeping.
- [ ] Execute notebook 15 end to end with outputs saved, per the house rule that
      training runs live in an executed notebook.
- [ ] `domaingap/` results are already committed (`725dc05`).

## 6. Explicitly NOT worth doing, with the evidence

Recorded so nobody spends a day rediscovering these.

| tried | result |
|---|---|
| Fine-tuning CBraMod end to end (Arm C) | worse than freezing on all 3 folds (0.683 vs 0.720); ~22 h/seed |
| Hyperparameter search, 16 configs | whole space spans 0.0084, less than half a fold SD |
| Richer pooling (mean+std, mean+std+min+max) | statistically EQUIVALENT, TOST p = 0.0000 |
| Wider heads (512 vs 256) | equivalent, TOST p = 0.0000 |
| Third EEG representation stream | equivalent on staging, significantly worse on respiratory |
| Clinical covariates into the respiratory head | significantly WORSE (-0.026 AUC, p = 0.0004) |
| Lesion location as a model input | unusable: brain region 22/100, free text with typos |

The staging ceiling is not a framing problem. 27+ model families all land
0.60-0.75, the learning curve is flat, and one architecture scores 0.855 on
healthy sleep against 0.585 here. Effort belongs on the respiratory head, which
the learning curve shows is still climbing.
