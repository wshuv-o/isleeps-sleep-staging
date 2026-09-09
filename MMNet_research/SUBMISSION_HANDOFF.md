# Submission handoff — everything needed to finish the paper elsewhere

Written 9 Sep 2026, at the end of the RTX 5080 training session. **All training is
finished.** Nothing in this document requires a GPU, this machine, or re-running
anything. Every number you need is written out below, so you can write the paper
from a laptop with only the repository and a LaTeX editor.

Branch: `foundation-model-experiment`. `master` is untouched at `9db9d3d`.

MM-Net is **under review (round-2 revision), not published.** "Submitted model"
below means the configuration the current manuscript reports. Never write
"published" of MM-Net.

---

## 0. The one-paragraph summary

The submitted model used 188 engineered EEG features and 14 engineered
cardiorespiratory features. The final model keeps the 188, adds a **frozen
LaBraM embedding** (200-d) alongside them, and replaces the 14 cardio features
with a **learned CNN over the raw 7 x 750 signal**. It is better on every
headline metric, and the largest single effect in the paper is that the
cardiorespiratory branch is worth **0.128 AUC**.

Two manuscript claims changed and must be rewritten: the SpO2 attribution (C3)
is now an **effort** attribution, and the healthy-model-failure claim (C2) is
now an **asymmetric transfer** claim, which is stronger.

---

## 1. Headline numbers — replace these everywhere

Three seeds x ten patient-independent folds = 30 fold-values, mean +- SD.

| metric | submitted | **final model** |
|---|---|---|
| accuracy | 0.7275 | **0.7394 +- 0.0202** |
| macro-F1 | 0.6536 | **0.6701 +- 0.0204** |
| Cohen kappa | 0.6110 | **0.6340 +- 0.0240** |
| respiratory AUC | 0.7070 | **0.7818 +- 0.0378** |
| respiratory AP | 0.3370 | **0.4193 +- 0.1046** |

All four improvements are significant (Wilcoxon, 30 fold-values).

The LaTeX macros in `paper/multimodal_access.tex` are already updated:
`\mmAcc 0.739`, `\mmMf 0.670`, `\mmK 0.634`, `\mmAuc 0.782`, `\mmAp 0.419`.

**Numbers derived from pooled seed-42 predictions** (these are single-seed and
will differ slightly from the 30-fold means above — that is expected, not an
error; the confusion matrix and curves must come from ONE prediction set):

- staging acc 0.7338, mF1 0.6696, kappa 0.6281
- per-class F1: W 0.784, N1 **0.310**, N2 0.796, N3 0.712, R 0.746
- respiratory AUC 0.7775, AP 0.4186, event prevalence 0.157
- precision at sensitivity: 0.90 -> 0.220, **0.80 -> 0.270**, 0.70 -> 0.321, 0.60 -> 0.369
- AHI correlation **rho 0.491, p 2.46e-07, n 99** (was 0.315)
- staging by severity: Normal 0.769, Mild 0.747, Moderate 0.722, Severe 0.719

Source of truth: `results/revision/runs/final/derived_seed42.json`.

---

## 2. Model description for the Methods section

Trainable **2,764,774** parameters (submitted model: 773,254), plus **5,819,936
frozen** LaBraM parameters.

| block | params |
|---|---|
| EEG encoder, FeatMLP 388 -> 128 | 66,816 |
| Cardio encoder, CardioCNN 5250 -> 64 | 155,232 |
| Fusion, concat 192 -> 128 | 24,704 |
| BiLSTM, 2 layers, h=256 -> 512 | 2,367,488 |
| Staging head | 2,565 |
| Respiratory head | 147,969 |

Config: `arm="A+labram"`, `cardio="raw_cnn"`, `cardio_mode="concat"`,
`temporal="lstm"`, `hidden=256`, `drop=0.3`, `lr=3e-4`, `wd=1e-4`.
Authoritative in `foundation/run_final_model.py`.

Neural input 388-d = 188 engineered + 200 LaBraM. LaBraM sees only the four EEG
derivations (C4:M1 C3:M2 O2:M1 O1:M2 -> C4 C3 O2 O1); EOG and EMG have no
canonical 10-20 name and are dropped. Epochs are resampled 100 -> 200 Hz and
split into two 15 s sub-windows of 3000 samples, encoded and mean-pooled.

Cardio input 5250-d = 7 channels x 750 samples at 25 Hz (ECG, Flow, Thorax,
Abdomen, Effort, SpO2, Pulse), per-subject per-channel z-scored.

---

## 3. C4 — modality attribution. **REWRITE: SpO2 is no longer the anchor.**

Nine conditions x 3 seeds x 10 folds. Full model 0.7394 acc / 0.7818 AUC.
Table: `results/revision/runs/final/ablation_table.json`.

| removed | staging acc | delta | resp AUC | delta |
|---|---|---|---|---|
| EEG | 0.6800 | **-0.0593** | 0.7741 | -0.0078 |
| EOG | 0.7309 | -0.0084 | 0.7774 | -0.0044 |
| EMG | 0.7347 | -0.0047 | 0.7778 | -0.0040 |
| SpO2 | 0.7415 | +0.0021 | 0.7796 | -0.0023 |
| pulse/HRV | 0.7406 | +0.0012 | 0.7847 | +0.0029 |
| ECG | 0.7373 | -0.0021 | 0.7773 | -0.0045 |
| airflow | 0.7398 | +0.0005 | 0.7777 | -0.0041 |
| **effort** | 0.7411 | +0.0017 | 0.7439 | **-0.0379** |
| **all cardio** | 0.7409 | +0.0016 | 0.6536 | **-0.1282** |

**The claim to make**, and it is a better one than the manuscript's:

> The five individual cardiorespiratory removals sum to -0.0458 AUC, but removing
> the branch entirely costs -0.1282 — **2.80x the sum**. The learned encoder
> distributes respiratory information redundantly across channels: each is
> individually dispensable because the others overlap it, while the ensemble
> carries 0.128 AUC. Only effort is individually necessary.

This is clinically meaningful, not a consolation: 12 of 100 patients have
missing channels, and the model degrades gracefully when one is absent.

**Do NOT write that SpO2 drives the decision.** Under the learned encoder SpO2 is
the single most expendable channel (p = 0.61). Effort is the anchor —
physiologically coherent, since the belts show paradoxical thoraco-abdominal
motion *during* an obstructive event whereas desaturation is a lagged
consequence.

### DECISION YOU MUST MAKE — multiplicity family

EOG and EMG change verdict depending on the correction family:

| | EOG | EMG |
|---|---|---|
| Holm across all 9 conditions | 0.0795 **ns** | 0.1297 **ns** |
| Holm within family (3 neural on staging, 5 cardio on respiratory) | 0.0199 **sig** | 0.0199 **sig** |

The family-wise split is defensible — C4 is a modality-to-outcome mapping by
design, and nobody expects chin EMG to predict apnea — **but it is the more
favourable option, so declare it as pre-specified in the text.** Do not adopt it
silently. If you are not comfortable declaring it, report the pooled correction
and describe EOG/EMG as directionally consistent but not significant.

---

## 4. C2 — domain gap. **REWRITE around asymmetry; this is the strongest claim.**

The manuscript says healthy-sleep deep models fail here. That is true but
one-directional and now beatable. Measure both directions:

| direction | accuracy | kappa |
|---|---|---|
| healthy-trained -> **stroke** (3 models) | **0.313 +- 0.010** | 0.09-0.11 |
| **stroke-trained** -> healthy (Sleep-EDF) | **0.794** | **0.717** |

> Transfer is strongly asymmetric. A model trained on pathological sleep
> generalises to healthy sleep; the reverse collapses.

All three healthy models label 65-78% of stroke epochs Wake, N2 recall 0.03.
Report BOTH bounds so the montage confound cannot be used against you:
pathology alone ~0.19 from in-domain models; montage shift plus pathology 0.56.

Also true and worth keeping: a public checkpoint, frozen, reaches parity with
the 188 features (TOST p = 0.008), so do not claim deep models simply fail.
The sharper version is that *training from scratch* fails on 99 patients while
*pretraining transfers*.

Source: `MMNet_research/domaingap/`.

---

## 5. C5 — new claim: representational complementarity

Two independently pretrained encoders (CBraMod, LaBraM) each add significantly
to the engineered features; a third stream adds nothing.

- frozen SSL vs features alone: **+0.0081 acc, p = 0.003**
- frozen SSL vs its own random-init control: **+0.0203 acc, p = 0.0001**
- frozen CBraMod is equivalent to the 188 features (TOST p = 0.008)
- a third EEG representation stream: equivalent on staging, worse on respiratory

This is the novelty the round-2 referee said the architecture itself does not
supply.

---

## 6. Referee Finding 2 — the attribution figure. **Statistics changed.**

Two attributions sharing no machinery. Permutation importance was recomputed on
the final model: `results/revision/runs/final/permutation_importance_final.json`.

| modality | ablation dAUC | permutation dAUC |
|---|---|---|
| **effort** | **0.0379** | **0.0707** |
| airflow | 0.0041 | 0.0269 |
| SpO2 | 0.0023 | 0.0156 |
| pulse/HRV | -0.0029 | 0.0121 |
| ECG | 0.0045 | 0.0024 |

Staging drops: EEG 0.2842, EOG 0.0295, EMG 0.0063 — same order as the ablation.

**Report per metric, not pooled:**

- staging **Spearman rho 0.833, p 0.010** (manuscript reported 0.810)
- respiratory **rho 0.381, p 0.35 — NOT significant**
- pooled rho 0.691, p 0.003

The respiratory correlation fails for a diagnosable reason: four of five cardio
channels have retraining effects indistinguishable from zero, so their rank
order is noise and correlating noise returns nothing. **The respiratory panel
should claim effort's dominance under both methods — which is real and large —
not a rank correlation.** Do not report 0.691 as if it replaced 0.810.

`figures/fig_attribution_agreement.py` still reads the OLD
`permutation_importance.json` and `foundation/ablation_shards/`. It must be
pointed at `final/permutation_importance_final.json` and
`final/ablation_shards/` before the figure is regenerated.

---

## 7. Learning curve — supports "collect more patients"

`results/revision/runs/final/lc_shards/`, 4 fractions x 10 folds, seed 42.

| train patients | final acc | submitted acc | **final resp AUC** | submitted AUC |
|---|---|---|---|---|
| 20 | 0.7010 | 0.6929 | 0.7320 | 0.6556 |
| 40 | 0.7212 | 0.7135 | 0.7504 | 0.6852 |
| 59 | 0.7260 | 0.7213 | 0.7519 | 0.6957 |
| 79 | 0.7346 | 0.7227 | **0.7853** | 0.7111 |

The final model wins at every training-set size, so the gain is not a
large-sample artefact. The respiratory head's **steepest segment is the last
one** (+0.033 from 59 to 79 patients) — it is not saturated and more patients
would still pay. Staging is nearly flat, which keeps the manuscript's argument
that the staging ceiling is the cohort rather than the architecture.

---

## 8. External validation

**Sleep-EDF, zero-shot, staging only** (no respiratory annotations in the corpus):

| | acc | mF1 | kappa |
|---|---|---|---|
| final model | 0.7940 | **0.7157** | **0.7174** |
| submitted model | 0.8002 | 0.7086 | 0.7121 |

Equivalent; marginally better on both class-balanced metrics.

Caveats to state, both already in `preprocessing/build_sleepedf_external.py`:
- **Montage substitution.** Sleep-EDF has EEG Fpz-Cz and Pz-Oz only, duplicated
  into four slots. iSLEEPS is central/occipital, Sleep-EDF frontal/occipital.
  This is a confound on any drop, so the number is a lower bound.
- **No cardiorespiratory channels**, so that branch is fed zeros.

A control settles whether zeroing only at test inflates the number. Training
with the branch zeroed too gives **0.7735 — lower** than 0.7940, so the reported
figure is not a train/test-mismatch artefact.
(`external_sleepedf_matched.json`.)

**ISRUC is NOT done.** No `data/isruc_mm/` on this machine; the host was
unreachable and the HuggingFace mirrors are 6-channel. Either drop it, or run it
on the other machine per `foundation/EXTERNAL_VALIDATION_INSTRUCTIONS.md` —
the code is ready and takes ~5 minutes once the data exists.

---

## 9. What NOT to claim — recorded so nobody rediscovers it

| tried | result |
|---|---|
| **A designed cardio CNN as architectural novelty** | **Refuses it.** The trained CNN is *significantly worse* than the same architecture with random weights (p = 0.0155). Do not write "we designed the encoder." |
| Fine-tuning CBraMod end to end (Arm C) | worse than freezing on all 3 folds (0.683 vs 0.720); ~22 h/seed |
| Hyperparameter search, 16 configs | whole space spans 0.0084, under half a fold SD |
| Richer pooling, wider heads | statistically EQUIVALENT (TOST p = 0.0000) |
| Third EEG representation stream | equivalent on staging, worse on respiratory |
| Clinical covariates into respiratory head | significantly WORSE (-0.026 AUC, p = 0.0004) |
| Lesion location as input | unusable: brain region 22/100, free text with typos |

MOMENT (341 M) reaches 0.8058 respiratory AUC versus the CNN's 0.7816. **You
decided to report the CNN**, which is the honest and reproducible choice. If a
referee asks whether a larger encoder helps, say yes and cite the number — the
repository is public and the result is in
`results/revision/runs/foundation/`. Do not conceal it.

---

## 10. Remaining tasks, in priority order

Writing only. None of these needs a GPU.

1. **Rewrite C3** from SpO2 to effort + redundancy (section 3 above).
2. **Rewrite C2** around asymmetric transfer (section 4).
3. **Add C5** (section 5).
4. **Decide the multiplicity family** and declare it (section 3).
5. **Repoint `fig_attribution_agreement.py`** at the final-model inputs,
   regenerate, rewrite the caption per section 6, swap in at
   `paper/multimodal_access.tex:994` (referee Finding 2).
6. **Swap the architecture figure** at `paper/multimodal_access.tex:354` for
   `figures/mm_architecture_v4.drawio` (export from draw.io) or the
   ready-made `figures/fig_architecture_v4.pdf`. **Do not use v3** — it draws a
   model that was never trained.
7. **Remove the `\stale{}` red marking** from the 11 captions once each number
   is updated. `\stale` is defined in the preamble; delete the wrapper, keep the
   text.
8. Regenerate any figure whose underlying run changed (confusion, AHI, ROC/PR,
   per-class F1 — all derive from `derived_seed42.json`).
9. Optional: ISRUC (section 8), title (Finding 1, author's call).

---

## 11. Environment notes for the other machine

- `requirements.txt` is **missing three packages** now required: `pyedflib`
  (data build), `momentfm` (install with `--no-deps`; its pinned
  huggingface_hub will not build on 3.12), and `braindecode` (LaBraM).
- **Pin `scipy==1.15.2`.** Windows Smart App Control blocks unsigned scientific
  wheels; a `pip install --force-reinstall` resets the file's reputation and
  makes the block *worse*. Pin widely-deployed versions instead. See the 5080
  handoff doc.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` for anything importing `mmnet_core`.
- Runs exiting with `rc=3221226505` (0xC0000409) *after* printing their result
  are fine — that is a CUDA/OpenMP teardown crash after the shard is written.
  Check the shard, not the return code.
- **`gh` is currently authenticated as `wshuv-o`, not `EsmeAbha`.** Run
  `gh auth switch --user EsmeAbha` on this machine when you return.

## 12. Repository hygiene, still open

- Cherry-pick `34d4152` to master — it corrects three README data-build errors
  and is unrelated to this experiment.
- Execute notebook 15 end to end with outputs saved, per the house rule that
  training runs live in an executed notebook.
- Decide whether the foundation work lands in this paper or a follow-up.
