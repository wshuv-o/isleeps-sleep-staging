# MM-Net build notes — IEEE Access round 2

Where the current PDF is, what is inside it, what changed to get here, and what is
still waiting on a decision. Written 11 September 2026 against commit `26ec52b`.

---

## The file

**`MMNet_research/paper/multimodal_access.pdf`**

| | |
|---|---|
| Pages | 18 |
| Page size | US Letter, 612 × 792 pt |
| Size | 1,131,220 bytes (1.08 MB) |
| Built | 11 September 2026 |
| Commit | `26ec52b` |
| Abstract | 239 words |
| Figures / Tables / References | 7 / 13 / 50 |

A second file, `multimodal_access_PREVIEW_2026-09-10.pdf`, sits beside it and is
**byte-identical** — same build, kept only because that filename was the one being
opened on GitHub. Either works.

**Ignore these four** in the same folder. They are from 7 September and predate all
of this work: `multimodal.pdf`, `multimodal_diff.pdf`, `multimodal_elsevier.pdf`,
`multimodal_elsevier_diff.pdf`.

---

## Headline results

Ten-fold patient-independent cross-validation on iSLEEPS, three seeds, reported as
the mean over all thirty fold-values. Cohort is 99 recordings, 92,560 epochs.

| Metric | Value | SD |
|---|---|---|
| Staging accuracy | **0.739** | ± 0.020 |
| Cohen's kappa | **0.634** | ± 0.024 |
| Respiratory AUC | **0.782** | ± 0.038 |
| Average precision | **0.419** | ± 0.105 |

Re-derived from `results/revision/runs/final/final_model.json` and matching the
manuscript macros exactly.

MM-Net is **numerically ahead of all eight** baselines we ran ourselves, by +0.019
to +0.093 accuracy, winning eight to ten folds of ten in every comparison. Paired on
the ten fold-means with Holm correction, the margin is statistically separable for
five of the eight. AttnSleep (+0.023), the corpus LSTM (+0.019) and
1D-ResNet-SE-LSTM (+0.032) sit inside what ten folds can resolve, and §VI-D says so
rather than glossing it.

---

## What changed in this round

Fourteen commits. Two of them corrected claims the paper was making that the code
did not support.

| Commit | Change |
|---|---|
| `26ec52b` | **Preview build patcher fixed.** The local tectonic build needs `\RequirePackage{spotcolor}` commented out as well as the three spot-colour primitives; without it xcolor fails under XeTeX. |
| `4bf754b` | **Suva's biography names fields, not paper titles**, matching how Abha's and Sajin's are written. |
| `ba4cd57` | **Suva's biography rewritten from his publication record.** Third authorship on the IEEE TAI paper checked against arXiv 2411.01641 rather than assumed. |
| `f65ea95` | **Author photographs.** New portrait for Abha; Suva's re-cropped from the 1254×1254 original so it fills the frame at the same scale as the others. |
| `fbcb390` | **Cohort bookkeeping settled from the data.** See below. |
| `b7052e7` | **Figure 2 now shows the fusion the model runs.** See below. |
| `54dc19f` | **Both ablation tables use the same test unit.** `tab:repr` moved onto fold-means; every verdict held. |
| `d486b37` | **Every paired test moved to fold-means.** Three seeds within a fold share patients, so they are re-rolls of one experiment, not three observations. Effect sizes unchanged; two claims moved. |

### The two substantive corrections

**Figure 2 drew a model we never ran.** The fusion block showed a four-head
cross-attention transformer — tokenize, `softmax(QKᵀ/√d_k)V`, `concat·W_o`,
residual, LayerNorm, feed-forward. That is `CrossFusion` at `mmnet_core.py:101`,
reachable only via `fusion="cross"`. The published model calls
`run_10fold(fusion="concat")`, which reaches `mmnet_core.py:125`:

```python
self.fuse = nn.Sequential(nn.Linear(d + d_card, d), nn.GELU(), nn.Dropout(drop))
fz = self.fuse(torch.cat([e, c], -1))          # 128 + 64 -> 128
```

Confirmed by reconstructing the exact final model rather than by reading: **2,764,774
trainable parameters**, matching the paper to the digit, with **zero**
`MultiheadAttention` modules and `fuse = Linear(192→128)`. The figure was redrawn
and two pieces of prose that leaned on the old panel were corrected — including a
caption and a §IV-C claim that both cited §VI-A for a fusion comparison §VI-A does
not contain. That comparison does exist (`results/headline_concat.json` vs
`results/attention_cross.json`) and is now quoted with its numbers: attention is
indistinguishable on staging (+0.005, p = 0.49) and slightly worse on the
respiratory head (−0.014 AUC, p = 0.06).

**`N = 96` was stale and matched nothing.** Not 99 usable, not 87 with all seven
channels, not the 98 an older filter would have kept. What settled it:
`mmnet_core.load_data()` drops only the SN28 duplicate and applies **no** channel
filter, and `run_10fold` scores every test subject with no mask. So both tasks run
on all 99, with absent channels zero-filled — there is no reduced complete-montage
subset anywhere in the pipeline, which means the sentence describing one was wrong
about the design and not only about the number. Fixed in §III and §VII; the AHI
sentence said "three recordings" where it is twelve.

Left alone deliberately: `tab:resp`'s "96 annotated patients". That 96 is real and
unrelated — `derived_seed42.json` records
`per_event_type_note = "96 subjects aligned"`, the subjects whose typed event
annotations aligned for the by-event-type AUC.

---

## Figures

| | Caption |
|---|---|
| Fig 1 | Sleep-disordered-breathing burden across the cohort, over the 99 with respiratory labels |
| Fig 2 | The two-stream, multi-task model — **redrawn this round** |
| Fig 3 | A representative held-out night (patient SN90) |
| Fig 4 | Row-normalised confusion matrix, pooled ten-fold test epochs |
| Fig 5 | Respiratory-event detection, pooled over the ten test folds |
| Fig 6 | Predicted per-patient event burden against the clinical AHI |
| Fig 7 | Learning curve over training-patient fraction |

## Tables

| | Caption |
|---|---|
| Tab 1 | Stage distribution and respiratory-event prevalence (N = 99) |
| Tab 2 | Training configuration, shared across folds, baselines and ablations |
| Tab 3 | Reimplementation check on Sleep-EDF Cassette subjects 0–20 |
| Tab 4 | Sleep staging on iSLEEPS against every architecture evaluated |
| Tab 5 | Per-class staging F1 at the hidden-Markov operating point |
| Tab 6 | Modality-ablation grid, leave-one-out — **Holm p column added** |
| Tab 7 | Event-level detection, pooled over the ten folds |
| Tab 8 | Respiratory-event detection, ten-fold patient-independent |
| Tab 9 | Staging accuracy by sleep-disordered-breathing severity (AASM AHI bands) |
| Tab 10 | What each stream should be represented as — **moved to fold-means** |
| Tab 11 | External validation, zero-shot, no dataset-specific tuning |
| Tab 12 | Model size and clinical capability |
| Tab 13 | Recent single-channel methods: reported accuracy vs. accuracy here |

---

## Submission package

IEEE Access typesets from source. Everything lives in `MMNet_research/paper/`, except
the figures, which resolve to `MMNet_research/figures/` through `\graphicspath`.

| File | Size | What it is |
|---|---|---|
| `multimodal_access.tex` | 97 KB | The manuscript |
| `references.bib` | 24 KB | 50 references, all carrying DOIs |
| `ieeeaccess.cls` | 52 KB | Journal class, **unmodified** |
| `figures/*.pdf` | — | 7 vector figures |
| `bio_suva.jpg`, `bio_abha.jpg`, `bio_sajin.jpg`, `bio_mobin.jpg` | — | 4 author photographs, 420 px wide |

`ieeeaccess.cls` in the repository is untouched and is what should be submitted. The
local preview builds with tectonic / XeTeX, which cannot handle the class's
pdfTeX-only Pantone spot-colour block, so `patch_cls_preview.py` neutralises it on a
scratch copy and substitutes a CMYK approximation of PANTONE 3015 C. On Overleaf with
pdfLaTeX the real spot colour is used and no patch is needed.

---

## Still open

**Biography spacing on page 18 — needs a decision.** The gaps between biographies are
not a fault in our file. `ieeeaccess.cls` wraps each one in glue carrying
`plus 1fil` — infinite stretch — which absorbs all the slack needed to fill the final
page; the gap after Suva's measures 129 pt. `\raggedbottom` cannot compete (it offers
0.0001 fil) and verifiably changed nothing: biography positions were identical before
and after. This is IEEE Access house style. Leaving it is defensible; tightening it
means patching their class, which is best confined to the preview build.

**Review items 1 and 18 — closed by author decision.** Item 1 asked for an MDPI
citation and was ruled out. Item 18 proposed retitling away from "physiologically
interpretable"; the suggestion on file was *A Multimodal Multi-Task Model with Causal
Modality Attribution*.

**U-Sleep — dropped.** Never ran; the API token did not arrive. Nothing in the
manuscript claims it: U-Sleep appears once, in Related Work, as a cited prior work.
U-Time, which *was* reimplemented and benchmarked, is a different model.

**Everything else is clear.** The 25-item review is otherwise closed. No dangling
references, no unbalanced environments, no unreferenced labels, zero prose em-dashes
— all 19 in the rendered text are "not applicable" cells inside tables. The four
remaining overfull boxes are small and pre-existing (9.3, 15.4, 1.0 and 2.8 pt).

---

## Repository state

Branch `foundation-model-experiment` at `26ec52b`, pushed.

`master` was merged through `083e26e` and is now **6 commits behind**. It has the
cohort fix and the corrected Figure 2, but not the author photographs, Suva's
biography, or the build-patcher fix. Worth fast-forwarding before anyone pulls from
it.
