# External validation of the final model — instructions for the 2060 box

The final model changed, so the external-validation table (ISRUC and Sleep-EDF)
is stale. It **cannot be re-run on the 5080** because `data/isruc_mm/` and
`data/sleep_edf_mm/` were never copied across — everything else in the paper has
been re-run there, this is the one gap.

This is ~15 MB of data and about 30 minutes of compute. Run it wherever those two
directories already exist.

---

## What changed, and why the old numbers no longer apply

The manuscript's external validation used the feature-based configuration:
188 engineered EEG features + 14 engineered cardiorespiratory features. The final
model replaces both inputs:

| stream | current manuscript | final model |
|---|---|---|
| EEG | 188 engineered features | 188 features **+ frozen LaBraM embedding (200-d)** |
| cardio | 14 engineered features | **learned CNN over raw 7-channel signal** |

Both changes alter what the model expects at inference, so the external corpora
have to be pushed through the same two encoders before the model can score them.

---

## Step 1 — confirm the corpora are present

```bash
ls data/isruc_mm/*.npz    | wc -l     # expect 8
ls data/sleep_edf_mm/*.npz | wc -l    # expect 20
```

If they are missing, they are rebuilt by the same preprocessing that produced
them originally; see `MMNet_research/preprocessing/`.

## Step 2 — pull the branch with the final model

```bash
git fetch origin
git checkout foundation-model-experiment
pip install -r requirements.txt          # adds braindecode, pyedflib
```

`scipy` must be pinned to `1.15.2`. Newer builds are blocked by Smart App Control
on the 5080; if the 2060 has no such policy any recent scipy is fine, but 1.15.2
is what every number on the branch was produced with.

## Step 3 — build the LaBraM cache for the external corpora

The EEG stream needs the frozen embedding. `build_labram_cache.py` reads
`data/processed7/`, so point it at each external corpus in turn:

```bash
python MMNet_research/foundation/fetch_cbramod.sh      # if not already fetched
python MMNet_research/foundation/build_labram_cache.py --variant labram \
       --source data/isruc_mm     --out data/labram_ext/isruc
python MMNet_research/foundation/build_labram_cache.py --variant labram \
       --source data/sleep_edf_mm --out data/labram_ext/sleep_edf
```

**Montage check before you run it.** LaBraM requires canonical 10-20 names and
takes only the four EEG derivations. iSLEEPS maps
`C4:M1, C3:M2, O2:M1, O1:M2 -> C4, C3, O2, O1`. If either external corpus uses a
different montage the mapping in `build_labram_cache.CH_NAMES` must be edited to
match, and **the mapping used should be reported in the paper** — a silent
mismatch here would inflate or deflate the external numbers with no error raised.

## Step 4 — run the external validation

```bash
KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/foundation/run_external_validation.py \
       --corpus isruc     --out results/revision/runs/final/external_isruc.json
KMP_DUPLICATE_LIB_OK=TRUE python MMNet_research/foundation/run_external_validation.py \
       --corpus sleep_edf --out results/revision/runs/final/external_sleepedf.json
```

Each trains on all 99 iSLEEPS patients and evaluates on the external corpus with
no adaptation, which is the protocol the manuscript already describes.

## Step 5 — send back

Commit the two JSON files and push, or send them directly. Each contains pooled
accuracy, macro-F1, kappa, per-class F1 and, where the corpus has respiratory
annotations, AUC and AP.

---

## Expected outcome, stated in advance

The final model should perform **similarly or slightly better** on the external
corpora than the feature-based one. The cardio CNN gains are large on iSLEEPS
(+0.067 AUC) but ISRUC and Sleep-EDF have different cardiorespiratory montages,
so do not be surprised if the respiratory gain does not transfer.

**If external accuracy drops sharply**, that is a real finding and not a bug:
it would mean the added encoders overfit iSLEEPS-specific signal characteristics,
and it belongs in the paper's limitations rather than being tuned away.

## What NOT to do

Do not tune anything on the external corpora. They are held out; the whole value
of the table is that nothing was fitted to them.
