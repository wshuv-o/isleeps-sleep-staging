# How to undo the foundation-model work

Everything from the 5080 foundation-model experiment is isolated so it can be
discarded without touching the published pipeline.

## What is where

| Change | Location | On master? |
|---|---|---|
| CBraMod adapter + cache builder | `MMNet_research/foundation/` | no — branch only |
| README data-build corrections | `README.md` (commit `34d4152`) | no — branch only |
| 40-subject rebuilt arrays | `data/processed7`, `data/multimodal`, `data/mm_features` | not tracked (`data/` is git-ignored) |
| Frozen CBraMod embeddings | `data/cbramod_emb/` | not tracked |
| Raw Zenodo download | `data/zenodo/` (6.9 GB) + `data/Dataset` junction | not tracked |
| Path junction | `MMNet_research/data` -> repo-root `data` | not tracked |
| Extra package | `pyedflib` (installed with pip) | n/a |

`master` is untouched and still points at `9db9d3d`, exactly as the other machine
left it. Nothing here has been pushed.

## Revert the code

```bash
git checkout master                      # already clean; nothing to undo
git branch -D foundation-model-experiment
```

## Revert the data

```bash
rm -rf data/cbramod_emb data/processed7 data/multimodal data/mm_features
rm -rf data/Dataset MMNet_research/data      # junctions, not real directories
rm -rf data/zenodo                           # 6.9 GB raw download, re-fetchable
```

Each rebuilt directory carries a `PROVENANCE.txt` saying it is the 40-subject
open subset and not the published cohort. Read it before deleting or before
copying the full cohort in — the build scripts skip subjects that already exist,
so copying 100 subjects on top of these 40 leaves a silently mixed cache.

## Worth keeping even if the experiment is dropped

Commit `34d4152` is a README correctness fix, not experiment code: the 7-channel
montage comes from `build_npz_full.py` rather than `build_npz.py`, the two build
stages read different raw directories, and all three scripts resolve paths under
`MMNet_research/` rather than the repo root. It is unrelated to the foundation
model and applies to anyone rebuilding the arrays. To keep just that:

```bash
git checkout master && git cherry-pick 34d4152
```
