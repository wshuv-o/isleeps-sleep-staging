# RESUME — 7-channel (EEG+EOG+EMG) best-model run

**Saved state (safe to shut down):** the 7-channel preprocessing is ~84/100 done; the
completed `data/processed7/SN*.npz` persist on disk. Everything below is resumable.

## Where we are
- **Current best model:** Ensemble+HMM on 4-channel EEG (full cohort) = **0.727 acc / kappa 0.612**
  (within 2 pts of the published LSTM 0.747; beats their Transformer 0.674 and CNN 0.617).
  Result in `results/ensemble_all.json`, recorded in `RESULTS.md`.
- **In progress:** adding EOG + chin EMG (the standard sleep montage) to try to push past 0.727.
  7-channel npz -> `data/processed7/` (float16 to save disk). Trainer ready:
  `train_ensemble_full.py` (XGBoost on GPU, +LightGBM +HistGB, +HMM).

## Resume steps (use the GPU env + the OMP flag)
Env: `d:/EEG-TransNet/testenv/python.exe`, always `export KMP_DUPLICATE_LIB_OK=TRUE`.

1. **Check disk first** (`df -h /d`). It was ~1.1 GB free. If < ~1 GB, free space safely:
   - the 4-channel `data/processed/` (3.9 GB) is REDUNDANT with the 7-ch npz (7-ch[:,:4] = the
     4 EEG channels) — deletable if needed, or delete raw EDF `data/full100` (11 GB, re-downloadable).

2. **Finish 7-ch preprocessing** (skips the ~84 already done, does the last ~16):
   ```
   cd /d/MOB-EEG
   python -u preprocess/build_npz_full.py 1>data/proc7d.log 2>data/proc7d.err
   ```
   Confirm: `ls data/processed7/*.npz | wc -l` == 100 (99 usable; SN28 is a known duplicate).

3. **Run the 7-channel ensemble (best-model attempt):**
   ```
   KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe -u train_ensemble_full.py --context 3 \
     1>results/run_ens7.log 2>results/run_ens7.err
   ```
   Do NOT run other heavy jobs at the same time (RAM ~15 GB total; concurrent jobs cause OOM).
   Watch: `grep -E "Fold|=====" results/run_ens7.log`. Result -> `results/ensemble7_all.json`.

4. **Compare** the 7-ch `ensemble + HMM` acc vs the 4-ch 0.727. Honest expectation: +2 to +5 pts
   (mostly REM/N1 from EOG/EMG) -> roughly 0.75, i.e. match/edge the published 0.747. Will NOT hit
   80% (that is above human inter-scorer agreement on this disrupted cohort).

5. **If it wins:** update `RESULTS.md` best-model table + regenerate the deck leaderboard figure
   (`presentation/make_figures.py`) + rebuild `presentation/iSLEEPS_proposal.pptx`.

## Reminder on framing
Accuracy is the crowded, saturated metric (ceiling ~0.747 here). The proposal's real value is the
lesion-resolved finding (model as instrument). A ~0.75 stager that matches the published tool is a
credible instrument; it does not need to "win" on accuracy. See `FINAL_PROPOSAL.md`.
