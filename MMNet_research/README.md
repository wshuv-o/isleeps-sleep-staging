# MM-Net: Multimodal Joint Sleep Staging + Respiratory-Event Detection (iSLEEPS)

Paper `paper/multimodal.pdf`. `notebooks/` reproduce every number live. Model in
`model/mm_feature_net.py`; 10-fold engine `train/mmnet_repro.py`; metrics/artifacts in `results/`.
Verification log in `figures/VERIFICATION_LOG.md`.

## Data (single shared folder — not duplicated)
All data lives in one place at the parent level: **`../data/`**
- `../data/mm_features/SN*.npz`  — 188 EEG + 14 cardio features/epoch (feature model input)
- `../data/multimodal/`          — raw windows for the raw-CNN baseline
- `../data/Dataset/`             — raw iSLEEPS EDFs
Rebuild features with `preprocessing/extract_mm_features.py` + `cardio_features.py`.
