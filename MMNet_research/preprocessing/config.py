# Paths and modality used by the vendored official StagingPreprocess.
# Driver scripts (build_npz.py) override these explicitly; the placeholders
# only need to exist so `from config import *` in staging_preprocess.py succeeds.
raw_file_path = 'data/zenodo'
output_data_path = 'data/processed'
modality = ['eeg']
