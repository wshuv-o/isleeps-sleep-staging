# Standard library imports
import os
import glob
import warnings
from typing import List, Any, Dict, Union

# Third-party imports
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count, parallel_backend

# Local application/library-specific imports
from config import *  # open config.py to set your paths and modality
from channel_mapping import *

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Set MNE logging level
mne.set_log_level(verbose='WARNING')



class StagingPreprocess:
    def __init__(
        self,
        raw_path: str,
        ann_path: str,
        channel_mapping: Dict[str, str],
        modality: List[str],
        window_size: float,
        sfreq: int,
        preload: bool = False,
        crop_wake_mins: int = 0,
        crop: Union[Any, None] = None
    ):
        if not isinstance(raw_path, str) or not isinstance(ann_path, str):
            raise Exception(f"raw_path and ann_path must be strings, found raw_path: {type(raw_path)} ann_path: {type(ann_path)}")
        
        self.window_size = window_size
        self.sfreq = sfreq
        raw, desc = self._load_raw(
            raw_path,
            ann_path,
            channel_mapping,
            modality,
            preload=preload,
            crop_wake_mins=crop_wake_mins,
            crop=crop,
        )
        
        self._raw = raw
        self._description = desc
    
    @property
    def raw(self):
        return self._raw
    
    @property
    def description(self):
        return self._description
        
    def read_annotations(self, ann_fname):
        labels = []
        ann = pd.read_excel(ann_fname, sheet_name="Sleep profile")[8:]
        ann.reset_index(inplace=True, drop=True)
        ann.columns = ["timestamp", "stage"]
        ann_list = ann["stage"].tolist()
        timestamps = ann["timestamp"].tolist() # to be used

        for lbl in ann_list:
            if lbl == "Wake":
                labels.append('W')
            elif lbl == "N1":
                labels.append('N1')
            elif lbl == "N2":
                labels.append('N2')
            elif lbl == "N3":
                labels.append('N3')
            elif lbl == "REM":
                labels.append('R')
            elif lbl == "A":
                labels.append('A')
            else:
                labels.append('BAD_?')

        labels = np.asarray(labels)
#         print(labels)
        onsets = [self.window_size * i for i in range(len(labels))]
        onsets = np.asarray(onsets)
        durations = np.repeat(float(self.window_size), len(labels))
        annots = mne.Annotations(onsets, durations, labels)
        return annots

    def _set_channel_types(self, raw, channel_mapping):
        available_channels = raw.info['ch_names']
        updated_mapping = {channel: channel_mapping[channel] for channel in available_channels if channel in channel_mapping}
        raw.set_channel_types(updated_mapping)
        return raw
    
    def _load_raw(
        self,
        raw_fname,
        ann_fname,
        channel_mapping,
        modality,
        preload,
        crop_wake_mins,
        crop,
    ):  
        raw = mne.io.read_raw_edf(raw_fname, preload=preload, include=channel_mapping)
        raw = self._set_channel_types(raw, channel_mapping)
        raw.pick(modality)
        annots = self.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)
        raw.resample(self.sfreq, npad="auto")

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "R"] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        if crop is not None:
            raw.crop(*crop)

        raw_basename = os.path.basename(raw_fname)
        subj_nb = int(raw_basename.split('.')[0][2:])
        desc = pd.DataFrame(
            {
                "subject_id": [subj_nb],
            }
        )
        return raw, desc
    
    @staticmethod
    def create_windows(raw, description, window_size: float = 30., window_stride: float = 30., label_mapping=None, drop_last: bool = False, drop_bad: bool = False):
        assert isinstance(window_size, (int, np.integer, float, np.floating)), "window_size has to be an integer or float"
        assert isinstance(window_stride, (int, np.integer, float, np.floating)), "window_stride has to be an integer or float"
                
        window_size_samples = int(window_size*raw.info['sfreq'])
        window_stride_samples = int(window_stride*raw.info['sfreq'])

        assert len(raw.annotations.description) > 0, "No annotations found in raw"
        assert window_size_samples > 0, "window size has to be larger than 0"
        assert window_stride_samples > 0, "window stride has to be larger than 0"
        
        if label_mapping is None:
            label_mapping = dict()
            unique_events = np.unique(raw.annotations.description)
            filtered_unique_events = [event for event in unique_events if not event.startswith("BAD_")]
            label_mapping.update(
                {v: k for k, v in enumerate(filtered_unique_events)}
            )
            
        events, events_id = mne.events_from_annotations(raw, event_id=label_mapping) # type: ignore
        onsets = events[:, 0] # starts compared to original start of recording
        targets = events[:, -1]
        filtered_durations = np.array(
            [ann['duration'] for ann in raw.annotations
                if ann['description'] in events_id]
        )
        stops = onsets + (filtered_durations * raw.info['sfreq']).astype(int)
        
        if window_size_samples is None:
            window_size_samples = stops[0] - onsets[0]
            if window_stride_samples is None:
                window_stride_samples = window_size_samples   
        
        if drop_last:
            if (stops-onsets)[-1] != window_size_samples:
                stops = stops[:-1]
                onsets = onsets[:-1]
                targets = targets[:-1]
                events = events[:-1]
                
        events = [[start, window_size_samples, targets[i_start]] for i_start, start in enumerate(onsets)] # new events
#         events = np.array([[start, window_stride_samples, targets[i_start]] for i_start, start in enumerate(onsets)], dtype=int)  # new events

        metadata = pd.DataFrame({
            'start': onsets,
            'stop': stops,
            'size': window_size_samples,
            'stride': window_stride_samples,
            'target': targets})    
        
        desc_columns = list(description.columns)
        for col in desc_columns:
            metadata[col] = description[col].values[0]

        mne_epochs = mne.Epochs(
                raw, events, events_id, baseline=None, tmin=0,
                tmax=(window_size_samples - 1) / raw.info['sfreq'],
                metadata=metadata, preload=True, verbose=False)
        
        if drop_bad:
            mne_epochs.drop_bad()
        
        return mne_epochs
    
def _get_channels(raw_path, ann_path, window_size, sfreq, preprocessed_path, modality, label_mapping):
    save_data = dict()
    ds = StagingPreprocess(raw_path, ann_path, CHANNEL_MAPPING, modality, window_size, sfreq, preload=True)
    epochs_data = StagingPreprocess.create_windows(
        ds.raw, ds.description, window_size=window_size, 
        window_stride=window_size, label_mapping=label_mapping, drop_last=True,
        drop_bad=True
    )
    # print('hi sexy')
    func = lambda x: x * 1e6
    epochs_data.apply_function(func)
    
    cnt = 1
    sub_id = int(epochs_data.metadata.subject_id[0])
    subject_folder = os.path.join(preprocessed_path, f"SN_{sub_id}")
    
    # Create the subject folder if it does not exist
    os.makedirs(subject_folder, exist_ok=True)
    # breakpoint()
    for ep_idx in range(len(epochs_data)):
        for ch, pick in zip(epochs_data.ch_names, epochs_data.picks):
            data = epochs_data[ep_idx].get_data()
            save_data[ch] = data[:, pick, :]  # Shape of (1, 3000)
        save_data['target'] = int(epochs_data[ep_idx].metadata.target)
        
        # Save the data into the subject-specific folder
        temp_save_path = os.path.join(subject_folder, f"{sub_id}_{cnt}.npz")
        np.savez(temp_save_path, **save_data)
        cnt += 1

def _preprocess_dataset(raw_paths, ann_paths, k, N, preprocessed_path, window_size, sfreq, modality, label_mapping):
    raw_paths_core = [f for i, f in enumerate(raw_paths) if i%N==k]
    ann_paths_core = [f for i, f in enumerate(ann_paths) if i%N==k]
    # print('preprocess dataset!!!!!')
    # breakpoint()
    for raw_path, ann_path in tqdm(zip(raw_paths_core, ann_paths_core), desc="Dataset preprocessing...", total=len(raw_paths_core)):
        _get_channels(raw_path, ann_path, window_size, sfreq, preprocessed_path, modality, label_mapping)  
        
def generate(modality: Union[List[str], None] = None, data_path: Union[str, None] = None, preprocessed_path: Union[str, None] = None, n_jobs: int = -1, label_mapping: Union[Dict[str, int], None] = None):
    # Validate input arguments
    assert isinstance(data_path, (str, type(None))), f"{data_path} should be of type str"
    assert modality is None or (isinstance(modality, list) and all(isinstance(item, str) for item in modality)), "Provided modality is not a list of str"
    
    # Set default data and preprocessed paths if not provided
    if data_path is None:
        data_path = os.path.expanduser("~/.nimhans/edf_data")
        print("L1")
        
    if preprocessed_path is None:
        preprocessed_path = os.path.expanduser("~/.nimhans/preprocessed")
        print("L2")
    # Create directories if they do not exist
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    # Set default modality and label mapping if not provided
    if modality is None:
        modality = ['eeg']
        
    if label_mapping is None:
        label_mapping = LABEL_MAPPING
        
    # Set number of jobs to the number of CPUs if not provided
    if n_jobs == -1:
        n_jobs = cpu_count()
    # breakpoint()
    # Find EDF and annotation files
    edf_files = glob.glob(f'{data_path.split(".")[0]}.edf')
    ann_files = glob.glob(f'{data_path.split(".")[0]}.xlsx')
    # Define preprocessing parameters
    window_size = 30.0
    sfreq = 100
    # Parallel processing of datasets
    with parallel_backend(backend='loky'):     
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_preprocess_dataset)(
                edf_files, ann_files, k, n_jobs, preprocessed_path, 
                window_size, sfreq, modality, label_mapping
            ) for k in range(n_jobs)
        )
        
        
def process_data(modality, raw_file_path, output_data_path):
    """
    Processes data for all subjects in the raw data path and generates the required output.
    
    Parameters:
        modality (str): The modality to be used in the generate function.
        raw_file_path (str): The base path to the raw data files.
        output_data_path (str): The base path where the preprocessed data should be saved.
    """
    # List all files/directories in the raw data path
    all_files = os.listdir(raw_file_path)
    
    # Filter files/directories that match the subject ID pattern 'SN<number>'
    subject_ids = [f for f in all_files if f.startswith('SN') and f.split('.')[0][2:].isdigit()]

    # Sort subject IDs by the numeric value after 'SN'
    subject_ids_sorted = sorted(subject_ids, key=lambda x: int(x.split('.')[0][2:]))
    # Process each subject ID in sorted order
    for subject_id in subject_ids_sorted:
        print(f"Processing data for subject: {subject_id}")
        generate(
            modality=modality, 
            data_path=os.path.join(raw_file_path, subject_id), 
            preprocessed_path=os.path.join(output_data_path, 'nimhans_np/'),
            n_jobs=10
        )
