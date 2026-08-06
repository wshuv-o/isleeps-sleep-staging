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
from staging_preprocess import *

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Set MNE logging level
mne.set_log_level(verbose='WARNING')

def combine_npz_files(input_folder, output_file, modality):
    # List all .npz files in the folder and sort them by the number in the filename
    file_list = sorted(
        [f for f in os.listdir(input_folder) if f.endswith('.npz')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    # Initialize lists to hold the data for 'x' and 'y'
    x_data_list = []
    y_data_list = []

    # Define a dictionary to map alternative keys to standard keys
    key_mapping = {
        'E1:M2': 'E1:M2',
        'E2:M2': 'E2:M2',
        'C4:M1': 'C4:M1',
        'C3:M2': 'C3:M2',
        'O2:M1': 'O2:M1',
        'O1:M2': 'O1:M2',
        'EOG1:A2': 'E1:M2',
        'EOG2:A2': 'E2:M2',
        'F4:A1': 'C4:M1',
        'F3:A2': 'C3:M2',
        'C4:A1': 'C4:M1',
        'C3:A2': 'C3:M2',
        'O2:A1': 'O2:M1',
        'O1:A2': 'O1:M2'
    }

    # Load and process each .npz file
    for file_name in file_list:
        file_path = os.path.join(input_folder, file_name)
        data = np.load(file_path)
        data_keys = set(list(data.keys()))
        print("The data has the following headers: ")
        print(data_keys)
        fil_keys = set([key for key, value in CHANNEL_MAPPING.items() if value in modality])
        print("Required Headers as per configuration: ")
        print(fil_keys)
        # Initialize a list to hold the data for the current file
        x_data = []
        
        # Extract the data using the standard keys or their mapped alternatives
        req_keys = list(data_keys.intersection(fil_keys))
        print("Extracted Data Headers for preprocessing: ")
        print(req_keys)
        for key in req_keys:
            if key in data:
                x_data.append(np.squeeze(data[key]))
            else:
                alt_key = next((k for k, v in key_mapping.items() if v == key and k in data), None)
                if alt_key:
                    x_data.append(np.squeeze(data[alt_key]))
                else:
                    raise KeyError(f"Neither {key} nor its alternative found in {file_name}")
        
        # Stack the current file's data along the last axis
        x_data_list.append(np.stack(x_data, axis=-1))
        
        # Append the target data
        y_data_list.append(data['target'])

    # Convert lists to numpy arrays
    x_data_array = np.stack(x_data_list, axis=0)
    y_data_array = np.array(y_data_list)

    # Save the stacked data into a new .npz file
    np.savez(output_file, x=x_data_array, y=y_data_array)
    
    print(f"Combined data saved to {output_file}")
    
def process_folders(base_folder, modality):
    
    input_base_folder = os.path.join(base_folder, 'nimhans_np/')
    output_base_folder = os.path.join(base_folder, 'numpy_subjects/')
    
    # Ensure the output directory exists
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # List all directories in the input base folder
    subfolders = [d for d in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, d))]
    subfolders_sorted = sorted(subfolders, key=lambda x: int(x[3:]))
    for folder in subfolders_sorted:
        # Construct input and output paths
        input_folder = os.path.join(input_base_folder, folder)
        output_file = os.path.join(output_base_folder, f'{folder}.npz')
        combine_npz_files(input_folder, output_file, modality)
#         try:
#             # Process the current folder
#             combine_npz_files(input_folder, output_file, modality)
# #             print(f"Successfully processed: {input_folder} -> {output_file}")
#             print(f"Successfully processed: {output_file}")
#         except Exception as e:
#             print(f"Error processing folder {input_folder}: {e}")
