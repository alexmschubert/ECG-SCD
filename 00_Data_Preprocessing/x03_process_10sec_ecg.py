import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Read CSV
    covariate_df = pd.read_feather("covariate_df.feather")
    
    # Output directory
    ECG_DIR = "10_sec_ecgs"
    os.makedirs(ECG_DIR, exist_ok=True)  # Create directory if it doesn't exist
    
    # Iterate through each row in the dataframe
    for idx, row in tqdm(covariate_df.iterrows()):
        # Load the ECG array
        ecg_array = np.load(row["file_path"])[row["npy_index"]]
        
        # Construct output path and save
        output_path = os.path.join(ECG_DIR, f'{row["studyId"]}.npy')
        np.save(output_path, ecg_array)