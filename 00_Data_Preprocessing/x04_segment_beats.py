# This script creates a new directory of segmented ECGs that have been filtered based on a variety of criteria. 
# It also creates a waterfall table that shows how many ECGs remain after each filtering step.
import os
import sys

import pandas
import torch
from tqdm import tqdm

from ekg_scd.datasets import Hallandata
from ekg_scd.segmented_ecg import SegmentedECG, z_distance

BEAT_OUT_DIR = f"ecg_beats/"
WINDOW = (-100, 200)


if __name__ == '__main__':
    print("Starting beat segmentation and filtering")
    # Ensure output directories are created
    if not os.path.exists(BEAT_OUT_DIR):
        os.makedirs(BEAT_OUT_DIR)

    # Load data
    print("Loading data")
    covariate_df = pandas.read_feather("covariate_df.feather")

    #########################################
    # Obtain ECGs for Beat segmentation
    #########################################
    
    ecg_ids = covariate_df['studyId'].values

    # Create a dataloader to iterate over
    data = Hallandata(
        ids=ecg_ids,
        outputs=['scd1'],
        regress=[False],
        train=False,
        aug=None,
        preprocessing=['zero_median'],
        covariate_df="covariate_df.feather",
        x_dir = '10_sec_ecgs/'
    )

    #######################################
    # Beat level filtering
    #######################################

    # Calculate beat-level statistics to filter against
    for i, (ecg, label) in enumerate(tqdm(data, dynamic_ncols=True, file=sys.stdout)):
        cur_id = data.ids[i]
        ecg = SegmentedECG(ecg, cur_id, window_lims=WINDOW)
        try:
            ecg.identify_beats()
            ecg.segment_signal()
            ecg.filter_beats()
            ecg.save_beats(BEAT_OUT_DIR)
        except Exception as e:
            print(f"Error with {cur_id}:  {e}")
            continue

