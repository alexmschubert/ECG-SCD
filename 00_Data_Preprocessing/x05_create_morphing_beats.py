# This script creates a new directory of segmented ECGs that have been filtered based on a variety of criteria. 
# It also creates a waterfall table that shows how many ECGs remain after each filtering step.
import os
import sys

import pandas
import torch
from tqdm import tqdm

from ekg_scd.datasets import Hallandata
from ekg_scd.segmented_ecg import SegmentedECG, z_distance

BEAT_OUT_DIR = f"ecg_beats_morphing/"
WATERFALL_OUTPUT_DIR = "ecg_beats_morphing"
N_MEDIAN_SAMPLE = 10000 # How many beats to use when calculating global median beat
FLAGS = ["ef_below_35", "VTVF", "ICD_VT", "ICD_VF", "Phi_VT", "ST_elevation",
        "ST_depression", "qtc500", "LBBB", "antiarr", "deltawave",
        "CHF_Cardiomyopathy", "Hypertension", "CAD", "Diabetes",
        "Hyperlipidemia", "TROPT40", "Acute_MI", "Old_MI",
        "heartRate_above_120", "pDuration_above_120", "pDuration_below_20",
        "qrsDuration_above_120", "qrsDuration_below_20", "qtcb_abnormal", "qtcf_abnormal",
        "rrInterval_abnormal"]
WINDOW = (-100, 200)
DISTANCE_THRESHOLD = 100 # Threshold for filtering beats based on distance from median beat


if __name__ == '__main__':
    print("Starting beat segmentation and filtering")
    # Ensure output directories are created
    if not os.path.exists(BEAT_OUT_DIR):
        os.makedirs(BEAT_OUT_DIR)

    if not os.path.exists(WATERFALL_OUTPUT_DIR):
        os.makedirs(WATERFALL_OUTPUT_DIR)

    # Load data
    print("Loading data")
    covariate_df = pandas.read_feather("covariate_df.feather")

    # Create flags that don't yet exist
    covariate_df["ef_below_35"] = covariate_df["most_recent_ef"] < 35
    print(covariate_df["VTVF"])
    covariate_df["VTVF"] = covariate_df["VTVF"].astype(int)
    covariate_df["heartRate_above_120"] = covariate_df["heartRate"] > 120
    covariate_df["pDuration_above_120"] = covariate_df["pDuration"] > 120
    covariate_df["pDuration_below_20"] = covariate_df["pDuration"] < 20
    covariate_df["qrsDuration_above_120"] = covariate_df["qrsDuration"] > 120
    covariate_df["qrsDuration_below_20"] = covariate_df["qrsDuration"] < 20
    covariate_df["qtcb_abnormal"] = (covariate_df["qtcb"] > 450) | (covariate_df["qtcb"] < 350)
    covariate_df["qtcf_abnormal"] = (covariate_df["qtcf"] > 450) | (covariate_df["qtcf"] < 350)
    covariate_df["rrInterval_abnormal"] = (covariate_df["rrInterval"] > 1500) | (covariate_df["rrInterval"] < 350)

    # Create dictionary to hold observation counts
    obs_dict = {}
    obs_dict['total'] = len(covariate_df)

    ###################################
    # ECG level filtering
    ###################################

    # Create a second dataframe to filter
    covariate_df_filtered = covariate_df.copy()

    # Filter on heartrate
    covariate_df_filtered = covariate_df_filtered[(covariate_df_filtered['heartRate'] >= 30) & (covariate_df_filtered['heartRate'] <= 90)]
    obs_dict['heartRate'] = len(covariate_df_filtered)

    # Filter on age
    covariate_df_filtered = covariate_df_filtered[(covariate_df_filtered['age'] >= 18) & (covariate_df_filtered['age'] <= 80)]
    obs_dict['age'] = len(covariate_df_filtered)

    # Filter on other flags
    for flag in FLAGS:
        covariate_df_filtered = covariate_df_filtered[covariate_df_filtered[flag] == 0]
        obs_dict[flag] = len(covariate_df_filtered)

    # Get remaining IDs
    ecg_ids = covariate_df_filtered['studyId'].values

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

    ###################################
    # Beat level filtering
    ###################################

    # Calculate beat-level statistics to filter against
    ecgs = []
    for i, (ecg, label) in enumerate(tqdm(data, dynamic_ncols=True, file=sys.stdout)):
        cur_id = data.ids[i]
        ecg = SegmentedECG(ecg, cur_id, window_lims=WINDOW)
        try:
            ecg.identify_beats()
            ecg.segment_signal()
            ecg.filter_beats()
            # Add beats that pass filters to ecgs
            for beat in ecg.filtered_beats:
                ecgs.append(beat)
            if len(ecgs) > N_MEDIAN_SAMPLE:
                break
        except Exception as e:
            print(f"Error with {cur_id}:  {e}")
            continue


    # Calculate median and standard deviation for each lead
    # Note that beats should be aligned, since windows are defined relative to the R-peak
    median_beat = torch.median(torch.stack(ecgs), dim=0).values
    std_beat = torch.std(torch.stack(ecgs), dim=0)


    # Filter on beat segmentation and signal issues
    beat_tracking = {
        'n_beats_theory': [],
        'n_beats_found': [],
        'n_post_segmentation': [],
        'n_post_range_var_auto': [],
        'n_post_median_filter': [],
    }
    
    for i, (ecg, label) in enumerate(tqdm(data, dynamic_ncols=True, file=sys.stdout)):
        cur_id = data.ids[i]

        # Set up SegmentedECG object
        ecg = SegmentedECG(ecg, cur_id, window_lims=WINDOW)

        # Identify beats and segment signal
        try:
            ecg.identify_beats()
            ecg.segment_signal()

            # Filter beats (first, last, and any that contain peaks from other beats)
            ecg.filter_beats()

            # Drop beats that are too different from the median beat
            distances = z_distance(torch.stack(ecg.filtered_beats), median_beat, std_beat)
            n_post_median_filter = len([d for d in distances if d < DISTANCE_THRESHOLD])
            n_beats_found = len(ecg.beats) + 2 # We always drop the first and last, so add 2
            n_post_segmentation = len(ecg.beats) - sum(ecg.duplicate_peaks)
            n_post_range_var_auto = len(ecg.filtered_beats)
            ecg.save_beats(BEAT_OUT_DIR)

        except Exception as e:
            print(f"Error with {cur_id}:  {e}")
            n_beats_found = 0
            n_post_segmentation = 0
            n_post_range_var_auto = 0
            n_post_median_filter = 0
            continue
        
        # How many beats should there be based on heart rate?
        n_beats_theory = round(covariate_df[covariate_df['studyId'] == cur_id]['heartRate'].item() / 60 * 10)

        beat_tracking['n_beats_theory'].append(n_beats_theory)
        beat_tracking['n_beats_found'].append(n_beats_found)
        beat_tracking['n_post_segmentation'].append(n_post_segmentation)
        beat_tracking['n_post_range_var_auto'].append(n_post_range_var_auto)
        beat_tracking['n_post_median_filter'].append(n_post_median_filter)


    # Calculate results
    n_beats_theory = sum(beat_tracking['n_beats_theory'])
    n_beats_found = sum(beat_tracking['n_beats_found'])
    n_beats_segmentation = sum(beat_tracking['n_post_segmentation'])
    n_ecg_segmentation = sum([n > 0 for n in beat_tracking['n_post_segmentation']])
    n_beats_range_var_auto = sum(beat_tracking['n_post_range_var_auto'])
    n_ecg_range_var_auto = sum([n > 0 for n in beat_tracking['n_post_range_var_auto']])
    n_beats_median_filter = sum(beat_tracking['n_post_median_filter'])

    ###################################
    # Generate table
    ###################################

    table_data = {
        'Filters': ['Initial', 'Heart rate', 'Age'] + FLAGS + [ 'Segmentation', 'Range, variance, and autocorrelation', 'Median beat'], #'Drop missing files',
        'ECG count': [obs_dict['total'], obs_dict['heartRate'], obs_dict['age']] + [obs_dict[f] for f in FLAGS] + [round(obs_dict['rrInterval_abnormal'] * n_ecg_segmentation / i), round(obs_dict['rrInterval_abnormal'] * n_ecg_range_var_auto / i), round(obs_dict['rrInterval_abnormal'] * n_beats_median_filter / i)],
    }

    print(table_data)
    table_df = pandas.DataFrame(table_data)
    
    table_df.to_csv(f'{WATERFALL_OUTPUT_DIR}/x05_waterfall_table_results.csv', index=False)
