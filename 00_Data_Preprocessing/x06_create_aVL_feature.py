import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from ekg_scd.helpers import preproc
from biosppy.signals import ecg as bioecg

def detect_s_peaks(ecg_data_avl_raw, first_diff_smooth_raw, sampling_rate=500, speak_window=61, threshold=0.01, consecutive_points=3, window_size=20):
    rpeaks, = bioecg.hamilton_segmenter(ecg_data_avl_raw, sampling_rate=sampling_rate)
    
    speaks = []
    for rpeak in rpeaks:
        s_found = False
        for i in range(rpeak - window_size//2 + 1, min(rpeak - window_size//2 + speak_window, len(ecg_data_avl_raw) - 1)):
            if all(abs(first_diff_smooth_raw[j]) < threshold for j in range(i, min(i + consecutive_points, len(first_diff_smooth_raw)))):
                speaks.append(i + window_size//2)
                s_found = True
                break
        if not s_found:
            speaks.append(min(rpeak - window_size//2 + speak_window, len(ecg_data_avl_raw) - 1))

    return rpeaks, speaks

def compute_rs_differences_and_rpeak_features(ecg_data_avl):
    
    # Min-max normalize the waveform
    ecg_data_avl = (ecg_data_avl - np.min(ecg_data_avl)) / (np.max(ecg_data_avl) - np.min(ecg_data_avl))
    
    # Apply smoothing
    first_diff = np.diff(ecg_data_avl)
    second_diff = np.diff(first_diff)
        
    # Detect R and S peaks
    rpeaks, speaks = detect_s_peaks(ecg_data_avl, first_diff, window_size=1)

    # Calculate aVL_rs_diff_new and aVL_rs_diff_2_new
    rs_diffs = []
    rs_diffs_2 = []
    for r, s in zip(rpeaks, speaks):
        r = r - 1//2
        s = s - 1//2
        rs_diff = np.mean(np.abs(first_diff[r:s]))
        rs_diff_2 = np.mean(np.abs(second_diff[r:s-1]))
        rs_diffs.append(rs_diff)
        rs_diffs_2.append(rs_diff_2)

    avg_rs_diff = rs_diffs[0] if rs_diffs else 0
    avg_rs_diff_2 = rs_diffs_2[0] if rs_diffs_2 else 0
    
    # Average the features across all beats
    if len(rpeaks) == 0:
        return [np.NaN] * 2  # Return NaN for all features if no R peaks are found
    else:
        return [
            avg_rs_diff, avg_rs_diff_2
        ]


# Function to process each entry
def process_entry(entry):
    idx, scaling_factor = entry['studyId'], entry['scaling_factor']
    file_path = f"10_sec_ecgs/{idx}.npy"
    results = {
                'aVL_rs_diff': np.NaN,
                'aVL_rs_diff_2': np.NaN,
               }
    
    if os.path.exists(file_path):
        
        #Load ECG
        ########################################
        ecg = np.load(file_path)
        ecg = ecg / scaling_factor
        ecg = preproc.apply_preprocessing(ecg, preprocessing=['zero_mode'], to_mV=False)
        sampling_rate = 500
        
        #Obtain median beat
        ########################################
        rpeaks, = bioecg.hamilton_segmenter(ecg[4], sampling_rate=sampling_rate)  # Adjust sampling rate

        # Define a window around the R-peak to extract beats (adjust window size as needed)
        window_size = 500  
        half_window = window_size // 2

        # Extract individual beats
        beats = []
        for r_peak in rpeaks:
            start = max(0, r_peak - half_window)
            end = min(len(ecg[0]), r_peak + half_window)
            beat = ecg[:, start:end]
            # Pad beat if necessary to ensure shape (12, 500)
            if beat.shape[1] < window_size:
                pad_width = window_size - beat.shape[1]
                beat = np.pad(beat, ((0, 0), (0, pad_width)), 'constant', constant_values=0)
            
            beats.append(beat)

        # Ignore the first and last beat to avoid edge effects
        beats = beats[1:-1]

        # Calculate the median beat from the segmented beats
        median_beat = np.median(np.array(beats), axis=0)
        abs_median_beat = np.abs(median_beat)
        
        try:
            # Calculate the new features for aVL lead
            ecg_data_avl = median_beat[4] 
            feature_values = compute_rs_differences_and_rpeak_features(ecg_data_avl)
            
            for key, value in zip(results.keys(), feature_values):
                results[key] = value
            
        except Exception as e:
            print(f"Error processing new features for studyId {idx}: {e}")

    return results

def main():
    covariate_df = pd.read_feather(f"covariate_df.feather")
    entries = covariate_df[['studyId', 'scaling_factor']].to_dict('records')

    num_processes = max(1, cpu_count() // 2)
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_entry, entries), total=len(entries)))

    # Update dataframe with all results
    for key in results[0].keys():  # Assuming all results dictionaries have the same keys
        covariate_df[key] = [result[key] for result in results]

    covariate_df.to_feather("covariate_df.feather")

if __name__ == "__main__":
    main()
