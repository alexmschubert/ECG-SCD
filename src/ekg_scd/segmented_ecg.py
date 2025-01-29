# This script defines a class `SegmentedECG` with methods for identifying beats in a full length ECG, segmenting it into beats, and filtering out beats that are problematic.
import os

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from pytorch_forecasting.utils import autocorrelation
import torch


class SegmentedECG():
    def __init__(
        self, signal, id, window_lims, identification_lead=0, sampling_rate=500, 
        range_cutoff=0.5, sd_cutoff=0.06, autocorr_cutoff=0.75
    ):
        self.signal = signal
        self.id = id
        self.window_lims = window_lims
        self.beats = []
        self.beat_windows = []
        self.identification_lead = identification_lead
        self.sampling_rate = sampling_rate
        self.range_cutoff = range_cutoff
        self.sd_cutoff = sd_cutoff
        self.autocorr_cutoff = autocorr_cutoff
        self.signal_processed = False

    def identify_beats(self):
        self.signals_df, self.rpeaks = nk.ecg_process(self.signal[self.identification_lead], sampling_rate=self.sampling_rate)

    def segment_signal(self):
        if self.beat_windows == []:
            self.get_beat_windows()

        # Clip signals into beats using the beat windows
        self.beats = [self.signal[:, l:r].clone().detach() for (l, r) in self.beat_windows]

    def drop_beats(self, beat_indices: list):
        """
        Drops beats from the ECG signal.
        """
        self.beats = [beat for i, beat in enumerate(self.beats) if i not in beat_indices]
        self.beat_windows = [window for i, window in enumerate(self.beat_windows) if i not in beat_indices]

    def get_beat_windows(self):
        """
        Generates beat windows based on R-peaks and a specified window range.
        """
        left, right = self.window_lims
        self.beat_windows = [(rpeak + left, rpeak + right) for rpeak in self.rpeaks['ECG_R_Peaks']]
        
        # Drop windows which would be out of bounds
        self.beat_windows = [window for window in self.beat_windows if window[0] >= 0 and window[1] <= len(self.signal[0])]

    def filter_beats(self):
        """
        Filters out beats that:
        - Are the first or last beat in the ECG signal
        - Contain more than one of any kind of peak (i.e. overlap with another beat)
        - Have abnormal range, sd, or autocorrelations
        """
        if self.beats == []:
            self.segment_signal()

        # Drop first and last beats
        self.beats = self.beats[1:-1]
        self.beat_windows = self.beat_windows[1:-1]

        # Identify problematic beat windows
        self.duplicate_peaks = self.identify_duplicate_peaks()
        
        # Check exclusion criteria
        # get per lead range
        max_vals = [torch.max(beat, dim=1)[0] for beat in self.beats]
        min_vals = [torch.min(beat, dim=1)[0] for beat in self.beats]
        ecg_range_per_lead = [max_val - min_val for max_val, min_val in zip(max_vals, min_vals)]
        mean_range = [torch.mean(r).item() for r in ecg_range_per_lead]
        self.low_range_flag = [r <= self.range_cutoff for r in mean_range]

        # Get mean sd
        beat_sds = [round(torch.mean(torch.std(beat, dim=1)).item(), 4) for beat in self.beats] # mean of per lead SD
        self.low_sd_flag = [sd <= self.sd_cutoff for sd in beat_sds]

        # Check autocorrelation
        beat_autocorr_per_lead = [autocorrelation(beat, dim=1)[:, :5] for beat in self.beats]

        autocorr_beats = [torch.nanmean(torch.nanmean(autocorrs, dim=1), dim=0).item() for autocorrs in beat_autocorr_per_lead]
        self.low_autocorr_flag = [autocorr <= self.autocorr_cutoff for autocorr in autocorr_beats]

        # Flag beats that have any of the above issues
        flags_zip = zip(self.duplicate_peaks, self.low_range_flag, self.low_sd_flag, self.low_autocorr_flag)
        self.beat_flags = [dup or low_r or low_sd or low_autocorr for dup, low_r, low_sd, low_autocorr in flags_zip]

        # Update beats and beat windows
        self.filtered_beats = [beat for i, beat in enumerate(self.beats) if not self.beat_flags[i]]
        self.filtered_beat_windows = [window for i, window in enumerate(self.beat_windows) if not self.beat_flags[i]]

    def identify_duplicate_peaks(self):
        """
        Identifies windows that contain more than one of any kind of peak (P, Q, R, S, T).
        """
        peak_types = ['ECG_R_Peaks'] #'ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks' 
        #print(self.rpeaks)
        return [any(sum(1 for peak in self.rpeaks[peak_type] if l < peak < r) > 1 for peak_type in peak_types) for (l, r) in self.beat_windows]

    def process_signal(self, scaling_factor=None):
        """
        Processes the ECG signal, normalizing each lead so that it is centered on zero and so that
        """
        if not self.signal_processed:
            # Convert to mV if needed
            if scaling_factor is not None:    
                self.signal = self.signal/float(scaling_factor)

            # Zero-center signal, using the median of the middle 80% of the signal to avoid weirdness around the start and end
            ten_percent_idx = int(self.signal.shape[1]*0.1)
            median = np.median(self.signal[:, ten_percent_idx:-ten_percent_idx], axis=1).reshape(12, 1)
            self.signal = self.signal - median
            self.signal_processed = True
        else:
            raise ValueError("Signal has already been processed.")
    
    def plot_ecg(self, windows=False, filtered=False):
        """
        Plots the full ECG signal, optionally with beat windows highlighted.
        """
        fig, axes = plt.subplots(nrows=signal.shape[0], ncols=1, sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(self.signal[i])
            if i == 0:
                ax.set_title(f'ECG {self.id if self.id is not None else ""}')
            if windows:
                if filtered:
                    windows = self.filtered_beat_windows
                else:
                    windows = self.beat_windows
                for (l, r) in windows:
                    ax.axvspan(l, r, alpha=0.2, color='red')
        ax.set_title(f'ECG {self.id}')
        ax.set_xlabel(f'Window: {sanitize_str(str(self.window_lims))}')
        return fig, ax
    
    def save_beats(self, out_dir, filtered=True):
        """
        Saves the beats to a directory.
        """
        if filtered:
            beats = self.filtered_beats
        else:
            beats = self.beats

        if os.path.exists(f"{out_dir}/{self.id}"):
            # Delete existing directory and its contents
            for file in os.listdir(f"{out_dir}/{self.id}"):
                os.remove(f"{out_dir}/{self.id}/{file}")

        os.makedirs(f"{out_dir}/{self.id}", exist_ok=True)

        for i, beat in enumerate(beats):
            np.save(os.path.join(out_dir, self.id, f"{self.id}_{i}.npy"), beat.numpy())
            
def z_distance(ecg_data, median_heartbeat, std_heartbeat):
    distances = []
    for heartbeat in ecg_data:
        # Compute the distance
        dist = np.linalg.norm((heartbeat - median_heartbeat) / std_heartbeat)
        distances.append(dist)
    return distances