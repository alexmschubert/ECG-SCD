""" functions for preprocessing raw EKG signal

e.g. de-trending EKG leads using gaussian smoothing, squashing the range into 
[-1, 1]preprocess EKG signal functions

primarily used in dataset class definition
"""
import math
import pickle
from platform import machine
import random
import wave

import biosppy
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
from biosppy.signals import ecg as becg
from scipy import ndimage
from scipy.signal import butter, filtfilt


def segment_to_channels(waveform, fs_training, n_template_beats=8, length=600):
    """Segment EKG into beats and stack to form n_template_beats*n_channels total channels"""

    def segment_beats(waveform, fs_training=1000):

        lead_II = waveform[1].numpy()

        try:
            rpeaks = nk.ecg_findpeaks(lead_II, sampling_rate=fs_training, method="neurokit")["ECG_R_Peaks"]
        except:
            return None, None

        if len(rpeaks) < 8:
            return None, None
        else:
            _, waves_peak = nk.ecg_delineate(lead_II, rpeaks, sampling_rate=fs_training, method="dwt")
            qrs_on = [qrs for qrs in waves_peak["ECG_R_Onsets"]]
            rpeaks = [rpeaks[i] for i in range(len(rpeaks)) if ~np.isnan(qrs_on[i])]
            qrs_on = [qrs for qrs in qrs_on if ~np.isnan(qrs)]

            q_on = [
                x - int(fs_training / 10) for x in qrs_on
            ]  # Get a bit more room by shifting back 1/10th of a second

            beat_idx = [[q_on[i], q_on[i + 1]] for i in range(len(q_on) - 1)]

            return beat_idx, rpeaks

    def fill_template(waveform, beat_idx, rpeaks, template):
        """Fill beat template with beat-segmented signal,
        locating r peaks in the middle"""

        if beat_idx is None:
            raise ValueError("segfault")

        (n_total_channels, length) = template.shape
        n_ecg_channels = 12
        n_beats = len(beat_idx)
        n_template_beats = int(n_total_channels / n_ecg_channels)
        midpoint = length // 2

        # Shorten n_beats if we have too many
        if n_template_beats < n_beats:
            beat_idx = beat_idx[:n_template_beats]

        for i, idx in enumerate(beat_idx):
            start_idx, end_idx = idx[0], idx[1]
            on_to_peak_distance = rpeaks[i] - start_idx
            peak_to_off_distance = end_idx - rpeaks[i]

            if on_to_peak_distance > midpoint:
                start_idx = rpeaks[i] - midpoint
            if peak_to_off_distance > midpoint:
                end_idx = rpeaks[i] + midpoint

            start_loc, end_loc = midpoint - on_to_peak_distance, midpoint + peak_to_off_distance

            for c in range(waveform.shape[0]):
                channel = waveform[c]

                beat = channel[start_idx:end_idx]

                template[i * 12 + c][start_loc:end_loc] = beat

        return template

    n_ecg_channels = waveform.shape[0]

    beat_idx, rpeaks = segment_beats(waveform, fs_training)

    template = np.zeros((n_template_beats * n_ecg_channels, length))

    waveform = fill_template(waveform, beat_idx, rpeaks, template)

    return waveform


def detrend_raw(Yraw, ts):
    """univariate detrend"""
    sampfreq = 1.0 / (ts[1] - ts[0])

    # detrend w/ a gaussian kernel
    Ykern = ndimage.gaussian_filter1d(Yraw, sigma=sampfreq / 4.0)
    Y = (Yraw - Ykern) + Ykern.mean()
    return Y


def cut(X, sig_length):
    """Standardize length: Cut down long ECGs"""

    if X.shape[1] > sig_length:
        X = X[:, :sig_length]

    return X


def random_cut(X, sig_length):
    start_idx = random.randint(0, X.shape[1] - sig_length)

    if X.shape[1] > sig_length:
        X = X[:, start_idx : start_idx + sig_length]

    return X


def pad(X, sig_length):
    """Standardize length: Lengthen short ECGs"""

    if X.shape[1] < sig_length:
        padding = np.zeros(
            (
                X.shape[0],
                sig_length - X.shape[1],
            )
        )
        X = np.concatenate([X, padding], axis=1)

    return X


def detrend_raw_multi_lead(Yraw, ts):
    return np.array([detrend_raw(Yraw[i], ts) for i in range(Yraw.shape[0])])


def preproc_raw(Yraw, ts):
    """preproc a univariate example"""
    # detrend w/ a gaussian kernel
    Y = detrend_raw(Yraw, ts)

    # re-scale so that the total range is between [-1 and 1]
    ymax, ymin = Y.max(), Y.min()
    Yproc = 2 * (Y - ymin) / (ymax - ymin) - 1
    return Yproc, (ymin, ymax)


def preproc_raw_multi_lead(Yraw, ts):
    """preproc a C-lead example"""
    return np.array([preproc_raw(Yraw[i], ts)[0] for i in range(Yraw.shape[0])])


def segment_beat(X, tgrid, alg="christov-aligned", grid_len=100, detrend=True):
    if alg == "christov":
        samp_rate = 1.0 / (tgrid[1] - tgrid[0])
        rpeaks = becg.christov_segmenter(X[0], samp_rate)
        bmat = np.dstack(
            [
                becg.extract_heartbeats(Xr, rpeaks=rpeaks["rpeaks"], sampling_rate=samp_rate, before=0.3, after=0.4)[
                    "templates"
                ]
                for Xr in X
            ]
        )
        return bmat

    elif alg == "christov-aligned":
        # first detect R peaks (using preprocessed first lead)
        samp_rate = 1.0 / (tgrid[1] - tgrid[0])
        Xfix = preproc_raw(X[0], tgrid)[0]
        rpeaks = becg.christov_segmenter(Xfix, samp_rate)

        # then extract irregularly lengthed beats and resample
        if detrend:
            Xdet = detrend_raw_multi_lead(X, tgrid)
        else:
            Xdet = X

        # actually extract beats
        bmat, lens = extract_irregular_beats(Xdet, rpeaks=rpeaks["rpeaks"], grid_len=grid_len)
        return bmat, lens


def extract_irregular_beats(X, rpeaks, grid_len):
    # start points are 1/3 cycle before the rpeak, ignore first one
    lens = np.diff(rpeaks)
    if len(lens) == 0:
        return np.array([]), np.array([])
    starts = rpeaks[:-1] + np.floor((2.0 / 3.0) * lens).astype(int)
    ends = starts + lens
    if ends[-1] > X.shape[1]:
        starts, ends = starts[:-1], ends[:-1]

    # segment each beat and interpolate to a fixed grid
    bgrid = np.linspace(0, 1, grid_len)
    beatmat = np.zeros((len(starts), X.shape[0], grid_len))
    for n, (s, e) in enumerate(zip(starts, ends)):
        beat = X[:, s:e]
        bg = np.linspace(0, 1, beat.shape[1])
        for c in range(X.shape[0]):
            beatmat[n, c, :] = np.interp(bgrid, bg, beat[c])

    return beatmat, ends - starts


def apply_amplitude_scaling(X, y):
    """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
    if y["rpeaks"]:
        # for channel_rpeaks in y['rpeaks']:
        if y["rpeaks"][0]:
            # remove baseline
            for i in range(12):
                X[:, 0] -= np.median(X[:, 0])
            return X / np.median(X[y["rpeaks"][0], 0] + 0.001)

    for i in range(12):
        X[:, 0] -= np.median(X[:, 0])

    return X / (X[:, 0].std() + 0.001)


def FIR_filter(X):
    sampling_rate = 1000
    filter_order = 0.3

    order = int(filter_order * sampling_rate)
    for i in range(X.shape[0]):
        X[i], _, _ = biosppy.tools.filter_signal(
            signal=X[i], ftype="FIR", band="bandpass", order=order, frequency=[3, 45], sampling_rate=sampling_rate
        )
    return X


# Preprocessing (applied both to training and validation) ------
def apply_preprocessing(waveform, ehr = None, preprocessing=None, to_mV=False, mode=None):

    options = ["project", "zero_median", "zero_mode", "clamp", "scale", 'calibration', 'butter_bandpass']
    for p in preprocessing:
        if p not in options:
            raise ValueError(f"Invalid preprocessing method provided. Available methods are {options}")

    if "project" in preprocessing:
        waveform = _project(waveform)
    
    if "zero_mode" in preprocessing:
        waveform = _zero_mode(waveform, mode=mode)
    
    if "butter_bandpass" in preprocessing:
        waveform = _apply_butter_bandpass_filter(waveform, lowcut=0.2, highcut=25.0, fs=500.0, order=4)

    if "zero_median" in preprocessing:
        waveform = _zero_median(waveform)

    if "clamp" in preprocessing:
        waveform = _clamp(waveform)
    
    if "scale" in preprocessing:
        if to_mV:
            scale_factors = pickle.load(open("path/to/ECG_scale_factors_mV.pkl", "rb"))
            scale_factors = scale_factors * (waveform.shape[0] // 12)  # accounting for stacked case
            waveform = _scale(waveform, scale_factors=scale_factors)
        else:
            scale_factors = pickle.load(open("path/to/ECG_scale_factors.pkl", "rb"))
            scale_factors = scale_factors * (waveform.shape[0] // 12)  # accounting for stacked case
            waveform = _scale(waveform, scale_factors=scale_factors)
    
    if 'calibration' in preprocessing and isinstance(ehr, pd.DataFrame):
        machine_id = ehr['sourceId'].item()
        if machine_id in [6,21]:
            waveform = waveform/200
        elif machine_id == 12:
            waveform = waveform/1000

    return waveform


def _resample(X, fs, fs_resampled, sig_length):
    new_X = [[] for i in range(X.shape[0])]
    for c in range(X.shape[0]):
        new_X[c] = nk.signal_resample(X[c], sampling_rate=fs, desired_sampling_rate=fs_resampled, method="numpy")[
            :sig_length
        ]

    new_X = np.array(new_X)

    return new_X


def _scale(X, scale_factors):
    for c in range(X.shape[0]):
        X[c] = X[c] / scale_factors[c] * 100

    return X


def _project(X):
    # NOTE: leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    angles = [0.0, 60.0, 120.0, -150.0, -30.0, 90.0]
    # This implementation assumes order
    for c, angle in enumerate(angles):

        degreediff = angles[1] - angle
        multiplier = math.cos(math.pi * degreediff / 180)
        multiplier = round(multiplier, 6)
        X[c] = X[c] * multiplier

    return X


def _zero_median(X):
    for channel in range(X.shape[0]):
        s = X[channel]
        median = np.median(s)
        X[channel] = X[channel] - median
    return X


def _zero_mode(X, mode=None):
    for channel in range(X.shape[0]):
        s = X[channel]
        if mode == None:
            mode = scipy.stats.mode(s, keepdims=False)[0]
        else:
            mode = mode
        X[channel] = X[channel] - mode
    return X

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _apply_butter_bandpass_filter(data, lowcut=0.2, highcut=25.0, fs=500.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def _clamp(X):
    """Clamp ECG to [0,100]"""
    for channel in range(X.shape[0]):

        s = X[channel]
        maxi = s.max()
        mini = s.min()
        if maxi != mini:
            s = (s - mini) / (maxi - mini)
            X[channel] = s * 100
        else:
            X[channel] = X[channel] + 0.5

    return X


# Augmentations (applied only to training) -------


def apply_augmentation(waveform, meta_data=None, fs=None, aug=None):

    options = ["resample", "scale", "vertical", "horizontal", "noise", "shuffle", 'butter_bandpass']
    for p in aug:
        if p not in options:
            raise ValueError(f"Invalid augmentation method provided. Available methods are {options}")

    # Random resampling to another (reasonable) HR
    if "resample" in aug:
        waveform = _random_resample(waveform=waveform, meta_data=meta_data, fs_training=1000, probability=0.25)
    # Random amplitude scale
    if "scale" in aug:
        waveform = _random_scale(waveform=waveform, probability=0.5)

    if "vertical" in aug:
        waveform = _shift_vertical(waveform=waveform, probability=0.5)

    if "horizontal" in aug:
        waveform = _shift_horizontal(waveform=waveform, sig_length=19000, probability=0.5)

    # Apply synthetic noise
    if "noise" in aug:
        waveform = _add_synthetic_noise(waveform=waveform, fs_training=300, probability=0.15)

    # Permute signal
    if "shuffle" in aug:
        waveform = _shuffle(waveform=waveform, fs_training=1000, probability=0.5)

    return waveform


def _shuffle(waveform, fs_training, probability):
    """Shuffle beats within a given ECG. Maintain reference across leads."""

    ecg = waveform[1]

    try:
        rpeaks = nk.ecg_findpeaks(ecg, sampling_rate=fs_training, method="neurokit")["ECG_R_Peaks"]
    except:
        return waveform

    if len(rpeaks) > 3 and _coin_flip(probability=probability):
        _, waves_peak = nk.ecg_delineate(ecg, rpeaks, sampling_rate=fs_training, method="dwt")
        r_on = [r for r in waves_peak["ECG_R_Onsets"] if not np.isnan(r)]
        q_on = [x - int(fs_training / 10) for x in r_on]

        beat_idx = [[q_on[i], q_on[i + 1]] for i in range(len(q_on) - 1)]
        random.shuffle(beat_idx)

        new_X = []
        for c in range(waveform.shape[0]):
            channel = waveform[c]
            beats = np.array([channel[idx[0] : idx[1]] for idx in beat_idx])
            # initialize random magnitude at which to start modified signal
            start = random.randint(-500, 500)
            new_channel = np.array([])
            for beat in beats:
                # set beat to desired starting magnitude
                beat = (start - beat[0]) + beat
                new_channel = np.concatenate((new_channel, beat))
                # align start of next beat, with a small bit of noise, to the end of current beat
                start = beat[-1] + random.randint(-5, 5)

            new_X.append(new_channel)

        return np.array(new_X)
    else:
        return waveform


def _shift_vertical(waveform, probability):
    """Randomly shift signal some amount up or down."""

    shift = random.randint(-1000, 1000)

    if _coin_flip(probability=probability):
        # add constant signal shift to all points of ECG
        waveform = waveform + shift

    return waveform


def _shift_horizontal(waveform, probability):
    """Randomly shift signal some amount left or right."""

    shift = random.randint(-1000, 1000)

    if _coin_flip(probability=probability):
        if shift < 0:
            waveform = waveform[:, :shift]
        if shift > 0:
            waveform = waveform[:, shift:]

    return waveform


def _random_resample(waveform, meta_data, fs_training, probability):
    """Randomly resample waveform."""
    # TODO: ---
    if _coin_flip(probability=probability):
        try:
            rpeaks = nk.ecg_findpeaks(waveform[1], sampling_rate=fs_training, method="neurokit")["ECG_R_Peaks"]
        except:
            return waveform

        rrint_mean = np.mean([rpeaks[i + 1] - rpeaks[i] for i in range(len(rpeaks) - 1)])
        hr = (fs_training / rrint_mean) * 60

        # Get waveform duration
        duration = waveform.shape[1] / fs_training

        # Get new heart rate
        hr_new = int(hr * np.random.uniform(1, 1.25))
        if hr_new > 300:
            hr_new = 300
        elif hr_new < 40:
            hr_new = 40
        else:
            pass

        # Get new duration
        duration_new = duration * hr / hr_new

        # Get number of samples
        samples = int(duration_new * fs_training)

        # Resample waveform
        waveform = signal.resample_poly(waveform, samples, waveform.shape[1], axis=0).astype(np.float32)

        return waveform
    else:
        return waveform


def _random_scale(waveform, probability):
    """Apply random scale factor between 0.25 and 3 to the waveform amplitudes."""
    # Get random scale factor
    scale_factor = random.uniform(0.25, 3.0)

    if _coin_flip(probability):
        return waveform * scale_factor
    return waveform

def _add_synthetic_noise(waveform, fs_training, probability):
    """Add different kinds of synthetic noise to the signal."""
    
    fs_options = [10, 25, 75, 125]
    # Randomly select one item from the list
    fs_training = random.choice(fs_options)
    
    for idx in range(waveform.shape[0]):
        if waveform.shape[0]==1:
            waveform = _generate_high_frequency_noise(waveform=waveform, fs=fs_training, probability=0.25)
            waveform = _generate_pulse_noise(waveform=waveform, probability=0.25)
        else:
            waveform[idx] = _generate_high_frequency_noise(waveform=waveform[idx], fs=fs_training, probability=probability)
            waveform[idx] = _generate_gaussian_noise(waveform=waveform[idx], probability=probability)
            waveform[idx] = _generate_pulse_noise(waveform=waveform[idx], probability=probability)
    return waveform


def _generate_high_frequency_noise(waveform, fs, probability=0.5):
    """Adds high frequency sinusoidal noise to the input signal."""
    if _coin_flip(probability):
         
        # Generate time array
        time = np.arange(waveform.shape[1]) / fs

        # Add noise
        waveform += random.uniform(0.001, 0.2) * np.sin(
            2 * np.pi * random.uniform(0.05, 0.2) * time #+ random.uniform(0, 0.3)
        )

    return waveform


def _generate_gaussian_noise(waveform, probability=0.5):
    """Adds white noise noise to the input signal."""
    if _coin_flip(probability):
        waveform += np.random.normal(loc=0.0, scale=random.uniform(0.01, 0.15), size=waveform.shape[1])

    return waveform


def _generate_pulse_noise(waveform, probability=0.5):
    """Adds gaussian pulse to the input signal."""
    if _coin_flip(probability):

        # Get pulse
        pulse = signal.gaussian(int(waveform.shape[1] * random.uniform(0.05, 0.010)), std=random.randint(50, 200))
        pulse = np.diff(pulse)

        # Get remainder
        remainder = waveform.shape[1] - len(pulse)
        if remainder >= 0:
            left_pad = int(remainder * random.uniform(0.0, 1.0))
            right_pad = remainder - left_pad
            pulse = np.pad(pulse, (left_pad, right_pad), "constant", constant_values=0)
            pulse = pulse / pulse.max()

        waveform += pulse * random.uniform(waveform.max() * 1.5, waveform.max() * 2)

    return waveform

def _coin_flip(probability):
    if random.random() < probability:
        return True
    return False
