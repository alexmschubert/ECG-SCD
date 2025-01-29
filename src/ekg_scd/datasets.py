""" Define the Hallandata dataset class. """

import pickle
import pathlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ekg_scd.helpers import preproc

from scipy.signal import resample
import random


class Hallandata(Dataset):
    def __init__(
        self,
        ids,
        outputs,
        regress,
        train=False,
        aug=None,
        preprocessing=None,
        sig_length=5000,
        calibration_sig_length=500,
        to_mV = True,
        covariate_df = "path/to/patient_data",
        covariate_conditioning = None,
        conditioning_mean = 0,
        conditioning_std = 1,
        x_dir = "path/to/ecgs",
        downsample = None
    ):

        self.ids = [str(x) for x in ids]
        self.outputs = outputs
        self.regress = regress
        self.train = train
        self.aug = aug
        self.preprocessing = preprocessing
        self.sig_length = sig_length
        self.calibration_sig_length = calibration_sig_length
        self.to_mV = to_mV
        self.covariate_conditioning = covariate_conditioning
        self.conditioning_mean = conditioning_mean
        self.conditioning_std = conditioning_std
        self.x_dir = x_dir
        self.downsample = downsample

        if isinstance(covariate_df, str) or isinstance(covariate_df, pathlib.Path):
            self.covariate_df = pd.read_feather(covariate_df)
        else:
            self.covariate_df = covariate_df
        if self.covariate_conditioning is not None:
            self.conditioning = self.covariate_df[self.covariate_conditioning]

        new_cols = ['scd3mo_6mo', 'scd1_2', 'scd6mo_1', 'scd3mo_6mo_x', 'scd1_2_x', 'scd6mo_1_x', 'scd3mo_6mo_y', 'scd1_2_y', 'scd6mo_1_y']
        for col in new_cols:
            if col in self.covariate_df.columns:
                self.covariate_df = self.covariate_df.drop(col, axis=1)
                print('col dropped')

        self.covariate_df['scd3mo_6mo'] = (self.covariate_df['scd6mo'] == 1) & (self.covariate_df['scd3mo'] == 0)
        self.covariate_df['scd6mo_1'] = (self.covariate_df['scd6mo'] == 0) & (self.covariate_df['scd1'] == 1)
        self.covariate_df['scd1_2'] = (self.covariate_df['scd1'] == 0) & (self.covariate_df['scd2'] == 1)

        # Coerce ids to string
        self.ids = [str(id) for id in self.ids]
        self.covariate_df['studyId'] = self.covariate_df['studyId'].astype(str)

        self.covariate_df = self.covariate_df[self.covariate_df['studyId'].isin(self.ids)]
        self.covariate_df['studyId2'] = self.covariate_df['studyId']

        # Set 'studyId' as the DataFrame index
        self.covariate_df = self.covariate_df.set_index('studyId2')

        # Reorder DataFrame rows to match the order of 'ids'
        common_ids = [id for id in self.ids if id in self.covariate_df.index]
        if len(common_ids) != len(self.ids):
            print("Warning: not all ids found in covariate_df. Dropping missing ids.")
        self.ids = common_ids
        self.covariate_df = self.covariate_df.loc[self.ids]

        # Reset the index of the new DataFrame.
        self.covariate_df.reset_index(drop=True, inplace=True)

        if self.covariate_conditioning is not None:
            self.conditioning_data = self.covariate_df[self.covariate_conditioning]
            self.conditioning = [[self.conditioning_data[self.covariate_conditioning[x]][i] for x in range(len(self.covariate_conditioning))] for i in range(len(self.ids))] #potentially to be cleaned up
            if self.train:
                self.conditioning_mean = self.conditioning_data.mean()
                self.conditioning_std = self.conditioning_data.std()  
        self.targets = [[self.covariate_df[self.outputs[x]][i] for x in range(len(self.outputs))] for i in range(len(self.ids))] #potentially to be cleaned up
        self.covariate_df = self.covariate_df[['studyId', 'scaling_factor', 'scd1']]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.covariate_conditioning is not None:
            data, covars, label = self.load_data(idx)
            return data, covars, label
        else:
            data, label = self.load_data(idx)
            return data, label

    def load_data(self, idx):
        # get studyId
        ecg_id = self.ids[idx]
        
        data = np.load(f"{self.x_dir}/{ecg_id}.npy")
        
        #convert waveforms to mV
        if self.to_mV:
            scaling_factor = self.covariate_df[self.covariate_df['studyId']==ecg_id]['scaling_factor'].values
            if len(scaling_factor)==0 or np.isnan(scaling_factor):
                scaling_factor = 1 # do not scale if scaling factor is missing

            scaling_factor = float(scaling_factor)
            data = data/scaling_factor
            
        label = self.targets[idx]

        if self.calibration_sig_length is not None:
            data = preproc.cut(data, data.shape[1] - self.calibration_sig_length)

        data = self.process_X(X=data, y=label)

        # Assuming ECG signal is sampled at 500 Hz
        if self.downsample is not None:
            # Use scipy's resample function to downsample the signal
            data = self.downsample_data(data, original_rate = 500, target_rate = self.downsample)


        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        if self.covariate_conditioning is not None:
            conditioning = self.conditioning[idx]
            normalized_conditioning = [(x - self.conditioning_mean[i]) / self.conditioning_std[i] for i, x in enumerate(conditioning)]
            conditioning = torch.tensor(normalized_conditioning, dtype=torch.float)

            return data, conditioning, label

        return data, label

    def downsample_data(self, data, original_rate=500, target_rate=300):
    # Calculate the new number of samples per channel
        num_original_samples = data.shape[-1]
        new_num_samples = int(num_original_samples * target_rate / original_rate)

        # Check if the data is multi-channel
        if data.ndim > 1 and data.shape[0] > 1:
            # If multi-channel, iterate over channels
            downsampled_data = np.array([
                resample(channel, new_num_samples) for channel in data
            ])
        else:
            # If single-channel, directly resample
            downsampled_data = resample(data, new_num_samples)

        return downsampled_data


    def scale_labels(self, label):
        factors = pickle.load(open("out/y_scale_factors.pkl", "rb"))

        y_scale_factors = [factors[output] if regress else None for regress, output in zip(self.regress, self.outputs)]

        # If a regression output, min max scale (for the sake of keeping losses in similar magnitude between classification and regression)
        label = [(l - f[0]) / (f[1] - f[0]) if f is not None else l for l, f in zip(label, y_scale_factors)]

        return label

    def process_X(self, X, y):
        """Preprocess / Augment ECG signals. Preprocessing is applied to train & val while augmentation is only applied to val"""

        if self.train:  # Make sure random cutting doesn't yield majority zeros
            trim_length = len(np.trim_zeros(X[0], "b"))
            X = X[:, :trim_length]

        if X.shape[1] > self.sig_length:
            if self.train:
                X = preproc.random_cut(X, self.sig_length)
            else:
                X = preproc.cut(X, self.sig_length)

        if X.shape[1] < self.sig_length:
            X = preproc.pad(X, self.sig_length)

        if self.aug is not None:
            if not self.train:
                raise ValueError("Augmentation has been provided for a non-training dataset. Please check this.")

            X = preproc.apply_augmentation(X, y, fs=self.fs, aug=self.aug)

        if self.preprocessing is not None:
            X = preproc.apply_preprocessing(X, preprocessing=self.preprocessing, to_mV=self.to_mV)

        return X

class HallandEKGBeats(torch.utils.data.TensorDataset):
    def __init__(
        self,
        ids,
        root: pathlib.Path,
        sig_length:int =512,
        sampling_rate: int = 100,
        singlebeat: bool = True,
        time_max: int = 5000,
        outputs:list = ["meanqtc","scd1"],
        regress:list = [True,False],
        aug:list = None,
        preprocessing = None,
        transform: bool=False,
        train: bool=True,
        clean_filenames=None,
        clean_filepaths=None,
        mean=None,
        std=None,
        targetmean = 0,
        targetstd = 1,
        one_beat = False,
        covariate_df = "path/to/patient_metadata.feather",
        to_mV = False, 
        source_labels = False,
    ):
        self.root = root
        self.ids = ids
        self.regress = regress
        self.outputs = outputs
        self.time_max = time_max
        self.singlebeat = singlebeat
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.preprocessing = preprocessing
        self.clean_filenames = clean_filenames
        self.clean_filepaths = clean_filepaths
        self.mean = mean
        self.std = std
        self.aug = aug
        self.targetmean = targetmean
        self.targetstd = targetstd
        self.train = train 
        self.one_beat = one_beat
        self.to_mV = to_mV
        self.covariate_df = pd.read_feather(covariate_df)
        self.sig_length = sig_length
        self.source_labels = source_labels

        # build dataset using root
        self.build_dataset()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        
        with open(self.filepaths[idx], "rb") as f:
            signal = np.load(f, allow_pickle=True)
        
        if self.to_mV:
            study_id = self.filepaths[idx].name.split("_")[0]
            scaling_factor = self.covariate_df[self.covariate_df['studyId']==study_id]['scaling_factor'].values
            if len(scaling_factor)==0 or np.isnan(scaling_factor.iloc[0]):
                scaling_factor = 1 # do not scale in this case
            scaling_factor = float(scaling_factor)
            signal = signal/scaling_factor

        if self.transform:
            signal = self.transform_func(signal) 
        else:
            signal = torch.Tensor(signal)

        target = self.targets[idx]
        signal = self.process_X(X=signal, y=target)
        target = torch.tensor(target)
        
        if self.source_labels:
            return signal.float(), target.float() ,[str(self.filepaths[idx])]
        else:
            return signal.float(), target.float()

    def build_dataset(self):
        """Find all EKGs in self.root and creates dataset from them."""

        #Extact beats matching the given IDs
        if self.clean_filenames and self.clean_filepaths:
            self.filenames = self.clean_filenames
            self.filepaths = self.clean_filepaths
        elif self.one_beat:
            filepath = []
            for Path, subdirs, files in os.walk(self.root):
                if len(files)>=1:
                    index = random.randint(0, len(files)-1)
                    name = files[index]
                    beat = pathlib.Path(Path, name)
                    filepath.append(beat)

            self.filepaths = [path for path in filepath if int(path.name.split("_")[0]) in self.ids]
            self.filenames = [path.name for path in self.filepaths]
        else:
            filepath = [pathlib.Path(Path, name) for Path, subdirs, files in os.walk(self.root) for name in files]
            self.filepaths = [path for path in filepath if path.name.split("_")[0] in self.ids]
            self.filenames = [path.name for path in self.filepaths]
            
        
        #Extract labels for these beats
        self.covariate_df = self.covariate_df.loc[self.covariate_df["studyId"].isin(self.ids)]
        
        new_cols = ['scd3mo_6mo', 'scd1_2', 'scd6mo_1', 'scd3mo_6mo_x', 'scd1_2_x', 'scd6mo_1_x', 'scd3mo_6mo_y', 'scd1_2_y', 'scd6mo_1_y']
        for col in new_cols:
            if col in self.covariate_df.columns:
                self.covariate_df = self.covariate_df.drop(col, axis=1)

        self.covariate_df['scd3mo_6mo'] = (self.covariate_df['scd6mo'] == 1) & (self.covariate_df['scd3mo'] == 0)
        self.covariate_df['scd6mo_1'] = (self.covariate_df['scd6mo'] == 0) & (self.covariate_df['scd1'] == 1)
        self.covariate_df['scd1_2'] = (self.covariate_df['scd1'] == 0) & (self.covariate_df['scd2'] == 1)

        outputs = ["studyId"] + self.outputs 
        covariate_df = self.covariate_df[outputs]
        
        dataset = pd.DataFrame(self.filenames,columns=["filenames"])
        dataset[['studyId', 'beat']] = dataset['filenames'].str.split('_', expand=True)
        dataset = dataset.merge(covariate_df, how="left", on="studyId")

        # standardising non-binary labels
        if self.train:
            self.targetmean = []
            self.targetstd = []      
            for i in range(len(self.regress)): # loop over column names to make more error proof
                if self.regress[i]:
                    self.targetmean.append(dataset.iloc[:,i+3].mean())
                    self.targetstd.append(dataset.iloc[:,i+3].std())
                    dataset.iloc[:,i+3] = (np.array(dataset.iloc[:,i+3]) - dataset.iloc[:,i+3].mean())/dataset.iloc[:,i+3].std()

        else:
            for i in range(len(self.regress)):
                if self.regress[i]:
                    if (type(self.targetmean) == int) and (type(self.targetstd) == int):
                        dataset.iloc[:,i+3] = (np.array(dataset.iloc[:,i+3]) - self.targetmean)/self.targetstd
                    else:
                        dataset.iloc[:,i+3] = (np.array(dataset.iloc[:,i+3]) - self.targetmean[i])/self.targetstd[i]

        #Add target values to dataloader attribute
        dataset = dataset.iloc[:,3:]
        cols = dataset.columns

        if "scd1" in cols:
            dataset["scd1"] = dataset["scd1"].astype(int)
        self.targets = [[dataset[cols[x]][i] for x in range(len(cols))] for i in range(len(self.filenames))] 
        del filepath
    
    def transform_func(self, input_t):

        output = []
        input_t = torch.from_numpy(input_t).type(dtype=torch.FloatTensor)

        for channel in range(12):
            channel = (input_t[channel]-self.mean[channel].item()) / self.std[channel].item()
            output.append(channel)

        output_tensor = torch.cat(output, 0).view(12,-1)
        
        return output_tensor
    
    def process_X(self, X, y):
        """Preprocess / Augment ECG signals. Preprocessing is applied to train & val while augmentation is only applied to val"""

        if X.shape[1] > self.sig_length:
            if self.train:
                X = preproc.random_cut(X, self.sig_length)
            else:
                X = preproc.cut(X, self.sig_length)

        if X.shape[1] < self.sig_length:
            X = preproc.pad(X, self.sig_length)

        if self.aug is not None:
            if not self.train:
                raise ValueError("Augmentation has been provided for a non-training dataset. Please check this.")
            X = preproc.apply_augmentation(X, y, fs=self.fs, aug=self.aug)

        if self.preprocessing is not None:
            X = preproc.apply_preprocessing(X, preprocessing=self.preprocessing, to_mV = self.to_mV)

        return torch.Tensor(X)