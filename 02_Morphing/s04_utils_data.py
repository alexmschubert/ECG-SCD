import warnings

import numpy
import pandas
import tqdm

from sklearn.model_selection import train_test_split

from ekg_scd.datasets import HallandEKGBeats


def load_dataset(ecg_data_dirname, covariate_data_filepath, split, outcome_names, outcome_regress):
    
    # use saved covariate_df to identify relevant sample
    covariate_df = pandas.read_feather(covariate_data_filepath)
    
    covariate_df = covariate_df[covariate_df['include_modelling']==True]
    pt_ids = covariate_df['ptId'].tolist()
    train_ids, remainder = train_test_split(
        pt_ids,
        train_size=split[0],     
        random_state=42,    
        shuffle=True
    )
    val_ids, _ = train_test_split(
        remainder,
        train_size=split[1]/(1-split[0]),
        random_state=42,
        shuffle=True
    )
    val_ids = covariate_df[covariate_df['ptId'].isin(val_ids)]['studyId'].tolist()
    train_ids = covariate_df[covariate_df['ptId'].isin(train_ids)]['studyId'].tolist()
    print(f"This model uses {len(train_ids)} ids for training")

    if len(train_ids)==0:
        train_data = []
    else:
        train_data = HallandEKGBeats(
            train_ids,
            ecg_data_dirname,
            300,
            True,
            5000,
            outputs=outcome_names,
            regress=outcome_regress,
            preprocessing=["zero_mode"],
            covariate_df=covariate_data_filepath,
            transform=False,
            one_beat=False,
            to_mV=False
        )

    print(f"This model uses {len(val_ids)} ids for validation")

    val_data = HallandEKGBeats(
        val_ids,
        ecg_data_dirname,
        300,
        True,
        5000,
        outputs=outcome_names,
        regress=outcome_regress,
        preprocessing=["zero_mode"],
        transform=False,
        train=False,
        one_beat=False,
        to_mV=False,
        covariate_df=covariate_data_filepath
    )

    return train_data, val_data


def load_data(ecg_data_dirname, covariate_data_filepath, split, outcome_names, outcome_regress):
    train_data, val_data = load_dataset(ecg_data_dirname, covariate_data_filepath, split, outcome_names, outcome_regress)

    # Establish parameters based on sample file
    print(f"x shape: {train_data[0][0].shape}, y shape: {train_data[0][1].shape}")

    # loading all in chunks
    chunk_indices = numpy.linspace(0, len(train_data), len(train_data) // 1000)
    chunk_indices = tqdm.tqdm(zip(chunk_indices[:-1], chunk_indices[1:]), total=len(chunk_indices) - 1, desc="X data")
    _x_chunks = [numpy.stack([train_data[i][0] for i in range(int(jlo), int(jhi))]) for jlo, jhi in chunk_indices]

    # Repeat for y
    warnings.warn("Using hard-coded nonsense on y targets, needs fixing.")
    chunk_indices = numpy.linspace(0, len(train_data), len(train_data) // 1000)
    chunk_indices = tqdm.tqdm(zip(chunk_indices[:-1], chunk_indices[1:]), total=len(chunk_indices) - 1, desc="Y data")
    _y_chunks = [numpy.stack([train_data[i][1][:4] for i in range(int(jlo), int(jhi))]) for jlo, jhi in chunk_indices]
    X = numpy.concatenate(_x_chunks, axis=0)
    y = numpy.concatenate(_y_chunks, axis=0)

    return X, y
