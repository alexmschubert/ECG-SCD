#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Train a model on Halland ECG data """
import gc
import json
import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from ekg_scd.datasets import HallandEKGBeats
from ekg_scd.helpers import models
    

def train_beats(
    model,
    outputs,
    regress: list,
    data_dir: str = "/path/to/processed_ECG_beats",
    aug: list = None,
    preprocessing: list = None,
    split: tuple = (10, 5),
    min_epochs: int = 40,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-2,
    batch_size: int = 256,
    weight_decay: float = 1e-5,
    lr_reduce_interval: int = 10,
    lr_sched_gamma: float = 0.5,
    warm_start_path: str = None,
    warm_start_from_zero: bool = False,
    model_name: str = "experiment",
    rep_mp: int = 4,
    ensemble_models: list = None,
    ensemble_paths: list = None,
    only_deaths: bool = False,
    one_beat: bool = False,
    cumulative_predictor: bool = False,
    covariate_df_path: str = "path/to/patient_data", 
):
    save_path = f"modelfits_beat/{model_name}"

    if os.path.isfile(f"modelfits_beat/{model_name}.json"):
        warnings.warn(
            f"Path to which results will be saved already exists. If allowed to finish, this fit will overwrite previous fit of model_name '{model_name}'"
        )

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if len(regress) != len(outputs):
        raise ValueError("Length of regress kwarg != N outputs")
    
    # use saved covariate_df to manage kwarg filter conditions
    covariate_df = pd.read_feather(covariate_df_path)
    
    if only_deaths:
        covariate_df = covariate_df[covariate_df['dead1']==True]

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
        train_size=split[1]/(1-split[0] ),
        random_state=42,
        shuffle=True
    )
    
    val_ids = covariate_df[covariate_df['ptId'].isin(val_ids)]['studyId'].tolist()
    train_ids = covariate_df[covariate_df['ptId'].isin(train_ids)]['studyId'].tolist()

    

    print("\n")
    print("_______________________________________________________________________")
    print(f"Model name: {model_name}")
    print(f"This model uses {len(train_ids)} ids for training and {len(val_ids)} ids for validation")
    train_covariates = covariate_df[covariate_df['studyId'].isin(train_ids)]
    test_covariates = covariate_df[covariate_df['studyId'].isin(val_ids)]

    train_pt_ID = train_covariates["ptId"]
    train_age = train_covariates["age"]
    train_scd1 = train_covariates["scd1"]

    test_pt_ID = test_covariates["ptId"]
    test_age = test_covariates["age"]
    test_scd1 = test_covariates["scd1"]

    print(f"Train labels: There are {len(pd.unique(train_pt_ID))} unique patients with mean age {train_age.mean()} and mean one-year SCD rate of {train_scd1.mean()}")
    print(f"Test labels: There are {len(pd.unique(test_pt_ID))} unique patients with mean age {test_age.mean()} and mean one-year SCD rate of {test_scd1.mean()}")

    print("_______________________________________________________________________")
    print("\n")


    train = HallandEKGBeats(train_ids, data_dir, 300, True, 5000, outputs=outputs, regress=regress, preprocessing=preprocessing, transform=False, 
        one_beat=one_beat, to_mV=False, covariate_df=covariate_df_path, aug=aug) 
    gc.collect()
    val = HallandEKGBeats(val_ids, data_dir, 300, True, 5000, outputs=outputs, regress=regress, preprocessing=preprocessing, transform=False, train=False, 
        one_beat=one_beat, to_mV=False, covariate_df=covariate_df_path, aug=aug) 

    # Establish parameters based on sample file
    param_X, param_y = train[0]
    n_channels, n_samples, n_outputs, dim_wide = models.establish_params(param_X, param_y)
    

    m = models.establish_model(
        model=model,
        regress=regress,
        n_channels=n_channels,
        n_samples=n_samples,
        n_outputs=n_outputs,
        dim_wide=dim_wide,
        ensemble_models=ensemble_models,
        ensemble_paths=ensemble_paths,
        cumulative_predictor=cumulative_predictor,
        output_names = outputs,
        rep_mp = rep_mp
    )

    # Train ---

    # Estimate an optimal number of workers
    available_cores = os.cpu_count()
    num_workers = max(1, int(available_cores / 2))  # using half of the available cores

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers) 

    print(f"Training {model}, regress = {regress}.")
    m.fit(
        train_loader,
        val_loader,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_reduce_interval=lr_reduce_interval,
        lr_sched_gamma=lr_sched_gamma,
        patience=patience,
        batch_size=batch_size,
        warm_start_path=warm_start_path,
        warm_start_from_zero=warm_start_from_zero,
        save_path=save_path,
    )

    save_all(params=locals())

def save_all(params: dict) -> None:
    """Save parameters and results to 'out/results.json' dataframe"""

    performance = get_performance(params["save_path"])
    params["init_path"] = None if params["warm_start_from_zero"] is False else params["warm_start_path"]
    all_params = {
        "date": date.today().strftime("%m-%d-%Y"),
        **params,
        **performance,
    }

    all_params = {k:v for k,v in all_params.items() if isinstance(v, (type(None), tuple, int, bool, float, str, list))}
    result_path = f"modelfits_beat/{params['model_name']}.json"
    json.dump(all_params, open(result_path, "w"))


def get_performance(save_path):
    """Grab performance statistics from model save"""

    d = torch.load(f"{save_path}/model.best.pth.tar")
    performance = {k: d[k] for k in ["best_loss", "best_stat", "best_epoch"]}
    return performance


if __name__ == "__main__":

    #########################################
    ### Beat model training
    #########################################

    time_defs = ["3mo", "6mo", "1", "2"] #, "3"
    output_categories = ["scd", "dead"] #"in_scd", "unrelated"
    outputs = [f"{cat}{t}" for cat in output_categories for t in time_defs]
    outputs += ["ICD_VT1", "ICD_VF1", "Phi_VT1", "TROPT1", "Acute_MI1"]
    regress = [False] * len(outputs)

    outputs += ["age","female","qrsDuration","qtInterval","qrsFrontAxis","qrsHorizAxis","rrInterval","pDuration","atrialrate","meanqtc"] 
    regress += [True, False, True, True, True, True, True, True, True, True]

    train_beats(
        "ResNet",
        outputs=outputs,
        learning_rate=1e-4,
        data_dir= "ecg_beats",
        model_name="Beatmodel_2024_03_11_filter_tropt_ami",
        regress=regress,
        preprocessing=["zero_mode"],
        split=(0.30, 0.20),
        max_epochs = 2,
        rep_mp=5, 
        one_beat=False,
        covariate_df_path = 'covariate_df.feather'
    )