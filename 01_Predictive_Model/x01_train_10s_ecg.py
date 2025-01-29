#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Train a model on Halland ECG data """
import json
import os
import warnings
from datetime import date

import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from ekg_scd.datasets import Hallandata
from ekg_scd.helpers import models


def train(
    model,
    outputs,
    regress: list,
    aug: list = None,
    preprocessing: list = None,
    split: tuple = (30, 30),
    min_epochs: int = 40,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-2,
    batch_size: int = 256,
    weight_decay: float = 0,
    lr_reduce_interval: int = 10,
    lr_sched_gamma: float = 0.5,
    warm_start_path: str = None,
    warm_start_from_zero: bool = False,
    model_name: str = "experiment",
    ensemble_models: list = None,
    ensemble_paths: list = None,
    only_deaths: bool = False,
    cumulative_predictor: bool = False,
    covariate_df_path: str = "path/to/patient_data", 
    dropout: float = 0.5,
    covariate_conditioning: list = None,
    x_dir: str = "path/to/ecgs",
    pretrain_path: str = None,
    attention: bool = False,
    rep_mp: int = 4,
    num_rep_blocks:int =32,
    conv_channels: int = 128,
    downsample: int =None
):
    save_path = f"modelfits_ecg/{model_name}"

    if os.path.isfile(f"modelfits_ecg/{model_name}.json"):
        warnings.warn(
            f"Path to which results will be saved already exists. If allowed to finish, this fit will overwrite previous fit of model_name '{model_name}'"
        )

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if len(regress) != len(outputs):
        raise ValueError("Length of regress kwarg != N outputs")

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
    print(f"Validation labels: There are {len(pd.unique(test_pt_ID))} unique patients with mean age {test_age.mean()} and mean one-year SCD rate of {test_scd1.mean()}")

    print("_______________________________________________________________________")
    print("\n")

    train = Hallandata(
        ids=train_ids,
        outputs=outputs,
        regress=regress,
        train=True,
        aug=aug,
        preprocessing=preprocessing,
        covariate_df=covariate_df_path,
        covariate_conditioning = covariate_conditioning,
        x_dir = x_dir,
        downsample = downsample
    )

    val = Hallandata(
        ids=val_ids,
        outputs=outputs,
        regress=regress,
        train=False,
        aug=None,
        preprocessing=preprocessing,
        covariate_df=covariate_df_path,
        covariate_conditioning = covariate_conditioning,
        conditioning_mean=train.conditioning_mean,
        conditioning_std=train.conditioning_std,
        x_dir = x_dir,
        downsample = downsample
    )

    # Establish parameters based on sample file
    if covariate_conditioning is not None:
        param_X, _, param_y = train[0]
    else:
        param_X, param_y = train[0]


    n_channels, n_samples, n_outputs, dim_wide = models.establish_params(param_X, param_y)

    if covariate_conditioning is not None:
        param_X, _, param_y = train[0]
    else:
        param_X, param_y = train[0]

    
    if pretrain_path == None:
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
            dropout = dropout, 
            covariate_conditioning = covariate_conditioning,
            attention = attention,
            rep_mp = rep_mp,
            num_rep_blocks = num_rep_blocks,
            conv_channels=conv_channels
        )
    else:
        m = torch.load(pretrain_path)
        last_dim = m.net.num_last_active
        m.out = nn.Linear(last_dim, n_outputs)

    # Train ---

    # Estimate an optimal number of workers
    available_cores = os.cpu_count()
    num_workers = max(1, int(round(0.5*available_cores)))  

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    print(f"Training {model}, regress = {regress}")

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
        covariate_conditioning = covariate_conditioning is not None,
        attention = attention
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

    """
    for k,v in all_params.items():
        if not isinstance(v, (type(None), int, bool, float, str, list)):
            print(k)
            print(type(v))
    """

    all_params = {k:v for k,v in all_params.items() if isinstance(v, (type(None), tuple, int, bool, float, str, list))}
    result_path = f"modelfits_ecg/{params['model_name']}.json"
    json.dump(all_params, open(result_path, "w"))


def get_performance(save_path):
    """Grab performance statistics from model save"""

    d = torch.load(f"{save_path}/model.best.pth.tar")
    performance = {k: d[k] for k in ["best_loss", "best_stat", "best_epoch"]}
    return performance


if __name__ == "__main__":

    #########################################
    ### SCD model fitting 
    #########################################
        
    time_defs = ["3mo", "6mo", "1", "2"] 
    output_categories = ["scd", "dead"] 
    outputs = [f"{cat}{t}" for cat in output_categories for t in time_defs]
    outputs += ["ICD_VT1", "ICD_VF1", "Phi_VT1","TROPT1", "Acute_MI1"] 
    outputs += ['scd3mo_6mo', 'scd6mo_1', 'scd1_2']

    regress = [False] * len(outputs)

    train(
        "ResNet",
        outputs=outputs,
        learning_rate=1e-5, 
        model_name="07_08_scd1_model_dropDefib_agesex_pretrain",
        regress=regress,
        preprocessing=["zero_mode"], 
        split=(0.30, 0.30),
        max_epochs=1000,
        warm_start_from_zero=True,
        cumulative_predictor=True,
        dropout = 0.4,
        covariate_conditioning=['age', 'female'],
        covariate_df_path = "covariate_df.feather",
        x_dir = "10_sec_ecgs",
        conv_channels = 128
    )

    #########################################
    ### Conditional SCD model fitting 
    #########################################

    time_defs = ["3mo", "6mo", "1", "2"] 
    output_categories = ["scd"] 
    outputs = [f"{cat}{t}" for cat in output_categories for t in time_defs]
    outputs += ["TROPT1", "Acute_MI1"] 
    outputs += ['scd3mo_6mo', 'scd6mo_1', 'scd1_2']
    regress = [False] * len(outputs)

    train(
        "ResNet",
        outputs=outputs,
        learning_rate=1e-5, 
        model_name="10_01_death_model_fliterTropt_dropDefib_agesex_pretrain",
        regress=regress,
        preprocessing=["zero_mode"],
        split=(0.30, 0.30),
        max_epochs=1000,
        warm_start_from_zero=True,
        only_deaths = True,
        cumulative_predictor=True,
        dropout = 0.4,
        covariate_conditioning=['age', 'female'],
        covariate_df_path = "covariate_df.feather",
        x_dir = "10_sec_ecgs", 
    )

    #########################################
    ### Low EF model fitting 
    #########################################

    outputs = ['low_ef'] 

    regress = [False] * len(outputs)

    train(
        "ResNet",
        outputs=outputs,
        learning_rate=1e-4, 
        model_name="10_01_low_ef_model_fliterTropt_dropDefib_agesex_pretrain_no_scale_v4",
        regress=regress,
        preprocessing=["zero_mode"], 
        split=(0.30, 0.30),
        max_epochs=1000,
        min_epochs=80,
        lr_reduce_interval=20,
        warm_start_from_zero=True,
        cumulative_predictor=True,
        dropout = 0.4,
        covariate_conditioning=None,
        covariate_df_path = "covariate_df.feather",
        x_dir = "10_sec_ecgs",
        num_rep_blocks = 12,
        conv_channels = 128,
    )

