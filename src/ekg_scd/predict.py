"""
Utility function for running a forward pass of a model over a provided set of IDs.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pathlib
import itertools

from ekg_scd.datasets import Hallandata, HallandEKGBeats
from ekg_scd.helpers import models

def predict_beats(
    val_ids,
    outputs,
    regress,
    model,
    model_path,
    covariate_df_path='path/to/patient_data',
    data_dir='path/to/data',
    aug=None,
    preprocessing=None,
    sig_length=5000,
    ensemble_models=None,
    ensemble_paths=None,
    stack=False,
    oversample_scd = True,
    cumulative_predictor=False,
    train_ids=None,
    in_sample = True,
    one_beat: bool = True,
    clean_beats_path: str = 'path/to/beats',
    rep_mp = 5
):

    # subset to only those IDs which successfully stack
    if stack:
        stackable_ids = [int(idx[:-4]) for idx in os.listdir(f"{DATA_DIR}/stacked_X")]
        val_ids = np.array(list(set(val_ids).intersection(set(stackable_ids))))
    
    # normalization_data = np.load(f'scale_factors/normalization_stats_clean.pickle', allow_pickle=True)
    
    if in_sample and train_ids:
        val = HallandEKGBeats(train_ids, data_dir, 300, True, 5000, outputs=outputs, regress=regress, preprocessing=preprocessing, transform=False, train=True,  covariate_df=covariate_df_path,
            # mean=normalization_data['mean'], std=normalization_data['std'], 
            # targetmean=normalization_data['targetmean'],targetstd=normalization_data['targetstd'], oversample_scd=oversample_scd,
            one_beat=True, to_mV=False, source_labels=True) #TODO consider standardizing targets if necessary
    else:
        val = HallandEKGBeats(val_ids, data_dir, 300, True, 5000, outputs=outputs, regress=regress, preprocessing=preprocessing, transform=False, train=True,  covariate_df=covariate_df_path,
            # mean=normalization_data['mean'], std=normalization_data['std'], 
            # clean_beats_path=clean_beats_path,targetmean=0,targetstd=1, oversample_scd=oversample_scd,
                              one_beat=False, to_mV=False, source_labels=True) #TODO consider standardizing targets if necessary

    # establish model parameters
    param_X, param_y, _ = val[0] #
    n_channels, n_samples, n_outputs, dim_wide = models.establish_params(param_X, param_y)
    print(f"dim_wide: {dim_wide} n_outputs: {n_outputs}")
    # establish models with which to predict
    m = models.establish_model(
        model,
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
    state_dict = torch.load(f"{model_path}/model.best.pth.tar")["best_state"]
    m.load_state_dict(state_dict)

    if in_sample:
        pred_df = make_predictions_beats(m, train_ids, val, outputs) #, normalization_data
    else:
        pred_df = make_predictions_beats(m, val_ids, val, outputs) #, normalization_data

    return pred_df


def predict(
    val_ids,
    outputs,
    regress,
    model,
    model_path,
    aug=None,
    preprocessing=None,
    sig_length=5000,
    ensemble_models=None,
    ensemble_paths=None,
    # stack=False,
    cumulative_predictor=False,
    train_ids=None,
    in_sample = True,
    dropout = 0.5,
    covariate_conditioning: list = None,
    covariate_df_path = f"covariate_df.feather",
    x_dir = '//lthalland.se/vdata/ECG/X_new',
    conv_channels = 128,
    attention = 'max',
    rep_mp = 4,
    num_rep_blocks = 12,
):

    # subset to only those IDs which successfully stack
    # if stack:
    #     stackable_ids = [int(idx[:-4]) for idx in os.listdir(f"{DATA_DIR}/stacked_X")]
    #     val_ids = np.array(list(set(val_ids).intersection(set(stackable_ids))))
    # establish dataset for sampling later
    if train_ids is not None and in_sample:

        train = Hallandata(
            ids=train_ids,
            outputs=outputs,
            regress=regress,
            train=True,
            aug=aug,
            preprocessing=preprocessing,
            covariate_df=covariate_df_path,
            # stack=stack,
            covariate_conditioning = covariate_conditioning,
            x_dir = x_dir
        )

        val = Hallandata(
            ids=train_ids,
            outputs=outputs,
            regress=regress,
            train=False,
            aug=None,
            preprocessing=preprocessing,
            # stack=stack,
            covariate_df=covariate_df_path,
            covariate_conditioning = covariate_conditioning,
            conditioning_mean=train.conditioning_mean,
            conditioning_std=train.conditioning_std,
            x_dir = x_dir
        )

    else:
        train = Hallandata(
            ids=train_ids,
            outputs=outputs,
            regress=regress,
            train=True,
            aug=aug,
            preprocessing=preprocessing,
            # stack=stack,
            covariate_conditioning = covariate_conditioning,
            covariate_df=covariate_df_path,
            x_dir = x_dir
        )

        val = Hallandata(
            ids=val_ids,
            outputs=outputs,
            regress=regress,
            train=False,
            aug=None,
            preprocessing=preprocessing,
            # stack=stack,
            covariate_conditioning = covariate_conditioning,
            conditioning_mean=train.conditioning_mean,
            conditioning_std=train.conditioning_std,
            covariate_df=covariate_df_path,
            x_dir = x_dir
        )

    # Establish parameters based on sample file
    if covariate_conditioning is not None:
        param_X, _, param_y = val[0]
    else:
        param_X, param_y = val[0]

    n_channels, n_samples, n_outputs, dim_wide = models.establish_params(param_X, param_y)

    # establish models with which to predict
    m = models.establish_model(
        model,
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
        conv_channels=conv_channels,
        attention = attention,
        rep_mp = rep_mp,
        num_rep_blocks = num_rep_blocks,
    )
    state_dict = torch.load(f"{model_path}/model.best.pth.tar")["best_state"]
    m.load_state_dict(state_dict)

    if in_sample:
        pred_df = make_predictions(m, train_ids, val, outputs, covariate_conditioning)
    else:
        pred_df = make_predictions(m, val_ids, val, outputs, covariate_conditioning)

    return pred_df


def make_predictions(m, val_ids, val, outputs, covariate_conditioning=None):
    # Set model to eval mode to turn off dropout layers, etc.
    m.eval()
    
    # Decide on device (GPU if available, otherwise CPU).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m.to(device)
    
    # Create DataLoader
    batch_size = 256
    val_loader = torch.utils.data.DataLoader(
        val, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # Pin memory only makes sense if CUDA is available.
        sampler=None
    )

    true_ls = []
    pred_ls = []

    # If covariates are used
    if covariate_conditioning is not None:
        for (data, cov, target) in tqdm(val_loader, total=len(val_loader)):
            data = data.to(device)
            cov = cov.to(device)
            target = target.to(device)
            preds = m(data, cov)
            trues = target.cpu().tolist()
            preds = preds.cpu().tolist()
            true_ls.extend(trues)
            pred_ls.extend(preds)

    else:
        # If no covariates used
        for (data, target) in tqdm(val_loader, total=len(val_loader)):
            data = data.to(device)
            target = target.to(device)
            preds = m(data)
            trues = target.cpu().tolist()
            preds = preds.cpu().tolist()
            true_ls.extend(trues)
            pred_ls.extend(preds)

    # Prepare output DataFrame
    model_pred_cols = [f"{o}_hat" for o in outputs]
    out_ls = [
        [val_ids[i]] + true_ls[i] + pred_ls[i] 
        for i in range(len(true_ls))]
    pred_df = pd.DataFrame(out_ls, columns=["studyId"] + outputs + model_pred_cols)

    return pred_df


def make_predictions_beats(m, val_ids, val, outputs):
    # Set model to evaluation mode (turns off dropout, etc.)
    m.eval()
    
    # Decide on device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m.to(device)

    # Prepare DataLoader
    batch_size = 256
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # Pin memory only if CUDA is available
        sampler=None
    )
    
    filenames = []
    true_ls = []
    pred_ls = []

    # Loop over validation loader
    for (data, target, filename) in tqdm(val_loader, total=len(val_loader)):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            preds = m(data)
        trues = target.cpu().tolist()
        preds = preds.cpu().tolist()
        filename = [list(f) for f in filename]  
        filename = sum(filename, [])
        true_ls.extend(trues)
        pred_ls.extend(preds)
        filenames.extend(filename)

    # Build the DataFrame
    model_pred_cols = [f"{o}_hat" for o in outputs]
    val_ids = [name.split('_')[0] for name in filenames]
    
    out_ls = [
        [val_ids[i]] + [filenames[i]] + true_ls[i] + pred_ls[i]
        for i in range(len(true_ls))
    ]
    pred_df = pd.DataFrame(out_ls, columns=["studyId", "filename"] + outputs + model_pred_cols)

    return pred_df
