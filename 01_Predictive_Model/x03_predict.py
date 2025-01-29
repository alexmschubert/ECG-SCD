import json
import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from ekg_scd import predict

def main():
    parser = argparse.ArgumentParser(description="Script to generate ECG predictions.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to be used for prediction (without file extension)."
    )
    parser.add_argument(
        "--covariate_df_path",
        type=str,
        required=True,
        help="Path to the Feather file containing the covariate DataFrame."
    )
    parser.add_argument(
        "--ecg_dir",
        type=str,
        required=True,
        help="Path to the directory containing the ECG data."
    )
    
    parser.add_argument(
        "--split",
        nargs=2,          
        type=float,       
        default=[0.3, 0.3],
        help="Train/Val split proportion. Usage example: --split 0.3 0.3"
    )
    
    parser.add_argument(
        "--beat",
        action="store_true",
        help="If set, predictions will be generated per-beat instead of per-segment."
    )
    parser.add_argument(
        "--in_sample",
        action="store_true",
        help="If set, predictions will be generated in-sample (training set) rather than out-of-sample (validation set)."
    )

    args = parser.parse_args()

    model_name = args.model_name
    covariate_df_path = args.covariate_df_path
    ecg_dir = args.ecg_dir
    beat = args.beat
    in_sample = args.in_sample
    split = args.split
    
    # Read in DataFrame
    covariate_df = pd.read_feather(covariate_df_path)
    covariate_df = covariate_df[covariate_df["include_modelling"] == True]
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
    
    val_ids = covariate_df[covariate_df['ptId'].isin(val_ids)]['studyId']
    train_ids = covariate_df[covariate_df['ptId'].isin(train_ids)]['studyId']
    
    # Load model parameters
    parameters = json.load(open(f"modelfits_ecg/{model_name}.json", "r"))
    for k, v in parameters.items():
        if isinstance(v, float) and pd.isna(v):
            parameters[k] = None

    # Determine output path
    if in_sample:
        prediction_save_path = f"predictions/{parameters['model_name']}_1/predictions_in_sample.feather"
    else:
        prediction_save_path = f"predictions/{parameters['model_name']}_1/predictions.feather"

    # Only run predictions if they haven't been generated already
    if not os.path.isfile(prediction_save_path):
        covariate_conditioning = parameters.get("covariate_conditioning", None)
        
        if not beat:
            predictions = predict.predict(
                val_ids=val_ids if not in_sample else train_ids,
                outputs=parameters["outputs"],
                preprocessing=parameters["preprocessing"],
                regress=parameters["regress"] if "regress" in parameters else [False] * len(parameters["outputs"]),
                model=parameters["model"],
                model_path=f"modelfits_ecg/{parameters['model_name']}",
                sig_length=5000,
                cumulative_predictor=parameters["cumulative_predictor"],
                train_ids=train_ids,
                in_sample=in_sample,
                dropout=parameters["dropout"],
                covariate_conditioning=covariate_conditioning,
                covariate_df_path=covariate_df_path,
                x_dir=ecg_dir,
                conv_channels=parameters["conv_channels"],
                attention=parameters["attention"],
                rep_mp=parameters["rep_mp"],
                num_rep_blocks=parameters["num_rep_blocks"],
            )
        else:
            # If beat-level predictions are requested
            predictions = predict.predict_beats(
                val_ids=val_ids,
                data_dir=ecg_dir,
                outputs=parameters["outputs"],
                preprocessing=parameters["preprocessing"],
                regress=parameters["regress"] if "regress" in parameters else [False] * len(parameters["outputs"]),
                model=parameters["model"],
                model_path=f"modelfits_beat/{parameters['model_name']}",
                ensemble_models=parameters["ensemble_models"],
                ensemble_paths=parameters["ensemble_paths"],
                covariate_df_path=covariate_df_path,
                sig_length=300,
                cumulative_predictor=False,
                train_ids=train_ids,
                in_sample=in_sample,
                rep_mp=5
            )

        pred_cols = [f"{o}_hat" for o in parameters["outputs"]]
        os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)

        if beat:
            predictions = predictions[["studyId", "filename"] + pred_cols]
        else:
            predictions = predictions[["studyId"] + pred_cols]

        predictions.to_feather(prediction_save_path)
        print(f"Predictions saved to {prediction_save_path}")
    else:
        print(f"Predictions already exist at {prediction_save_path}. Skipping.")

if __name__ == "__main__":
    main()
