""" Ensembling step to create the final SCD model"""

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

# Local imports
from ekg_scd.helpers.regress import sm_regress, tt_split

if __name__ == "__main__":

    # Configuration
    random_seed = 42
    split = (0.30, 0.30)

    # Model names
    eval_model_name = '07_08_scd1_model_dropDefib_agesex_pretrain'
    conditional_model_name = "10_01_death_model_fliterTropt_dropDefib_agesex_pretrain"
    ef_model_name = "10_01_low_ef_model_fliterTropt_dropDefib_agesex_pretrain_no_scale_v4"

    # Read covariates
    covariate_df = pd.read_feather("covariate_df.feather")

    # Create a dummy indicator for missing EF and median-impute EF
    covariate_df['ef_na'] = covariate_df["most_recent_ef"].isna()
    ef_median = covariate_df.loc[covariate_df['ef_na'] == 0, 'most_recent_ef'].median()
    covariate_df['most_recent_ef_imputed'] = covariate_df['most_recent_ef']
    covariate_df.loc[covariate_df['ef_na'] == 1, 'most_recent_ef_imputed'] = ef_median

    # Where final predictions are saved
    predictions_savepath = "predictions/scd1_ensemble_model"
    os.makedirs(predictions_savepath, exist_ok=True)

    # Ensemble configuration
    share = 0.50
    ENSEMBLE_VARS = [
        "scd1_hat0", "cond_scd1_hat", "scd2_hat", "scd3mo_hat", 
        "scd6mo_hat", "TROPT1_hat", "Acute_MI1_hat", "low_ef_hat",
        "age", "female"
    ]

    ##############################################################################
    # Split Data into Train/Validation
    ##############################################################################
    covariate_df = covariate_df[covariate_df['include_modelling'] == True]
    pt_ids = covariate_df['ptId'].tolist()

    train_ids, remainder = train_test_split(
        pt_ids, 
        train_size=split[0], 
        random_state=random_seed, 
        shuffle=True
    )

    val_ids, _ = train_test_split(
        remainder,
        train_size=split[1] / (1 - split[0]),
        random_state=random_seed,
        shuffle=True
    )
    val_ids = covariate_df.loc[covariate_df['ptId'].isin(val_ids), 'studyId']
    train_ids = covariate_df.loc[covariate_df['ptId'].isin(train_ids), 'studyId']

    # Further split validation set into two halves for ensemble
    val_pts = covariate_df.loc[covariate_df['studyId'].isin(val_ids), 'ptId']
    val_ids_half1, val_ids_half2 = train_test_split(
        val_pts, 
        train_size=share, 
        random_state=random_seed, 
        shuffle=True
    )
    val_ids_half1 = covariate_df.loc[covariate_df['ptId'].isin(val_ids_half1), 'studyId']
    val_ids_half2 = covariate_df.loc[covariate_df['ptId'].isin(val_ids_half2), 'studyId']

    ##############################################################################
    # Load and Merge Model Predictions
    ##############################################################################
    # SCD model
    eval_predictions = pd.read_feather(f"predictions/{eval_model_name}_1/predictions.feather")
    eval_predictions = eval_predictions[["studyId"] + [c for c in eval_predictions.columns if c.endswith("_hat")]]

    # Conditional SCD model
    if conditional_model_name is not None:
        conditional_predictions = pd.read_feather(f"predictions/{conditional_model_name}_1/predictions.feather")
        conditional_predictions = conditional_predictions.rename(columns={"scd1_hat": "cond_scd1_hat"})
        conditional_predictions = conditional_predictions[["studyId", "cond_scd1_hat"]]

    # Low EF model
    if ef_model_name is not None:
        ef_predictions = pd.read_feather(f"predictions/{ef_model_name}_1/predictions.feather")
        ef_predictions = ef_predictions[["studyId", "low_ef_hat"]]

    val_df = eval_predictions.merge(conditional_predictions, how="left", on="studyId")
    if ef_model_name is not None:
        val_df = val_df.merge(ef_predictions, how="left", on="studyId")

    # Attach covariates for final modeling
    val_df = val_df.merge(covariate_df, how='left', on='studyId')

    ##############################################################################
    # Fit Ensemble Model on Each Half
    ##############################################################################

    # First half
    val_df_half1 = val_df.loc[val_df["studyId"].isin(val_ids_half1)].copy()
    val_df_half1 = val_df_half1.rename(columns={"scd1_hat": "scd1_hat0"})
    val_df_half1 = val_df_half1.dropna(subset=ENSEMBLE_VARS).reset_index(drop=True)

    _, val_df_half1["scd1_hat"] = sm_regress(
        train_df=val_df_half1,
        covariates=ENSEMBLE_VARS,             # Ensemble variables
        pred_var="scd1",
        cluster_var="ptId",
        model_save_path=f"ensembles/{eval_model_name}_ensemble_val_half1.pkl",
        print_vars=ENSEMBLE_VARS,               # Printed var names
        logreg=True,
        scale=False
    )

    # Second half
    val_df_half2 = val_df.loc[val_df["studyId"].isin(val_ids_half2)].copy()
    val_df_half2 = val_df_half2.rename(columns={"scd1_hat": "scd1_hat0"})
    val_df_half2 = val_df_half2.dropna(subset=ENSEMBLE_VARS).reset_index(drop=True)

    _, val_df_half2["scd1_hat"] = sm_regress(
        train_df=val_df_half2,
        covariates=ENSEMBLE_VARS,
        pred_var="scd1",
        cluster_var="ptId",
        model_save_path=f"ensembles/{eval_model_name}_ensemble_val_half2.pkl",
        print_vars=ENSEMBLE_VARS,
        logreg=True,
        scale=False
    )

    ##############################################################################
    # Obtain Ensemble Predictions
    ##############################################################################

    # Load both halves of the ensemble
    with open(f'ensembles/{eval_model_name}_ensemble_val_half1.pkl', 'rb') as file:
        model_half1 = pickle.load(file)

    with open(f'ensembles/{eval_model_name}_ensemble_val_half2.pkl', 'rb') as file:
        model_half2 = pickle.load(file)

    # Predict scd1_hat for the first half using the second-half model
    train_df = val_df_half1.copy()
    validate_df = val_df_half1.copy()
    pred_var = 'scd1'

    X_train, y_train, X_test, y_test = tt_split(
        train_df, 
        validate_df, 
        ENSEMBLE_VARS, 
        pred_var, 
        scale=False
    )
    prediction_frame = model_half2.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=0.05)
    y_pred_proba = np.array(prediction_frame["mean"])
    val_df_half1['scd1_hat_ensemble'] = y_pred_proba

    # Predict scd1_hat for the second half using the first-half model
    train_df = val_df_half2.copy()
    validate_df = val_df_half2.copy()
    pred_var = 'scd1'

    X_train, y_train, X_test, y_test = tt_split(
        train_df, 
        validate_df, 
        ENSEMBLE_VARS, 
        pred_var, 
        scale=False
    )
    prediction_frame = model_half1.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=0.05)
    y_pred_proba = np.array(prediction_frame["mean"])
    val_df_half2['scd1_hat_ensemble'] = y_pred_proba

    # Merge Results and Evaluate
    val_df = pd.concat([val_df_half1, val_df_half2])
    y_true = val_df['scd1'].values
    y_pred = val_df['scd1_hat_ensemble'].values
    auc = roc_auc_score(y_true, y_pred)
    print(f"Ensemble AUC: {auc:.4f}")
    
    # Create a histogram of the predicted values
    plt.figure(figsize=(6, 4))
    plt.hist(y_pred, bins=500, alpha=0.7, color='blue')
    plt.title("Distribution of Predicted SCD risk values")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('Distribution_of_predicted_scd_risk.png')
    plt.show()

    # Save Predictions
    val_df.reset_index(drop=True).to_feather(f"{predictions_savepath}/predictions.feather")
    print(f"Predictions saved to '{predictions_savepath}/predictions.feather'")
