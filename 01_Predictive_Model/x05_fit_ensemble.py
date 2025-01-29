""" Ensembling step to create the final SCD model"""

import os

import numpy as np
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split

from ekg_scd.helpers.regress import sm_regress

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm


def tt_split(train_df, val_df, covariates, pred_var, scale=True):
    """
    Given train DF and val DF, scale regression inputs and return arrays.
    """

    # establish training data
    X_train = train_df[covariates]
    y_train = train_df[pred_var].astype(float)
    # establish val data
    X_test = val_df[covariates]
    y_test = val_df[pred_var].astype(float)
    
    # scale everything
    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

################################################################
## Configuration inputs
################################################################

random_seed = 42
split = (0.30, 0.30)

eval_model_name = '07_08_scd1_model_dropDefib_agesex_pretrain'
conditional_model_name = "10_01_death_model_fliterTropt_dropDefib_agesex_pretrain"
ef_model_name = "10_01_low_ef_model_fliterTropt_dropDefib_agesex_pretrain_no_scale_v3"
covariate_df = pd.read_feather("covariate_df.feather")

# Create dummy values when EF is not available and median impute EF  
covariate_df['ef_na'] = covariate_df["most_recent_ef"].isna()
ef_median = covariate_df[covariate_df['ef_na']==0]['most_recent_ef'].median()
covariate_df['most_recent_ef_imputed'] = covariate_df['most_recent_ef']
covariate_df.loc[covariate_df['ef_na']==1, 'most_recent_ef_imputed'] = ef_median

# Savepaths
predictions_savepath = "predictions/scd1_ensemble_model"

# Ensembling configuration
share = 0.50
ENSEMBLE_VARS = ["scd1_hat0", 'cond_scd1_hat', 'scd2_hat', 'scd3mo_hat', 'scd6mo_hat', 'TROPT1_hat', 'Acute_MI1_hat', 'low_ef_hat', 'age', 'female']

################################################################
## Ensembling step
################################################################

# Obtain ids for evaluation
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

val_ids = covariate_df[covariate_df['ptId'].isin(val_ids)]['studyId']
train_ids = covariate_df[covariate_df['ptId'].isin(train_ids)]['studyId']

# Split val_ids into two sets for ensemble training and out of sample prediction
val_pts = covariate_df[covariate_df['studyId'].isin(val_ids)]['ptId']
val_ids_half1, val_ids_half2 = train_test_split(val_pts, train_size=share,random_state=42, shuffle=True)
val_ids_half1 = covariate_df[covariate_df['ptId'].isin(val_ids_half1)]['studyId']
val_ids_half2 = covariate_df[covariate_df['ptId'].isin(val_ids_half2)]['studyId']

# Load SCD model predictions
eval_predictions = pd.read_csv(f"predictions/{eval_model_name}/predictions.feather")
eval_predictions = eval_predictions[["studyId"] + [c for c in eval_predictions.columns if c[-4:] == "_hat"]]

# Load conditional SCD model predictions
if conditional_model_name != None:
    conditional_predictions = pd.read_csv(f"predictions/{conditional_model_name}/predictions.feather")
    conditional_predictions = conditional_predictions.rename(columns={"scd1_hat": "cond_scd1_hat"})
    conditional_predictions = conditional_predictions[["studyId", "cond_scd1_hat"]] 

# Load low EF model predictions
if ef_model_name != None:
    ef_predictions = pd.read_feather(f"predictions/{ef_model_name}/predictions.feather")
    ef_predictions = ef_predictions[["studyId", "low_ef_hat"]]

# Combine preditions in a dataframe
val_df = eval_predictions.merge(
    conditional_predictions, how="left", on="studyId"
)

if ef_model_name != None:
    val_df = val_df.merge(
        ef_predictions, how="left", on="studyId"
    )

val_df = val_df.merge(covariate_df, how='left', on='studyId')

print('###########################################################')
print('fitting ensemble model')

# Format first half of validation set for prediction
val_df_half1 = val_df.loc[val_df["studyId"].isin(val_ids_half1)]
val_df_half1 = val_df_half1.rename(columns={"scd1_hat": "scd1_hat0"})
val_df_half1 = val_df_half1.dropna(subset=ENSEMBLE_VARS).reset_index(drop=True)

_, val_df_drop_defib["scd1_hat"] = sm_regress(
    train_df=val_df_drop_defib,
    covariates=ENSEMBLE_VARS,
    pred_var="scd1",
    cluster_var="ptId",
    model_save_path=f"ensembles/{eval_model_name}_ensemble_val_half1.pkl",
    print_vars=covariates,
    logreg=True,
    scale=False
)

############################

val_df_half2 = val_df.loc[val_df["studyId"].isin(val_ids_half2)]
val_df_half2 = val_df_half2.rename(columns={"scd1_hat": "scd1_hat0"})
val_df_half2 = val_df_half2.dropna(subset=ENSEMBLE_VARS).reset_index(drop=True)

_, val_df_drop_defib["scd1_hat"] = sm_regress(
    train_df=val_df_drop_defib,
    covariates=covariates,
    pred_var="scd1",
    cluster_var="ptId",
    model_save_path=f"ensembles/{eval_model_name}_ensemble_val_half2.pkl",
    print_vars=covariates,
    logreg=True,
    scale=False
)


print('#########################################################')
print('############# Obtain ensemble predictions')
print('#########################################################')

with open(f'../../out/{eval_model_name}_ensemble_val_half1.pkl', 'rb') as file:
    model_half1 = pickle.load(file)

with open(f'../../out/{eval_model_name}_ensemble_val_half2.pkl', 'rb') as file:
    model_half2 = pickle.load(file)

# Predict scd1_hat for the first half
train_df = val_df_half1.copy()
validate_df = val_df_half1.copy()
pred_var = 'scd1'

X_train, y_train, X_test, y_test = tt_split(
    train_df, 
    validate_df, 
    covariates, 
    pred_var, 
    scale=False
)
prediction_frame = model_half2.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=0.05)
y_pred_proba = np.array(prediction_frame["mean"])
val_df_half1['scd1_hat_ensemble'] = y_pred_proba


# Predict scd1_hat for the second half
train_df = val_df_half2.copy()
validate_df = val_df_half2.copy()
pred_var = 'scd1'

X_train, y_train, X_test, y_test = tt_split(
    train_df, 
    validate_df, 
    covariates, 
    pred_var, 
    scale=False
)
prediction_frame = model_half1.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=0.05)
y_pred_proba = np.array(prediction_frame["mean"])
val_df_half2['scd1_hat_ensemble'] = y_pred_proba


# Merge the data
val_df = pd.concat([val_df_half1, val_df_half2])

# Calculate AUC and print it
y_true = val_df['scd1'].values
y_pred = val_df['scd1_hat_ensemble'].values
auc = roc_auc_score(y_true, y_pred)
print(f"Ensemble AUC: {auc:.4f}")

# save predictions
os.makedirs(predictions_savepath, exist_ok=True)
val_df.reset_index(drop=True).to_feather("predictions/scd1_ensemble_model/predictions.feather")
print("Predictions saved to 'predictions/scd1_ensemble_model/predictions.feather'")
