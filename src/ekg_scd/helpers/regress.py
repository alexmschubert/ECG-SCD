""" Regression utilities. """

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt


def calculate_nested_f_statistic(small_model, big_model):
    # Copied from https://stackoverflow.com/questions/27328623/anova-test-for-glm-in-python/60769343#60769343
    """Given two fitted GLMs, the larger of which contains the parameter space of the smaller,
    return the F Stat and P value corresponding to the larger model adding explanatory power"""
    addtl_params = big_model.df_model - small_model.df_model
    f_stat = (small_model.deviance - big_model.deviance) / (addtl_params * big_model.scale)
    df_numerator = addtl_params
    # use fitted values to obtain n_obs from model object:
    df_denom = big_model.fittedvalues.shape[0] - big_model.df_model
    p_value = stats.f.sf(f_stat, df_numerator, df_denom)
    return f"({round(f_stat,3)}){star_str(p_value)}"

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

def fit_sm_regression(X_train, y_train, logreg, cluster=None):
    """Fit logistic or linear regressions, clustering if provided."""
    # fit rgression
    if logreg:
        model = sm.GLM(endog=y_train, exog=sm.add_constant(X_train), family=sm.families.Binomial())
    else:
        model = sm.OLS(endog=y_train, exog=sm.add_constant(X_train), hasconst=True)
    if cluster is not None:
        fit = model.fit(cov_kwds={"groups": cluster})
    else:
        fit = model.fit()
    return fit


def star_str(p_val):
    """Define statistical significance indicator for convenience."""

    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""

def clean_print(model, pred_var, covariates):
    """Define an easy-to-read output of provided covariates given an already fit regression."""

    df_results = pd.read_html(
        model.summary(yname=pred_var, xname=["intercept"] + covariates).tables[1].as_html(), header=0, index_col=0
    )[0].reset_index()
    df_results.columns = ["var", "coef", "se", "z", "p", "25", "75"]
    df_results["clean_print"] = [
        f'{row["coef"]} ({row["se"]}){star_str(row["p"])}' for row in df_results.to_dict("records")
    ]
    return df_results[["var", "clean_print", 'se']]

def convert_bool_to_int(*arrays):
    return [array.astype(float) for array in arrays]

def sm_regress(
    train_df: pd.DataFrame,
    covariates: list,
    pred_var: str,
    val_df: pd.DataFrame = None,
    print_vars: list = None,
    logreg: bool = True,
    f_covars: list = None,
    cluster_var: str = None,
    model_save_path: str = None,
    model_to_load: str = None,
    return_error: bool = False,
    continous: bool = False, 
    cutoff: float = 0.035,
    print_standard: bool = False,
    return_coef: bool = False,
    return_coef_name: str = None,
    scale: bool = True,
    show: bool = True
):
    """
    Regress pred_var ~ covariates.
    Additionally, print in an easy to copy-paste format if provided.
    Returns: (trues, preds, preds_standard_error)
    """

    # evaluate on train_df if no val_df is provided
    if val_df is None:
        val_df = train_df

    X_train, y_train, X_test, y_test = tt_split(train_df, val_df, covariates, pred_var, scale=scale)
    X_train, y_train, X_test, y_test = convert_bool_to_int(X_train, y_train, X_test, y_test)
    if not model_to_load:
        cluster = train_df[cluster_var] if cluster_var is not None else None
        model = fit_sm_regression(X_train=X_train, y_train=y_train, logreg=logreg, cluster=cluster)
        if model_save_path:
            model.save(model_save_path)
    else:
        model = sm.load(model_to_load)

    # predict
    X_test = X_test.astype(float)
    prediction_frame = model.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=0.05)
    y_pred_proba = np.array(prediction_frame["mean"])
    # calculate f-statistic
    if f_covars:
        big_model = model
        X_train_s, y_train_s, _, _ = tt_split(
            train_df, val_df, list(set(covariates).difference(set(f_covars))), pred_var, scale=scale
        )
        small_model = fit_sm_regression(X_train=X_train_s, y_train=y_train_s, logreg=logreg)
        f_stat = calculate_nested_f_statistic(small_model, big_model)
        
    if print_standard:
        print(model.summary())
        return y_test, y_pred_proba
    if print_vars:
        f_stat = f_stat if f_covars else None
        coefs = clean_print(model, pred_var, covariates)
        print_order = pd.DataFrame(print_vars, columns=["var"])
        if continous:
            stats = pd.DataFrame(
                [
                    ["f_stat", f_stat, ""],
                    #["ROC", round(roc_auc_score(y_test, y_pred_proba), 4)],
                    ["R2", round(model.pseudo_rsquared(), 4), ""],
                ],
                columns=coefs.columns,
            )
            print_df = print_order.merge(coefs, how="left").append(stats)
            if show:
                print(print_df.fillna(""))
        else:
            highrisk = np.where(y_pred_proba>cutoff,1,0)
            stats = pd.DataFrame(
                [
                    ["f_stat", f_stat, ""],
                    ["ROC", round(roc_auc_score(y_test, y_pred_proba), 4), ""],
                ],
                columns=coefs.columns,
            )
            print_df = pd.concat(
                            [print_order.merge(coefs, how="left"), stats],
                            ignore_index=True
                        )
            if show:
                print(print_df.fillna(""))
                print('Precision Recall F-Score Support')
                print(precision_recall_fscore_support(y_test, highrisk))

    if return_coef:
        return print_df[print_df['var']==return_coef_name]['clean_print'], print_df[print_df['var']==return_coef_name]['se']

    if return_error:
        return y_test, y_pred_proba, np.array(prediction_frame["mean_se"])
    else:
        return y_test, y_pred_proba