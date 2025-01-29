""" Functions for the builds of project-specific table outputs. """

from email.mime import base
from email.mime import base
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportions_ztest
import statsmodels
from torch import frac


"""
import os
os.environ['R_HOME'] = 'C:/ProgramData/Anaconda3/envs/rstudio/lib/R'

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
stats = importr('stats')
"""

from .definitions import CARDIAC_CONTROLS, DEMO_COVARIATES, MECHANISM_VARS, MECH_CONTROLS, MECH_VARS_POST_PRE, MEDS30, ECG_HEADERMEASURES
from .regress import sm_regress, star_str, sm_regress_mechanism

from tabulate import tabulate
from tabulate import tabulate


def print_worries_table(df: pd.DataFrame, sub_80: bool = False) -> None:
    """
    Build worries table which shows performance statistics when removing
    each of a set of known risk indicators.
    """

    df["age80"] = df["age"] >= 80
    df["EF_low"] = (df["most_recent_ef"] > 0) & (df["most_recent_ef"] <= 35)
    df["EF_high"] = (df["most_recent_ef"] > 35) & (df["most_recent_ef"] <= 100)
    df["EF_na"] = (df["EF_low"] == 0) & (df["EF_high"] == 0)
    df["True"] = True

    worries = [
        "True",
        "age80",
        "Acute_MI_death",
        "TROPT40_death",
        "EF_low_v_high",
        "EF_low_v_high_or_na",
        "VFVT",
        "qtc500",
        "LBBB",
        "CHF_Cardiomyopathy",
        "CAD",
    ]
    if sub_80:
        df = df.loc[df["age80"] == 0]
        worries.remove("age80")

    hrg = df.loc[df["y_pred_proba"] > 0.035]

    out = []
    for w in worries:
        values = []
        values.append(w)
        base_df = df
        if w == "EF_low_v_high":
            base_df = base_df.loc[df["EF_na"] == 0]
        if w in ["EF_low_v_high", "EF_low_v_high_or_na"]:
            w = "EF_low"
        values.append(round(sum(hrg[w]) / len(hrg), 3))  # frac_of_hrg
        values.append(round(sum((hrg[w]) & (hrg["dead1"])) / sum(hrg["dead1"]), 3))  # frac_of_hrg_deaths
        values.append(round(sum((base_df[w]) & (base_df["scd1"])) / sum(base_df[w]), 3))  # SCD_frac_in_W
        if w == "True":
            base_df[w] = 0
            hrg[w] = 0
        df_drop_w = base_df.loc[base_df[w] == 0]
        values.append(round(roc_auc_score(df_drop_w["scd1"], df_drop_w["y_pred_proba"]), 3))  # AUC in W=0
        values.append(
            round(sum(hrg.loc[hrg[w] == 0, "scd1"]) / len(hrg.loc[hrg[w] == 0]), 3)
        )  # SCD rate in new high risk group
        _, df_drop_w["scd1_hat"] = sm_regress(
            train_df=df_drop_w, covariates=["scd1_hat0", "cond_scd1_hat"], pred_var="scd1", cluster_var="ptId"
        )
        _, df_drop_w["new_proba"] = sm_regress(
            train_df=df_drop_w, covariates=["scd1_hat"], pred_var="scd1", cluster_var="ptId"
        )
        new_hrg = df_drop_w.loc[df_drop_w["new_proba"] > 0.035]
        values.append(round(len(new_hrg) / len(df_drop_w), 3))
        out.append(values)

    table = pd.DataFrame(
        out,
        columns=[
            "W",
            "frac_of_hrg",
            "frac_of_hrg_deaths",
            "SCD_frac_in_W",
            "AUC_out_W",
            "SCD_frac_in_hrg_out_W",
            "rereg_hrg_prop",
        ],
    )
    print(table)

def compute_scd_stats(df: pd.DataFrame, percentile: float = 0.02, models: list = ['scd1_hat', 'scd1_hat_baseline_new_imp', 'random'], outcome_var: str = 'scd1'):
    
    stats_list = [percentile * df.shape[0]]
    
    all_scds = df[outcome_var].sum().item()
    
    for model in models:
        if model == 'random':
            df_size = df.shape[0]
            num_samples = int(percentile * df_size)
            df_highrisk = df.sample(n=num_samples, random_state=42)
        else:
            df = df.sort_values(by=model, ascending=False)
            df_size = df.shape[0]
            df_highrisk = df.iloc[: int(percentile * df_size)]
        stats_list.extend([df_highrisk[outcome_var].sum().item()/(df_highrisk.shape[0]),
                           df_highrisk[outcome_var].sum().item()/all_scds])
    
    return stats_list
    

def print_highrisk_table_by_subpops(
    df: pd.DataFrame, percentile: float = 0.02, models: list = ['scd1_hat', 'scd1_hat_baseline_new_imp'], outcome_var: str ='scd1',  savepath: str = None,
    return_df = False):
    """
    Print high risk table which shows scd rates in high risk groups defined using our predictor for different subpopulations:
    (1) Define percentile of the population considered as highrisk based on our predictor
    (2) Split population into subpopulations e.g. patients below 80
    (3) For each subpopulation compute scd_rate P(SCD=1| Highrisk=1) and share of SCDs captured P(Highrisk=1 | SCD=1)
    """
    
    subpops = [
        "all",
        "age below 80",
        "age above 80",
        "ef below 35",
        "ef above 35",
        "ef not available",
        "AMI within 40 days of ECG",
        "No AMI within 40 days of ECG"
    ]
    
    parameters = ["SCD rate HR", "% SCDs captured"]
    
    column_names = ["N"]
    statistics = []
    
    for model in models:
        for parameter in parameters:
            name = model + " - \n" + parameter
            column_names.append(name)
    
    #all
    df_all = df
    statistics.append(compute_scd_stats(df_all, percentile, models, outcome_var=outcome_var))
    #sub80
    sub80 = df[df['age']<80]
    statistics.append(compute_scd_stats(sub80, percentile, models, outcome_var=outcome_var))
    #above80
    above80 = df[df['age']>=80]
    statistics.append(compute_scd_stats(above80, percentile, models, outcome_var=outcome_var))
    #ef below 35
    df['most_recent_ef'] = df['most_recent_ef'].replace(-10, np.nan)
    efsub35 = df[df['most_recent_ef']<35]
    statistics.append(compute_scd_stats(efsub35, percentile, models, outcome_var=outcome_var))
    #ef above 35
    df['most_recent_ef'] = df['most_recent_ef'].replace(-10, np.nan)
    efabove35 = df[df['most_recent_ef']>=35]
    statistics.append(compute_scd_stats(efabove35, percentile, models, outcome_var=outcome_var))
    #ef_na
    df['most_recent_ef'] = df['most_recent_ef'].replace(-10, np.nan)
    efna = df[df['most_recent_ef'].isna()]
    statistics.append(compute_scd_stats(efna, percentile, models, outcome_var=outcome_var))
    #Acute MI
    ami = df[df['Acute_MI']==1]
    statistics.append(compute_scd_stats(ami, percentile, models, outcome_var=outcome_var))
    #No Acute MI
    noami = df[df['Acute_MI']==0]
    statistics.append(compute_scd_stats(noami, percentile, models, outcome_var=outcome_var))
    
    output_df = pd.DataFrame(
        statistics, columns=column_names, index=subpops
    )
    output_df = output_df.round(decimals=3)
    
    # Adjust display options
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.width', None)        # Automatically adjust display width to fit content
    
    print(output_df)

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        output_df.to_csv(savepath)
    if return_df:
        return output_df
        

def print_highrisk_table(
    df: pd.DataFrame, cut_value: float = 0.035, savepath: str = None, pred_var: str = "y_pred_proba", outcome_var: str ='scd1', exclude_marginals: bool = True
) -> None:
    """
    Print high risk table which shows the stats of high risk groups defined by:
    (1) Proportions of the population in range(.02, .22, .02)
    (2) HRG defined by the point where cumulative incidence drops below cut_value
    (3) HRG defined by the point where predicted risk drops below cut_value
    """

    highrisk_ls = []
    # proportion based cuts
    for prop in np.arange(0.02, 0.22, 0.02):
        highrisk_df = get_highrisk_df(df, outcome_var, pred_var, cut_value=prop, cut_type="prop")
        highrisk_ls.append(
            [
                len(highrisk_df) / len(df),
                sum(highrisk_df[outcome_var]) / sum(df[outcome_var]),
                sum(highrisk_df[outcome_var]) / len(highrisk_df),
                np.min(highrisk_df[pred_var]),
                proportions_ztest(count=sum(highrisk_df[outcome_var]), nobs=len(highrisk_df), value=cut_value, prop_var=0.5)[
                    -1
                ],
            ]
        )
    # cumulative scd rate based cut
    c_highrisk_df = get_highrisk_df(df, outcome_var, pred_var, cut_value=cut_value, cut_type="mean")
    if c_highrisk_df is None:
        highrisk_ls.append([float("NaN") for i in range(5)])
    else:
        highrisk_ls.append(
            [
                len(c_highrisk_df) / len(df),
                sum(c_highrisk_df[outcome_var]) / sum(df[outcome_var]),
                sum(c_highrisk_df[outcome_var]) / len(c_highrisk_df),
                np.min(c_highrisk_df[pred_var]),
                proportions_ztest(
                    count=sum(c_highrisk_df[outcome_var]), nobs=len(c_highrisk_df), value=cut_value, prop_var=0.5
                )[-1],
            ]
        )
    # marginal scd rate based cut
    m_highrisk_df = get_highrisk_df(df, outcome_var, pred_var, cut_value=cut_value, cut_type="marginal")
    if m_highrisk_df is None or len(m_highrisk_df)==0:
        highrisk_ls.append([float("NaN") for i in range(5)])
    else:
        highrisk_ls.append(
            [
                len(m_highrisk_df) / len(df),
                sum(m_highrisk_df[outcome_var]) / sum(df[outcome_var]),
                sum(m_highrisk_df[outcome_var]) / len(m_highrisk_df),
                np.min(m_highrisk_df[pred_var]),
                proportions_ztest(
                    count=sum(m_highrisk_df[outcome_var]), nobs=len(m_highrisk_df), value=cut_value, prop_var=0.5
                )[-1],
            ]
        )

    output_df = pd.DataFrame(
        highrisk_ls, columns=["prop", "recall", "precision (mean SCD)", "marginal_SCD", "p"]
    ).sort_values("prop")
    output_df = output_df.round(decimals=3)
    
    #Filter out marginal parts for now
    if exclude_marginals:
        output_df = output_df[["prop", "recall (TPR)", "precision (mean SCD)"]]

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        output_df.to_csv(savepath)
    else:
        print(output_df)


def get_highrisk_df(df: pd.DataFrame, true_var: str, pred_var: str, cut_value: float = 0.035, cut_type: str = "mean"):
    """Define HRG given a cut type in ['mean', 'marginal', 'prop']"""

    df = df.sort_values(pred_var, ascending=False)

    if cut_type == "mean":
        # Define "cum_mean" variable in df
        df = cum_mean(df, true_var, pred_var)
        # Get cum_mean crossing points over CUTOFF
        cross = crossing_points(df, "cum_mean", cut_value)

        # Subtract 1 because crossing_points() returns all values involved in crossing of threshold
        # We keep those ECGs ordered before the last cross over CUTOFF threshold
        n_highrisk = int(np.max(cross["order"]) - 1) if not pd.isna(np.max(cross["order"])) else float("NaN")

        if pd.isna(n_highrisk):
            return None
        else:
            return df.iloc[:n_highrisk]
    elif cut_type == "marginal":
        return df.loc[df[pred_var] >= cut_value]
    elif cut_type == "prop":
        return df.iloc[: int(cut_value * len(df))]
    else:
        raise ValueError("One of cut_type ('mean',  'marginal', 'prop') must be provided.")


def cum_mean(df: pd.DataFrame, true_var: str, pred_var: str) -> pd.DataFrame:
    """
    Define cumulative incidence of true_var for all ECGs
    at or below pred_var value.
    Mainly, show cumulative SCD incidence for patients riskier
    than a given risk level.
    """

    df = df.sort_values(pred_var, ascending=False)
    df["order"] = range(len(df))
    df["cum_mean"] = df[true_var].expanding().mean()

    tails = df.groupby(pred_var)[[pred_var, "cum_mean"]].tail(1)

    df = df.drop(columns=["cum_mean"])
    df = df.merge(tails, how="left", on=pred_var)

    return df


def crossing_points(df: pd.DataFrame, var: str, cutoff: float) -> pd.DataFrame:
    """Find point where cumulative indicence drops below cutoff"""

    df["use_var"] = df[var] - cutoff
    out = df[np.sign(df["use_var"]).diff().fillna(0).ne(0)].copy()
    out = out.drop(columns=["use_var"])

    return out


def regression_table(val_df: pd.DataFrame, ROC_curve: bool = False, covariate_ls: list = None, print_vars: list = None, print_standard=False, pred_var ='scd1') -> None:
    """
    Build primary regression table,
    showing 'scd1_hat' predictive performance relative to demo covariates
    """
    
    if covariate_ls == None:
        covariate_ls = [["scd1_hat"],
                        ["scd1_hat", 'cond_scd1_hat'],
                        ["scd1_hat", 'cond_scd1_hat', 'scd2_hat', 'scd3mo_hat', 'scd6mo_hat'], 
                        ["scd1_hat", 'cond_scd1_hat', 'scd2_hat', 'scd3mo_hat', 'scd6mo_hat', 'TROPT1_hat', 'Acute_MI1_hat'],
                        ['heartRate', 'pDuration', 'qrsDuration', 'qtInterval', 'qtcb', 'qtcf', 'rrInterval', "ST_elevation", "ST_depression"],
                        DEMO_COVARIATES + CARDIAC_CONTROLS,  
                        DEMO_COVARIATES + CARDIAC_CONTROLS + ["scd1_hat"], 
                        DEMO_COVARIATES + CARDIAC_CONTROLS +["scd1_hat", 'cond_scd1_hat', 'scd2_hat', 'scd3mo_hat', 'scd6mo_hat']]
        roc_curves = [True,False, False, False, False, False,False,False]
    else:
        roc_curves = [False]*len(covariate_ls)
    

    if print_vars == None:
        print_vars= ["age", "female", "most_recent_ef", "ef_na", "scd1_hat", 'cond_scd1_hat', 'scd2_hat', 'scd3mo_hat', 'scd6mo_hat', 'TROPT1_hat', 'Acute_MI1_hat']

    for covariates, roc_curve in zip(covariate_ls,roc_curves):
        print(covariates)
        f_covars = [c for c in CARDIAC_CONTROLS if c in covariates]
        val_df = val_df.dropna(subset=covariates)
        sm_regress(
            val_df,
            covariates=covariates,
            pred_var=pred_var,
            f_covars=f_covars,
            cluster_var="ptId",
            print_vars= print_vars, #['heartRate', 'pDuration', 'qrsDuration', 'qtInterval', 'qtcb', 'qtcf', 'rrInterval', "ST_elevation", "ST_depression"],
            roc_curve_graph=roc_curve,
            print_standard=print_standard
        )
        print("LENGTH: ", len(val_df))
        print("N SCD: ", sum(val_df["scd1"]))

def regression_table_scd1_only(val_df: pd.DataFrame) -> None:
    """
    Build primary regression table,
    showing 'scd1_hat' predictive performance relative to demo covariates
    """

    covariate_ls = [["scd1_hat"]]

    for covariates in covariate_ls:
        print(covariates)
        f_covars = [c for c in CARDIAC_CONTROLS if c in covariates]
        sm_regress(
            val_df,
            covariates=covariates,
            pred_var="scd1",
            f_covars=f_covars,
            cluster_var="ptId",
            print_vars=["age", "female", "most_recent_ef", "ef_na", "scd1_hat"],
        )
        print("LENGTH: ", len(val_df))
        print("N SCD: ", sum(val_df["scd1"]))

def regression_table_meds(val_df: pd.DataFrame) -> None:
    """
    Build regression table for medication analysis,
    Regress on scdhat <- (studyAcquiredHour).FE + ptId.FE + months.FE + age + 
    {dummies for each medication filled in 30 days before ECG} + AcuteMI (double-check definition) 
    + check table if anything off the shelf

    Need to build medication dummies in data

    """
    month_dummies = pd.get_dummies(val_df[['studyId','month']], columns=['month'], drop_first=True)
    val_df = val_df.merge(month_dummies, on='studyId', how='left')
    month_dummies = month_dummies.drop('studyId', axis=1)
    month_dummies_names = list(month_dummies.columns)

    hour_dummies = pd.get_dummies(val_df[['studyId','studyAcquiredHour']], columns=['studyAcquiredHour'], drop_first=True)
    val_df = val_df.merge(hour_dummies, on='studyId', how='left')
    hour_dummies = hour_dummies.drop('studyId', axis=1)
    hour_dummies_names = list(hour_dummies.columns)

    patient_dummies = pd.get_dummies(val_df[['studyId','ptId']], columns=['ptId'], drop_first=True)
    val_df = val_df.merge(patient_dummies, on='studyId', how='left')
    patient_dummies = patient_dummies.drop('studyId', axis=1)
    patient_dummies_names = list(patient_dummies.columns)

    print(month_dummies_names)
    print(hour_dummies_names)

    covariate_ls = [MEDS30, MEDS30 +['Acute_MI', 'most_recent_ef', 'ef_na'] + DEMO_COVARIATES, 
    MEDS30 +['Acute_MI', 'most_recent_ef', 'ef_na'] + DEMO_COVARIATES + month_dummies_names + hour_dummies_names
    ] #MEDS30 +['Acute_MI', 'most_recent_ef', 'ef_na'] + DEMO_COVARIATES + month_dummies_names + hour_dummies_names + patient_dummies_names

    print_vars = MEDS30 + ['age', 'female','Acute_MI', 'most_recent_ef', 'ef_na']

    label = ['Meds only', 'Meds + demo + AMI & EF', 'Meds + demo + AMI & EF + month FE + hour FE',
    'Meds + demo + AMI & EF + month FE + hour FE + patient FE']

    for i, covariates in enumerate(covariate_ls):
        print(label[i])
        #f_covars = [c for c in CARDIAC_CONTROLS if c in covariates] #skip F-stat calculation for now
        sm_regress(
            val_df,
            covariates=covariates,
            pred_var="scd1_hat",
            #f_covars=f_covars,
            cluster_var="ptId",
            print_vars=print_vars,
            continous= True
        )
        print("LENGTH: ", len(val_df))
        print("N SCD: ", sum(val_df["scd1"]))


def print_twobytwo(df: pd.DataFrame, row_var: str, col_var: str, incidence_var: str):
    """Build and print 2x2"""

    true_df = df.loc[df[incidence_var] == 1]
    true_count = pd.crosstab(true_df[row_var], true_df[col_var]).to_numpy()
    total_count = pd.crosstab(df[row_var], df[col_var]).to_numpy()
    calc_tbl(true_count, total_count)
    print('Observations per bucket')
    print(total_count)


def calc_tbl(true_count: np.array, total_count: np.array) -> None:
    """Calculate and assemble 2x2"""

    base_tbl = true_count / total_count
    se = np.sqrt(((true_count / total_count) * (1 - true_count / total_count)) / total_count)

    out = [[], []]
    for row in range(2):
        for col in range(2):
            out[row].append(f"{round(base_tbl[row,col],3)} ({round(se[row,col],3)})")
    out_df = pd.DataFrame(out)
    row_p = []
    col_p = []
    for row in range(2):
        p = proportions_ztest(true_count[row, :], total_count[row, :])[1]
        row_p.append(f"{round(p,3)}{star_str(p)}")
    for col in range(2):
        p = proportions_ztest(true_count[:, col], total_count[:, col])[1]
        col_p.append(f"{round(p,3)}{star_str(p)}")
    out_df["row_p"] = pd.Series(row_p)
    col_p.append("")
    # print(pd.DataFrame(col_p, columns=out_df.columns))
    out_df = out_df.append(pd.DataFrame([col_p], columns=out_df.columns))
    print(out_df)

def print_threebytwo(df: pd.DataFrame, row_var: str, col_var: str, incidence_var: str):
    """Build and print 3x2"""

    true_df = df.loc[df[incidence_var] == 1]
    print(pd.crosstab(true_df[row_var], true_df[col_var]))
    true_count = pd.crosstab(true_df[row_var], true_df[col_var]).to_numpy()
    total_count = pd.crosstab(df[row_var], df[col_var]).to_numpy()
    #print(true_count)
    #print(total_count)
    calc_tbl_3x2(true_count, total_count)


def calc_tbl_3x2(true_count: np.array, total_count: np.array) -> None:
    """Calculate and assemble 2x2"""

    base_tbl = true_count / total_count
    #print(base_tbl)
    se = np.sqrt(((true_count / total_count) * (1 - true_count / total_count)) / total_count)
    #print(se)

    out = [[], [], []]
    for row in range(3):
        for col in range(2):
            out[row].append(f"{round(base_tbl[row,col],3)} ({round(se[row,col],3)})")
    out_df = pd.DataFrame(out)
    row_p = []
    #col_p = []
    for row in range(3):
        p = proportions_ztest(true_count[row, :], total_count[row, :])[1]
        row_p.append(f"{round(p,3)}{star_str(p)}")
    """
    for col in range(2): #Since we have more than 2 samples we need to use the R function for proportions tests
        prop = stats.prop_test(x=true_count[:, col], n=total_count[:, col])
        print(prop)
        p = prop[1]
        col_p.append(f"{round(p,3)}{star_str(p)}")
    """
    out_df["row_p"] = pd.Series(row_p)
    #col_p.append("")
    # print(pd.DataFrame(col_p, columns=out_df.columns))
    #out_df = out_df.append(pd.DataFrame([col_p], columns=out_df.columns))
    print(out_df)
    print("3 sample proportions test functionality not available in any python package implementation possible using R if needed")


def calc_tbl(true_count: np.array, total_count: np.array) -> None:
    """Calculate and assemble 2x2"""

    base_tbl = true_count / total_count
    se = np.sqrt(((true_count / total_count) * (1 - true_count / total_count)) / total_count)

    out = [[], []]
    for row in range(2):
        for col in range(2):
            out[row].append(f"{round(base_tbl[row,col],3)} ({round(se[row,col],3)})")
    out_df = pd.DataFrame(out)
    row_p = []
    col_p = []
    for row in range(2):
        p = proportions_ztest(true_count[row, :], total_count[row, :])[1]
        row_p.append(f"{round(p,3)}{star_str(p)}")
    for col in range(2):
        p = proportions_ztest(true_count[:, col], total_count[:, col])[1]
        col_p.append(f"{round(p,3)}{star_str(p)}")
    out_df["row_p"] = pd.Series(row_p)
    col_p.append("")
    # print(pd.DataFrame(col_p, columns=out_df.columns))
    out_df = out_df.append(pd.DataFrame([col_p], columns=out_df.columns))
    print(out_df)
def print_threebytwo(df: pd.DataFrame, row_var: str, col_var: str, incidence_var: str):
    """Build and print 3x2"""

    true_df = df.loc[df[incidence_var] == 1]
    print(pd.crosstab(true_df[row_var], true_df[col_var]))
    true_count = pd.crosstab(true_df[row_var], true_df[col_var]).to_numpy()
    total_count = pd.crosstab(df[row_var], df[col_var]).to_numpy()
    #print(true_count)
    #print(total_count)
    calc_tbl_3x2(true_count, total_count)


def calc_tbl_3x2(true_count: np.array, total_count: np.array) -> None:
    """Calculate and assemble 2x2"""

    base_tbl = true_count / total_count
    #print(base_tbl)
    se = np.sqrt(((true_count / total_count) * (1 - true_count / total_count)) / total_count)
    #print(se)

    out = [[], [], []]
    for row in range(3):
        for col in range(2):
            out[row].append(f"{round(base_tbl[row,col],3)} ({round(se[row,col],3)})")
    out_df = pd.DataFrame(out)
    row_p = []
    #col_p = []
    for row in range(3):
        p = proportions_ztest(true_count[row, :], total_count[row, :])[1]
        row_p.append(f"{round(p,3)}{star_str(p)}")
    """
    for col in range(2): #Since we have more than 2 samples we need to use the R function for proportions tests
        prop = stats.prop_test(x=true_count[:, col], n=total_count[:, col])
        print(prop)
        p = prop[1]
        col_p.append(f"{round(p,3)}{star_str(p)}")
    """
    out_df["row_p"] = pd.Series(row_p)
    #col_p.append("")
    # print(pd.DataFrame(col_p, columns=out_df.columns))
    #out_df = out_df.append(pd.DataFrame([col_p], columns=out_df.columns))
    print(out_df)
    print("3 sample proportions test functionality not available in any python package implementation possible using R if needed")


def calc_tbl(true_count: np.array, total_count: np.array) -> None:
    """Calculate and assemble 2x2"""

    base_tbl = true_count / total_count
    se = np.sqrt(((true_count / total_count) * (1 - true_count / total_count)) / total_count)

    out = [[], []]
    for row in range(2):
        for col in range(2):
            out[row].append(f"{round(base_tbl[row,col],3)} ({round(se[row,col],3)})")
    out_df = pd.DataFrame(out)
    row_p = []
    col_p = []
    for row in range(2):
        p = proportions_ztest(true_count[row, :], total_count[row, :])[1]
        row_p.append(f"{round(p,3)}{star_str(p)}")
    for col in range(2):
        p = proportions_ztest(true_count[:, col], total_count[:, col])[1]
        col_p.append(f"{round(p,3)}{star_str(p)}")
    out_df["row_p"] = pd.Series(row_p)
    col_p.append("")
    # print(pd.DataFrame(col_p, columns=out_df.columns))
    out_df = out_df.append(pd.DataFrame([col_p], columns=out_df.columns))
    print(out_df)

def defib_regression_table(df: pd.DataFrame, depvar: str, covariate_ls=None, interaction_terms=None, print_vars=None, print_standard=False) -> None:
    """
    Build regression table showing ability of defibrillators to reduce death risk
    conditional on predicted risk and HRG assignment.
    """

    if print_vars == None:
        print_vars = ["age", "female", "most_recent_ef","ef_na","defib_proc_history", "highrisk", 
         "scd1_hat", "highrisk*defib_proc_history", "scd1_hat*defib_proc_history"]

    if covariate_ls == None:
        covariate_ls = [
            DEMO_COVARIATES + CARDIAC_CONTROLS + ["highrisk"],
            DEMO_COVARIATES + CARDIAC_CONTROLS + ["scd1_hat"],
            DEMO_COVARIATES + CARDIAC_CONTROLS + ["highrisk", "defib_proc_history", "highrisk*defib_proc_history"],
            DEMO_COVARIATES + CARDIAC_CONTROLS + ["scd1_hat", "defib_proc_history", "scd1_hat*defib_proc_history"],
        ]
    
    if interaction_terms==None:
        interaction_terms = ["highrisk", "scd1_hat"]
    #df["highrisk"] = df["highrisk"].astype(int)
    for term in interaction_terms:
        df[term + "*defib_proc_history"] = df[term] * df["defib_proc_history"]

    for covariates in covariate_ls:
        f_covars = [c for c in CARDIAC_CONTROLS if c in covariates]
        sm_regress(
            df,
            covariates=covariates,
            pred_var=depvar,
            f_covars=f_covars,
            print_vars=print_vars,
            print_standard=print_standard
        )
    print("LENGTH:", len(df))
    print("N DV:", sum(df[depvar]))
    print("N Defib in DV:", len(df.loc[(df[depvar] == 1) & (df["defib_proc_history"] == 1)]))

def defib_scdhat_regression_table(df: pd.DataFrame, depvar: str) -> None:
    """
    Build regression table showing ability of our scdhat and HRG indicator to predict
    life-saving shocks given by an ICD.  
    """

    covariate_ls = [
        DEMO_COVARIATES + CARDIAC_CONTROLS + ["highrisk"],
        DEMO_COVARIATES + CARDIAC_CONTROLS + ["scd1_hat"],
    ]
    #DEMO_COVARIATES + CARDIAC_CONTROLS + ["highrisk", "defib_proc_history", "highrisk*defib_proc_history"],
    # DEMO_COVARIATES + CARDIAC_CONTROLS + ["scd1_hat", "defib_proc_history", "scd1_hat*defib_proc_history"],

    # Slice into EF datasets
    print(df['most_recent_ef'].mode())
    print(df.shape)
    ef_df = df[df['most_recent_ef']!=-10]
    no_ef_df = df[df['most_recent_ef']==-10]

    print(ef_df.shape)
    print(no_ef_df.shape)

    labels = ["ef data", "no ef data"]
    frame_df = [ef_df,no_ef_df]

    # interaction_terms = ["highrisk", "scd1_hat"]
    # df["highrisk"] = df["highrisk"].astype(int)
    # for term in interaction_terms:
    #     df[term + "*defib_proc_history"] = df[term] * df["defib_proc_history"]
    for label, frame in zip(labels,frame_df):
        print(label)
        print()
        if label == "no ef data":
            drop_elements = ['most_recent_ef','ef_na']
            covariate_ls_new = [[x for x in y if x not in drop_elements] for y in covariate_ls]
        else:
            covariate_ls_new = covariate_ls
        for covariates in covariate_ls_new:
            f_covars = [c for c in CARDIAC_CONTROLS if c in covariates]
            sm_regress(
                frame,
                covariates=covariates,
                pred_var=depvar,
                f_covars=f_covars,
                print_vars=[
                    "age",
                    "female",
                    "most_recent_ef",
                    "ef_na",
                    "defib_proc_history",
                    "highrisk",
                    "scd1_hat",
                    "highrisk*defib_proc_history",
                    "scd1_hat*defib_proc_history",
                ],
            )
    print("LENGTH:", len(df))
    print("N DV:", sum(df[depvar]))
    print("N Defib in DV:", len(df.loc[(df[depvar] == 1) & (df["defib_proc_history"] == 1)]))

def defib_regression_table_ef(df: pd.DataFrame, depvar: str) -> None:
    """
    Build regression table showing ability of defibrillators to reduce death risk
    conditional on predicted risk and HRG assignment.
    """
    demo = DEMO_COVARIATES

    demo.remove('most_recent_ef')
    covariate_ls = [
        demo + CARDIAC_CONTROLS + ["ef_below_35"],
        demo + CARDIAC_CONTROLS + ["most_recent_ef"],
        demo + CARDIAC_CONTROLS + ['ef_below_35', "defib_proc_history", "ef_below_35*defib_proc_history"],
        demo + CARDIAC_CONTROLS + ["most_recent_ef", "defib_proc_history", "most_recent_ef*defib_proc_history"],
    ]
    interaction_terms = ["highrisk", "scd1_hat"]
    df["highrisk"] = df["highrisk"].astype(int)
    for term in interaction_terms:
        df[term + "*defib_proc_history"] = df[term] * df["defib_proc_history"]

    for covariates in covariate_ls:
        f_covars = [c for c in CARDIAC_CONTROLS if c in covariates]
        sm_regress(
            df,
            covariates=covariates,
            pred_var=depvar,
            f_covars=f_covars,
            print_vars=[
                "age",
                "female",
                "defib_proc_history",
                "ef_na",
                "ef_below_35",
                "most_recent_ef",
                "ef_below_35*defib_proc_history",
                "most_recent_ef*defib_proc_history",
            ],
        )
    print("LENGTH:", len(df))
    print("N DV:", sum(df[depvar]))
    print("N Defib in DV:", len(df.loc[(df[depvar] == 1) & (df["defib_proc_history"] == 1)]))

### Table 1

"""
# Mapping from CARDIAC_CONTROLS to labels in table
{
    "defib_proc_history": "Defibrillator Procedure History",
    "ST_elevation": "ST Elevation",
    "ST_depression": "ST Depression",
    "qtc500": "QTc>500",
    "LBBB": "Prior LBBB Label",
    "antiarr": "Prior Anti-Arrythmic Drug Prescription",
    "deltawave": "Prior Delta Wave Label",
    "CHF_Cardiomyopathy": "CHF & Cardiomyopathy Within Prior 365 Days",
    "Hypertension": "Hypertension Within Prior 365 Days",
    "CAD": "CAD Within Prior 365 Days",
    "Diabetes": "Diabetes Within Prior 365 Days",
    "Hyperlipidemia": "Hyperlipidemia Within Prior 365 Days",
    "TROPT40": "Troponin 34 ng/L+ [M] or 16 ng/L+ [W] (w/in 40 days prior)",
    "Acute_MI": "Acute MI (w/in 40 days prior)",
    "Old_MI": "Old MI"
}
"""

def build_summaries(df, last_ecg=False, median=False):

    def get_summaries(df, vars, median=False):
        def mean_se(col):
            return f"{round(np.mean(col), 3)} ({round(np.std(col),4)})"
        
        def median_calc(col):
            return f"{round(np.median(col), 3)}"

        scd = df.loc[df["scd1"] == 1]
        noscd = df.loc[df["scd1"] == 0]
        out = [["N", len(df), len(df), sum(df["scd1"]), sum(~df["scd1"])]]
        if median:
            for var in vars:
                n = sum(~df[var].isna())
                out.append([var, n, median_calc(df[var]), median_calc(scd[var]), median_calc(noscd[var])])
            print(pd.DataFrame(out, columns=["Var", f"N (total = {len(df)})", "All", "SCD=1", "SCD=0"]))
        else:
            for var in vars:
                n = sum(~df[var].isna())
                out.append([var, n, mean_se(df[var]), mean_se(scd[var]), mean_se(noscd[var])])
            print(pd.DataFrame(out, columns=["Var", f"N (total = {len(df)})", "All", "SCD=1", "SCD=0"]))

    # Return EF 0s to N
    # ef_dummy would create misleading statistics if not handled
    df.loc[df["most_recent_ef"] <= 0, "most_recent_ef"] = float("NaN")
    df["ef_available"] = ~df["most_recent_ef"].isna()  

    if last_ecg:
        # take most recent ECG by patient and summarize
        df = df.sort_values("days", ascending=False)
        df = df.groupby("ptId").head(1)
        pt_vars = (
            ["scd1", 'dead1', "age", "female", "ef_available", "most_recent_ef", "defib_proc_history", "VFVT"]
            + [c for c in CARDIAC_CONTROLS if c not in ["ICD_VT", "ICD_VF", "Phi_VT"]]
            + ["VFVT", "Acute_MI_death", "TROPT40_death"]
            + ["outpatient_ED","inpatient"]
            + ["age","prInterval","qrsDuration","qtInterval","qrsFrontAxis","qrsHorizAxis","rrInterval","pDuration","atrialrate","meanqrsdur"] #"sex","atrialratestddev",
        )
        get_summaries(df, pt_vars, median)
    else:
        pass # ignoring the rest because it's complicated and you don't need it

def risk_factors_high_vs_low_risk(df,risk_factors=[c for c in CARDIAC_CONTROLS if c not in ["ICD_VT", "ICD_VF", "Phi_VT"]]
            + ["VFVT", "Acute_MI_death", "TROPT40_death"]):
    df["risk_factor_sum"] = 0
    for c in risk_factors:
        df["risk_factor_sum"] += df[c]
    
    highrisk_df = df.loc[df["highrisk"]==1]
    rest_df = df.loc[df["highrisk"]==0]

    out = {"Highrisk patients":[np.median(highrisk_df["risk_factor_sum"])],"Other patients":[np.median(rest_df["risk_factor_sum"])]}

    print(pd.DataFrame(out))

"""
def ef35_vs_other_scd_rate(df):

    def ef_based_scd(df,totals_df):
        ef_35_df = df[df['most_recent_ef']>=35]
        ef_below_35_df = df[(df['most_recent_ef']<35) & (df['ef_na']==0)]
        no_ef = df[df['ef_na']==1]

        scd_rate_list = [np.sum(ef_35_df['scd1'])/ef_35_df.shape[0],np.sum(ef_below_35_df['scd1'])/ef_below_35_df.shape[0], np.sum(no_ef['scd1'])/no_ef.shape[0]]

        frac_of_sample_list = [ef_35_df.shape[0]/totals_df.shape[0],ef_below_35_df.shape[0]/totals_df.shape[0],no_ef.shape[0]/totals_df.shape[0]]

        string_list = [f'{scd_rate}  ({frac})' for scd_rate, frac in zip(scd_rate_list,frac_of_sample_list)]

        return scd_rate_list, frac_of_sample_list, string_list
    
    
    
    highrisk_df = df.loc[df["highrisk"]==1]
    rest_df = df.loc[df["highrisk"]==0]

    highrisk_scd, highrisk_frac, highrisk_str = ef_based_scd(highrisk_df,df)
    rest_scd, rest_frac, rest_str = ef_based_scd(rest_df,df)

    out_str = {"Highrisk patients":highrisk_str,"Other patients":rest_str}

    print('scd rate (fraction of sample)')
    print(pd.DataFrame(out_str, index=['ef >=35', 'ef<35', 'no ef']))
"""

def ef35_vs_other_scd_rate(df: pd.DataFrame, row_var: str, col_var: str, incidence_var: str):
    """Build and print 3x2"""

    true_df = df.loc[df[incidence_var] == 1]
    cross_tab_for_labels = pd.crosstab(true_df[row_var], true_df[col_var],margins=True)
    true_count = pd.crosstab(true_df[row_var], true_df[col_var],margins=True).to_numpy()
    total_count = pd.crosstab(df[row_var], df[col_var],margins=True).to_numpy()
    total = df.shape[0]
    print(total_count)
    #print(true_count)
    #print(total_count)
    calc_tbl_incidence_share(true_count, total_count,cross_tab_for_labels, total)


def calc_tbl_incidence_share(true_count: np.array, total_count: np.array, cross_tab_for_labels: pd.DataFrame, total) -> None:
    """Calculate and assemble 4x3" (incl.marginals)"""

    total_count = total_count + 0.0000000000000000000000000000000000001 #avoid division error
    base_tbl = true_count / total_count #incidence per subgroup
    total_share = total_count / total
    #print(base_tbl)

    out = [[], [], [], []]
    for row in range(base_tbl.shape[0]):
        for col in range(base_tbl.shape[1]):
            out[row].append(f"{round(base_tbl[row,col],3)} ({round(total_share[row,col],3)})")
    out_df = pd.DataFrame(out,index=cross_tab_for_labels.index, columns=cross_tab_for_labels.columns)
    
    print(out_df)
    
#WORK IN PROGRESS
def scd_mechanism_analysis_table(val_df,vars_of_interest=MECHANISM_VARS, CUTOFF=0.035, mechanism_controls = False, condition_at_ECG = False):

    #TODO Dropping variables currently not available
    if 'Malaise_fatigue1' in vars_of_interest: vars_of_interest.remove("Malaise_fatigue1")

    output_table = pd.DataFrame(columns=['var','SCD_hat coef', 'SE', 'p', 'AUC', 'Share of highrisk', 'Share of all'])

    if mechanism_controls:
        new_controls = MECH_CONTROLS
        if 'Malaise_fatigue' in new_controls: new_controls.remove("Malaise_fatigue")
        covariates =  DEMO_COVARIATES + new_controls + ["scd1_hat"]
    else:
        covariates =  DEMO_COVARIATES + ["scd1_hat"]

    for i,var in enumerate(vars_of_interest):
        if condition_at_ECG:
            control = MECH_VARS_POST_PRE[var]
            covariates += [control]

        covariates_view = [var] + covariates
        test_df = val_df[covariates_view]
        #print(test_df.corr())

        #perform regression analysis
        try:
            coef_scdhat, SE_val, p, AUC_val = sm_regress_mechanism(val_df,cluster_var="ptId",covariates=covariates,pred_var=var)
        except FloatingPointError:
            coef_scdhat, SE_val, p, AUC_val = (np.NaN,np.NaN,np.NaN,'Floating point error')
        except statsmodels.tools.sm_exceptions.PerfectSeparationError:
            coef_scdhat, SE_val, p, AUC_val = (np.NaN,np.NaN,np.NaN,'Perfect Separation Error Check Feature Engineering')

        #calculate share of condition in highrisk and full populations
        highrisk_df = val_df.loc[val_df["y_pred_proba"] > CUTOFF]
        var_highrisk_share = highrisk_df.loc[highrisk_df[var]==1].shape[0]/highrisk_df.shape[0]
        var_all_share = val_df.loc[val_df[var]==1].shape[0]/val_df.shape[0]
        
        #store metrics
        row = [var,coef_scdhat, SE_val, p, AUC_val,var_highrisk_share,var_all_share]
        output_table.loc[i] = row
    
    print(tabulate(output_table, headers='keys', tablefmt='psql'))

def print_twobytwo_incidence(df: pd.DataFrame, row_var: str, col_var: str, incidence_var: str):
    """Build 2x2"""

    true_df = df.loc[df[incidence_var] == 1]

    true_df = true_df.sort_values("days", ascending=False)
    true_df = true_df.groupby("ptId").head(1)

    true_count = pd.crosstab(true_df[row_var], true_df[col_var], margins=True).to_numpy()
    #total_count = pd.crosstab(df[row_var], df[col_var],margins=True).to_numpy()
    share_table = true_count/np.array([[true_df.shape[0]]])
    some_crosstab = pd.crosstab(true_df[row_var], true_df[col_var],margins=True)
    print(some_crosstab)
    print(some_crosstab.index)
    out = [[],[],[]]
    for row in range(true_count.shape[0]):
        for col in range(true_count.shape[1]):
            out[row].append(f'{round(share_table[row,col],3)} ({round(true_count[row,col],3)})')
    out_df = pd.DataFrame(out,index=some_crosstab.index)
    print(out_df)