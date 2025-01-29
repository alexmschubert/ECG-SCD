""" Functions related to plotting in analysis. """

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from math import ceil, sqrt

from ekg_scd.helpers.preproc import cut 


def plot_binned_calibration(df, true_var, pred_var, savepath, n_bins=50):
    """Build binned calibration plot given true outcomes and predictions."""
    _fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    auc = roc_auc_score(df[true_var], df[pred_var])
    # initialize lists
    out = []
    err = []
    x_mean = []
    risk_bin = []

    df["risk_bin"] = pd.qcut(df[pred_var], n_bins, labels=False)
    # iterate through risk bins, saving stats as we go
    for b in range(n_bins):
        use = df.loc[df["risk_bin"] == b]
        x = use[true_var]
        try:
            mean = x.mean()
            y_err = x.std() * np.sqrt(1 / len(x) + np.sum((x - x.mean())) ** 2 / np.sum((x - x.mean()) ** 2))
            x_m = use[pred_var].mean()
        except:
            mean = float("NaN")
            y_err = float("NaN")
        percentile = b * 100/n_bins
        risk_bin.append(percentile)
        out.append(mean)
        err.append(y_err)
        x_mean.append(x_m)

    ax.plot([0, 1], [0, 1], ":", color="red", alpha=0.5)
    ax.errorbar(x_mean, out, err, capsize=3, fmt="o", markersize=4)
    ax.set_xlabel(pred_var)
    ax.set_ylabel(f"AUC = {round(auc,3)} \n\n mean({true_var})")
    ax.grid(linestyle="-", linewidth=1, alpha=0.3)
    ax.set_xlim(0 - 0.001, x_mean[-1] + 0.001)
    ax.set_ylim(0, out[-1] + err[-1] + 0.001)
    # save and clear
    plt.savefig(savepath)
    plt.clf()

def percentile_data(df,bin_var, bins, y_var):
    # create deciles based on 'y_pred_proba' column
    df['decile'] = pd.qcut(df[bin_var], q=bins, labels=False)

    # calculate mean 'VFVT1' for each decile
    mean_vfvt1 = df.groupby('decile')[y_var].mean()
    count_vfvt1 = df.groupby('decile')[y_var].count()
    std_vfvt1 = df.groupby('decile')[y_var].std()
    #mean_error.append(1.96 * np.sqrt((p * (1 - p)) / len(highrisk_df)))
    y_err = std_vfvt1 * np.sqrt(1 / count_vfvt1 + np.sum((mean_vfvt1 - mean_vfvt1.mean())) ** 2 / np.sum((mean_vfvt1 - mean_vfvt1.mean()) ** 2))
    return mean_vfvt1, count_vfvt1, std_vfvt1, y_err

def percentile_plot(df,savepath=None, bin_var = 'y_pred_proba', bin_var_comp = None, y_var='VFVT1', bins = 10, show=False):
    
    mean_vfvt1, count_vfvt1, std_vfvt1, y_err = percentile_data(df,bin_var, bins, y_var)
    
    if bin_var_comp:
        mean_vfvt1_comp, count_vfvt1_comp, std_vfvt1_comp, y_err_comp = percentile_data(df,bin_var_comp, bins, y_var)

    multiplier = 100/bins
    # create bar chart of mean 'VFVT1' per decile
    plt.clf()
    plt.plot(mean_vfvt1.index*multiplier, mean_vfvt1.values, color='violet', label='Model')
    plt.fill_between(mean_vfvt1.index*multiplier, mean_vfvt1.values - y_err.values, mean_vfvt1.values + y_err.values, color="violet", alpha=0.7)
    if bin_var_comp:
        plt.plot(mean_vfvt1_comp.index*multiplier, mean_vfvt1_comp.values, color='grey', label='Baseline')
        plt.fill_between(mean_vfvt1_comp.index*multiplier, mean_vfvt1_comp.values - y_err_comp.values, mean_vfvt1_comp.values + y_err_comp.values, color="grey", alpha=0.7)
    plt.xlabel('Risk percentile')
    plt.ylabel(f'Incidence {y_var}')
    plt.legend()
    if show:
        plt.show()
    # save and clear
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.clf()

def create_calibration_series(df, target, value_range, y_outcome, se_name):
    # initialize lists
    marginal_rates = []
    mean_rates = []
    mar_error = []
    mean_error = []
    
    df = df.sort_values(target, ascending=False)
    for prop in value_range:
        highrisk_df = df.head(round(prop * len(df)))
        p = sum(highrisk_df[y_outcome]) / len(highrisk_df)
        mean_rates.append(p)
        marginal_patient = highrisk_df.tail(1)
        marginal_rates.append(marginal_patient.iloc[0][target])
        mean_error.append(1.96 * np.sqrt((p * (1 - p)) / len(highrisk_df)))
        mar_error.append(1.96 * marginal_patient.iloc[0][se_name])
    # convert to array for error bar calculations
    marginal_rates = np.array(marginal_rates)
    mean_rates = np.array(mean_rates)
    mar_error = np.array(mar_error)
    mean_error = np.array(mean_error)

    return marginal_rates, mean_rates, mar_error, mean_error

def plot_highrisk_table(df, savepath, cut_value=0.035, y_outcome = 'scd1', plot_marginals = True, cum_label = "cumulative SCD rate", y_lim=0.1,
                        target: str = "y_pred_proba", comparable_target: str = None, cum_label_comp = "cumulative SCD rate - comp model",
                        se_name: str = 'mean_se', se_name_comp: str = 'mean_se'):
    """Build plot of highrisk population SCD incidence as HRG size is increased."""

    # iterate through range of values to get HRG stats as we make group larger
    value_range = np.arange(0.0025, 0.25, 0.0025)

    marginal_rates, mean_rates, mar_error, mean_error = create_calibration_series(df, target, value_range, y_outcome, se_name)

    if comparable_target:
        marginal_rates_comp, mean_rates_comp, mar_error_comp, mean_error_comp = create_calibration_series(df, comparable_target, value_range, y_outcome, se_name_comp)

    # plot lines
    plt.clf()
    if plot_marginals:
        plt.plot(value_range, marginal_rates, label=f"marginal SCD rate {target}", alpha=0.7, c="b")
    plt.plot(value_range, mean_rates, label=cum_label, alpha=0.7, c="orange")
    if comparable_target:
        if plot_marginals:
            plt.plot(value_range, marginal_rates_comp, label=f"marginal SCD rate {comparable_target}", alpha=0.7, c="violet")
        plt.plot(value_range, mean_rates_comp, label=cum_label_comp, alpha=0.7, c="grey")
    # plot errrors
    if plot_marginals:
        plt.fill_between(value_range, marginal_rates - mar_error, marginal_rates + mar_error, color="b", alpha=0.3)
    plt.fill_between(value_range, mean_rates - mean_error, mean_rates + mean_error, color="orange", alpha=0.3)

    if comparable_target:
        if plot_marginals:
            plt.fill_between(value_range, marginal_rates_comp - mar_error_comp, marginal_rates_comp + mar_error_comp, color="violet", alpha=0.3)
        plt.fill_between(value_range, mean_rates_comp - mean_error_comp, mean_rates_comp + mean_error_comp, color="grey", alpha=0.3)
    # add horizontal indicator for cutoff value
    plt.hlines(y=cut_value, xmin=0, xmax=np.max(value_range), colors=["black"])
    plt.legend()
    plt.xlabel("Proportion of Population Highrisk")
    plt.ylabel("Incidence")
    plt.ylim(0, y_lim)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()


def plot_hrg_5_5(df, savepath, cut_value=0.035):
    """Build plot of highrisk population SCD incidence as HRG size is increased."""

    # initialize lists
    marginal_rates = []
    mar_error = []
    # iterate through range of values to get HRG stats as we make group larger
    value_range = np.arange(0.0025, 0.25, 0.0025)
    df = df.sort_values("y_pred_proba", ascending=False)
    for prop in value_range:
        highrisk_df = df.head(round(prop * len(df)))
        marginal_patient = highrisk_df.tail(1)
        marginal_rates.append(marginal_patient.iloc[0]["y_pred_proba"])
        mar_error.append(1.96 * marginal_patient.iloc[0]["mean_se"])
    # convert to array for error bar calculations
    marginal_rates = np.array(marginal_rates)
    mar_error = np.array(mar_error)
    # plot lines
    plt.plot(value_range, marginal_rates, label="marginal SCD rate", alpha=0.7, c="b")
    # plot errrors
    plt.fill_between(value_range, marginal_rates - mar_error, marginal_rates + mar_error, color="b", alpha=0.3)
    # add horizontal indicator for cutoff value
    plt.hlines(y=cut_value, xmin=0, xmax=np.max(value_range),linestyle='dashed', colors=["black"])
    plt.legend()
    plt.xlabel("Proportion of Population Highrisk")
    plt.ylabel("Incidence")
    plt.ylim(0, 0.1)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()

def plot_captured_SCD_share(df, savepath, cut_value=0.035, risk_range=0.2, ticksteps=0.05):
    """Build plot of share of SCDs in HRG to total SCDs as HRG cut-off increases."""

    #Calculate total SCDs
    total_scd1 = sum(df["scd1"])

    # initialize lists
    scd_share = []
    mar_error = []
    mean_error = []
    # iterate through range of values to get HRG stats as we make group larger
    value_range = np.arange(0.00, risk_range+0.00000000001, 0.01)
    value_range = np.arange(0.00, risk_range+0.00000000001, 0.01)
    df = df.sort_values("y_pred_proba", ascending=False)
    for prop in value_range:
        highrisk_df = df.head(round(prop * len(df)))
        if round(prop * len(df)) == 0:
            p=0
        p = sum(highrisk_df["scd1"]) / total_scd1
        scd_share.append(p)
        #marginal_patient = highrisk_df.tail(1)
        #marginal_rates.append(marginal_patient.iloc[0]["y_pred_proba"])
        N = len(pd.unique(highrisk_df["ptId"]))
        if N==0:
            mean_error.append(0)
        else:
            mean_error.append(1.96 * np.sqrt((p * (1 - p)) / N))
        N = len(pd.unique(highrisk_df["ptId"]))
        if N==0:
            mean_error.append(0)
        else:
            mean_error.append(1.96 * np.sqrt((p * (1 - p)) / N))
        #mar_error.append(1.96 * marginal_patient.iloc[0]["mean_se"])
    #Calculate key stats
    twenty_percent_riskiest = scd_share[list(value_range).index(0.2)]*100
    highrisk_df = df.loc[df["y_pred_proba"]>0.035]
    highrisk_share_patients = len(pd.unique(highrisk_df["ptId"])) / len(pd.unique(df["ptId"]))*100
    highrisk_share_ECGs = len(highrisk_df["ptId"]) / len(df["ptId"])*100
    p_highrisk_cutoff = (sum(highrisk_df["scd1"]) / total_scd1)*100
    print(f"The riskiest 20% of patients account for {twenty_percent_riskiest}% of the SCDs within one year")
    print(f"The patients with a predicted SCD risk above 3.5% account for {p_highrisk_cutoff}% of the SCDs within one year")
    print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_patients}% of all patients")
    print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_ECGs}% of all ECGs")

    #Calculate key stats
    twenty_percent_riskiest = scd_share[list(value_range).index(0.2)]*100
    highrisk_df = df.loc[df["y_pred_proba"]>0.035]
    highrisk_share_patients = len(pd.unique(highrisk_df["ptId"])) / len(pd.unique(df["ptId"]))*100
    highrisk_share_ECGs = len(highrisk_df["ptId"]) / len(df["ptId"])*100
    p_highrisk_cutoff = (sum(highrisk_df["scd1"]) / total_scd1)*100
    print(f"The riskiest 20% of patients account for {twenty_percent_riskiest}% of the SCDs within one year")
    print(f"The patients with a predicted SCD risk above 3.5% account for {p_highrisk_cutoff}% of the SCDs within one year")
    print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_patients}% of all patients")
    print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_ECGs}% of all ECGs")

    # convert to array for error bar calculations
    scd_share = np.array(scd_share)
    #mar_error = np.array(mar_error)
    mean_error = np.array(mean_error)
    #convert valuee range to risk percentiles
    value_range = [1-x for x in value_range]
    # plot lines
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = ['Arial']
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = ['Arial']
    plt.plot(value_range, scd_share, alpha=0.7, c="b")
    # plot errrors
    #plt.fill_between(value_range, marginal_rates - mar_error, marginal_rates + mar_error, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share - mean_error, scd_share + mean_error, color="b", alpha=0.3)
    # add horizontal indicator for cutoff value
    num_ticks = int((risk_range*100)//(ticksteps*100) + 1)
    ticks = [1-x*ticksteps for x in range(num_ticks)]
    labels = [int(100*t) for t in ticks]
    #plt.vlines(x=1-cut_value, ymin=0, ymax=np.max(scd_share), linestyle='dashed', label="Highrisk cutoff", colors=["black"])
    plt.legend()
    plt.xlabel("Algorithm predicted risk")
    plt.ylabel("Fraction of sudden cardiac deaths")
    plt.ylim(0, np.max(scd_share + mean_error))
    plt.xlim(np.min(value_range),np.max(value_range))
    plt.gca().invert_xaxis()
    plt.xticks(ticks, labels)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()

def plot_captured_SCD_share_ef(df, savepath, cut_value=0.035, risk_range=0.2, ticksteps=0.05):
    """Build plot of share of SCDs in HRG to total SCDs as HRG cut-off increases."""

    def calc_scd_share(df, cut_value, risk_range, ticksteps):

        #Calculate total SCDs
        total_scd1 = sum(df["scd1"])

        # initialize lists
        scd_share = []
        mar_error = []
        mean_error = []
        # iterate through range of values to get HRG stats as we make group larger
        value_range = np.arange(0.00, risk_range+0.00000000001, 0.01)
        df = df.sort_values("y_pred_proba", ascending=False)
        for prop in value_range:
            highrisk_df = df.head(round(prop * len(df)))
            if round(prop * len(df)) == 0:
                p=0
            p = sum(highrisk_df["scd1"]) / total_scd1
            scd_share.append(p)
            #marginal_patient = highrisk_df.tail(1)
            #marginal_rates.append(marginal_patient.iloc[0]["y_pred_proba"])
            N = len(pd.unique(highrisk_df["ptId"]))
            if N==0:
                mean_error.append(0)
            else:
                mean_error.append(1.96 * np.sqrt((p * (1 - p)) / N))
            #mar_error.append(1.96 * marginal_patient.iloc[0]["mean_se"])
        #Calculate key stats
        twenty_percent_riskiest = scd_share[list(value_range).index(0.2)]*100
        highrisk_df = df.loc[df["y_pred_proba"]>0.035]
        highrisk_share_patients = len(pd.unique(highrisk_df["ptId"])) / len(pd.unique(df["ptId"]))*100
        highrisk_share_ECGs = len(highrisk_df["ptId"]) / len(df["ptId"])*100
        p_highrisk_cutoff = (sum(highrisk_df["scd1"]) / total_scd1)*100
        print(f"The riskiest 20% of patients account for {twenty_percent_riskiest}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {p_highrisk_cutoff}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_patients}% of all patients")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_ECGs}% of all ECGs")
        return np.array(scd_share), np.array(mean_error), value_range

    #create metrics for ef >=35
    print()
    print('ECGs with ef >= 35')
    print()
    df_35 = df[df['most_recent_ef']>=35]
    scd_share_above35, mean_error_above35, value_range_above35 = calc_scd_share(df_35, cut_value, risk_range, ticksteps)

    #create metrics for ef na
    print()
    print('ECGs with no ef info')
    print()
    df_na = df[df['ef_na']==1]
    scd_share_na, mean_error_na, value_range_na = calc_scd_share(df_na, cut_value, risk_range, ticksteps)

    #create metrics for ef < 35
    print()
    print('ECGs with ef < 35')
    print()
    df_below35 = df[(df['ef_na']==0)&(df['most_recent_ef']<35)]
    scd_share_below35, mean_error_below35, value_range_below35 = calc_scd_share(df_below35, cut_value, risk_range, ticksteps)
    
    #convert valuee range to risk percentiles
    value_range = [1-x for x in value_range_above35]
    # plot lines
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = ['Arial']
    
    plt.plot(value_range, scd_share_above35, alpha=0.7, c="b", label= 'EF above 35')
    plt.plot(value_range, scd_share_below35, alpha=0.7, c="g", label= 'EF below 35')
    plt.plot(value_range, scd_share_na, alpha=0.7, c="violet", label= 'EF not available')

    # plot errrors
    #plt.fill_between(value_range, marginal_rates - mar_error, marginal_rates + mar_error, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share_above35 - mean_error_above35, scd_share_above35 + mean_error_above35, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share_below35 - mean_error_below35, scd_share_below35 + mean_error_below35, color="g", alpha=0.3)
    plt.fill_between(value_range, scd_share_na - mean_error_na, scd_share_na + mean_error_na, color="violet", alpha=0.3)


    # add horizontal indicator for cutoff value
    num_ticks = int((risk_range*100)//(ticksteps*100) + 1)
    ticks = [1-x*ticksteps for x in range(num_ticks)]
    labels = [int(100*t) for t in ticks]
    #plt.vlines(x=1-cut_value, ymin=0, ymax=np.max(scd_share), linestyle='dashed', label="Highrisk cutoff", colors=["black"])
    plt.legend()
    plt.xlabel("Algorithm predicted risk")
    plt.ylabel("Fraction of sudden cardiac deaths")
    plt.ylim(0, np.max(scd_share_below35 + mean_error_below35))
    plt.xlim(np.min(value_range),np.max(value_range))
    plt.gca().invert_xaxis()
    plt.xticks(ticks, labels)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()

def plot_captured_SCD_share_ami40(df, savepath, cut_value=0.035, risk_range=0.2, ticksteps=0.05):
    """Build plot of share of SCDs in HRG to total SCDs as HRG cut-off increases."""

    def calc_scd_share(df, cut_value, risk_range, ticksteps):

        #Calculate total SCDs
        total_scd1 = sum(df["scd1"])

        # initialize lists
        scd_share = []
        mar_error = []
        mean_error = []
        # iterate through range of values to get HRG stats as we make group larger
        value_range = np.arange(0.00, risk_range+0.00000000001, 0.01)
        df = df.sort_values("y_pred_proba", ascending=False)
        for prop in value_range:
            highrisk_df = df.head(round(prop * len(df)))
            if round(prop * len(df)) == 0:
                p=0
            p = sum(highrisk_df["scd1"]) / total_scd1
            scd_share.append(p)
            #marginal_patient = highrisk_df.tail(1)
            #marginal_rates.append(marginal_patient.iloc[0]["y_pred_proba"])
            N = len(pd.unique(highrisk_df["ptId"]))
            if N==0:
                mean_error.append(0)
            else:
                mean_error.append(1.96 * np.sqrt((p * (1 - p)) / N))
            #mar_error.append(1.96 * marginal_patient.iloc[0]["mean_se"])
        #Calculate key stats
        twenty_percent_riskiest = scd_share[list(value_range).index(0.2)]*100
        highrisk_df = df.loc[df["y_pred_proba"]>0.035]
        highrisk_share_patients = len(pd.unique(highrisk_df["ptId"])) / len(pd.unique(df["ptId"]))*100
        highrisk_share_ECGs = len(highrisk_df["ptId"]) / len(df["ptId"])*100
        p_highrisk_cutoff = (sum(highrisk_df["scd1"]) / total_scd1)*100
        print(f"The riskiest 20% of patients account for {twenty_percent_riskiest}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {p_highrisk_cutoff}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_patients}% of all patients")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_ECGs}% of all ECGs")
        return np.array(scd_share), np.array(mean_error), value_range

    #create metrics for ami40
    df_ami40 = df[df['Acute_MI']==1]
    scd_share_ami40, mean_error_ami40, value_range_ami40 = calc_scd_share(df_ami40, cut_value, risk_range, ticksteps)

    #create metrics for no heartattack
    df_ami_no = df[df['Acute_MI']==0]
    scd_share_ami_no, mean_error_ami_no, value_range_ami_no = calc_scd_share(df_ami_no, cut_value, risk_range, ticksteps)

    #convert value range to risk percentiles
    value_range = [1-x for x in value_range_ami_no]
    # plot lines
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = ['Arial']
    
    plt.plot(value_range, scd_share_ami40, alpha=0.7, c="b", label= 'AMI in past 40 days')
    plt.plot(value_range, scd_share_ami_no, alpha=0.7, c="g", label= 'No AMI')

    # plot errrors
    #plt.fill_between(value_range, marginal_rates - mar_error, marginal_rates + mar_error, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share_ami40 - mean_error_ami40, scd_share_ami40 + mean_error_ami40, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share_ami_no - mean_error_ami_no, scd_share_ami_no + mean_error_ami_no, color="g", alpha=0.3)

    # add horizontal indicator for cutoff value
    num_ticks = int((risk_range*100)//(ticksteps*100) + 1)
    ticks = [1-x*ticksteps for x in range(num_ticks)]
    labels = [int(100*t) for t in ticks]
    labels = [int(100*t) for t in ticks]
    #plt.vlines(x=1-cut_value, ymin=0, ymax=np.max(scd_share), linestyle='dashed', label="Highrisk cutoff", colors=["black"])
    plt.legend()
    plt.xlabel("Algorithm predicted risk")
    plt.ylabel("Fraction of sudden cardiac deaths")
    plt.ylim(0, np.max(scd_share_ami40 + mean_error_ami40))
    plt.xlim(np.min(value_range),np.max(value_range))
    plt.gca().invert_xaxis()
    plt.xticks(ticks, labels)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()


def plot_captured_SCD_share_ef(df, savepath, cut_value=0.035, risk_range=0.2, ticksteps=0.05):
    """Build plot of share of SCDs in HRG to total SCDs as HRG cut-off increases."""

    def calc_scd_share(df, cut_value, risk_range, ticksteps):

        #Calculate total SCDs
        total_scd1 = sum(df["scd1"])

        # initialize lists
        scd_share = []
        mar_error = []
        mean_error = []
        # iterate through range of values to get HRG stats as we make group larger
        value_range = np.arange(0.00, risk_range+0.00000000001, 0.01)
        df = df.sort_values("y_pred_proba", ascending=False)
        for prop in value_range:
            highrisk_df = df.head(round(prop * len(df)))
            if round(prop * len(df)) == 0:
                p=0
            p = sum(highrisk_df["scd1"]) / total_scd1
            scd_share.append(p)
            #marginal_patient = highrisk_df.tail(1)
            #marginal_rates.append(marginal_patient.iloc[0]["y_pred_proba"])
            N = len(pd.unique(highrisk_df["ptId"]))
            if N==0:
                mean_error.append(0)
            else:
                mean_error.append(1.96 * np.sqrt((p * (1 - p)) / N))
            #mar_error.append(1.96 * marginal_patient.iloc[0]["mean_se"])
        #Calculate key stats
        twenty_percent_riskiest = scd_share[list(value_range).index(0.2)]*100
        highrisk_df = df.loc[df["y_pred_proba"]>0.035]
        highrisk_share_patients = len(pd.unique(highrisk_df["ptId"])) / len(pd.unique(df["ptId"]))*100
        highrisk_share_ECGs = len(highrisk_df["ptId"]) / len(df["ptId"])*100
        p_highrisk_cutoff = (sum(highrisk_df["scd1"]) / total_scd1)*100
        print(f"The riskiest 20% of patients account for {twenty_percent_riskiest}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {p_highrisk_cutoff}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_patients}% of all patients")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_ECGs}% of all ECGs")
        return np.array(scd_share), np.array(mean_error), value_range

    #create metrics for ef >=35
    print()
    print('ECGs with ef >= 35')
    print()
    df_35 = df[df['most_recent_ef']>=35]
    scd_share_above35, mean_error_above35, value_range_above35 = calc_scd_share(df_35, cut_value, risk_range, ticksteps)

    #create metrics for ef na
    print()
    print('ECGs with no ef info')
    print()
    df_na = df[df['ef_na']==1]
    scd_share_na, mean_error_na, value_range_na = calc_scd_share(df_na, cut_value, risk_range, ticksteps)

    #create metrics for ef < 35
    print()
    print('ECGs with ef < 35')
    print()
    df_below35 = df[(df['ef_na']==0)&(df['most_recent_ef']<35)]
    scd_share_below35, mean_error_below35, value_range_below35 = calc_scd_share(df_below35, cut_value, risk_range, ticksteps)
    
    #convert valuee range to risk percentiles
    value_range = [1-x for x in value_range_above35]
    # plot lines


def plot_captured_SCD_share_ef(df, savepath, cut_value=0.035, risk_range=0.2, ticksteps=0.05):
    """Build plot of share of SCDs in HRG to total SCDs as HRG cut-off increases."""

    def calc_scd_share(df, cut_value, risk_range, ticksteps):

        #Calculate total SCDs
        total_scd1 = sum(df["scd1"])

        # initialize lists
        scd_share = []
        mar_error = []
        mean_error = []
        # iterate through range of values to get HRG stats as we make group larger
        value_range = np.arange(0.00, risk_range+0.00000000001, 0.01)
        df = df.sort_values("y_pred_proba", ascending=False)
        for prop in value_range:
            highrisk_df = df.head(round(prop * len(df)))
            if round(prop * len(df)) == 0:
                p=0
            p = sum(highrisk_df["scd1"]) / total_scd1
            scd_share.append(p)
            #marginal_patient = highrisk_df.tail(1)
            #marginal_rates.append(marginal_patient.iloc[0]["y_pred_proba"])
            N = len(pd.unique(highrisk_df["ptId"]))
            if N==0:
                mean_error.append(0)
            else:
                mean_error.append(1.96 * np.sqrt((p * (1 - p)) / N))
            #mar_error.append(1.96 * marginal_patient.iloc[0]["mean_se"])
        #Calculate key stats
        twenty_percent_riskiest = scd_share[list(value_range).index(0.2)]*100
        highrisk_df = df.loc[df["y_pred_proba"]>0.035]
        highrisk_share_patients = len(pd.unique(highrisk_df["ptId"])) / len(pd.unique(df["ptId"]))*100
        highrisk_share_ECGs = len(highrisk_df["ptId"]) / len(df["ptId"])*100
        p_highrisk_cutoff = (sum(highrisk_df["scd1"]) / total_scd1)*100
        print(f"The riskiest 20% of patients account for {twenty_percent_riskiest}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {p_highrisk_cutoff}% of the SCDs within one year")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_patients}% of all patients")
        print(f"The patients with a predicted SCD risk above 3.5% account for {highrisk_share_ECGs}% of all ECGs")
        return np.array(scd_share), np.array(mean_error), value_range

    #create metrics for ef >=35
    print()
    print('ECGs with ef >= 35')
    print()
    df_35 = df[df['most_recent_ef']>=35]
    scd_share_above35, mean_error_above35, value_range_above35 = calc_scd_share(df_35, cut_value, risk_range, ticksteps)

    #create metrics for ef na
    print()
    print('ECGs with no ef info')
    print()
    df_na = df[df['ef_na']==1]
    scd_share_na, mean_error_na, value_range_na = calc_scd_share(df_na, cut_value, risk_range, ticksteps)

    #create metrics for ef < 35
    print()
    print('ECGs with ef < 35')
    print()
    df_below35 = df[(df['ef_na']==0)&(df['most_recent_ef']<35)]
    scd_share_below35, mean_error_below35, value_range_below35 = calc_scd_share(df_below35, cut_value, risk_range, ticksteps)
    
    #convert valuee range to risk percentiles
    value_range = [1-x for x in value_range_above35]
    # plot lines
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = ['Arial']
    
    plt.plot(value_range, scd_share_above35, alpha=0.7, c="b", label= 'EF above 35')
    plt.plot(value_range, scd_share_below35, alpha=0.7, c="g", label= 'EF below 35')
    plt.plot(value_range, scd_share_na, alpha=0.7, c="violet", label= 'EF not available')

    # plot errrors
    #plt.fill_between(value_range, marginal_rates - mar_error, marginal_rates + mar_error, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share_above35 - mean_error_above35, scd_share_above35 + mean_error_above35, color="b", alpha=0.3)
    plt.fill_between(value_range, scd_share_below35 - mean_error_below35, scd_share_below35 + mean_error_below35, color="g", alpha=0.3)
    plt.fill_between(value_range, scd_share_na - mean_error_na, scd_share_na + mean_error_na, color="violet", alpha=0.3)


    # add horizontal indicator for cutoff value
    num_ticks = int((risk_range*100)//(ticksteps*100) + 1)
    ticks = [1-x*ticksteps for x in range(num_ticks)]
    labels = [int(100*t) for t in ticks]
    #plt.vlines(x=1-cut_value, ymin=0, ymax=np.max(scd_share), linestyle='dashed', label="Highrisk cutoff", colors=["black"])
    plt.legend()
    plt.xlabel("Algorithm predicted risk")
    plt.ylabel("Fraction of sudden cardiac deaths")
    plt.ylim(0, np.max(scd_share_below35 + mean_error_below35))
    plt.xlim(np.min(value_range),np.max(value_range))
    plt.gca().invert_xaxis()
    plt.xticks(ticks, labels)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()


def _ax_plot(ax, x, y, secs=10, lwidth=1.20, amplitude_ecg = 5.0, time_ticks =0.2, miny=0, maxy=3.5):
    ax.set_xticks(np.arange(0,11,time_ticks))    
    ax.set_yticks(np.arange(-ceil(amplitude_ecg),ceil(amplitude_ecg),0.5))

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    ax.minorticks_on()
    
    # Set major and minor grid lines on X
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=.2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(base=0.2 / 5.))

    ax.yaxis.set_major_locator(mticker.MultipleLocator(base=.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(base=0.5 / 5.))

    ax.set_ylim(miny-0.5, maxy+0.5)
    ax.set_xlim(0, secs)
    ax.set_aspect(0.4) 
    #ax.set_aspect('auto') 

    ax.grid(which='major', linestyle='-', linewidth='1.5', color='red')
    ax.grid(which='minor', linestyle='--', linewidth='1.25', color=(1, 0.7, 0.7))

    ax.plot(x,y, linewidth=lwidth, color='black')


lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
def plot_12(
        ecg, 
        sample_rate = 500,
        title       = 'ECG 12', 
        lead_index  = lead_index, 
        lead_order  = None,
        columns     = 1,
        savepath    = None,
        fig_size    = (5, 15),
        show        = False,
        conversion_factor = 200
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 1
        conversion_factor: Factor to scale beat values to mV
    """
    #ecg = ecg/conversion_factor
    
    if not lead_order:
        lead_order = list(range(0,len(ecg)))

    leads = len(lead_order)
    seconds = len(ecg[0])/sample_rate

    print(f'Seconds {seconds}')

    min_y = np.min(ecg)
    max_y = np.max(ecg)

    #print(min_y)
    #print(max_y)

    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(
        ceil(len(lead_order)/columns),columns,
        sharex=True, 
        sharey=True,
        # subplot_kw=dict(box_aspect=1)
        figsize=fig_size
        )
    fig.subplots_adjust(
        hspace = 0.00, 
        wspace = 0.04,
        left   = 0.04,  # the left side of the subplots of the figure
        right  = 0.98,  # the right side of the subplots of the figure
        bottom = 0.06,  # the bottom of the subplots of the figure
        top    = 0.95
        )
    fig.suptitle(title)
    # plt.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()

    step = 1.0/sample_rate

    for i in range(0, len(lead_order)):
        if(columns == 1):
            t_ax = ax[i]
        else:
            t_ax = ax[i//columns,i%columns]
        t_lead = lead_order[i]
        t_ax.set_ylabel("%s /mV" % lead_index[t_lead])
        t_ax.tick_params(axis='x',rotation=90)

        #print(ecg[t_lead][-100:-50])

        #t_ax.plot(np.arange(0,len(ecg[t_lead])),ecg[t_lead], color='black') #linewidth=lwidth,
       
        _ax_plot(ax=t_ax, x=np.arange(0, len(ecg[t_lead])*step, step), y=ecg[t_lead], secs=seconds, miny=min_y, maxy=max_y)
        plt.xlabel("time(s)")
    
    #fig.tight_layout()
    
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)

    if show:
        plt.show()
    
    plt.clf()

lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
def plot_ecg_no_grid(
        ecg, 
        sample_rate = 500,
        title       = 'ECG 12', 
        lead_index  = lead_index, 
        lead_order  = None,
        columns     = 1,
        savepath    = None,
        fig_size    = (5, 15),
        show        = False,
        conversion_factor = 200
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 1
        conversion_factor: Factor to scale beat values to mV
    """
    ecg = ecg/conversion_factor
    
    if not lead_order:
        lead_order = list(range(0,len(ecg)))

    leads = len(lead_order)
    seconds = len(ecg[0])/sample_rate

    print(f'Seconds {seconds}')

    min_y = np.min(ecg)
    max_y = np.max(ecg)

    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(
        ceil(len(lead_order)/columns),columns,
        sharex=True, 
        sharey=True,
        # subplot_kw=dict(box_aspect=1)
        figsize=fig_size
        )
    fig.subplots_adjust(
        hspace = 0.00, 
        wspace = 0.04,
        left   = 0.04,  # the left side of the subplots of the figure
        right  = 0.98,  # the right side of the subplots of the figure
        bottom = 0.06,  # the bottom of the subplots of the figure
        top    = 0.95
        )
    fig.suptitle(title)
    # plt.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()

    step = 1.0/sample_rate

    for i in range(0, len(lead_order)):
        if(columns == 1):
            t_ax = ax[i]
        else:
            t_ax = ax[i//columns,i%columns]
        t_lead = lead_order[i]
        t_ax.set_ylabel("%s /mV" % lead_index[t_lead])
        t_ax.tick_params(axis='x',rotation=90)
       
        _ax_plot(ax=t_ax, x=np.arange(0, len(ecg[t_lead])*step, step), y=ecg[t_lead], secs=seconds, miny=min_y, maxy=max_y)
        plt.xlabel("time(s)")
    
    #fig.tight_layout()
    
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)

    if show:
        plt.show()
    
    plt.clf()

def plot_binned_PrICD_EF(df: pd.DataFrame, x_var: str, y_var: str, savepath, bin_value=5, title="ICD rate by EF bin", x_label ="5 percent bins of EF", y_label= "Share of patients with ICD", max_x = 100):
    """Build binned plot of 5% EF ranges to ICD share"""

    #Create bins for X variable
    cut_bins = list(range(0,max_x,bin_value))
    cut_labels = cut_bins[1:]
    df["x_bins"] = pd.cut(df[x_var], bins=cut_bins, labels=cut_labels)

    out = []
    err = []
    x_mean = []
    obs = []

    # iterate through risk bins, saving stats as we go
    for b in cut_labels:
        use = df.loc[df["x_bins"] == b]
        y = use[y_var]
        try:
            mean = y.mean()
            y_error = sqrt((mean*(1-mean))/use.shape[0])
            #y_err = x.std() * np.sqrt(1 / len(x) + np.sum((x - x.mean())) ** 2 / np.sum((x - x.mean()) ** 2))
            x_m = use[x_var].mean()
        except:
            mean = float("NaN")
            y_error = float("NaN")
        out.append(mean)
        err.append(y_error)
        x_mean.append(x_m)
        obs.append(use.shape[0])
    
    # convert to array for error bar calculations
    bin_avg = np.array(out)
    bin_error = np.array(err)
    # plot lines
    plt.plot(cut_labels, bin_avg, label="Avg. ICD rate", alpha=0.7, c="b")
    # plot errrors
    plt.fill_between(cut_labels, bin_avg - bin_error, bin_avg + bin_error, color="b", alpha=0.3)
    # add horizontal indicator for cutoff value
    #plt.hlines(y=cut_value, xmin=0, xmax=np.max(value_range),linestyle='dashed', colors=["black"])
    #plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    #plt.ylim(0, 0.1)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()

    print('obs per bin')
    print(obs)

#plotting.plot_binned_PrICD_EF_by_risk(general_df_ef_sub80, 'risk', 'defib_proc_history', 'low_EF',savepath=f"{table_path}/ICD_by_scdhat_bin_low_high_sub80.png", 
    #    bin_value=1, title="ICD rate by scd_hat bin - sub80 patients", x_lab="one percent bin predicted scd risk", one_lab= "ICD rate - low EF patients", zero_lab="ICD rate - normal EF patients",
    #    max_x = 10
    # )

def plot_binned_PrICD_EF_by_risk(df: pd.DataFrame, x_var: str, y_var: str, highrisk, savepath, bin_value=5, title="ICD rate by EF bin",
x_lab = "5 percent bins of EF", zero_lab = "Avg. ICD rate - low scd risk", one_lab = "Avg. ICD rate - high scd risk", max_x = 60):
    """Build binned plot of 5% EF ranges to ICD share"""

    #Create bins for X variable
    cut_bins = list(range(0,max_x,bin_value))
    cut_labels = cut_bins[1:]
    df["x_bins"] = pd.cut(df[x_var], bins=cut_bins, labels=cut_labels)

    out = []
    err = []
    x_mean = []
    obs = []

    # Todo update for my problem
    # iterate through risk bins, saving stats as we go
    for a in range(2):
        out_help = []
        err_help = []
        x_mean_help = []
        obs_help = []
        for b in cut_labels:
            use = df.loc[(df["x_bins"] == b) & (df[highrisk]==a)]
            y = use[y_var]
            try:
                mean = y.mean()
                y_error = sqrt((mean*(1-mean))/use.shape[0])
                #y_err = x.std() * np.sqrt(1 / len(x) + np.sum((x - x.mean())) ** 2 / np.sum((x - x.mean()) ** 2))
                x_m = use[x_var].mean()
            except:
                mean = float("NaN")
                y_error = float("NaN")
            out_help.append(mean)
            err_help.append(y_error)
            x_mean_help.append(x_m)
            obs_help.append(use.shape[0])
        out.append(out_help)
        err.append(err_help)
        x_mean.append(x_mean_help)
        obs.append(obs_help)
    
    # convert to array for error bar calculations
    bin_avg_low = np.array(out[0])
    bin_error_low = np.array(err[0])
    bin_avg_high = np.array(out[1])
    bin_error_high = np.array(err[1])

    # plot lines
    plt.plot(cut_labels, bin_avg_low, label=zero_lab, alpha=0.7, c="b") 
    plt.plot(cut_labels, bin_avg_high, label=one_lab, alpha=0.7, c="g") 
    # plot errrors
    plt.fill_between(cut_labels, bin_avg_low - bin_error_low, bin_avg_low + bin_error_low, color="b", alpha=0.3)
    plt.fill_between(cut_labels, bin_avg_high - bin_error_high, bin_avg_high + bin_error_high, color="g", alpha=0.3)
    # add horizontal indicator for cutoff value
    #plt.hlines(y=cut_value, xmin=0, xmax=np.max(value_range),linestyle='dashed', colors=["black"])
    plt.legend()
    plt.xlabel(x_lab)
    plt.ylabel("Share of patients with ICD")
    plt.title(title)
    #plt.ylim(0, 0.1)
    # save and clear
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.clf()

    print('obs per bin - label = 1')
    print(obs[1])
    print('obs per bin - label = 0')
    print(obs[0])