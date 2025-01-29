#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load in dataframe for "pre" events of cardiac arrests, w/in 1year, labelling with outcomes
"""

import os

import numpy as np
import pandas as pd

CARDIAC_CONTROLS = ["qtc500", "lbbb", "stelev", "stdown", "dltwv", "ventricular_rate", "atrial_rate", "qrs_duration", "qt_interval", "qtc", "qtc_frederica", "avg_rr"]

CONTROL_COLS = [
    "patient_ngsci_id",
    "ecg_id",
    "npy_index",
    "days_to_event",
    "days_to_event_og",
    "file_path",
    "age",
    "sex",
] + CARDIAC_CONTROLS
COLS = CONTROL_COLS + ["cause_of_cardiac_arrest", "VTVF"]


def collapse_lead_df(lead_df):
    lead_df["stelev"] = lead_df["stelev"] == 8
    lead_df["stdown"] = lead_df["stdown"] == 4
    lead_df["dltwv"] = lead_df["dltwv"] == 32
    lead_df_collapsed = lead_df.groupby("ecg_id").max().reset_index()
    return lead_df_collapsed


def process_code_df(code_df):
    code_df["lbbb"] = code_df["stmt_text"].str.contains("left bundle branch block")
    code_df["paced"] = code_df["stmt_text"].str.contains("paced")
    code_df_collapsed = code_df[["ecg_id", "lbbb", "paced"]].groupby("ecg_id").max().reset_index()
    return code_df_collapsed


def process_study_df(study_path):
    # base df
    group_df = pd.read_csv(f"{study_path}/ecg-waveforms/waveform-npy.csv")[["ecg_id", "npy_index", "ecg_timetag"]]
    group_df = group_df.loc[group_df["ecg_timetag"] == "[0]pre"]
    # get additional information
    ecg_df = pd.read_csv(f"{study_path}/ecg.csv")[
        ["patient_ngsci_id", "ecg_id", "acquisition_datetime_offset", "ecg_timetag"]
    ]
    cohort_df = pd.read_csv(f"{study_path}/cohort.csv")[["ecg_id", "age", "sex", "rosc_datetime_offset"]]
    # use hospitalization for study because it's more complete - we do not have for control cases
    outcomes_df = pd.read_csv(f"{study_path}/rosc-outcomes.csv")[
        ["ecg_id", "cause_of_cardiac_arrest", "hospitalization_datetime_offset", "initial_rhythm"]
    ]
    outcomes_df["VTVF"] = outcomes_df["initial_rhythm"] == "VT/VF"
    # lead level measurement
    lead_df = pd.read_csv(f"{study_path}/ecg-metadata/measurement-matrix-per-lead.csv")[["ecg_id", "stelev", "stdown", "dltwv"]]
    lead_df_collapsed = collapse_lead_df(lead_df)
    # qtc measurement
    global_df = pd.read_csv(f"{study_path}/ecg-metadata/measurement-matrix-global.csv")[
        ["ecg_id", "qtc", "avg_rr"]
    ]
    global_df["qtc500"] = global_df["qtc"] > 500
    # other resting measurements
    resting_df = pd.read_csv(f"{study_path}/ecg-metadata/resting-ecg-measurements.csv")[
        ["ecg_id", "ventricular_rate", "atrial_rate", "qrs_duration", "qt_interval", "qtc_frederica"]
    ]
    # code processing
    code_df = pd.read_csv(f"{study_path}/ecg-metadata/diagnosis.csv")[["ecg_id", "stmt_text"]]
    code_df_collapsed = process_code_df(code_df)
    # merge
    group_df = (
        group_df.merge(ecg_df[["patient_ngsci_id", "ecg_id", "acquisition_datetime_offset"]], how="left", on="ecg_id")
        .merge(cohort_df, how="left")
        .merge(outcomes_df, how="left")
        .merge(lead_df_collapsed, how="left")
        .merge(global_df, how="left")
        .merge(resting_df, how="left")
        .merge(code_df_collapsed, how="left")
    )
    
    group_df["days_to_event"] = (
        pd.to_datetime(group_df["rosc_datetime_offset"], format='mixed')
        - pd.to_datetime(group_df["acquisition_datetime_offset"])
    ).dt.days
    
    group_df["days_to_event_og"] = (
        pd.to_datetime(group_df["hospitalization_datetime_offset"], format='mixed')
        - pd.to_datetime(group_df["acquisition_datetime_offset"])
    ).dt.days
    
    group_df["file_path"] = f"{study_path}/ecg-waveforms/waveform-rhythm.npy"
    return group_df[COLS]


def process_control_df(control_path):
    # base df
    group_df = pd.read_csv(f"{control_path}/pre/ecg-waveforms/waveform-npy.csv")[["ecg_id", "npy_index"]]
    # get additional information
    ecg_df = pd.read_csv(f"../../datasets/arrest-ntuh-ecg/v1/control-group/ecg-cohort.csv")[
        ["ecg_id", "patient_ngsci_id", "acquisition_datetime_offset", "ecg_timetag", "age"]
    ]
    cohort_df = pd.read_csv(f"../../datasets/arrest-ntuh-ecg/v1/control-group/ecg-cohort.csv")[["ecg_id", "sex"]]
    # subset to "pre" ECGs and get date of control event
    pre_df = ecg_df.loc[ecg_df["ecg_timetag"] == "pre"]
    control_df = ecg_df.loc[ecg_df["ecg_timetag"] == "control"][
        ["patient_ngsci_id", "acquisition_datetime_offset"]
    ].rename(columns={"acquisition_datetime_offset": "control_datetime"})
    final_df = pre_df.merge(control_df, how="left")
    
    final_df["days_to_event"] = (
        pd.to_datetime(
            final_df["control_datetime"], format='mixed') -
            pd.to_datetime(final_df["acquisition_datetime_offset"], format='mixed')
        ).dt.days
    
    final_df["days_to_event_og"] = (
        pd.to_datetime(
            final_df["control_datetime"], format='mixed') -
            pd.to_datetime(final_df["acquisition_datetime_offset"], format='mixed')
        ).dt.days

    # lead level measurements
    lead_df = pd.read_csv(f"{control_path}/pre/ecg-metadata/measurement-matrix-per-lead.csv")[
        ["ecg_id", "stelev", "stdown", "dltwv"]
    ]
    lead_df_collapsed = collapse_lead_df(lead_df)
    # global measurements
    global_df = pd.read_csv(f"{control_path}/pre/ecg-metadata/measurement-matrix-global.csv")[
        ["ecg_id", "qtc", "avg_rr"]
    ]
    global_df["qtc500"] = global_df["qtc"] > 500
    # other resting measurements
    resting_df = pd.read_csv(f"{control_path}/pre/ecg-metadata/resting-ecg-measurements.csv")[
        ["ecg_id", "ventricular_rate", "atrial_rate", "qrs_duration", "qt_interval", "qtc_frederica"]
    ]
    # code processing
    code_df = pd.read_csv(f"{control_path}/pre/ecg-metadata/diagnosis.csv")[["ecg_id", "stmt_text"]]
    code_df_collapsed = process_code_df(code_df)
    # merge
    group_df = (
        group_df.merge(final_df, how="left", on="ecg_id")
        .merge(cohort_df, how="left", on="ecg_id")
        .merge(lead_df_collapsed, how="left")
        .merge(global_df, how="left")
        .merge(resting_df, how="left")
        .merge(code_df_collapsed, how="left")
    )
    group_df["file_path"] = f"{control_path}/pre/ecg-waveforms/waveform-rhythm.npy"
    return group_df[CONTROL_COLS]


def calculate_rr_std(ecg):
    """calculate RR variance using first lead"""
    from biosppy.signals import ecg as becg

    rpeaks = becg.christov_segmenter(ecg[0])[0]
    diffs = [rpeaks[i] - rpeaks[i - 1] for i in range(1, len(rpeaks))]

    return np.std(diffs)

def select_from_multiple_ecgs(group):
    valid_rows = group[group['days_to_event'] >= 2]
    
    if not valid_rows.empty:
        return valid_rows.loc[valid_rows['days_to_event'].idxmin()]
    else:
        return group.loc[group['days_to_event'].idxmin()]


if __name__ == "__main__":
    study_path = "../../datasets/arrest-ntuh-ecg/v1/study-group"
    study_group_df = process_study_df(study_path)
    study_group_df["group"] = "study"
    # initialize full DF, starting by appending arrest patients
    cardiac_df = pd.DataFrame(columns=["group"] + COLS)
    cardiac_df = pd.concat([cardiac_df, study_group_df], ignore_index=True)

    # establish control group by year and append to full DF
    for year in os.listdir("../../datasets/arrest-ntuh-ecg/v1/control-group/"):
        if year != "ecg-cohort.csv":
            control_path = f"../../datasets/arrest-ntuh-ecg/v1/control-group/{year}"
            control_group_df = process_control_df(control_path)
            control_group_df["group"] = year
            cardiac_df = pd.concat([cardiac_df, control_group_df], ignore_index=True)
            
    # Drop rows where days_to_event is NaN
    cardiac_df = cardiac_df[~pd.isnull(cardiac_df['days_to_event'])]
    cardiac_df['days_to_event'] = cardiac_df['days_to_event'].astype(int)
    
    # In cases (should be 2) where patients have more than one ECG reading, take the one that is closest to the date of the ECG reading,
    # but at least 2 days after the reading.
    cardiac_df = cardiac_df.groupby('patient_ngsci_id').apply(select_from_multiple_ecgs).reset_index(drop=True)

    # define SCD outcome as any study who had an SCD more than one day out from visit and within 1 year,
    # to be as consistent with Halland variable definition as possible (no visit information available in this dataset)
    cardiac_df["scd3mo"] = (
        (cardiac_df["group"] == "study")
        & (cardiac_df["days_to_event"] <= 91)
        & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] == "cardiac event")
    )
    
    cardiac_df["scd6mo"] = (
        (cardiac_df["group"] == "study")
        & (cardiac_df["days_to_event"] <= 182)
        & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] == "cardiac event")
    )
    
    cardiac_df["scd1"] = (
        (cardiac_df["group"] == "study")
        & (cardiac_df["days_to_event"] <= 365)
        & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] == "cardiac event")
    )
    
    # repeat for scd2 (within 2 years)
    cardiac_df["scd2"] = (
        (cardiac_df["group"] == "study")
         & (cardiac_df["days_to_event"] <= 730)
         & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] == "cardiac event")
    )
    # repeat for scd (no maximum time between ECG reading and cardiac event)
    cardiac_df["scd"] = (
        (cardiac_df["group"] == "study")
         & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] == "cardiac event")
    )
    # For further analysis, identify SCD outcomes *not caused by a cardiac event* (i.e. opioid overdose)
    # Our algorithm should *not* perform well on this
    cardiac_df["arrest_noncardiac"] = (
        (cardiac_df["group"] == "study")
        & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] != "cardiac event")
    )
    cardiac_df["arrest_noncardiac1"] = (
        (cardiac_df["group"] == "study")
        & (cardiac_df["days_to_event"] <= 365)
        & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] != "cardiac event")
    )
    # repeat for scd2 (within 2 years)
    cardiac_df["arrest_noncardiac2"] = (
        (cardiac_df["group"] == "study")
         & (cardiac_df["days_to_event"] <= 730)
         & (cardiac_df["days_to_event"] >= 2)
        & (cardiac_df["cause_of_cardiac_arrest"] != "cardiac event")
    )
    
    cardiac_df["any_arrest"] = (cardiac_df["scd"] | cardiac_df["arrest_noncardiac"])
    cardiac_df["any_arrest1"] = (cardiac_df["scd1"] | cardiac_df["arrest_noncardiac1"])

    cardiac_df[["lbbb", "qtc500", "stelev", "stdown", "dltwv"]] = cardiac_df[
        ["lbbb", "qtc500", "stelev", "stdown", "dltwv"]
    ].fillna(False)
    cardiac_df["age"] = cardiac_df["age"].str.extract("(\d+)", expand=False).astype(float)
    
    # align patient sex definition to Halland set-up
    cardiac_df["female"] = float("NaN")
    cardiac_df.loc[cardiac_df["sex"] == "male", "female"] = 0
    cardiac_df.loc[cardiac_df["sex"] == "female", "female"] = 1
    
    cardiac_df.to_feather("covariate_df.feather")