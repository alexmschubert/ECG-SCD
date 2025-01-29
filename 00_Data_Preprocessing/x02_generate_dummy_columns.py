"""
This script augments the covariate_df with columns not present in the NTUH dataset
but necessary to run parts of the training and morphing code generated within the Halland environment.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

def sample_pduration(n):
    # Define intervals and their probabilities
    intervals = [(0, 20), (20, 120), (120, 160)]
    probs = [0.025, 0.95, 0.025]  # 10% in [0,20], 80% in [20,120], 10% in [120,160]

    # Randomly choose which interval each row should come from
    chosen_intervals = np.random.choice([0, 1, 2], size=n, p=probs)

    # For each row, sample from the chosen interval
    result = np.empty(n, dtype=float)
    for i, interval_idx in enumerate(chosen_intervals):
        low, high = intervals[interval_idx]
        result[i] = np.random.uniform(low, high)
    return result

def main(input_file: str = "covariate_df.feather", output_file: str = "covariate_df"):
    # Read input CSV and rename columns
    df = pd.read_feather(input_file).rename(columns={"ecg_id": "studyId", "qtc_frederica":"qtcf", 'qtc':'qtcb', 'qrs_duration': 'qrsDuration', 'avg_rr':'rrInterval', 'stelev':"ST_elevation", 'stdown': "ST_depression", 'atrial_rate': "atrialrate", "qt_interval":"qtInterval"})
    
    # Add scaling factor to convert NTUH ECGs to mV
    df["scaling_factor"] = 1000
    
    # Add patient Id column
    df["ptId"] = df["patient_ngsci_id"]
    
    # Implement last inclusion criteria
    df["include_modelling"] = (
        (~df["age"].isna()) &
        (~df["female"].isna()) &
        (df["age"] < 80) &
        (df['days_to_event'] >= 2)
    ).astype(int)
    
    # Define scd intervals
    df["scd3mo_6mo"] = (df["scd6mo"] == 1) & (df["scd3mo"] == 0)
    df["scd6mo_1"]   = (df["scd6mo"] == 0) & (df["scd1"] == 1)
    df["scd1_2"]     = (df["scd1"] == 0) & (df["scd2"] == 1)
    
    # Impute false for missing VFVT diagnosis
    df['VTVF'] = df['VTVF'].fillna(False)
    df['VFVT'] = df['VTVF'].fillna(False)

    # Dummy all-cause mortality outcomes
    mortality_outcomes = ["dead3mo", "dead6mo", "dead1", "dead2"]
    df[mortality_outcomes] = False
    death_rates = {
        "dead3mo": 0.02,  
        "dead6mo": 0.03,  
        "dead1": 0.04,     
        "dead2": 0.05      
    }
    for outcome, rate in death_rates.items():
        random_numbers = np.random.rand(len(df))
        random_deaths = random_numbers < rate
        
        scd_column = 'scd' + outcome.replace('dead', '')
        scd_deaths = df[scd_column] == 1
        df[outcome] = random_deaths | scd_deaths
    
    # Dummy additional outcomes
    additional_outcomes = ["low_ef", "ICD_VT1", "ICD_VF1", "Phi_VT1", "TROPT1", "Acute_MI1", "ICD_VT", "ICD_VF", "Phi_VT", "LBBB", "antiarr", "deltawave", "CHF_Cardiomyopathy", "Hypertension", "CAD", "Diabetes", "Hyperlipidemia", "TROPT40", "Acute_MI", "Old_MI"]
    
    df[additional_outcomes] = False
    
    # Continous dummy meaures
    df['heartRate'] = df['ventricular_rate']
    df["most_recent_ef"] = np.random.triangular(left=20, mode=55, right=70, size=len(df))
    df['pDuration'] = sample_pduration(len(df)) 
    df["meanqtc"] = df["qtcb"]
    df['qrsFrontAxis'] = np.random.uniform(low=-90, high=90, size=len(df))
    df['qrsHorizAxis'] = np.random.uniform(low=-90, high=90, size=len(df))
    
    # Save augmented dataframe
    df.to_csv(f'{output_file}.csv', index=False) 
    df.to_feather(f'{output_file}.feather')


if __name__ == "__main__":
    main()
    