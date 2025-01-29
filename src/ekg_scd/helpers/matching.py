""" 
Functions related to batch matching 

Primarily used in BatchShuffleSampler definition.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def get_matched_df(
    full_df,
    soft_match_vars: list,
    hard_match_vars: list,
    outcome_var: str,
    n_neighbors: int,
    n_matches: int,
    patient_limit: bool = False,
):
    """
    Define a DF where each positive sample gets n_matches.
    Matches can only be assigned within hard_match_vars bins and
    within those bins are assigned based upon XGB predicted risk
    proximity to the reference positive case.
    """

    vars_list = list(filter(None, [soft_match_vars, hard_match_vars, [outcome_var]]))
    all_vars = [item for sublist in vars_list for item in sublist]
    full_df = full_df.loc[~full_df[all_vars].isna().T.any()]

    if hard_match_vars:
        out_df = pd.DataFrame(columns=list(full_df.columns) + ["match"])
        hard_match_values = [full_df[v].unique().tolist() for v in hard_match_vars]
        hard_match_combinations = itertools.product(*hard_match_values)

        for combination in hard_match_combinations:
            use_df = full_df

            for i, v in enumerate(hard_match_vars):
                use_df = use_df.loc[use_df[v] == combination[i]]

            matched_df = soft_match(use_df, soft_match_vars, outcome_var, n_neighbors, n_matches, patient_limit)

            if matched_df is not None:
                out_df = out_df.append(matched_df)
    else:
        out_df = soft_match(full_df, soft_match_vars, outcome_var, n_neighbors, n_matches, patient_limit)

    return out_df


def soft_match(full_df, covariate_vars, outcome_var, n_neighbors, n_matches, patient_limit):
    """Get closest matches to positive cases on XGB predicted risk."""

    treated_df = full_df.loc[full_df[outcome_var] == 1].copy()
    non_treated_df = full_df.loc[full_df[outcome_var] == 0]
    # Ensure that non-SCD ECGs from future SCD patients are not candidates for matching
    non_treated_df = non_treated_df.loc[~non_treated_df["ptId"].isin(treated_df["ptId"])]

    if len(treated_df) == 0 or len(non_treated_df) == 0:
        return None

    # If only hard matches, randomly sample from non_treated
    if covariate_vars is None:
        treated_df["match"] = float("NaN")
        match_df = pd.DataFrame([], columns=treated_df.columns)
        for studyId in treated_df["studyId"]:
            if n_matches is None:
                matches = non_treated_df.copy()
            else:
                matches = non_treated_df.iloc[np.random.choice(len(non_treated_df), n_matches)].copy()
            matches["match"] = studyId
            match_df = match_df.append(matches, ignore_index=True)

        matched_df = treated_df.append(match_df, ignore_index=True)
        matched_df[outcome_var] = matched_df[outcome_var].astype(int)

        return matched_df

    if len(non_treated_df) < n_matches:
        return None

    # Fit StandardScaler() on treated X and apply to both treated and untreated
    treated_x = treated_df[covariate_vars]
    non_treated_x = non_treated_df[covariate_vars]

    scaler = StandardScaler()
    scaler.fit(treated_x)

    treated_x = scaler.transform(treated_x)
    non_treated_x = scaler.transform(non_treated_x)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(non_treated_x)

    # find neighbours in non-treated sample
    _, indices = nbrs.kneighbors(treated_x)

    treated_df.loc[:, "match"] = float("NaN")
    match_df = pd.DataFrame([], columns=treated_df.columns)
    for i, match_indices in enumerate(indices):
        matched = non_treated_df.iloc[match_indices]
        if not patient_limit:
            matches = matched.iloc[np.random.choice(n_neighbors, n_matches, replace=False)].copy()
        else:
            matches = matched.loc[matched["ptId"] != treated_df.iloc[i]["ptId"]]
            matches = matched.groupby("ptId").head(1)
            matches = matches.iloc[:n_matches]
        matches["match"] = treated_df.iloc[i]["studyId"]
        match_df = match_df.append(matches, ignore_index=True)

    matched_df = treated_df.append(match_df, ignore_index=True)

    matched_df[outcome_var] = matched_df[outcome_var].astype(int)

    return matched_df
