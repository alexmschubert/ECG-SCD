""" Functions for building out the dataset from SQL tables. """
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import string
import typing

import numpy as np
import pandas as pd
from tqdm import tqdm

from .definitions import CARDIAC_CONTROLS, DX_DICT, SCD_CODES, SYMP_DICT, SCD_CODES_AUG, CONTROL_DICT
from .regress import sm_regress
from .utils import log_print


def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.
    
    Does not support timeout or chunksize as executor.submit is used internally
    
    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm(as_completed(futures_list), total=len(futures_list), **kwargs):
        yield f.result()

def compute_scd1_conditional_ami(train_df):
    """Compute P(SCD1|AMI) for use in Sendhil's weighted predictor idea."""
    train_pts = train_df["ptId"].unique()

    codes = [code[:4] for code in DX_DICT["Acute_MI"]]

    q = f"SELECT * FROM v_ptDx"
    dx = pd.read_sql(q, con=set_cnxn())
    dx = dx.loc[(dx["dxCode"].str[:4].isin(codes)) & (dx["ptId"].isin(train_pts))]

    train_df["daysToDeathFull"] = train_df["days"] + train_df["daysToDeath"]
    scd_deaths = train_df.loc[(~train_df["daysToDeathFull"].isna()) & (train_df["scdall"] == 1)].groupby("ptId").head(1)

    dx_death = dx.merge(scd_deaths[["ptId", "daysToDeathFull"]], how="left")
    dx_death = dx_death.loc[~(dx_death["daysToDeathFull"] - dx_death["dxRelDays"] <= 1)]

    n_ami = len(dx_death)
    n_scd_following_ami = sum(dx_death["daysToDeathFull"] - dx_death["dxRelDays"] <= 365.25)

    return n_scd_following_ami / n_ami


def get_sm_predictor(covariate_df, train_df):
    """
    Sendhil predictor concept --
    Apply P(SCD1|AMI) to patients who have AMI within a year but not SCD1.
    This way, the model is not penalized as much for applying risk to truly risky patients.
    """

    scd1_conditional_ami = 0.00847457627118644  # compute_scd1_conditional_ami(train_df)

    covariate_df["y_sm"] = 0
    covariate_df.loc[(covariate_df["Acute_MI1"] == 1) & (covariate_df["scd1"] == 0), "y_sm"] = scd1_conditional_ami
    covariate_df.loc[covariate_df["scd1"] == 1, "y_sm"] = 1

    return covariate_df["y_sm"]


def get_jr_jl_predictor(full_df, train_df, jr=True):
    """
    James / Jack predictor concepts --
    Mainly - assign predicted risk of SCD1 from logistic regression so
    as to not penalize the model for predicting high risk in legitimately
    risky patients.

    In JL case, this basically means converting logistic regression weights
    to a mapping from ECGs.

    In JR case, set true 'scd1' patients to 1 so that model can learn
    signal beyond logistic predictions.
    """

    # Note: we train regression on non-defib patients and predict for all.
    _, full_df["y"] = sm_regress(
        train_df=train_df,
        val_df=full_df,
        covariates=["female", "most_recent_ef", "ef_na"] + CARDIAC_CONTROLS,
        pred_var="scd1",
        cluster=train_df["ptId"],
    )
    if jr:
        full_df.loc[full_df["scd1"] == 1, "y"] = 1
    return full_df["y"]


def append_conditional_heads(covariate_df):
    """Append new heads for SM, JR, and JL predictors"""

    fit_df = covariate_df.copy()
    fit_df = na_dummy(fit_df, "most_recent_ef", "ef_na")
    fit_df["female"] = fit_df["female"].fillna(np.mean(fit_df["female"]))
    train_ids, _ = get_ids((30, 30))
    # drop all defib patients from fitting
    train_ids = np.setdiff1d(train_ids, get_defib_ids(sub="all"))
    train_df = fit_df.loc[fit_df["studyId"].isin(train_ids)]
    covariate_df["y_sm"] = get_sm_predictor(fit_df, train_df)
    covariate_df["y_jr"] = get_jr_jl_predictor(fit_df, train_df, jr=True)
    covariate_df["y_jl"] = get_jr_jl_predictor(fit_df, train_df, jr=False)

    return covariate_df


def append_cardiac_controls(
        covariate_df, 
        control_dict: typing.Union[dict, None] = None, 
        VFVT: bool = True, 
        checkpoint: typing.Union[str, None] = None
):
    """Get new cardiac controls by code and timeframe."""
    if control_dict is None:
        control_dict = CONTROL_DICT

    cnxn = set_cnxn()
    for i, (tablename, var_details) in enumerate(control_dict.items()):
        print(f"Building variables from {tablename}. ({i}/{len(control_dict)})")
        table = pd.read_sql(f"SELECT * FROM {tablename}", cnxn)
        # iterate through variables
        for j, (varname, timeframe, codes, cutoffs) in enumerate(var_details):
            print(f"Building variable {varname}. ({j}/{len(var_details)})")
            if tablename == "v_ptDx":
                positives = processDx(codes, timeframe, covariate_df, table)
            elif tablename == "v_ptStatementcodes":
                positives = processStatementcodes(codes, timeframe, covariate_df, table)
            elif tablename == "v_ptMeds":
                positives = processMeds(codes, timeframe, covariate_df, table)  # only handles "pre" and "post_year"
            elif tablename == "v_ptCrossleadmeasurement":
                positives = processCrossleadmeasurement(timeframe, covariate_df, table) 
                covariate_df = covariate_df.merge(table[["studyId","meanqtc"]], how="left", on="studyId")
            elif tablename == "v_ptLeadmeasurement":
                positives = processLeadmeasurement(codes, timeframe, covariate_df, table)  # only handles "pre" and "post_year"
            elif tablename == "v_ptLabs":
                positives = processLabs(codes, timeframe, covariate_df, table, cutoffs)
            covariate_df[varname] = covariate_df["studyId"].isin(positives)

        if checkpoint is not None:
            covariate_df.to_feather(checkpoint)
    # Finally, append measurements of ventratestddev
    if VFVT:
        covariate_df["ventratestddev"] = get_ventratestddev(covariate_df)
        covariate_df["VFVT"] = covariate_df["ICD_VT"] | covariate_df["ICD_VF"] | covariate_df["Phi_VT"]
        covariate_df["VFVT1"] = covariate_df["ICD_VT1"] | covariate_df["ICD_VF1"] | covariate_df["Phi_VT1"]

    return covariate_df


def get_ventratestddev(covariate_df):
    """Get HR variability at ECG level."""

    print(f"Building variables from v_ptGroupmeasurement")
    table = pd.read_sql(f"SELECT * FROM v_ptGroupmeasurement", con=set_cnxn())
    table = table.loc[table["groupnumber"] == 1]
    # If ventratestddev exists, replace it
    covariate_df = covariate_df.drop(columns="ventratestddev", errors="ignore")
    covariate_df = covariate_df.merge(table[["studyId", "ventratestddev"]], how="left", on="studyId")

    return covariate_df["ventratestddev"]

def get_positives_dx(args):
    # Unpack args
    row, shm_name, shape, dtype, cols = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cov_df_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    cov_df = pd.DataFrame(cov_df_np)
    cov_df.columns = cols
    
    if row['timeframe'] == "pre":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["dxRelDays"]), "studyId"
        ].tolist()
    elif row['timeframe'] == "post_year":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & (cov_df["days"] < row["dxRelDays"])
            & (cov_df["days"] + 365.25 >= row["dxRelDays"]),
            "studyId",
        ].tolist()
    
    # Close shared memory
    existing_shm.close()

    if len(pos_ids) > 0:
        return pos_ids
    
def get_positives_meds(args):
    # Unpack args
    row, shm_name, shape, dtype, cols = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cov_df_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    cov_df = pd.DataFrame(cov_df_np)
    cov_df.columns = cols

    if row['timeframe'] == "pre":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["rxRelDays"]), "studyId"
            ].tolist()
    elif row['timeframe'] == 'pre30':
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["rxRelDays"])
            & (cov_df["days"] <= row["rxRelDays"] + 30), "studyId"
            ].tolist()

    elif row['timeframe'] == "post_year":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & (cov_df["days"] < row["rxRelDays"])
            & (cov_df["days"] + 365.25 >= row["rxRelDays"]),
            "studyId",
        ].tolist()

    # Close shared memory
    existing_shm.close()

    if len(pos_ids) > 0:
        return pos_ids
    
def get_positives_statements(args):
    # Unpack args
    row, shm_name, shape, dtype, cols = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cov_df_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    cov_df = pd.DataFrame(cov_df_np)
    cov_df.columns = cols

    if row['timeframe'] == "pre":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["days"]), "studyId"
        ].tolist()
    elif row['timeframe'] == "post_year":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & (cov_df["days"] < row["days"])
            & (cov_df["days"] + 365.25 >= row["days"]),
            "studyId",
        ].tolist()

    # Close shared memory
    existing_shm.close()

    if len(pos_ids) > 0:
        return pos_ids
    
def get_positives_crosslead(args):
    # Unpack args
    row, shm_name, shape, dtype, cols = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cov_df_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    cov_df = pd.DataFrame(cov_df_np)
    cov_df.columns = cols

    if row['timeframe'] == "pre":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["days"]), "studyId"
        ].tolist()
    elif row['timeframe'] == "post_year":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & (cov_df["days"] < row["days"])
            & (cov_df["days"] + 365.25 >= row["days"]),
            "studyId",
        ].tolist()

    # Close shared memory
    existing_shm.close()
    
    if len(pos_ids) > 0:
        return pos_ids

def get_positives_lead(args):
    # Unpack args
    row, shm_name, shape, dtype, cols = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cov_df_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    cov_df = pd.DataFrame(cov_df_np)
    cov_df.columns = cols

    if row['timeframe'] == "pre":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["days"]), "studyId"
        ].tolist()
    elif row['timeframe'] == "post_year":
        pos_ids = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & (cov_df["days"] < row["days"])
            & (cov_df["days"] + 365.25 >= row["days"]),
            "studyId",
        ].tolist()

    # Close shared memory
    existing_shm.close()

    if len(pos_ids) > 0:
        return pos_ids
    
def get_positives_labs(args):
    row, shm_name, shape, dtype, cols = args
    
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cov_df_np = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    cov_df = pd.DataFrame(cov_df_np)
    cov_df.columns = cols

    if row['timeframe'] == "pre":
        pos_df = cov_df.loc[
            (cov_df["ptId"] == row["ptId"]) & (cov_df["days"] >= row["labRelDays"])]
        
    elif row['timeframe'] == "post24":
        pos_df = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & ((cov_df["days"] == row["labRelDays"]) | (cov_df["days"] == row["labRelDays"] - 1))
        ]
    elif row['timeframe'] == "prior40":
        pos_df = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & ((cov_df["days"] >= row["labRelDays"]) & (cov_df["days"] - 40 <= row["labRelDays"]))
        ]
    elif row['timeframe'] == "prior40_death":
        death_df = cov_df.loc[(cov_df["days"] == 0) & ~(cov_df["daysToDeath"].isna())].groupby("ptId").head(1)
        tropt_death_df = death_df.loc[
            (death_df["ptId"] == row["ptId"])
            & (
                (death_df["daysToDeath"] - 1 > row["labRelDays"])
                & (death_df["daysToDeath"] - 40 <= row["labRelDays"])
            )
        ]
        pos_df = cov_df.loc[cov_df["ptId"].isin(tropt_death_df["ptId"])]
    elif row['timeframe'] == "post_year":
        pos_df = cov_df.loc[
            (cov_df["ptId"] == row["ptId"])
            & ((cov_df["days"] < row["labRelDays"]) & (cov_df["days"] + 365.25 >= row["labRelDays"]))
        ]

    # Close shared memory
    existing_shm.close()

    if len(pos_df) > 0:
        positives = pos_df["studyId"].tolist()
        labResult = [row['labResult']] * len(pos_df["studyId"].tolist())
        return positives, labResult
    
        
def processDx(codes, timeframe, cov_df, ptDx):
    """Process diagnosis codes into binary indicators based on diagnosis date relative to ECG."""
    cand_df = pd.DataFrame(columns=ptDx.columns)
    for code in codes:
        cands = ptDx.loc[(ptDx["ptId"].isin(cov_df["ptId"])) & (ptDx["dxCode"].str[: len(code)] == code)]
        cand_df = pd.concat([cand_df, cands], ignore_index=True)
    if timeframe == "pre":
        cand_df = cand_df.sort_values("dxRelDays", ascending=True)
        cand_df = cand_df.groupby("ptId").head(1)
    cand_df['timeframe'] = timeframe        
    cand_records = cand_df.to_dict("records")

    # Convert cov_df to numpy array for use in shared memory
    cov_df_np = cov_df.to_numpy()
    shape = cov_df_np.shape
    dtype = cov_df_np.dtype
    cols = cov_df.columns

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=cov_df_np.nbytes)
    # Create a NumPy array backed by shared memory
    shared_cov_df_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shared_cov_df_np, cov_df_np)

    # Prepare records
    records = [(record, shm.name, shape, dtype, cols) for record in cand_records]

    # Run in parallel
    log_print("Processing Dx in parallel...")
    try:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(get_positives_dx, records), total=len(cand_records)))
        log_print("Done processing Dx in parallel")
    except Exception as ex:
        log_print("Error processing Dx in parallel: " + str(ex))
    finally:
        shm.unlink()

    positives = [item for sublist in results if sublist is not None for item in sublist]
    
    return positives


def processMeds(codes, timeframe, cov_df, ptMeds):
    """Process medication codes into binary indicators based on pickup date relative to ECG."""

    cand_df = pd.DataFrame(columns=ptMeds.columns)
    for code in codes:
        cands = ptMeds.loc[(ptMeds["ptId"].isin(cov_df["ptId"])) & (ptMeds["rxCode"].str[: len(code)] == code)]
        cand_df = pd.concat([cand_df, cands], ignore_index=True)
    
    if timeframe == "pre":
            cand_df = cand_df.sort_values("rxRelDays", ascending=True)
            cand_df = cand_df.groupby("ptId").head(1)
    
    if timeframe == "pre":
            cand_df = cand_df.sort_values("rxRelDays", ascending=True)
            cand_df = cand_df.groupby("ptId").head(1)

    cand_df['timeframe'] = timeframe    
    cand_records = cand_df.to_dict("records")
    
    # Convert cov_df to numpy array for use in shared memory
    cov_df_np = cov_df.to_numpy()
    shape = cov_df_np.shape
    dtype = cov_df_np.dtype
    cols = cov_df.columns

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=cov_df_np.nbytes)
    # Create a NumPy array backed by shared memory
    shared_cov_df_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shared_cov_df_np, cov_df_np)

    # Prepare records
    records = [(record, shm.name, shape, dtype, cols) for record in cand_records]

    # Run in parallel
    log_print("Processing Meds in parallel...")
    try:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(get_positives_meds, records), total=len(cand_records)))
        log_print("Done processing Meds in parallel")
    except Exception as ex:
        log_print("Error processing Meds in parallel: " + str(ex))
    finally:
        shm.unlink()

    positives = [item for sublist in results if sublist is not None for item in sublist]

    return positives


def processStatementcodes(codes, timeframe, cov_df, ptStatementcodes):
    """Process statement codes into binary indicators based on date relative to ECG."""

    ptStatementcodes.columns = ptStatementcodes.columns.str.replace(" ", "")
    positives_at = ptStatementcodes.loc[(ptStatementcodes[codes] == 1).any(axis=1)]
    cov_df["codematch"] = cov_df["studyId"].isin(positives_at["studyId"])
    positive_df = cov_df.loc[cov_df["codematch"] == 1]
    if timeframe == "@":
        positives = list(positive_df["studyId"])
    else:
        if timeframe == "pre":
            positive_df = positive_df.sort_values("days", ascending=True)
            positive_df = positive_df.groupby("ptId").head(1)     
        positive_df['timeframe'] = timeframe
        positive_records = positive_df.to_dict("records")

        # Convert cov_df to numpy array for use in shared memory
        cov_df_np = cov_df.to_numpy()
        shape = cov_df_np.shape
        dtype = cov_df_np.dtype
        cols = cov_df.columns

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=cov_df_np.nbytes)
        # Create a NumPy array backed by shared memory
        shared_cov_df_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        np.copyto(shared_cov_df_np, cov_df_np)

        # Prepare records
        records = [(record, shm.name, shape, dtype, cols) for record in positive_records]
        # Run in parallel
        try:
            log_print("Processing Statements in parallel...")
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(get_positives_statements, records), total=len(positive_records)))
            log_print("Done processing Statements in parallel")
        except Exception as ex:
            log_print("Error processing Statements in parallel: " + str(ex))
        finally:
            shm.unlink()

        positives = [item for sublist in results if sublist is not None for item in sublist]

    return positives


def processCrossleadmeasurement(timeframe, cov_df, ptCrossleadmeasurement):
    """Get QTC>=500 indicator at ECG level."""
    positives = ptCrossleadmeasurement.loc[ptCrossleadmeasurement["meanqtc"] >= 500, "studyId"]
    print(positives.shape)
    print(type(positives))
    cov_df["codematch"] = cov_df["studyId"].isin(positives)
    positive_df = cov_df.loc[cov_df["codematch"] == 1]
        
    if timeframe == "@":
        pass
    else:
        if timeframe == "pre":
            positive_df = positive_df.sort_values("days", ascending=True)
            positive_df = positive_df.groupby("ptId").head(1)
        positive_df['timeframe'] = timeframe
        positive_records = positive_df.to_dict("records")
        
        # Convert cov_df to numpy array for use in shared memory
        cov_df_np = cov_df.to_numpy()
        shape = cov_df_np.shape
        dtype = cov_df_np.dtype
        cols = cov_df.columns

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=cov_df_np.nbytes)
        # Create a NumPy array backed by shared memory
        shared_cov_df_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        np.copyto(shared_cov_df_np, cov_df_np)

        # Prepare records
        records = [(record, shm.name, shape, dtype, cols) for record in positive_records]

        # Run in parallel
        try:
            log_print("Processing Crossleads in parallel...")
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(get_positives_crosslead, records), total=len(positive_records)))
            log_print("Done processing Crossleads in parallel")
        except Exception as ex:
            log_print("Error processing Crossleads in parallel: " + str(ex))
        finally:
            shm.unlink()

        positives = [item for sublist in results if sublist is not None for item in sublist]

    return positives


def processLeadmeasurement(codes, timeframe, cov_df, ptLeadmeasurement):
    """Process lead codes into binary indicators based on date relative to ECG."""

    positive_ids = ptLeadmeasurement.loc[(ptLeadmeasurement[codes] == "True").any(axis=1), "studyId"]
    positive_df = cov_df.loc[cov_df["studyId"].isin(positive_ids)]

    if timeframe == "pre":
        positive_df = positive_df.sort_values("days", ascending=True)
        positive_df = positive_df.groupby("ptId").head(1)

    if timeframe == "pre":
        positive_df = positive_df.sort_values("days", ascending=True)
        positive_df = positive_df.groupby("ptId").head(1)
    
    positive_df['timeframe'] = timeframe
    positive_records = positive_df.to_dict("records")
    
    # Convert cov_df to numpy array for use in shared memory
    cov_df_np = cov_df.to_numpy()
    shape = cov_df_np.shape
    dtype = cov_df_np.dtype
    cols = cov_df.columns

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=cov_df_np.nbytes)
    # Create a NumPy array backed by shared memory
    shared_cov_df_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shared_cov_df_np, cov_df_np)

    # Prepare records
    records = [(record, shm.name, shape, dtype, cols) for record in positive_records]

    # Run in parallel
    log_print("Processing Leadmeasurements in parallel...")
    try:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(get_positives_lead, records), total=len(positive_records)))
        log_print("Done processing Leadmeasurements in parallel")
    except Exception as ex:
        log_print("Error processing Leadmeasurements in parallel: " + str(ex))
    finally:
        shm.unlink()

    positives = [item for sublist in results if sublist is not None for item in sublist]

    return positives


def processLabs(codes, timeframe, cov_df, ptLabs, cutoffs, test_available_only=False, return_posdf = False):
    """Process lab codes into binary indicators based on date relative to ECG."""
    log_print("Processing labs...")
    sex_df = cov_df.groupby("ptId").head(1)[["ptId", "female"]]
    ptLabs = ptLabs.merge(sex_df, how="left", on="ptId")
    ptLabs["labResult"] = ptLabs["labResult"].str.replace("<", "")
    ptLabs["labResult"] = ptLabs["labResult"].str.replace(">", "")
    ptLabs["labResult"] = pd.to_numeric(ptLabs["labResult"], errors="coerce")
    cand_df = pd.DataFrame(columns=ptLabs.columns)
    if test_available_only:
        cands = ptLabs.loc[
                (ptLabs["ptId"].isin(cov_df["ptId"]))
                & (ptLabs["labCode"] == code)
            ]
        cand_df = pd.concat([cand_df, cands], ignore_index=True)
    else:
        if cutoffs == None:        
            for code in codes:
                log_print(f"Processing {code}")
                cands = ptLabs.loc[
                    (ptLabs["ptId"].isin(cov_df["ptId"]))
                    & (ptLabs["labCode"] == code)
                    & (
                        ((ptLabs["labResult"] >= 34) & (ptLabs["female"] == 0))
                        | ((ptLabs["labResult"] >= 16) & (ptLabs["female"] == 1))
                    )
                ]

                cand_df = pd.concat([cand_df, cands], ignore_index=True)
        else:
            if cutoffs[0] == "above":
                for code in codes:
                    cands = ptLabs.loc[
                        (ptLabs["ptId"].isin(cov_df["ptId"]))
                        & (ptLabs["labCode"] == code)
                        & (ptLabs["labResult"] > cutoffs[1]) 
                    ]
                    cand_df = pd.concat([cand_df, cands], ignore_index=True)
            elif cutoffs[0] == "smaller":
                for code in codes:
                    cands = ptLabs.loc[
                        (ptLabs["ptId"].isin(cov_df["ptId"]))
                        & (ptLabs["labCode"] == code)
                        & (ptLabs["labResult"] < cutoffs[1]) 
                    ]
                    cand_df = pd.concat([cand_df, cands], ignore_index=True)

    if timeframe == "pre":
        cand_df = cand_df.sort_values("labRelDays", ascending=True)
        cand_df = cand_df.groupby("ptId").head(1)   
    cand_df['timeframe'] = timeframe
    cand_records = cand_df.to_dict("records")
    # records_cov_df = [(record, cov_df) for record in cand_records]
    log_print(f"Length of records_cov_df: {len(cand_records)}")

    # Convert cov_df to numpy array for use in shared memory
    cov_df_np = cov_df.to_numpy()
    shape = cov_df_np.shape
    dtype = cov_df_np.dtype
    cols = cov_df.columns

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=cov_df_np.nbytes)
    # Create a NumPy array backed by shared memory
    shared_cov_df_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shared_cov_df_np, cov_df_np)

    # Prepare records
    records = [(record, shm.name, shape, dtype, cols) for record in cand_records]

    # Run in parallel
    log_print("Processing labs in parallel...")
    results = []
    try:
        with ProcessPoolExecutor() as executor:
            for result in tqdm_parallel_map(executor, lambda i: get_positives_labs(i), records):
                results += [result]
        log_print("Done processing labs in parallel")
    except Exception as ex:
        log_print("Error processing labs in parallel: " + str(ex))
    finally:
        shm.unlink()

    positives = [element for sublist, _ in results if sublist is not None for element in sublist]
    labResult = [element for _, sublist in results if sublist is not None for element in sublist]

    if return_posdf:
        return positives, labResult
    return positives




def append_dx_history(covariate_df):
    """Process diagnoses into binary indicators according to groupings defined in DX_DICT."""

    dx_dict_s = {k: pd.Series([i[:4] for i in v]).unique().tolist() for k, v in DX_DICT.items()}
    dx_dict_s["Acute_MI1"] = dx_dict_s["Acute_MI"]
    dx_dict_s["Acute_MI_pre"] = dx_dict_s["Acute_MI"]

    q = f"SELECT * FROM v_ptDx"
    cnxn = set_cnxn()
    dx = pd.read_sql(q, cnxn)

    for label, icd_codes in dx_dict_s.items():
        log_print(f"Processing {label}...")
        max_time = 365.25 if label != "Acute_MI" else 40
        positives = []
        dx_sub = pd.DataFrame(columns=dx.columns)
        for l in range(3, 5):
            codes_len = [c for c in icd_codes if len(c) == l]
            d = dx.loc[dx["dxCode"].str[:l].isin(codes_len)]
            dx_sub = pd.concat([dx_sub, d])
        dx_sub = dx_sub.drop_duplicates()
        covariate_df_sub = covariate_df.loc[covariate_df["ptId"].isin(dx_sub["ptId"].unique())]

        if label == "Acute_MI_pre":
            cands = dx_sub.sort_values("dxRelDays", ascending=True)
            cands = dx_sub.groupby("ptId").head(1)
            for row in tqdm(cands.to_dict("records")):
                pos_ids = covariate_df.loc[
                            (covariate_df["ptId"] == row["ptId"])
                            & (covariate_df["days"] > row["dxRelDays"]),'studyId'
                        ].tolist()
                if len(pos_ids) > 0:
                    positives += pos_ids
            covariate_df[label] = covariate_df["studyId"].isin(positives)

        else:
            for row in tqdm(covariate_df_sub.to_dict("records")):
                if not pd.isna(row["days"]):
                    if label == "Acute_MI1":
                        pt_dx = dx_sub.loc[
                            (dx_sub["ptId"] == row["ptId"])
                            & (dx_sub["dxRelDays"] > row["days"])
                            & (dx_sub["dxRelDays"] <= row["days"] + 365.25)
                        ]
                    #elif label == "Acute_MI_pre":
                        #pt_dx = dx_sub.loc[
                            #(dx_sub["ptId"] == row["ptId"])
                            #& (dx_sub["dxRelDays"] < row["days"])
                        #]
                    else:
                        pt_dx = dx_sub.loc[
                            (dx_sub["ptId"] == row["ptId"])
                            & (dx_sub["dxRelDays"] < row["days"])
                            & (dx_sub["dxRelDays"] >= row["days"] - max_time)
                        ]
                    if len(pt_dx) > 0:
                        positives.append(row["studyId"])
            covariate_df[label] = covariate_df["studyId"].isin(positives)

        if label == "Acute_MI_pre":
            cands = dx_sub.sort_values("dxRelDays", ascending=True)
            cands = dx_sub.groupby("ptId").head(1)
            for row in tqdm(cands.to_dict("records")):
                pos_ids = covariate_df.loc[
                            (covariate_df["ptId"] == row["ptId"])
                            & (covariate_df["days"] > row["dxRelDays"]),'studyId'
                        ].tolist()
                if len(pos_ids) > 0:
                    positives += pos_ids
            covariate_df[label] = covariate_df["studyId"].isin(positives)

        else:
            for row in tqdm(covariate_df_sub.to_dict("records")):
                if not pd.isna(row["days"]):
                    if label == "Acute_MI1":
                        pt_dx = dx_sub.loc[
                            (dx_sub["ptId"] == row["ptId"])
                            & (dx_sub["dxRelDays"] > row["days"])
                            & (dx_sub["dxRelDays"] <= row["days"] + 365.25)
                        ]
                    #elif label == "Acute_MI_pre":
                        #pt_dx = dx_sub.loc[
                            #(dx_sub["ptId"] == row["ptId"])
                            #& (dx_sub["dxRelDays"] < row["days"])
                        #]
                    else:
                        pt_dx = dx_sub.loc[
                            (dx_sub["ptId"] == row["ptId"])
                            & (dx_sub["dxRelDays"] < row["days"])
                            & (dx_sub["dxRelDays"] >= row["days"] - max_time)
                        ]
                    if len(pt_dx) > 0:
                        positives.append(row["studyId"])
            covariate_df[label] = covariate_df["studyId"].isin(positives)

        if label == "Acute_MI":
            pt_MI_deaths = []
            covariate_df["daysToDeathFull"] = covariate_df["days"] + covariate_df["daysToDeath"]
            patient_deaths = covariate_df.loc[~(covariate_df["daysToDeathFull"].isna())].groupby("ptId").head(1)
            for row in tqdm(patient_deaths.to_dict("records")):
                pt_dx = dx_sub.loc[
                    (dx_sub["ptId"] == row["ptId"])
                    & (dx_sub["dxRelDays"] < row["daysToDeathFull"] - 1)
                    & (dx_sub["dxRelDays"] >= row["daysToDeathFull"] - 40)
                ]
                if len(pt_dx) > 0:
                    pt_MI_deaths.append(row["ptId"])
            covariate_df["Acute_MI_death"] = covariate_df["ptId"].isin(pt_MI_deaths)
            covariate_df = covariate_df.drop(columns=["daysToDeathFull"])

    return covariate_df


def define_categoricals(df):
    """Define useful set of categoricals."""

    df.loc[df["most_recent_ef"].isna(), "EF_cat"] = "NoVal"
    df.loc[df["most_recent_ef"] <= 35, "EF_cat"] = "unhealthy"
    df.loc[(df["most_recent_ef"] > 35) & (df["most_recent_ef"] <= 50), "EF_cat"] = "mid"
    df.loc[df["most_recent_ef"] > 50, "EF_cat"] = "healthy"

    for i in range(0, int(np.max(df["age"])), 10):
        df.loc[(df["age"] >= i) & (df["age"] < i + 10), "age_bin"] = f"{i}-{i+10}"

    return df


def append_death_outcomes(covariate_df):
    """Define patient death outcomes. Most importantly, 'scd1'."""

    log_print("Merging outcomes...")

    i469 = ['I469']

    time_dict = {"all": None, "any_time": 20, "3mo": 0.25, "6mo": 0.5, "1": 1, "2": 2, "3": 3}
    code_dict = {
        "scd": [SCD_CODES, "outpatient_death"],
        "scd_new": [SCD_CODES_AUG, "outpatient_death"],
        "scdI469": [i469, "outpatient_death"],
        "in_scd": [SCD_CODES, "inpatient_death"],
        "all_scd": [SCD_CODES, "both"],
        "all_scd_new": [SCD_CODES_AUG, "both"],
        "all_scdI469": [i469, "both"],
        "unrelated": [get_unrelated_codes(SCD_CODES), "both"],
    }

    cod_query = "SELECT * FROM v_ptCauseofdeath"
    cod = pd.read_sql(cod_query, con=set_cnxn())

    detail_query = "SELECT * FROM v_ptECGDetails"
    details = pd.read_sql(detail_query, con=set_cnxn())

    ip_query = "SELECT * FROM v_ptLastIPVisit"
    ipvisit = pd.read_sql(ip_query, con=set_cnxn())

    covariate_df = covariate_df.merge(cod, how="left")
    covariate_df = covariate_df.merge(details, how="left")
    covariate_df = covariate_df.merge(ipvisit, how="left")

    covariate_df.loc[
        ((covariate_df["admissionRelDays"] == 0) | (covariate_df["dischargeRelDays"] < 0)), "outpatient_death"
    ] = True
    covariate_df.loc[
        ((covariate_df["admissionRelDays"] < 0) & (covariate_df["dischargeRelDays"] == 0)), "inpatient_death"
    ] = True

    for k, (codes, outpatient_str) in code_dict.items():
        for time in time_dict.keys():
            covariate_df[f"{k}{time}"] = False

        if outpatient_str in ["outpatient_death", "inpatient_death"]:
            for code in codes:
                covariate_df.loc[
                    (covariate_df["causeOfDeath"].str[: len(code)] == code) & (covariate_df[outpatient_str] == 1),
                    f"{k}all",
                ] = True
        else:
            for code in codes:
                covariate_df.loc[(covariate_df["causeOfDeath"].str[: len(code)] == code), f"{k}all"] = True

        for time, multi in time_dict.items():
            if time == "all":
                pass
            else:
                covariate_df.loc[
                    (covariate_df[f"{k}all"] == True)
                    & (covariate_df["daysToDeath"] >= 0)
                    & (covariate_df["daysToDeath"] < 365.25 * multi),
                    f"{k}{time}",
                ] = True

    covariate_df[f"deadall"] = ~(covariate_df["daysToDeath"].isna())
    for time, multi in time_dict.items():
        if time == "all":
            pass
        else:
            covariate_df[f"dead{time}"] = (covariate_df["daysToDeath"] < 365.25 * multi) & (
                covariate_df["daysToDeath"] >= 0
            )

    return covariate_df


def get_unrelated_codes(codes):
    """Define set of death codes that are explicitly unrelated to SCD."""

    def pad(n_str):
        if len(n_str) < 2:
            return f"0{n_str}"
        else:
            return n_str

    all_codes = list(string.ascii_uppercase)
    code_categories = set([code[0] for code in codes])
    # Initialize out_codes as any categories we don't touch
    out_codes = list(set(all_codes).difference(code_categories))
    for code in code_categories:
        full_ls = [f"{code}{pad(str(n))}" for n in range(100)]
        full_ls = [
            code for code in full_ls if (code[:2] not in codes and code[:3] not in codes)
        ]  # Removal of one and two figure matches
        out_codes += full_ls  # Removal of two figure matches

    return out_codes


def append_ecg_location(covariate_df):
    """Add information whether ECG has been taken during inpatient, outpatient or ED visit."""

    q = f"SELECT studyId, IP, OP, ED FROM v_ptECGHeader"
    cnxn = set_cnxn()
    locations = pd.read_sql(q, cnxn)

    covariate_df = covariate_df.merge(locations, how="left", on = "studyId")
    
    #covariate_df['outpatient']=covariate_df['OP']

    #Initialize other cleaned variables
    covariate_df["inpatient"]=0
    covariate_df["emergDept"]=0
    covariate_df["outpatient"]=0

    #Handle cases with more than one location per ECG
    covariate_df["visit_overlap"] =covariate_df["IP"] + covariate_df["OP"]+covariate_df["ED"]
    overlaps = covariate_df.loc[covariate_df["visit_overlap"]>1].copy(deep=True)

    #Identify ECGs with more than one location where the patient had multiple ECGs on the same day
    overlaps1 = overlaps.copy()
    overlaps_grouped = overlaps1.groupby(["ptId","days"]).count()
    overlaps_grouped_multiple = overlaps_grouped.loc[overlaps_grouped["studyId"]>1]
    overlaps_grouped_multiple.reset_index(inplace=True)

    #Create identifier for later use
    overlaps_grouped_multiple["same_day"] = overlaps_grouped_multiple["ptId"].astype(str)+' - '+overlaps_grouped_multiple["days"].astype('str')
    overlaps["same_day"] = overlaps["ptId"].astype(str)+' - '+overlaps["days"].astype('str')

    same_day_overlaps = overlaps.loc[overlaps["same_day"].isin(overlaps_grouped_multiple["same_day"])]

    #Sort so that ECGs on a given day are shown in chronological order
    same_day_overlaps=same_day_overlaps.sort_values(by = ['ptId', 'days','studyAcquiredHour'], ascending = [True, True,True], na_position = 'first')

    #Implement Ziads same day decision rule
    for arrival_day in list(overlaps_grouped_multiple["same_day"]):
        overlap_pt_day = same_day_overlaps[same_day_overlaps["same_day"]==arrival_day].copy()
        counter = 0
        for row in overlap_pt_day.to_dict("records"):
            if counter == 0:
                if row["OP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"outpatient"] =1
                elif row["ED"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
            elif counter == 1:
                if row["OP"] == 1 and row["ED"] == 1 and row["IP"]==1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
                elif row["OP"] == 1 and row["ED"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
                elif row["OP"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
                elif row["ED"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
            elif counter >= 2:
                if row["OP"] == 1 and row["ED"] == 1 and row["IP"]==1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
                elif row["OP"] == 1 and row["ED"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
                elif row["OP"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
                elif row["ED"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
            counter+= 1

    #Implement decision rule for overlaps where patient only has one ECG on each day
    different_day_overlaps = overlaps.loc[~overlaps["same_day"].isin(overlaps_grouped_multiple["same_day"])]
    for row in different_day_overlaps.to_dict("records"):
        if row["OP"] == 1 and row["ED"] == 1:
            covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"outpatient"] =1
        elif row["OP"] == 1 and row["IP"] == 1:
            covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"outpatient"] =1
        elif row["ED"] == 1 and row["IP"] == 1:
            covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1

    #Implement assignment to cleaned variables when there are no overlaps
    covariate_df.loc[~covariate_df["studyId"].isin(overlaps["studyId"]),"outpatient"]=covariate_df["OP"]
    covariate_df.loc[~covariate_df["studyId"].isin(overlaps["studyId"]),"emergDept"]=covariate_df["ED"]
    covariate_df.loc[~covariate_df["studyId"].isin(overlaps["studyId"]),"inpatient"]=covariate_df["IP"]

    #Classify ~6% of ECGs with missing location as outpatient (which is the most frequent location)
    covariate_df.loc[(covariate_df["OP"]==0) & (covariate_df["ED"]==0) & (covariate_df["IP"]==0), 'outpatient'] = 1

    #Create variable grouping outpatient and ED ECG locations
    covariate_df["outpatient_ED"] = covariate_df["outpatient"] + covariate_df["emergDept"]
    
    return covariate_df

def append_symptom_history(covariate_df):
    """Process Symptom ICD codes into binary indicators groupings defined in SYMP_DICT."""

    #dx_dict_s = {k: pd.Series([i[:4] for i in v]).unique().tolist() for k, v in DX_DICT.items()}
    #SYMP_DICT

    q = f"SELECT * FROM v_ptDx"
    cnxn = set_cnxn()
    dx = pd.read_sql(q, cnxn)

    for label, icd_code in SYMP_DICT.items():
        log_print(label)
        if label[-1]=='1':
            log_print('Dealing with next year variable')
            max_time = 365.25
            positives = []
            l = len(icd_code)
            cand_df = dx.loc[(dx["ptId"].isin(covariate_df["ptId"])) & (dx["dxCode"].str[: l] == icd_code)]

            for row in tqdm(cand_df.to_dict("records")):
                if not pd.isna(row["dxRelDays"]):
                    pos_ids = covariate_df.loc[
                        (row["ptId"] == covariate_df["ptId"])
                        & (row["dxRelDays"] > covariate_df["days"])
                        & (row["dxRelDays"] <= covariate_df["days"] + max_time), 'studyId'
                    ].tolist()
                    if len(pos_ids) > 0:
                        positives += pos_ids
            log_print(len(positives))
            covariate_df[label] = covariate_df["studyId"].isin(positives)
        else:
            log_print('Dealing with past oriented variable')
            positives = []
            l = len(icd_code)
            cand_df = dx.loc[(dx["ptId"].isin(covariate_df["ptId"])) & (dx["dxCode"].str[: l] == icd_code)]
            cand_df = cand_df.sort_values("dxRelDays", ascending=True)
            cand_df = cand_df.groupby("ptId").head(1)

            for row in tqdm(cand_df.to_dict("records")):
                if not pd.isna(row["dxRelDays"]):
                    pos_ids = covariate_df.loc[
                        (row["ptId"] == covariate_df["ptId"])
                        & (row["dxRelDays"] < covariate_df["days"]), 'studyId'
                    ].tolist()
                    if len(pos_ids) > 0:
                        positives += pos_ids
            print(len(positives))
            covariate_df[label] = covariate_df["studyId"].isin(positives)

    
    #covariate_df['outpatient']=covariate_df['OP']

    #Initialize other cleaned variables
    covariate_df["inpatient"]=0
    covariate_df["emergDept"]=0
    covariate_df["outpatient"]=0

    #Handle cases with more than one location per ECG
    covariate_df["visit_overlap"] =covariate_df["IP"] + covariate_df["OP"]+covariate_df["ED"]
    overlaps = covariate_df.loc[covariate_df["visit_overlap"]>1].copy(deep=True)

    #Identify ECGs with more than one location where the patient had multiple ECGs on the same day
    overlaps1 = overlaps.copy()
    overlaps_grouped = overlaps1.groupby(["ptId","days"]).count()
    overlaps_grouped_multiple = overlaps_grouped.loc[overlaps_grouped["studyId"]>1]
    overlaps_grouped_multiple.reset_index(inplace=True)

    #Create identifier for later use
    overlaps_grouped_multiple["same_day"] = overlaps_grouped_multiple["ptId"].astype(str)+' - '+overlaps_grouped_multiple["days"].astype('str')
    overlaps["same_day"] = overlaps["ptId"].astype(str)+' - '+overlaps["days"].astype('str')

    same_day_overlaps = overlaps.loc[overlaps["same_day"].isin(overlaps_grouped_multiple["same_day"])]

    #Sort so that ECGs on a given day are shown in chronological order
    same_day_overlaps=same_day_overlaps.sort_values(by = ['ptId', 'days','studyAcquiredHour'], ascending = [True, True,True], na_position = 'first')

    #Implement Ziads same day decision rule
    for arrival_day in list(overlaps_grouped_multiple["same_day"]):
        overlap_pt_day = same_day_overlaps[same_day_overlaps["same_day"]==arrival_day].copy()
        counter = 0
        for row in overlap_pt_day.to_dict("records"):
            if counter == 0:
                if row["OP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"outpatient"] =1
                elif row["ED"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
            elif counter == 1:
                if row["OP"] == 1 and row["ED"] == 1 and row["IP"]==1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
                elif row["OP"] == 1 and row["ED"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
                elif row["OP"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
                elif row["ED"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
            elif counter >= 2:
                if row["OP"] == 1 and row["ED"] == 1 and row["IP"]==1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
                elif row["OP"] == 1 and row["ED"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1
                elif row["OP"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
                elif row["ED"] == 1 and row["IP"] == 1:
                    covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"inpatient"] =1
            counter+= 1

    #Implement decision rule for overlaps where patient only has one ECG on each day
    different_day_overlaps = overlaps.loc[~overlaps["same_day"].isin(overlaps_grouped_multiple["same_day"])]
    for row in different_day_overlaps.to_dict("records"):
        if row["OP"] == 1 and row["ED"] == 1:
            covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"outpatient"] =1
        elif row["OP"] == 1 and row["IP"] == 1:
            covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"outpatient"] =1
        elif row["ED"] == 1 and row["IP"] == 1:
            covariate_df.loc[(covariate_df['studyId']==row["studyId"]),"emergDept"] =1

    #Implement assignment to cleaned variables when there are no overlaps
    covariate_df.loc[~covariate_df["studyId"].isin(overlaps["studyId"]),"outpatient"]=covariate_df["OP"]
    covariate_df.loc[~covariate_df["studyId"].isin(overlaps["studyId"]),"emergDept"]=covariate_df["ED"]
    covariate_df.loc[~covariate_df["studyId"].isin(overlaps["studyId"]),"inpatient"]=covariate_df["IP"]

    #Classify ~6% of ECGs with missing location as outpatient (which is the most frequent location)
    covariate_df.loc[(covariate_df["OP"]==0) & (covariate_df["ED"]==0) & (covariate_df["IP"]==0), 'outpatient'] = 1

    #Create variable grouping outpatient and ED ECG locations
    covariate_df["outpatient_ED"] = covariate_df["outpatient"] + covariate_df["emergDept"]
    
    return covariate_df

def append_symptom_history(covariate_df):
    """Process Symptom ICD codes into binary indicators groupings defined in SYMP_DICT."""
    log_print("Processing symptoms...")
    #dx_dict_s = {k: pd.Series([i[:4] for i in v]).unique().tolist() for k, v in DX_DICT.items()}
    #SYMP_DICT

    q = f"SELECT * FROM v_ptDx"
    cnxn = set_cnxn()
    dx = pd.read_sql(q, cnxn)

    for label, icd_code in SYMP_DICT.items():
        log_print(label)
        if label[-1]=='1':
            log_print('Dealing with next year variable')
            max_time = 365.25
            positives = []
            l = len(icd_code)
            cand_df = dx.loc[(dx["ptId"].isin(covariate_df["ptId"])) & (dx["dxCode"].str[: l] == icd_code)]

            for row in tqdm(cand_df.to_dict("records")):
                if not pd.isna(row["dxRelDays"]):
                    pos_ids = covariate_df.loc[
                        (row["ptId"] == covariate_df["ptId"])
                        & (row["dxRelDays"] > covariate_df["days"])
                        & (row["dxRelDays"] <= covariate_df["days"] + max_time), 'studyId'
                    ].tolist()
                    if len(pos_ids) > 0:
                        positives += pos_ids
            log_print(len(positives))
            covariate_df[label] = covariate_df["studyId"].isin(positives)
        else:
            log_print('Dealing with past oriented variable')
            positives = []
            l = len(icd_code)
            cand_df = dx.loc[(dx["ptId"].isin(covariate_df["ptId"])) & (dx["dxCode"].str[: l] == icd_code)]
            cand_df = cand_df.sort_values("dxRelDays", ascending=True)
            cand_df = cand_df.groupby("ptId").head(1)

            for row in tqdm(cand_df.to_dict("records")):
                if not pd.isna(row["dxRelDays"]):
                    pos_ids = covariate_df.loc[
                        (row["ptId"] == covariate_df["ptId"])
                        & (row["dxRelDays"] < covariate_df["days"]), 'studyId'
                    ].tolist()
                    if len(pos_ids) > 0:
                        positives += pos_ids
            log_print(len(positives))
            covariate_df[label] = covariate_df["studyId"].isin(positives)

    return covariate_df