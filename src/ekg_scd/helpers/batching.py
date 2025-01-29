"""
Functions for matching batches by a provided covariate set.
(as well as functions for optimizing hyperparameters in matching)
"""

import random

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Sampler

def get_sampler_weights(train_ids,output_to_balance,target_freq):
    covariate_df = pd.read_feather(f"covariate_df.csv")
    mask = covariate_df['studyId'].isin(train_ids)
    covariate_df = covariate_df.loc[mask]
    covariate_df = covariate_df[['studyId',output_to_balance]]

    print("WARNING Balancing data for scd1")

    class_counts = covariate_df.scd1.value_counts()

    non_target_freq = 1-target_freq

    target_proportions = {0:non_target_freq, 1:target_freq}

    sample_weights = [(target_proportions[i])*1/class_counts[i] for i in covariate_df.scd1.values]

    return sample_weights



def grid_search(X, y, ptIds):
    """Basic grid search for finding optimal propensity scoring parameters."""

    estimator = xgb.XGBClassifier(seed=90210, use_label_encoder=False, eval_metric="logloss")
    parameters = {
        "max_depth": range(1, 11, 2),
        "n_estimators": range(20, 260, 60),
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "alpha": [0, 1e-2, 1e-1, 1, 10],
        "colsample_bytree": np.arange(0.2, 1.01, 0.2),
    }
    # partition patient IDs into 5 groups and define as an iterable to match requirements
    # of GridSearchCV
    pt_list = partition_ptIds(ptIds, 5)
    pt_iterable = make_idx_iterable(ptIds=ptIds, pt_list=pt_list)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        n_jobs=50,
        cv=pt_iterable,
        verbose=3,
        scoring="neg_log_loss",
    )
    grid_search.fit(X, y)

    print(grid_search.best_score_)
    print(grid_search.cv_results_["params"][grid_search.best_index_])


def make_idx_iterable(ptIds, pt_list):
    """Create an iterable of indices based on patient partition."""

    ptIds.reset_index(drop=True, inplace=True)
    iterable_idx = []
    for f in range(len(pt_list)):
        # val_idx is one of 5 subsets, train_idx are remainder
        val_idx = ptIds.index[ptIds.isin(pt_list[f])].to_numpy()
        train_idx = ptIds.index[~ptIds.isin(pt_list[f])].to_numpy()
        iterable_idx.append((train_idx, val_idx))

    return iterable_idx


def partition_ptIds(ptIds, fold=5):
    """Randomly partition patient IDs into 5 folds for cross-validation."""
    pt_list = list(ptIds.unique())
    random.seed(90210)
    random.shuffle(pt_list)

    def partition(list, n):
        return [list[i::n] for i in range(n)]

    pt_bins = partition(pt_list, fold)
    return pt_bins


def cv(estimator, X, y, ptIds, fold):
    """Cross-validate estimator performance."""
    pt_bins = partition_ptIds(ptIds, fold)
    loglosses = []
    aucs = []
    for f in range(fold):
        val_locs = ptIds.isin(pt_bins[f])
        X_train = X.loc[~val_locs]
        X_test = X.loc[val_locs]
        y_train = y.loc[~val_locs]
        y_test = y.loc[val_locs]

        # fit xgb and predict
        estimator.fit(X_train, y_train)
        preds = estimator.predict_proba(X_test)[:, 1]
        # store outcomes for evaluation
        loglosses.append(log_loss(y_test, preds))
        aucs.append(roc_auc_score(y_test, preds))

    return np.mean(loglosses), np.mean(aucs)


def bayes_optimize(X, y, ptIds):
    """Implementation of bayes optimization for optimal XGB parameter search."""

    def bo_tune_xgb(max_depth, n_estimators, log_learning_rate, log_alpha, colsample_bytree):
        """XGB fitting function to meet BayestianOptimization class requirements."""
        parameters = {
            "objective": "binary:logistic",
            "max_depth": int(max_depth),
            "learning_rate": 10**log_learning_rate,
            "alpha": 10**log_alpha,
            "colsample_bytree": colsample_bytree,
            "n_estimators": int(n_estimators),
            "eval_metric": ["logloss", "auc"],
        }
        estimator = xgb.XGBClassifier(seed=90210, use_label_encoder=False)
        estimator.set_params(**parameters)
        # Cross validating with the specified parameters in 5 folds
        loss, auc = cv(estimator, X, y, ptIds, fold=5)
        # Return the negative logloss
        return -1.0 * loss

    # Instantiate BayesianOptimization and fit.
    # Modified bounds to "log_" to avoid sampling excessively from high end of distributions
    xgb_bo = BayesianOptimization(
        f=bo_tune_xgb,
        pbounds={
            "max_depth": (1, 5),
            "n_estimators": (10, 200),
            "log_learning_rate": (-1, -0.1),
            "log_alpha": (-2, 1),
            "colsample_bytree": (0, 1),
        },
        random_state=90210,
    )
    xgb_bo.maximize(n_iter=500, init_points=40, acq="ei", alpha=1e-4)
    # Print best parameters
    print(xgb_bo.max)


def propensity_score(X, y, ptIds):
    """
    Fit XGB according to found parameters and predict across 5 folds.
    This function forms the backbone of propensity score batch matching.
    """

    bo_params = {
        "colsample_bytree": 1.0,
        "alpha": 10**-1.1478107544038918,
        "learning_rate": 10**-0.45472748699703186,
        "max_depth": int(1.3918334199940172),
        "n_estimators": int(28.736130817323584),
    }
    estimator = xgb.XGBClassifier(seed=90210, use_label_encoder=False, eval_metric="logloss")
    estimator.set_params(**bo_params)
    pt_list = partition_ptIds(ptIds, 5)
    pt_iterable = make_idx_iterable(ptIds=ptIds, pt_list=pt_list)
    val_indices = [tv[1] for tv in pt_iterable]
    predictions = np.empty(len(X))
    for val_idx in val_indices:
        h_mask = np.isin(range(len(X)), val_idx)
        X_train = X[~h_mask]
        y_train = y[~h_mask]
        estimator.fit(X_train, y_train)
        predictions[h_mask] = estimator.predict_proba(X[h_mask])[:, 1]
    return predictions  # Predicted probability of positive class


def get_batch_locs(train_ids, pos_label="scd1", prop_match_vars=None, hard_match_vars=None):
    """
    Define batches based on XGB propensity scoring.
    For each positive sample (typically 'scd1'), get nearest n_matches to fill out batch.
    """
    covariate_df = pd.read_feather(f"covariate_df.csv")
    id_df = covariate_df.loc[covariate_df["studyId"].isin(train_ids)]
    id_df = hallandata.define_categoricals(id_df)
    id_df = hallandata.na_dummy(id_df, var="most_recent_ef", na_label="ef_na")
    id_df = id_df.dropna(subset=prop_match_vars)
    if prop_match_vars:
        id_df["propensity_score"] = propensity_score(
            X=id_df[prop_match_vars], y=id_df[pos_label].astype(int), ptIds=id_df["ptId"]
        )
    # Define a matched DF where each positive sample gets n_matches to fill out a batch.
    matched_df = matching.get_matched_df(
        id_df,
        soft_match_vars=None if prop_match_vars is None else ["propensity_score"],
        hard_match_vars=hard_match_vars,
        outcome_var=pos_label,
        n_neighbors=128,
        n_matches=128,
    )

    # Turn batches from DF (where each match gets 'match' == studyId of positive sample to which it's matched)
    # into list of indices to match BatchShuffleSampler requirements.
    match_ls = []
    for studyId in matched_df.loc[matched_df[pos_label] == True]["studyId"]:
        cand_df = matched_df.loc[matched_df["match"] == studyId]

        study_ls = [studyId] + cand_df["studyId"].tolist()

        # Convert studyIds to locs in ID set for indexing when loading
        idx_ls = [np.where(train_ids == idx)[0][0] for idx in study_ls]

        match_ls.append(idx_ls)

    return match_ls


class BatchShuffleSampler(Sampler):
    """
    Custom Sampler class which takes in batch_locs from above, randomly sampling batch_size from batches.
    This set-up ensures that we don't repeat batches through each epoch, while maintaining propensity score
    closeness.
    """

    def __init__(self, batch_locs: list, batch_size: int):
        self.batch_locs = batch_locs
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.batch_locs)

    def __iter__(self):
        n_batches = len(self.batch_locs)

        self.generator = torch.Generator()
        self.generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Randomly permute the order of the batches (may not be important)
        batch_order = torch.randperm(n_batches, generator=self.generator).tolist()

        # Get IDs in batch order
        sorted_locs = [self.batch_locs[i] for i in batch_order]
        # Randomly sample batch_size from n_matches in the batch. This ensures that we don't always feed the same batch at each epoch.
        selected_locs = [[batch[0]] + random.sample(batch[1:], len(batch) - 1)[: self.batch_size - 1] for batch in sorted_locs]

        yield from selected_locs
