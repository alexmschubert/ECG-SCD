import argparse
import pathlib
import pandas
import tqdm
import torch
import yaml
import typing
import numpy
import jax
import jax.random as jr
import jax.numpy as jnp

import s04_utils_data as utils_data
import s05_utils_vae as utils_vae
import s08_train_generator as generator


# paths to all important data to be loaded
MODEL_GEN_DIRNAME = pathlib.Path("generative_models/models")
MODEL_PRED_COEF_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami/model.best.pth.tar")
MODEL_PRED_METADATA_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami.json")
MORPH_OUTPUT_DIRNAME = pathlib.Path("morphing_outputs")

# make sure output directory exists
MORPH_OUTPUT_DIRNAME.mkdir(parents=True, exist_ok=True)


def parse_args(config_path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """Get configurations from config file"""
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def reconstruct_ecg(x, result, key):
    mu, sigmasq = result["apply_fn_enc"](result["params_enc"], x)
    z = utils_vae.gaussian_sample(key, mu, 0 * sigmasq)
    decode_fn = lambda z: result["apply_fn_dec"](result["params_dec"], z).reshape(x.shape)
    x_curr = decode_fn(z)
    return x_curr


def smooth_lead(lead, window_size):
    # Ensure window size is odd for symmetric smoothing
    if window_size % 2 == 0:
        raise ValueError(f"Input window_size must be odd, got '{window_size}' instead")

    # Pad the array to handle the borders
    pad_width = (window_size - 1) // 2
    padded_lead = jnp.pad(lead, (pad_width, pad_width), mode='edge')

    # Apply the moving average filter
    smoothed_lead = jnp.convolve(padded_lead, jnp.ones(window_size) / window_size, mode='valid')
    return smoothed_lead


def smooth_ecg(x, window_size: int = 5):
    return jax.vmap(lambda l: smooth_lead(l, window_size=window_size))(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_channels", type=int, default=12)  # number of channels to use
    parser.add_argument("--seed", type=int, default=0)  # random seed
    parser.add_argument("--gen_model", type=str, default="dr-vae", choices=["dr_vae", "baseline", "dsm", "real"])
    parser.add_argument("--hidden_width", type=int, default=100)  # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4)  # hidden layer depth
    parser.add_argument("--z_dim", type=int, default=512)  # latent dim

    args = parser.parse_args()

    ## Load dataset
    covariate_data_filepath = pathlib.Path(f"covariate_df.feather")
    outcome_regress = [False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, True, True, True, True, True, True]
    X_dataset, _ = utils_data.load_dataset(
        ecg_data_dirname=pathlib.Path("ecg_beats"),
        covariate_data_filepath=covariate_data_filepath,
        split=(0.30, 0.30), 
        outcome_names=['scd3mo', 'scd6mo', 'scd1', 'scd2', 'dead3mo', 'dead6mo', 'dead1', 'dead2', 'ICD_VT1', 'ICD_VF1', 'Phi_VT1', 'TROPT1', 'Acute_MI1', 'age', 'female', 'qrsDuration', 'qtInterval', 'qrsFrontAxis', 'qrsHorizAxis', 'rrInterval', 'pDuration', 'atrialrate', 'meanqtc'],
        outcome_regress=outcome_regress
    )

    # choose ECGs to morph
    key = jr.PRNGKey(0)
    idx = jr.choice(key, len(X_dataset), shape=(10,), replace=False)
    Xs = jnp.array(torch.stack([X_dataset[i][0] for i in idx]))

    # load generative model
    model_vae = generator.load_model(Xs, MODEL_GEN_DIRNAME, configs=args)

    # first, create a dataset of original ECG values, smoothed ECG values, and reconstructed ECG values
    results_single = {}
    for i, (xi, yi) in tqdm.tqdm(enumerate(X_dataset), total=len(X_dataset)):
        
        # ID variables
        filepath_curr_i = X_dataset.filepaths[i]

        # fix inputs
        xi = jnp.array(xi)
        xi_hat = reconstruct_ecg(jnp.array(xi), model_vae, jr.PRNGKey(0))

        # These are used to calculate within-beat, within-wave, and across-wave error terms
        wave_i, beat_i = filepath_curr_i.stem.split("_")
        beat_i = beat_i.replace(".npz", "") 

        # store all results
        results_single[i] = {"filepath": str(filepath_curr_i), "wave": wave_i, "beat": beat_i, "x_raw": xi, "x_hat": xi_hat}

        if i > 500:
            break

    # extract dataframe of results and convert into long format ready for cross-errors
    results_single_df = pandas.DataFrame.from_dict(results_single, orient="index")
    results_single_df = results_single_df.melt(id_vars=["filepath", "wave", "beat"], value_name="ecg_values", var_name="ecg_type")

    # take cross join and measure error types
    results_pairwise_df = results_single_df.join(results_single_df, how="cross", lsuffix="_i", rsuffix="_j")
    results_pairwise_df["err_mae"] = results_pairwise_df.apply(lambda r: abs(r["ecg_values_i"] - r["ecg_values_j"]).mean().item(), axis=1)
    results_pairwise_df["err_mse"] = results_pairwise_df.apply(lambda r: ((r["ecg_values_i"] - r["ecg_values_j"])**2).mean().item(), axis=1)
    results_pairwise_df["err_rmse"] = numpy.sqrt(results_pairwise_df["err_mse"])

    ## EXPORT DATA
    results_single_df["ecg_values"] = results_single_df["ecg_values"].apply(lambda x: x.tolist())
    results_single_df.to_feather(MORPH_OUTPUT_DIRNAME.joinpath("reconstruction-raw.feather"))

    # save a clean version of dataset
    results_pairwise_df = results_pairwise_df.drop(columns=["ecg_values_i", "ecg_values_j"])
    results_pairwise_df = results_pairwise_df.reset_index(drop=True)
    results_pairwise_df.to_feather(MORPH_OUTPUT_DIRNAME.joinpath("reconstruction-errors.feather"))
