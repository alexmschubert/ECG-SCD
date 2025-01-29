import argparse
import pathlib
import warnings
import json

import jax
import jax.random as jr
import jax.numpy as jnp
import tqdm
import torch
import yaml
import typing
import time
import pickle

import s04_utils_data as utils_data
import s05_utils_vae as utils_vae
import s08_train_generator as generator
import s09_convert_cnn_orig as convert_cnn

# paths to all important data to be loaded
MODEL_PRED_COEF_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami/model.best.pth.tar")
MODEL_PRED_METADATA_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami.json")
MODEL_GEN_DIRNAME = pathlib.Path("generative_models/models")
MORPH_OUTPUT_DIRNAME = pathlib.Path("morphing_outputs") 

MORPH_OUTCOME_NAME = "scd1"
MORPH_RECONSTRUCTION_ERR_MAX = 0.1
MORPH_YHAT_UPPER_BOUND = 1.0


if not MORPH_OUTPUT_DIRNAME.exists():
    MORPH_OUTPUT_DIRNAME.mkdir(parents=True)

def parse_args(config_path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """Get configurations from config file"""
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def get_latent_var(x, result, key):
    mu, sigmasq = result["apply_fn_enc"](result["params_enc"], x)
    z_pred = utils_vae.gaussian_sample(key, mu, sigmasq)

    return z_pred


def reconstruct(x, result):
    decode_fn = lambda z: result["apply_fn_dec"](result["params_dec"], z).reshape(x.shape)
    x_hat = decode_fn(get_latent_var(x, result, jr.PRNGKey(0)))
    return x_hat


def subspace_morph(x, params, apply_fn, result, n_steps=100, lr=5e-2, yhat_max=None):
    _pred_fn = lambda x: apply_fn(params, x)[0]
    decode_fn = lambda z: result["apply_fn_dec"](result["params_dec"], z).reshape(x.shape)
    x_curr = x.copy()
    z_curr = get_latent_var(x_curr, result, jr.PRNGKey(0))
    xs = []
    pbar = tqdm.tqdm(range(n_steps), desc="Prediction: " f"{100 * jax.nn.sigmoid(_pred_fn(x_curr)).item():.2f}%")
    for _ in pbar:
        _pred_fn_induced = lambda z: _pred_fn(decode_fn(z))
        z_delta = jax.grad(_pred_fn_induced)(z_curr)
        z_curr = z_curr + lr * z_delta
        x_curr = decode_fn(z_curr)
        xs.append(x_curr)
        yhat_curr = jax.nn.sigmoid(_pred_fn(x_curr)).item()
        pbar.set_description(f"Prediction: {100 * yhat_curr:.2f}%")
        if yhat_max is not None and yhat_curr > yhat_max:
            break

    return xs


if __name__ == "__main__":
    results_timestr = time.strftime("%Y%m%d%H%M")

    parser = argparse.ArgumentParser()

    # Specify the generative model to load
    parser.add_argument("--gen_model", type=str, default="dr-vae", choices=["dr_vae", "baseline", "dsm", "real"])
    parser.add_argument("--hidden_width", type=int, default=100)  # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4)  # hidden layer depth
    parser.add_argument("--z_dim", type=int, default=512)  # latent dim

    parser.add_argument("--seed", type=int, default=0)  # random seed

    parser.add_argument("--n_channels", type=int, default=12)  # number of channels to use
    parser.add_argument("--lr", type=float, default=1e-2)  # learning rate 
    parser.add_argument("--n_ecgs", type=int, default=1000)  # number of epochs to morph

    args = parser.parse_args()

    # make sure we can save results
    ecg_dir = pathlib.Path(MORPH_OUTPUT_DIRNAME, "morphed_ecgs")
    ecg_dir.mkdir(exist_ok=True)

    ## DISCRIMINATIVE MODEL
    warnings.warn("Use model JSON to extract head names")
    pytorch_metadata = json.load(open(MODEL_PRED_METADATA_FILEPATH, "r"))
    pytorch_results = torch.load(MODEL_PRED_COEF_FILEPATH, map_location=torch.device("cpu"))
    pytorch_state_dict = pytorch_results["best_state"]
    jax_state_dict_ported = convert_cnn.pytorch_to_jax(pytorch_state_dict)

    # load equivalent JAX model
    model_resnet_jax = convert_cnn.EKGResNetModel(out_features=pytorch_metadata['n_outputs'], output_names=pytorch_metadata['outputs'], rep_mp=5)

    # create a jax prediction function
    morph_outcome_index = pytorch_metadata["outputs"].index(MORPH_OUTCOME_NAME)
    
    @jax.jit
    def apply_fn(p, x):
        return model_resnet_jax.apply(p, x.reshape((1, *x.shape)).transpose(0, 2, 1), train=False)[:, morph_outcome_index]

    covariate_data_filepath = pathlib.Path(f"covariate_df.feather")
    outcome_regress = [False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, True, True, True, True, True, True]
    outcome_names=['scd3mo', 'scd6mo', 'scd1', 'scd2', 'dead3mo', 'dead6mo', 'dead1', 'dead2', 'ICD_VT1', 'ICD_VF1', 'Phi_VT1', 'TROPT1', 'Acute_MI1', 'age', 'female', 'qrsDuration', 'qtInterval', 'qrsFrontAxis', 'qrsHorizAxis', 'rrInterval', 'pDuration', 'atrialrate', 'meanqtc']
    _, X_dataset = utils_data.load_dataset(
        ecg_data_dirname=pathlib.Path("ecg_beats_morphing"),
        covariate_data_filepath=covariate_data_filepath,
        split=(0.01, 0.30),
        outcome_names=outcome_names,
        outcome_regress=outcome_regress
    )

    # choose ECGs to morph
    key = jr.PRNGKey(args.seed)
    idx = jr.choice(key, len(X_dataset), shape=(args.n_ecgs,), replace=False)
    Xs = jnp.array(torch.stack([X_dataset[i][0] for i in idx]))

    # load generative model
    result = generator.load_model(Xs, MODEL_GEN_DIRNAME, configs=args)

    # Morph
    for i, x in tqdm.tqdm(enumerate(Xs), total=len(Xs)):
        # setup
        filepath_curr = X_dataset.filepaths[i]

        # filter if necessary
        x_hat = reconstruct(x, result)
        err = ((x - x_hat) ** 2).mean().item()
        if err > MORPH_RECONSTRUCTION_ERR_MAX:
            print(f"Skipping ECG {i+1}/{args.n_ecgs} (err: {err})")
            continue

        # extract baseline prediction
        pred = jax.nn.sigmoid(apply_fn(jax_state_dict_ported, x)[0]).item()
        print(f"Morphing ECG {i+1}/{args.n_ecgs} (pred: {pred})")
        
        # Subspace morph
        xs_sm = subspace_morph(x, jax_state_dict_ported, apply_fn, result, lr=args.lr, n_steps=2000, yhat_max=MORPH_YHAT_UPPER_BOUND)

        # store raw data only
        x_data = {"x0": x, "x_sm": xs_sm, "filepath": filepath_curr}
        pickle.dump(x_data, open(pathlib.Path(ecg_dir, f"ecg_{i}.pickle"), "wb"))
