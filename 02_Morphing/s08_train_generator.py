import argparse
import pathlib
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import json
import torch
import numpy as np
import orbax.checkpoint as orbax_ckpt
from flax.training import orbax_utils
from jax.flatten_util import ravel_pytree

import s01_utils as utils
import s03_models as models
import s04_utils_data as utils_data
import s05_utils_vae as utils_vae
import s06_utils_dsm as utils_dsm
import s07_convert_cnn_orig as convert_cnn

# temporary variable while we load data
MODEL_DIRNAME = pathlib.Path("generative_models")
MODEL_PRED_COEF_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami/model.best.pth.tar")
MODEL_PRED_METADATA_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami.json")

CHANNEL_NAMES = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
MORPH_OUTCOME_NAME = "scd1"

def train_and_save_model(X, gen_ckpt_dir, configs, vae_pred_fn=None):
    X = jnp.array(X)

    if configs.gen_model == "dr-vae":
        result = utils_vae.train_dr_vae(
            vae_pred_fn,
            X,
            configs.alpha,
            configs.beta,
            configs.z_dim,
            configs.seed,
            configs.n_epochs,
            configs.batch_size,
            configs.hidden_width,
            configs.hidden_depth,
            configs.lr_init,
            configs.lr_peak,
            configs.lr_end,
            encoder_type="cnn",
            use_bias=False,
        )

        gen_ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path(gen_ckpt_dir, "params_enc.npy"), "wb") as f:
            jnp.save(f, result["params_enc"])
        with open(pathlib.Path(gen_ckpt_dir, "params_dec.npy"), "wb") as f:
            jnp.save(f, result["params_dec"])
        with open(pathlib.Path(gen_ckpt_dir, "mu_mean.npy"), "wb") as f:
            jnp.save(f, result["mu_mean"])
        with open(pathlib.Path(gen_ckpt_dir, "mu_std.npy"), "wb") as f:
            jnp.save(f, result["mu_std"])
        with open(pathlib.Path(gen_ckpt_dir, "sigmasq_mean.npy"), "wb") as f:
            jnp.save(f, result["sigmasq_mean"])
        with open(pathlib.Path(gen_ckpt_dir, "sigmasq_std.npy"), "wb") as f:
            jnp.save(f, result["sigmasq_std"])
        with open(pathlib.Path(gen_ckpt_dir, "losses.npy"), "wb") as f:
            jnp.save(f, result["losses"])
        with open(pathlib.Path(gen_ckpt_dir, "losses_rec.npy"), "wb") as f:
            jnp.save(f, result["losses_rec"])
        with open(pathlib.Path(gen_ckpt_dir, "losses_kl.npy"), "wb") as f:
            jnp.save(f, result["losses_kl"])
        with open(pathlib.Path(gen_ckpt_dir, "losses_dr.npy"), "wb") as f:
            jnp.save(f, result["losses_dr"])

    elif configs.gen_model == "dsm":
        X_tr = jnp.swapaxes(X, 1, 2)
        result, loss_history = utils_dsm.train_dsm(
            X_tr,
            key=configs.seed,
            n_epochs=configs.n_epochs,
        )
        ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
        save_args = orbax_utils.save_args_from_target(result)
        ckptr.save(gen_ckpt_dir, result, force=True, save_args=save_args)

    else:
        raise ValueError(f"Model {configs.gen_model} not supported.")

    return result


def generate_and_save_ecgs(X, result, gen_result_path, configs):
    gen_result_path = pathlib.Path(gen_result_path, "generated_ecgs")
    gen_result_path.mkdir(parents=True, exist_ok=True)
    key = jr.PRNGKey(configs.seed)
    key, subkey = jr.split(key)
    if configs.gen_model == "dr-vae":
        fn_dec, params_dec = result["apply_fn_dec"], result["params_dec"]
        mu_mean, mu_std = result["mu_mean"], result["mu_std"]
        sigmasq_mean, sigmasq_std = result["sigmasq_mean"], result["sigmasq_std"]
        ecgs = []
        for i in range(configs.n_ecgs):
            key1, key2, key3, key = jr.split(key, 4)
            mu_curr = mu_mean + mu_std * jr.normal(key1, shape=(configs.z_dim,))
            sigmasq_curr = sigmasq_mean + sigmasq_std * jr.normal(key2, shape=(configs.z_dim,))
            z = mu_curr + jnp.sqrt(sigmasq_curr) * jr.normal(key3, shape=(configs.z_dim,))
            x = fn_dec(params_dec, z)
            x = fn_dec(params_dec, z).reshape(configs.n_channels, -1)
            ecgs.append(x)
            fig, _ = utils.plot_ecg(x, CHANNEL_NAMES, configs.n_channels, (2 * configs.n_channels, 6))
            fig.savefig(pathlib.Path(gen_result_path, f"ecg_{i+1}.png"))
    elif configs.gen_model == "dsm":
        _, *x_dim = X.shape
        x_dim = jnp.insert(jnp.swapaxes(jnp.array(x_dim), 0, 1), 0, 1)
        ecgs = []
        for i in range(configs.n_ecgs):
            key1, key2, subkey = jr.split(subkey, 3)
            x_init = jr.uniform(key1, x_dim)
            xs = utils_dsm.sample_annealed_langevin(result.apply_fn, x_init, result.params, key2)
            x = jnp.swapaxes(xs[-1].squeeze(), 0, 1)
            ecgs.append(x)
            fig, _ = utils.plot_ecg(x, CHANNEL_NAMES, configs.n_channels, (2 * configs.n_channels, 6))
            fig.savefig(pathlib.Path(gen_result_path, f"ecg_{i+1}.png"))
    else:
        raise ValueError(f"Model {configs.gen_model} not supported.")
    ecgs = jnp.array(ecgs)
    # Visualize distribution
    ecg_mean, ecg_std = jnp.nanmean(ecgs, axis=0), jnp.nanstd(ecgs, axis=0)
    fig, _ = utils.plot_ecg(ecg_mean, CHANNEL_NAMES, 3, (4, 4), ecg_std)
    fig.savefig(pathlib.Path(gen_result_path, "ecg_dist.png"))


def load_model(X, gen_ckpt_dir, configs):
    key = jr.PRNGKey(configs.seed)

    if configs.gen_model == "dr-vae":
        result = {}
        with open(pathlib.Path(gen_ckpt_dir, "params_enc.npy"), "rb") as f:
            result["params_enc"] = jnp.load(f)
        with open(pathlib.Path(gen_ckpt_dir, "params_dec.npy"), "rb") as f:
            result["params_dec"] = jnp.load(f)
        with open(pathlib.Path(gen_ckpt_dir, "mu_mean.npy"), "rb") as f:
            result["mu_mean"] = jnp.load(f)
        with open(pathlib.Path(gen_ckpt_dir, "mu_std.npy"), "rb") as f:
            result["mu_std"] = jnp.load(f)
        with open(pathlib.Path(gen_ckpt_dir, "sigmasq_mean.npy"), "rb") as f:
            result["sigmasq_mean"] = jnp.load(f)
        with open(pathlib.Path(gen_ckpt_dir, "sigmasq_std.npy"), "rb") as f:
            result["sigmasq_std"] = jnp.load(f)
        _, *x_dim = X.shape
        x_dim = jnp.array(x_dim)

        hidden_feats = [configs.hidden_width] * configs.hidden_depth
        decoder_feats = [*hidden_feats, jnp.prod(x_dim)]

        key_enc, key_dec = jr.split(key)

        # Encoder
        encoder = utils_vae.CNNEncoder(configs.z_dim)
        params_enc = encoder.init(key_enc, jnp.ones(x_dim,),)["params"]
        params_enc, unflatten_fn_enc = ravel_pytree(params_enc)
        apply_fn_enc = lambda params, x: encoder.apply({"params": unflatten_fn_enc(params)}, x)

        # Decoder
        decoder = utils_vae.Decoder(decoder_feats, use_bias=False)
        params_dec = decoder.init(key_dec, jnp.ones(configs.z_dim,),)["params"]
        params_dec, unflatten_fn_dec = ravel_pytree(params_dec)
        apply_fn_dec = lambda params, x: decoder.apply({"params": unflatten_fn_dec(params)}, x)
        result["apply_fn_enc"] = apply_fn_enc
        result["apply_fn_dec"] = apply_fn_dec

    elif configs.gen_model == "dsm":
        model = models.NCSN(num_features=16)
        params = model.init(key, X[0:1], jnp.array([0]))
        flat_params, unflatten_fn = ravel_pytree(params)
        print(f"Number of parameters: {len(flat_params):,}")
        apply_fn = lambda flat_params, x, y: model.apply(unflatten_fn(flat_params), x, y)
        state = utils_dsm.create_train_state(flat_params, apply_fn, configs.lr_init)
        ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
        result = ckptr.restore(gen_ckpt_dir, item=state)
        
    else:
        raise ValueError(f"Model {configs.gen_model} not supported.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and generate ECGs")
    parser.add_argument("--load_model", action="store_true")

    # Specify the generative model to train
    parser.add_argument("--gen_model", type=str, default="dr-vae", choices=["dr_vae", "baseline", "dsm", "real"])

    # Specify model parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512)  # latent dim
    parser.add_argument("--alpha", type=float, default=0.01)  # disc. reg. weight
    parser.add_argument("--beta", type=float, default=0.01)  # disc. reg. weight
    parser.add_argument("--hidden_width", type=int, default=100)  # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4)  # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7)  # initial learning rate
    parser.add_argument("--lr_peak", type=float, default=1e-4)  # peak learning rate
    parser.add_argument("--lr_end", type=float, default=1e-7)  # end learning rate
    parser.add_argument("--target", type=str, default="sex")  # target for discriminator

    # Specify dataset
    parser.add_argument("--n_channels", type=int, default=12)  # number of channels to use

    # Specify training parameters
    parser.add_argument("--seed", type=int, default=0)  # random seed
    parser.add_argument("--n_epochs", type=int, default=100)  # number of epochs to train
    parser.add_argument("--batch_size", type=int, default=512)  # batch size

    # Specify number of generated ECGs
    parser.add_argument("--n_ecgs", type=int, default=100)  # number of ECGs to generate
    args = parser.parse_args()

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
    discriminative_pred_fn = lambda x: jax.nn.sigmoid(model_resnet_jax.apply(jax_state_dict_ported, x.reshape(1, *x.shape).transpose(0, 2, 1), train=False)[:, morph_outcome_index])


    ## MAIN FUNCTION
    covariate_data_filepath = pathlib.Path("covariate_df.feather")
    outcome_regress = [False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, True, True, True, True, True, True]
    
    # UPDATE: here we replace CLI args with ones from other local script for running on halland data
    X_tr, _ = utils_data.load_data(
        ecg_data_dirname=pathlib.Path("ecg_beats"),
        covariate_data_filepath=covariate_data_filepath,
        split=(0.30, 0.1),
        outcome_names=['scd3mo', 'scd6mo', 'scd1', 'scd2', 'dead3mo', 'dead6mo', 'dead1', 'dead2', 'ICD_VT1', 'ICD_VF1', 'Phi_VT1', 'TROPT1', 'Acute_MI1', 'age', 'female', 'qrsDuration', 'qtInterval', 'qrsFrontAxis', 'qrsHorizAxis', 'rrInterval', 'pDuration', 'atrialrate', 'meanqtc'],
        outcome_regress=outcome_regress
    )

    MODEL_OUTPUT_DIRNAME = pathlib.Path(MODEL_DIRNAME, "models")

    if args.load_model:
        result = load_model(X_tr, MODEL_OUTPUT_DIRNAME, configs=args)
    elif args.gen_model != "real":
        result = train_and_save_model(X_tr, MODEL_OUTPUT_DIRNAME, configs=args, vae_pred_fn=discriminative_pred_fn)

    if args.gen_model == "real":
        key = jr.PRNGKey(args.seed)
        idx = jr.choice(key, len(X_tr), shape=(args.n_ecgs,), replace=False)
        ecgs = X_tr[idx]
        for i, x in enumerate(ecgs):
            fig, _ = utils.plot_ecg(x, CHANNEL_NAMES, args.n_channels, (2 * args.n_channels, 6))
            drvae_result_path = pathlib.Path(MODEL_DIRNAME, "real_ecgs")
            drvae_result_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(pathlib.Path(drvae_result_path, f"ecg_{i+1}.png"))

        ecg_mean, ecg_std = jnp.nanmean(ecgs, axis=0), jnp.nanstd(ecgs, axis=0)

        fig, _ = utils.plot_ecg(ecg_mean, CHANNEL_NAMES, 3, (4, 4), ecg_std)
        fig.savefig(pathlib.Path(drvae_result_path, "ecg_dist.png"))
    else:
        ecgs = generate_and_save_ecgs(X_tr, result, MODEL_OUTPUT_DIRNAME, args)
