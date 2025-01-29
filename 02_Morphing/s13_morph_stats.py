import argparse
import pathlib
import warnings
import json
import pandas
import tqdm
import torch
import yaml
import typing
import numpy
import jax
import neurokit2

import s09_convert_cnn_orig as convert_cnn

MODEL_PRED_COEF_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami/model.best.pth.tar")
MODEL_PRED_METADATA_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami.json")
MORPH_OUTPUT_DIRNAME = pathlib.Path("morphing_outputs")

MORPH_OUTCOME_NAME = "scd1"

# segmentation parameters
SEGMENTATION_SAMPLING_RATE = 500


def parse_args(config_path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """Get configurations from config file"""
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def extract_indices(wave: numpy.ndarray, sampling_rate: int) -> typing.Dict[str, typing.Optional[int]]:
    assert wave.ndim == 1
    assert wave.shape[0] == 300
    wave_ext = numpy.concatenate([wave] * 10)
    try:
        wave_ext_proc = neurokit2.ecg_process(wave_ext, sampling_rate=sampling_rate, method="neurokit")
        wave_segment_keys = ["ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Peaks", "ECG_R_Offsets", "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"]
        wave_ext_indices = {
            k: min([n for n in wave_ext_proc[1][k] if n >= 300 and n < 600], default=None) for k in wave_segment_keys
        }
    except:
        wave_segment_keys = ["ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Peaks", "ECG_R_Offsets", "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"]
        wave_ext_indices = {
            k: None for k in wave_segment_keys
        }
    return wave_ext_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_channels", type=int, default=12)  # number of channels to use
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
    apply_fn = lambda p, x: model_resnet_jax.apply(p, x.reshape((1, *x.shape)).transpose(0, 2, 1), train=False)
    
    ## LOAD DATA
    # load morph results from stacked data
    ecg_dir = pathlib.Path(MORPH_OUTPUT_DIRNAME, "morphed_ecgs")
    ecg_morphed_df = pandas.read_feather(MORPH_OUTPUT_DIRNAME.joinpath("morph-ecg.feather"))
    ecg_morphed_df["ecg_type"] = ecg_morphed_df["type"].apply(lambda v: {"raw": "x_morph_raw", "t0": "x_morph_init", "tfinal": "x_morph_final"}[v])

    # load raw results also
    ecg_raw_df = pandas.read_feather(MORPH_OUTPUT_DIRNAME.joinpath("reconstruction-raw.feather"))
    ecg_raw_df["filename"] = ecg_raw_df["filepath"].apply(lambda p: pathlib.Path(p).stem)

    # create a combined dataset to extract all predictive results over
    ecg_combined_df = pandas.concat(
        [
            ecg_morphed_df[["filename", "ecg_type", "x"]], 
            ecg_raw_df.rename(columns={"ecg_values": "x"})[["filename", "ecg_type", "x"]]
        ]
    )
    ecg_combined_df["row_id"] = range(len(ecg_combined_df))

    # we can actually just iterate over rows for calculations
    results = {}
    for _, ecg_row in tqdm.tqdm(ecg_combined_df.iterrows(), total=ecg_combined_df.shape[0]):
        # get helpful filename suffix
        x = jax.numpy.stack([jax.numpy.array(r) for r in ecg_row["x"]])

        # extract predictions
        y = apply_fn(jax_state_dict_ported, x).tolist()[0]
        y = {f"yhat_{s}": y[n] for n, s in enumerate(pytorch_metadata["outputs"])}

        # extract waveform range
        lead_stats = {
            "wave_min": numpy.min(x).item(), 
            "wave_max": numpy.max(x).item(), 
            **{f"lead_{n}_min": v.item() for n, v in enumerate(numpy.min(x, axis=1))},
            **{f"lead_{n}_max": v.item() for n, v in enumerate(numpy.max(x, axis=1))}
        }

        # store results
        results[ecg_row["row_id"]] = {**y, **lead_stats}  

    # stack results into a dataframe
    ecg_pred_df = pandas.DataFrame.from_dict(results, orient="index")
    ecg_pred_df = ecg_combined_df.merge(ecg_pred_df, left_on="row_id", right_index=True)
    ecg_pred_df = ecg_pred_df.drop(columns=["x"])

    # Export
    ecg_pred_df.to_feather(MORPH_OUTPUT_DIRNAME.joinpath("morph-preds.feather"))
