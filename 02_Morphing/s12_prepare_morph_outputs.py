import argparse
import pathlib
import pandas
import yaml
import typing
import pickle

MODEL_PRED_COEF_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami/model.best.pth.tar")
MODEL_PRED_METADATA_FILEPATH = pathlib.Path("modelfits_beat/Beatmodel_2024_03_11_filter_tropt_ami.json")
MORPH_OUTPUT_DIRNAME = pathlib.Path("morphing_outputs")

def parse_args(config_path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """Get configurations from config file"""
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_channels", type=int, default=12)  # number of channels to use
    args = parser.parse_args()
    
    # load morph results
    ecg_dir = pathlib.Path(MORPH_OUTPUT_DIRNAME, "morphed_ecgs")
    morph_filepaths = list(ecg_dir.glob("ecg*.pickle"))

    # typehint just because of a bug in current pandas version
    dfs: typing.List[pandas.DataFrame] = []
    for morph_filepath in morph_filepaths:
        # extract metadata
        morph_n = int(morph_filepath.stem.split("_")[1])

        # load data
        x_data = pickle.load(open(morph_filepath, "rb"))

        df = pandas.DataFrame.from_records(
            [(x_data["x0"].tolist(), "raw"), (x_data["x_sm"][0].tolist(), "t0"), (x_data["x_sm"][-1].tolist(), "tfinal")],
            columns=["x", "type"]
        )
        df["morph_n"] = morph_n
        df["filepath"] = str(x_data["filepath"])
        df["filename"] = x_data["filepath"].stem
        df.to_feather(ecg_dir.joinpath(f"morph-ecg_{morph_n}.feather"))

        dfs.append(df)

    # also save all data in a single feather file, not in crowded morph folder
    df_all = pandas.concat(dfs)
    df_all.to_feather(MORPH_OUTPUT_DIRNAME.joinpath("morph-ecg.feather"))
