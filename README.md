# An ECG biomarker for sudden cardiac death discovered via deep learning

This repository contains the code accompanying the paper *An ECG biomarker for sudden cardiac death discovered via deep learning.*

Since the main dataset used to train the model in the paper cannot be publicly shared, we demonstrate the code by using the publicly available dataset based on the hospital-based cardiopulmonary arrest registry from National Taiwan University Hospital (Taipei), which has been used for external validation of the model proposed in the paper and is available via [Nightingale Open Science](https://docs.ngsci.org/datasets/arrest-ntuh-ecg/).

## Setup 

We recommend using Python 3.8 in a conda environment:

```bash
conda create -n ecg_scd_env python=3.8
conda activate ecg_scd_env
pip install -r requirements.txt
pip install .
```

You may want to follow the instructions to [create a virtual environment](https://ngsci.helpscoutdocs.com/article/12-create-a-new-virtual-environment) when working on the Nightingale platform. 

## Repository content

All scripts for data preprocessing are located in the `00_Data_Preprocessing/` directory. Run the scripts in sequential order to reproduce the preprocessing steps. Here the script `x02_generate_dummy_columns.py` creates dummy columns to accommodate variables that were available in our original dataset but are not part of the Nightingale data. This ensures downstream scripts can run smoothly.

The predictive modeling scripts are in `01_Predictive_Model/`. The script `prediction_commands.md` contains commands to generate predictions using the trained machine learning models.

We use morphing to investigate the risk drivers in our AI model. This procedure can help interpret the deep learning model by generating synthetic ECG waveforms with high predicted risk. The relevant scripts are in `02_Morphing/`. The script `s08_train_generator.py` trains the Variational Autoencoder (VAE) used to generate synthetic ECG waveforms. Our morphing procedure is implemented in `s10_morph_ecgs.py`. `s09_extract_horizontal_stats.py`, `s09_extract_horizontal_stats.py` and `s12_morph_stats.py` process the morphing outputs for analysis. Files `s01_utils.py`, `s02_model_layers.py`, `s03_models.py`, `s04_utils_data.py`, `s05_utils_vae.py`, and `s06_utils_dsm.py` contain helper functions and model architectures for the VAE training and morphing steps.
