## Commands to obtain model predictions 

For the 12-lead ECG models:

```bash
python 01_Predictive_Model/x03_predict.py --model_name 07_08_scd1_model_dropDefib_agesex_pretrain --covariate_df_path covariate_df.feather --ecg_dir 10_sec_ecgs

python 01_Predictive_Model/x03_predict.py --model_name 10_01_death_model_fliterTropt_dropDefib_agesex_pretrain --covariate_df_path covariate_df.feather --ecg_dir 10_sec_ecgs

python 01_Predictive_Model/x03_predict.py --model_name 10_01_low_ef_model_fliterTropt_dropDefib_agesex_pretrain_no_scale_v4 --covariate_df_path covariate_df.feather --ecg_dir 10_sec_ecgs
```

For the ECG beat model:

```bash
python 01_Predictive_Model/x03_predict.py --model_name Beatmodel_2024_03_11_filter_tropt_ami --covariate_df_path covariate_df.feather --ecg_dir ecg_beats --beat
```