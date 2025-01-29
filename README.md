# ECG-SCD

## Setup / Getting started 

create venv with python 3.9

## Data Pre-processing

The repository expects that the data has been pre-processed such that there exists a folder with ECG waveforms formatted as numpy arrays and sampled at 500 hz. 

Further it expects a dataframe that contains patient metadata (particularly `age` and `sex`), a column `scaling_factor` that indicates by which value the waveforms must be divided to be scaled to mV (alternatively this factor can be set to 1 in case the input waveforms have already been scaled to mV) as well as training labels for the intended prediction targets. 

## Predictive model training


## Morphing