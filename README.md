# SSSD

## Prerequisite
- [Miniconda installation](https://docs.anaconda.com/free/miniconda/miniconda-install/)

## Environment Installation
- Install `sssd` Conda env by
   ```bash
   make install
   ```

## Dataset
1. The NYISO dataset can be downloaded from https://www.nyiso.com/.
2. The cleaned data created by the author can be downloaded from [link](https://drive.google.com/drive/folders/1dwPkBIHSikhQ5ru3HPQiILSnaGAtP3Yr?usp=sharing).


- Note that the cleaned data is created following the scripts in `notebooks/dataset_script/nyiso-csv-to-pickle.ipynb` and `notebooks/dataset_script/nyiso-load-pickle-to-npy.ipynb`.

## Example: `config_SSSDS4-NYISO-3-mix.json`
1. Train a model: `make train-mix`
2. Generate one prediction for each sample in test data: `make inference-mix`
3. Generate 10 predictions for each sample in test data: `python sssd/inference/inference_multiples.py -c config/config_SSSDS4-NYISO-3-mix.json`


## Suggestion
1. Use `CUDA_VISIBLE_DEVICES` to specify the number of GPUs. Both training and inference require the same number of GPUs.
2. Use the sample size as the parameter `--num_samples` in the inference section.
  - e.g., `python sssd/infer.py -c config/config_SSSDS4-NYISO-3-mix.json --num_samples=803`
