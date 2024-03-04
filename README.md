# SSSD


## Dataset
NYISO cleaned data: [dounload](https://drive.google.com/drive/folders/1dwPkBIHSikhQ5ru3HPQiILSnaGAtP3Yr?usp=sharing)


## Implement
1. Train the model: `python3 train.py -c config/config_SSSDS4-NYISO-3-mix.json`
2. Generate one prediction for each sample in test data: `python3 inference.py -c config/config_SSSDS4-NYISO-3-mix.json --num_samples=803`
3. Generate 10 predictions for each sample in test data: `python3 inference_multiples.py -c config/config_SSSDS4-NYISO-3-mix.json`


## Seggestion
1. use `CUDA_VISIBLE_DEVICES` to specify the number of GPU, both train and inference need same number of GPU
2. use the sample size as the parameter `num_samples` in the inference section


