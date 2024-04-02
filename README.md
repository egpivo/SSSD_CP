# SSSD
<p align="left">
  <a href="https://github.com/egpivo/SSSD_CP/actions"><img src="https://github.com/egpivo/SSSD_CP/workflows/Test/badge.svg" alt="GitHub Actions"/></a>
  <a href="https://codecov.io/gh/egpivo/SSSD_CP"><img src="https://codecov.io/gh/egpivo/SSSD_CP/graph/badge.svg?token=gtKjUUupSz" alt="Codecov"/></a>
</p>

## Prerequisites
- [Local Environment] [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
- [Docker] [Install Docker](https://docs.docker.com/get-docker/)



## Environment Installation
- Install `sssd` Conda env by
   ```bash
   make install
   ```


## Docker Usage

To utilize Docker for running the SSSD application, follow these steps:

1. **Build Docker Image**: Run the following command to build the Docker image tagged as `sssd:latest`:
   ```bash
   make build-docker
   ```

2. Run Docker Container: After building the Docker image, execute the following command to run the Docker container:
  ```bash
  make run-docker
  ```
This command will start the SSSD application inside a Docker container. The configuration can be specified by modifying the docker-compose.yaml file. Ensure that the CONFIG_FILE environment variable in the `docker-compose.yaml` file points to the desired configuration file. By default, it is set to `config_SSSDS4-NYISO-3-mix.json`.

You can also customize other environment variables or volume mappings in the docker-compose.yaml file as needed.

3. Stopping Docker Container: To stop the Docker container, you can run:
  ```bash
  docker-compose down
  ```


## Dataset
1. The NYISO dataset can be downloaded from [here](https://www.nyiso.com/).
2. The cleaned data created by the author can be downloaded from [this link](https://drive.google.com/drive/folders/1dwPkBIHSikhQ5ru3HPQiILSnaGAtP3Yr?usp=sharing).

   Note that the cleaned data is created following the scripts in `notebooks/dataset_script/nyiso-csv-to-pickle.ipynb` and `notebooks/dataset_script/nyiso-load-pickle-to-npy.ipynb`.

## Example: `config_SSSDS4-NYISO-3-mix.json`
- To execute a diffusion process:
   ```bash
   make run-diffusion-mix
   ```

## Suggestions
1. Use `CUDA_VISIBLE_DEVICES` to specify the number of GPUs. Both training and inference require the same number of GPUs.
2. Use the sample size as the parameter --num_samples in the inference section.
   - Example:
```bash
python sssd/infer.py -c configs/config_SSSDS4-NYISO-3-mix.json --num_samples=128
``````
