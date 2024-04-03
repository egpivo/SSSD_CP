# SSSD
<p align="left">
  <a href="https://github.com/egpivo/SSSD_CP/actions"><img src="https://github.com/egpivo/SSSD_CP/workflows/Test/badge.svg" alt="GitHub Actions"/></a>
  <a href="https://codecov.io/gh/egpivo/SSSD_CP"><img src="https://codecov.io/gh/egpivo/SSSD_CP/graph/badge.svg?token=gtKjUUupSz" alt="Codecov"/></a>
  <a href="https://hub.docker.com/repository/docker/egpivo/sssd"><img src="https://img.shields.io/docker/automated/egpivo/sssd" alt="Docker build"/></a>
  <a href="https://hub.docker.com/repository/docker/egpivo/sssd"><img src="https://img.shields.io/docker/v/egpivo/sssd/main" alt="Docker tag"/></a>
</p>

## Prerequisites
- [Local Environment] [Install Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
- [Docker] [Install Docker](https://docs.docker.com/get-docker/)


## Dataset
1. The NYISO dataset can be downloaded from [here](https://www.nyiso.com/).
2. The cleaned data created by the author can be downloaded from [this link](https://drive.google.com/drive/folders/1dwPkBIHSikhQ5ru3HPQiILSnaGAtP3Yr?usp=sharing).

   Note that the cleaned data is created following the scripts in `notebooks/dataset_script/nyiso-csv-to-pickle.ipynb` and `notebooks/dataset_script/nyiso-load-pickle-to-npy.ipynb`.

## Usage
0. Download data from `google drive` or `S3` to `datasets/`
   - `S3`:
     - Enter the AWS credentials
     - Enter`aws s3 sync s3://sssd-cp/datasets/ /{repo}/datasets`
1. Run the process locally:
    ```shell
    bash scripts/diffusion_process.sh --config {CONFIG_FILE_PATH}
    ```
   - Example: `CONFIG_FILE_PATH=configs/config_toy_example.json`

2. Run in a container:
   - Adjust `CONFIG_FILE` in `docker-compose.yaml`
   - Trigger
       ```shell
       make run-docker
       ```
     ![img.png](docs/images/img.png)

####  Useful Commands

1. Stopping Docker Container: To stop the Docker container,
    ```bash
    docker compose down
    ```
   ![img.png](docs/images/img_5.png)
2. Check a Docker container status
   ```bash
   docker compose ps
   ```
   ![img_2.png](docs/images/img_2.png)
3. Check a Docker container logs
   ```bash
   docker compose logs
   ```
   ![img_1.png](docs/images/img_1.png)

4. Clean Docker cache
   ```bash
   docker system prune -f
   ```
   ![img_4.png](docs/images/img_4.png)

## Suggestions
1. Use `CUDA_VISIBLE_DEVICES` to specify the number of GPUs. Both training and inference require the same number of GPUs.
2. Use the sample size as the parameter --num_samples in the inference section.
   - Example:
        ```bash
        python sssd/infer.py -c configs/config_SSSDS4-NYISO-3-mix.json --num_samples=128
        ``````
