# Stage 1: Build stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

LABEL authors="Joseph Wang <egpivo@gmail.com>"

# Set the working directory in the container
WORKDIR /sssd

# Copy the project files
COPY . ./

# Build Conda environment
RUN bash envs/conda/build_conda_env.sh

# Find Conda environment directory and set it as a build argument
RUN CONDA_ENV_DIR=$(bash envs/conda/utils.sh find_conda_env_path sssd) \
    && echo "CONDA_ENV_DIR=$CONDA_ENV_DIR" >> /root/.conda_env_dir

# Stage 2: Final production image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /sssd

# Copy the Conda environment directory from the build argument
ARG CONDA_ENV_DIR
COPY --from=builder $CONDA_ENV_DIR /root/.conda/envs/sssd

# Set up Conda environment
RUN echo "source activate sssd" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "scripts/diffusion_process.sh", "--config", "config/config_SSSDS4-NYISO-3-mix.json"]
