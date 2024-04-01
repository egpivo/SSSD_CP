# Stage 1: Build stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

LABEL authors="Joseph Wang <egpivo@gmail.com>"

# Set the working directory in the container
WORKDIR /sssd

# Copy only necessary project files and Conda environment setup
COPY envs/conda envs/conda/
COPY bin bin/
COPY sssd sssd/
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Build Conda environment and cleanup unnecessary files
RUN bash envs/conda/build_conda_env.sh && \
    CONDA_ENV_DIR=$(bash envs/conda/utils.sh find_conda_env_path sssd) \
    && echo "CONDA_ENV_DIR=$CONDA_ENV_DIR" >> /root/.conda_env_dir && \
    rm -rf envs/conda/ sssd/pyproject.toml && \
    apt-get clean && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Stage 2: Final production image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /sssd

# Copy the Conda environment directory from the build argument
ARG CONDA_ENV_DIR
COPY --from=builder $CONDA_ENV_DIR /root/.conda/envs/sssd
COPY --from=builder /sssd/scripts/diffusion_process.sh scripts/
COPY --from=builder /sssd/config/config_SSSDS4-NYISO-3-mix.json config/

# Set up Conda environment
RUN echo "source activate sssd" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "scripts/diffusion_process.sh", "--config", "config/config_SSSDS4-NYISO-3-mix.json"]
