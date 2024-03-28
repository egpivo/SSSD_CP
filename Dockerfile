# Stage 1: Build stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

LABEL authors="Joseph Wang <egpivo@gmail.com>"

# Set the working directory in the container
WORKDIR /sssd

# Copy the project files
COPY . ./

# Build Conda environment
RUN bash envs/conda/build_conda_env.sh

# Pack Conda environment to sssd.zip (assuming it's generated in /sssd directory)
RUN bash envs/conda/pack_conda_env.sh


# Stage 2: Final production image
FROM continuumio/miniconda3:latest

# Install unzip utility
RUN apt-get update && apt-get install -y unzip

# Set the working directory in the container
WORKDIR /sssd

# Copy the zipped Conda environment from the builder stage
COPY --from=builder /sssd/sssd.zip .

# Unzip the Conda environment and set up
RUN mkdir -p /root/.conda/envs/sssd \
    && unzip sssd.zip -d /root/.conda/envs/sssd \
    && rm sssd.zip \
    && echo "source activate sssd" >> ~/.bashrc

# Copy the project files
COPY . ./

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "scripts/diffusion_process.sh", "--config", "config/config_SSSDS4-NYISO-3-mix.json"]
