# Stage 1: Build stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

LABEL authors="Joseph Wang <egpivo@gmail.com>"

# Set the working directory in the container
WORKDIR /sssd

# Copy the project files
COPY . ./

# Build Conda environment
RUN bash envs/conda/build_conda_env.sh

# Pack Conda environment to sssd.tar.gz (assuming it's generated in /sssd directory)
RUN bash envs/conda/pack_conda_env.sh --target_project_dir /sssd


# Stage 2: Final production image
FROM continuumio/miniconda3:latest

# Install tar utility
RUN apt-get update && apt-get install -y tar

# Set the working directory in the container
WORKDIR /sssd

# Copy the tar.gz Conda environment from the builder stage
COPY --from=builder /sssd/sssd.tar.gz .

# Extract the Conda environment and set up
RUN mkdir -p /root/.conda/envs/sssd \
    && tar -xzf sssd.tar.gz -C /root/.conda/envs/sssd \
    && rm sssd.tar.gz \
    && echo "source activate sssd" >> ~/.bashrc

# Copy the project files
COPY . ./

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "scripts/diffusion_process.sh", "--config", "config/config_SSSDS4-NYISO-3-mix.json"]
