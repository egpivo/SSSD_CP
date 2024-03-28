# Use the specified base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL authors="Joseph Wang <egpivo@gmail.com>"

# Set the working directory in the container
WORKDIR /sssd

COPY . ./

# Build Conda
RUN bash envs/conda/build_conda_env.sh

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash", "scripts/diffusion_process.sh", "--config", "config/config_SSSDS4-NYISO-3-mix.json"]
