# Use the specified base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL authors="Joseph Wang <egpivo@gmail.com>"

ENV POETRY_HOME=/root/.poetry \
    POETRY_VERSION=1.8.0 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory in the container
WORKDIR /sssd

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y make libsndfile1 && \
    apt-get install gcc -y && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry==${POETRY_VERSION} && \
    poetry config installer.max-workers 10

COPY . ./

# Build Conda
RUN bash envs/conda/build_conda_env.sh

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash", "scripts/diffusion_process.sh", "--config", "config/config_SSSDS4-NYISO-3-mix.json"]
