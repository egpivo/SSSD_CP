# Use the specified base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

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

# Copy only the necessary files
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-root

# Second stage: final environment with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /sssd

# Ensure the directory exists before copying
RUN mkdir -p /tmp/site-packages

# Copy installed dependencies from the first stage to a temporary directory
COPY --from=base /usr/local/lib/python3.10/site-packages/ /tmp/site-packages/

# Copy installed dependencies from the temporary directory to the final location
RUN cp -r /tmp/site-packages/* /usr/local/lib/python3.10/site-packages/

# Copy the rest of the project files
COPY . .

# Install project
RUN pip install dist/*.tar.gz

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]
