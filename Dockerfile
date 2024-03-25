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

RUN apt-get update && \
    apt-get install -y make libsndfile1 && \
    apt-get install gcc -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir poetry==${POETRY_VERSION} && \
    poetry config installer.max-workers 10 && \
    poetry install --only=main --no-root && \
    poetry build

RUN pip install dist/*.tar.gz

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]
