#
# Please check envs/dockers/Dockerfile.local for the original version
#
FROM egpivo/sssd:latest AS builder

LABEL authors="Joseph Wang <egpivo@gmail.com>" \
      version="0.0.14"

# Set the working directory in the container
WORKDIR /sssd

# Copy only necessary project files and Conda environment setup
COPY scripts/ scripts/
COPY envs/ envs/
COPY bin bin/
COPY notebooks notebooks/
COPY sssd sssd/
COPY pyproject.toml pyproject.toml

# Build Conda environment and cleanup unnecessary files
RUN bash envs/conda/build_conda_env.sh && \
    apt-get clean && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Stage 2: Final production image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /sssd

# Copy the Conda environment directory from the build stage to the appropriate location
COPY --from=builder /opt/conda/envs/sssd/ /opt/conda/envs/sssd/

# Set up Conda environment
ENV PATH /opt/conda/envs/sssd/bin:$PATH
RUN echo "conda activate sssd" >> ~/.bashrc

# Copy necessary files from the build stage
COPY --from=builder /sssd/bin bin/
COPY --from=builder /sssd/scripts scripts/
COPY --from=builder /sssd/envs envs/
COPY --from=builder /sssd/notebooks notebooks/
COPY --from=builder /sssd/sssd sssd/
COPY --from=builder /sssd/pyproject.toml pyproject.toml

SHELL ["conda", "run", "-n", "sssd", "poetry", "install"]

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "/bin/bash scripts/docker/$ENTRYPOINT_SCRIPT"]
