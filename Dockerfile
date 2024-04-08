#
# Please check envs/dockers/Dockerfile.local for the original version
#
FROM egpivo/sssd:latest AS builder

LABEL authors="Joseph Wang <egpivo@gmail.com>" \
      version="0.0.6"

# Set the working directory in the container
WORKDIR /sssd

# Copy only necessary project files and Conda environment setup
COPY scripts/ scripts/
COPY envs/conda envs/conda/
COPY bin bin/
COPY sssd sssd/
COPY pyproject.toml pyproject.toml
COPY README.md README.md

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

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "/bin/bash scripts/diffusion_process.sh --config configs/$CONFIG_FILE --trials ${TRIALS}"]
