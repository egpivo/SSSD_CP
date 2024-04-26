#!/bin/bash
#
# Install essentials on Ubuntu 22.04
#
# Essentials: Docker, NVIDIA driver 535 & CUDA 12.2, AWS CLI
#

install_docker_and_compose() {
    # Update package lists and install Docker
    sudo apt-get update > /dev/null 2>&1 || true

    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io > /dev/null 2>&1 || true

    # Install Docker Compose
    mkdir -p ~/.docker/cli-plugins/
    curl -SL https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
    chmod +x ~/.docker/cli-plugins/docker-compose
    # Add current user to the docker group
    sudo usermod -aG docker "$USER" || { echo "Error: Failed to add user to docker group."; return 1; }
    sudo chown "$USER":docker /var/run/docker.sock || { echo "Error: Failed to change ownership of docker.sock."; return 1; }
}

install_nvidia_drivers() {
    # References:
    # - https://man.twcc.ai/@twccdocs/doc-vcs-main-zh/https%3A%2F%2Fman.twcc.ai%2F%40twccdocs%2Fhowto-vcs-install-nvidia-gpu-driver-zh
    # - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    #

    # Install NVIDIA driver
    sudo apt update
    sudo DEBIAN_FRONTEND=noninteractive apt install -y nvidia-driver-535

    # Install CUDA
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i --force-all cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo DEBIAN_FRONTEND=noninteractive apt install -y cuda

    # Install NVIDIA Container Toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/ubuntu$(lsb_release -rs)/$(arch)/ /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt update
    sudo DEBIAN_FRONTEND=noninteractive apt install -y nvidia-container-toolkit

    # Configure NVIDIA Container Toolkit runtime for Docker
    sudo nvidia-ctk runtime configure --runtime=docker

    # Download and install NVIDIA driver
    wget https://us.download.nvidia.com/tesla/460.106.00/NVIDIA-Linux-x86_64-460.106.00.run
    sudo apt install -y linux-headers-$(uname -r)

    # Blacklist nouveau and set options
    echo "blacklist nouveau" | sudo tee -a /etc/modprobe.d/blacklist.conf
    echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist.conf
    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
}

main() {
    echo -e "Install Docker"
    install_docker_and_compose || { echo "Error: Failed to install Docker and Docker Compose."; exit 1; }

    echo -e "Install NVIDIA drivers"
    install_nvidia_drivers || { echo "Error: Failed to install NVIDIA drivers and CUDA."; exit 1; }
    sudo DEBIAN_FRONTEND=noninteractive apt install -y awscli
}

main
