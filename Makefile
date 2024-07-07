SHELL := /bin/bash
EXECUTABLE := poetry run
DOCKER_USERNAME := "egpivo"
MODEL_CONFIG ?= configs/model.yaml
TRAINING_CONFIG ?= configs/training.yaml
INFERENCE_CONFIG ?= configs/inference.yaml

.PHONY: clean clean-docker install-dev test run-local-diffusion build-docker push-docker run-docker-diffusion run-docker-diffusion-log run-local-jupyter run-docker-jupyter run-docker-jupyter-log help

## Clean up temporary files and Docker system
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name __pycache__ -delete
	@rm -rf build/ dist/ .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@rm -f .coverage
	@rm -rf .pytest_cache

clean-docker:
	@docker system prune -f

## Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	@$(SHELL) envs/conda/build_conda_env.sh -c sssd

## Run tests
test:
	@echo "Running tests..."
	@$(EXECUTABLE) pytest --cov=sssd

## Run diffusion process on local machine
run-local-diffusion:
	@echo "Running local diffusion process..."
	@$(SHELL) scripts/diffusion/diffusion_process.sh --model_config $(MODEL_CONFIG) --training_config $(TRAINING_CONFIG) --inference_config $(INFERENCE_CONFIG)

## Docker-related commands
build-docker:
	@echo "Building Docker image..."
	@docker build -t $(DOCKER_USERNAME)/sssd:latest -f Dockerfile .

push-docker:
	@echo "Pushing Docker image to Docker Hub..."
	@docker push $(DOCKER_USERNAME)/sssd:latest

run-docker-diffusion:
	@echo "Starting Docker Compose for diffusion process..."
	@docker compose -f services/diffusion-docker-compose.yaml up -d

run-docker-diffusion-log:
	@echo "Fetching logs for Docker diffusion process..."
	@docker compose -f services/diffusion-docker-compose.yaml logs -f

## Jupyter server commands
run-local-jupyter:
	@echo "Starting local Jupyter server..."
	@$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

run-docker-jupyter:
	@echo "Starting Jupyter server using Docker Compose..."
	@docker compose -f services/jupyter-docker-compose.yaml up -d

run-docker-jupyter-log:
	@echo "Fetching logs for Docker Jupyter server..."
	@docker compose -f services/jupyter-docker-compose.yaml logs -f

## Display help information
help:
	@echo "Available targets:"
	@echo "  clean                : Clean up temporary files and Docker system"
	@echo "  install-dev          : Install development dependencies"
	@echo "  test                 : Run tests"
	@echo "  run-local-diffusion  : Run diffusion process on local machine"
	@echo "  build-docker         : Build Docker image"
	@echo "  push-docker          : Push Docker image to Docker Hub"
	@echo "  run-docker-diffusion : Start Docker Compose for diffusion process"
	@echo "  run-docker-diffusion-log : Fetch logs for Docker diffusion process"
	@echo "  run-local-jupyter    : Start Jupyter server locally"
	@echo "  run-docker-jupyter   : Start Jupyter server using Docker Compose"
	@echo "  run-docker-jupyter-log : Fetch logs for Docker Jupyter server"
