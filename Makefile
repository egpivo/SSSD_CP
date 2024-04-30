SHELL := /bin/bash
EXECUTABLE := poetry run
DOCKER_USERNAME := "egpivo"
MODEL_CONFIG ?= configs/model.yaml
TRAINING_CONFIG ?= configs/training.yaml
INFERENCE_CONFIG ?= configs/inference.config

.PHONY: clean install activate-conda-env test run-local-diffusion build-docker push-docker run-docker-diffusion run-local-jupyter run-docker-jupyter help

## Clean targets
clean: clean-pyc clean-build clean-test-coverage clean-docker
clean-pyc:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
clean-build:
	rm -fr build/ dist/ .eggs/ find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +
clean-test-coverage:
	rm -f .coverage rm -rf .pytest_cache
clean-docker:
	docker system prune -f

## Installation
install:
	clean
	$(SHELL) envs/conda/build_conda_env.sh -c sssd

## Activate Conda environment
activate-conda-env:
	install
	eval "$$(conda shell.bash hook)" && conda activate sssd

## Testing
test:
	$(EXECUTABLE) pytest --cov=sssd

## Run diffusion process on local machine
run-local-diffusion:
	$(SHELL) scripts/diffusion/diffusion_process.sh \
		--model_config $(MODEL_CONFIG) \
		--training_config $(TRAINING_CONFIG) \
		--inference_config $(INFERENCE_CONFIG)

## Docker commands
build-docker:
	docker build -t $(DOCKER_USERNAME)/sssd:latest -f Dockerfile .

push-docker:
	docker tag $(DOCKER_USERNAME)/sssd:latest $(DOCKER_USERNAME)/sssd:latest
	docker push $(DOCKER_USERNAME)/sssd:latest

run-docker-diffusion:
	docker compose -f services/diffusion-docker-compose.yml up -d

## Jupyter server
run-local-jupyter:
	$(SHELL) envs/jupyter/start_jupyter_lab.sh --port 8501

run-docker-jupyter:
	docker compose -f services/jupyter-docker-compose.yml up -d

## Help
help:
	@echo "Available targets:"
	@echo "clean : Clean up temporary files"
	@echo "install : Install sssd with dependencies"
	@echo "activate-conda-env : Activate Conda environment"
	@echo "test : Run tests"
	@echo "run-local-diffusion : Run diffusion process on local machine"
	@echo "build-docker : Build Docker image"
	@echo "push-docker : Push Docker image to Docker Hub"
	@echo "run-docker-diffusion : Run Docker Compose for diffusion process"
	@echo "run-local-jupyter : Start Jupyter server locally"
	@echo "run-docker-jupyter : Start Jupyter server using Docker Compose"
