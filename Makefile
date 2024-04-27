SHELL := /bin/bash
EXECUTABLE := poetry run
DOCKER_USERNAME := "egpivo"

.PHONY: clean install activate-conda-env test run-local build-docker push-docker run-docker help

## Clean targets
clean: clean-pyc clean-build clean-test-coverage clean-docker

clean-pyc:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-build:
	rm -fr build/ dist/ .eggs/
	find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +

clean-test-coverage:
	rm -f .coverage
	rm -rf .pytest_cache

clean-docker:
	docker system prune -f

## Installation
install: clean
	$(SHELL) envs/conda/build_conda_env.sh -c sssd

## Activate Conda environment
activate-conda-env: install
	eval "$$(conda shell.bash hook)" && \
	conda activate sssd

## Testing
test: install
	$(EXECUTABLE) pytest --cov=sssd

## Run diffusion process on local machine
run-local:
	$(SHELL) scripts/diffusion/diffusion_process.sh \
		-m configs/model.yaml \
		-t configs/training.yaml \
		-i configs/inference.yaml \
		-u

## Docker commands
build-docker:
	docker build -t $(DOCKER_USERNAME)/sssd:latest -f Dockerfile .

push-docker:
	docker tag $(DOCKER_USERNAME)/sssd:latest $(DOCKER_USERNAME)/sssd:latest
	docker push $(DOCKER_USERNAME)/sssd:latest

run-docker:
	docker compose up -d

## Jupyter server
run-jupyter:
	$(SHELL) envs/notebook/start_jupyter_lab.sh --port 8501

## Help
help:
	@echo "Available targets:"
	@echo "clean              : Clean up temporary files"
	@echo "install            : Install sssd with dependencies"
	@echo "activate-conda-env : Activate Conda environment"
	@echo "test               : Run tests"
	@echo "run-local          : Run diffusion process on local machine"
	@echo "build-docker       : Build Docker image"
	@echo "push-docker        : Push Docker image to Docker Hub"
	@echo "run-docker         : Run diffusion process in Docker container"
