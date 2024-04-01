SHELL := /bin/bash
EXECUTABLE := poetry run
DOCKER_USERNAME := "egpivo"

.PHONY: clean install activate-conda-env test run-diffusion-mix clean-docker build-docker push-docker run-docker help

## Clean targets
clean: clean-pyc clean-build clean-test-coverage

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
	$(EXECUTABLE) poetry install

## Activate Conda environment
activate-conda-env: install
	eval "$$(conda shell.bash hook)" && \
	conda activate sssd

## Testing
test: install
	$(EXECUTABLE) pytest --cov=sssd

## Run diffusion mix
run-diffusion-mix: install
	$(EXECUTABLE) scripts/diffusion_process.sh --config config/config_SSSDS4-NYISO-3-mix.json

## Docker commands
build-docker:
	docker build -t $(DOCKER_USERNAME)/sssd:latest -f envs/docker/Dockerfile .

push-docker:
	docker tag $(DOCKER_USERNAME)/sssd:latest $(DOCKER_USERNAME)/sssd:latest
	docker push $(DOCKER_USERNAME)/sssd:latest

run-docker:
	docker compose up -d

## Help
help:
	@echo "Available targets:"
	@echo "clean              : Clean up temporary files"
	@echo "install            : Install sssd with dependencies"
	@echo "activate-conda-env : Activate Conda environment"
	@echo "test               : Run tests"
	@echo "run-diffusion-mix  : Run diffusion process with config/config_SSSDS4-NYISO-3-mix.json locally"
	@echo "build-docker       : Build Docker image"
	@echo "push-docker        : Push Docker image to Docker Hub"
	@echo "run-docker         : Run diffusion process in Docker container"
