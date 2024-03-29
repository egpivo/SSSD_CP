SHELL := /bin/bash
EXECUTABLE := poetry run

.PHONY: clean install activate-conda-env test run-diffusion-mix clean-docker build-docker run-docker

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

install: clean
	$(EXECUTABLE) poetry install

activate-conda-env: install
	eval "$$(conda shell.bash hook)" && \
	conda activate sssd

test: install
	$(EXECUTABLE) pytest --cov=sssd

run-diffusion-mix: install
	$(EXECUTABLE) scripts/diffusion_process.sh --config config/config_SSSDS4-NYISO-3-mix.json

build-docker:
	docker build -t sssd .

run-docker:
	docker run --gpus all -d sssd
