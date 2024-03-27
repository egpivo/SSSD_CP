SHELL := /bin/bash
EXECUTABLE := poetry run

.PHONY: clean install conda-env test diffusion-mix clean-docker docker-build docker-run

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

conda-env: install
	eval "$$(conda shell.bash hook)" && \
	conda activate sssd

test: install
	$(EXECUTABLE) pytest --cov=sssd

diffusion-mix: install
	$(EXECUTABLE) scripts/diffusion_process.sh --config config/config_SSSDS4-NYISO-3-mix.json

docker-build:
	docker build -t sssd-image .

docker-run:
	docker run --gpus all -d sssd-image
