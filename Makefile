SHELL := /bin/bash
EXECUTABLE := poetry run

.PHONY: clean install conda-env test diffusion-mix

clean: clean-pyc clean-build clean-test-coverage

clean-pyc:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-build:
	@rm -fr build/ dist/ .eggs/
	@find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +

clean-test-coverage:
	@rm -f .coverage
	@rm -rf .pytest_cache

install: clean
	@$(SHELL) envs/conda/build_conda_env.sh

conda-env: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate sssd

test: install
	@$(EXECUTABLE) pytest --cov=sssd

diffusion-mix: install
	@$(SHELL) scripts/diffusion_process.sh --config config/config_SSSDS4-NYISO-3-mix.json
