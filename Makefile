SHELL := /bin/bash

.PHONY: clean install conda-env train-mix inference-mix

clean: clean-pyc clean-build

clean-pyc:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-build:
	@rm -fr build/ dist/ .eggs/
	@find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +

install: clean
	@$(SHELL) envs/conda/build_conda_env.sh

conda-env: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate sssd

train-mix: conda-env
	python sssd/train.py -c config/config_SSSDS4-NYISO-3-mix.json

inference-mix: conda-env
	python sssd/inference.py -c config/config_SSSDS4-NYISO-3-mix.json
