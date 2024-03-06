SHELL := /bin/bash

.PHONY: clean install

clean: clean-pyc clean-build

clean-pyc:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-build:
	@rm -fr build/ dist/ .eggs/
	@find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +

install: clean
	@$(SHELL) envs/conda/build_conda_env.sh
