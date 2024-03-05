import sys
import warnings

from setuptools import find_packages, setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 9)

if REQUIRED_PYTHON != CURRENT_PYTHON:
    warnings.warn("Please use Python = 3.9+ for main functionality.")

setup(
    name="sssd",
    version="0.2.0.dev4",
    install_requires=[
        "h5py==3.10.0",
        "ipykernel==6.13.0",
        "ipython==8.3.0",
        "Jinja2==3.1.2",
        "jupyter==1.0.0",
        "matplotlib==3.5.2",
        "numpy",
        "pandas==1.4.2",
        "pytest==7.1.1",
        "pytorch-lightning==1.6.3",
        "PyYAML==6.0",
        "scikit-learn==1.0.2",
        "scipy==1.8.1",
        "seaborn==0.11.2",
        "torchaudio==0.11.0",
        "torchmetrics==0.8.2",
        "tqdm==4.64.0"
    ],
    extras_require={},
    packages=find_packages(),
    test_suite="tests",
    setup_requires=[
        "cython==0.29.24",
        "setuptools>=18.0",
    ],
    zip_safe=False,
)
