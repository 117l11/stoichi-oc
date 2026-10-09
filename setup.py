#!/usr/bin/env python
"""Setup script for STOICHI package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="stoichi",
    version="0.1.0",
    description="STOICHI: Stoichiometry-Constrained RNN with Native Mass Conservation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="117l11",
    url="https://github.com/117l11/stoichi-oc",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "fairchem": ["fairchem"],  # Optional FairChem integration
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
