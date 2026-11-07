"""Packaging configuration for the stochastic-rate-models library."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stochastic-rate-models",
    version="0.1.0",
    description=(
        "Institutional-grade stochastic short rate modelling: Vasicek, CIR, "
        "Hull-White and G2++ with pricing, calibration, risk and scenarios."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="rxj0102",
    url="https://github.com/rxj0102/stochastic-rate-models",
    license="MIT",
    packages=find_packages(
        exclude=("tests", "notebooks", "paper")
    ),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "streamlit>=1.30",
        "plotly>=5.18",
        "matplotlib>=3.7",
        "reportlab>=4.0",
        "statsmodels>=0.14",
    ],
    extras_require={"dev": ["pytest>=7.4", "black", "isort"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
