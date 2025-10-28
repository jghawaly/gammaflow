from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gammaflow",
    version="0.1.0",
    author="James Ghawaly Jr.",
    author_email="jamesghawaly@gmail.com",
    description="A Python library for working with time series of gamma ray spectra and listmode data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    url="https://github.com/jghawaly/gammaflow",
    project_urls={
        "Bug Tracker": "https://github.com/jghawaly/gammaflow/issues",
        "Documentation": "https://github.com/jghawaly/gammaflow/blob/main/README.md",
        "Source Code": "https://github.com/jghawaly/gammaflow",
    },
    keywords=["gamma ray", "spectroscopy", "nuclear", "physics", "time series", "radiation detection"],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)

