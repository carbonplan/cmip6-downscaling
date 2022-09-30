#!/usr/bin/env python

"""The setup script."""


import pathlib

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().strip().split("\n")

LONG_DESCRIPTION = pathlib.Path("README.md").read_text()
PYTHON_REQUIRES = ">=3.8"

description = "climate downscaling using cmip6 data"
maintainers = [
    "Oriana Chegwidden",
    "Raphael Hagen",
    "Joe Hamman",
    "Anderson Banihirwe",
    "Max Jones",
]
maintainer_emails = [
    "oriana@carbonplan.org",
    "raphael@carbonplan.org",
    "joe@carbonplan.org",
    "anderson@carbonplan.org",
    "max@carbonplan.org",
]

setup(
    name="cmip6-downscaling",
    description=description,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    maintainer=maintainers,
    maintainer_emails="joe@carbonplan.org",
    url="https://github.com/carbonplan/cmip6-downscaling",
    packages=find_packages(),
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "deepsd": ["tensorflow", "tensorflow-io"],
        "analysis": ["cartopy", "seaborn", "carbonplan[styles]"],
    },
    license="MIT",
    keywords="carbon, data, climate, downscaling",
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'dirty-tag',
    },
)
