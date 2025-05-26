import os
import sys
import setuptools 

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

if CURRENT_PYTHON < REQUIRED_PYTHON or CURRENT_PYTHON > MAX_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
CAPTAIN requires Python versions between 3.%s and 3.%s, but you're trying to
install it on Python %s.%s.
""" % ( REQUIRED_PYTHON[1], MAX_PYTHON[1], CURRENT_PYTHON[0], CURRENT_PYTHON[1]))
    sys.exit(1)


requirements_list = [
    "matplotlib>=3.5.3",
    "seaborn==0.12.2",
    "numpy==1.23.4",
    "pandas>=1.5.1",
    "scipy>=1.9.3",
    "baltic>=0.1.6",
    "gym>=0.26.2",
    "DendroPy>=4.5.2",
    "numba>=0.56.3",
    "plotly==5.15.0",
    "sparse==0.16.0a3",
    "tifffile==2023.3.21",
    "h5py>=3.7.0",
    "geopandas==0.14.3",
    "rioxarray==0.14.0",
    "rasterio==1.3.6",
    # "earthpy==0.9.4",
    "geopy==2.4.1",
    "shapely==2.0.1",
    "geocube==0.4.0",
    ]

setuptools.setup(
    name="CAPTAIN",
    version="1.0",
    author="Daniele Silvestro",
    description="CAPTAIN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=requirements_list,
)




