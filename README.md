# CAPTAIN v.2

CAPTAIN v.2 implements multi-objective reinforcement learning to optimize conservation or restoration planning with biodiversity and carbon targets while incorporating costs, and budget constraints. The method is described in our pre-print [Silvestro et al. 2025 bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.31.635975v2.abstract).  The program is released under [this license](https://github.com/captain-project/captain2/blob/main/CAPTAIN-License.pdf).

#### [Documentation and example files will be added soon...]

### Python setup and requirements

Create a Virtual Environment: open terminal or command prompt and navigate to the directory where the CAPTAIN library and scripts are:

`cd your_path/captain2-main`

Create a Python virtual environment (you'll need to have Python 3.11 or 3.12 already installed; you might have to type `python3.11` if you have multiple versions of python installed):

`python -m venv venv`

Activate the Virtual Environment using on macOS and Linux:

`source venv/bin/activate`

or on Windows:

`.\venv\Scripts\activate`

Install CAPTAIN:

`python -m pip install .`

Load CAPTAIN

`python`  
`>>> import captain as cp`  
`>>> cp.__version__ # print version number`










