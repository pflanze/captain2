# CAPTAIN library

CAPTAIN v.2 implements multi-objective reinforcement learning to optimize conservation or restoration planning with biodiversity and carbon targets while incorporating costs, and budget constraints. The method is described in our pre-print [Silvestro et al. 2025 bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.31.635975v2.abstract).  The program is released under [this license](https://github.com/captain-project/captain-project/blob/main/CAPTAIN-License.pdf).



### Python setup and requirements

Create a Virtual Environment: open terminal or command prompt and navigate to the directory where the CAPTAIN library and scripts are:

`cd captain_codes_data`

Create a Python virtual environment (you'll need to have Python 3.11 or higher already installed; you might have to type `python3.11` if you have multiple versions of python installed):

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










