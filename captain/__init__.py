__version__ = "2.02 - 20250923"
__citation__ = """
Silvestro, D., Goria, S., Sterner, T., and Antonelli, A. 
Improving biodiversity protection through artificial intelligence
(2022) Nature Sustainability, DOI:10.1038/s41893-022-00851-6
    
Silvestro, D., Goria, S., Groom, B., Sterner, T., and Antonelli, A. 
Using artificial intelligence to optimize ecological restoration for climate and biodiversity
(2025) bioRxiv, DOI:10.1101/2025.01.31.635975
    
"""
from . import biodivsim
from . import biodivinit
from . import algorithms
from . import agents
from .biodivinit import PhyloGenerator
from .biodivinit import SimulatorInit

from .biodivsim.CellClass import *
from .biodivsim.StateInitializer import *
from .biodivsim.BioDivEnv import *
from .biodivsim.SpeciesRiskClass import *
from .biodivsim.DisturbanceGenerator import *
from .biodivsim.ClimateGenerator import *
from .biodivinit.PhyloGenerator import *
from .algorithms.geneticStrategies import *
from .algorithms.geneticStrategiesRestore import *
from .algorithms.runOptimizedPolicy import *
from .algorithms.runOptimizedRestorePolicy import *
from .biodivsim.EmpiricalBioDivEnv import *
from .biodivinit.SimulatorInit import *
from .plot.plot_env import *
from .plot.plot_features import *
from .biodivsim.EmpiricalGrid import *
from .algorithms.runPolicyEmpirical import *
from .agents.policy import *
# from .utilities.empirical_data_parser import *
from .utilities.metrics import *
from .utilities.tf_nn import *
from . import plot

print("Loaded CAPTAIN", __version__)
