# This is the maim code to set the XPDL and simulation parameters file paths.
# Created: 2 Feb 2025
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # Add parent directory to sys.path
from simulation_engine import SimulationEngine

simulator = SimulationEngine(
    xpdl_path=Path('C://Users/kcmclau.EVOFORGE/Repos/bizagi_simulations/Bizagi/5.5_1/5.5.13 Real Property-Monthly Reviews-2.xpdl'),
    excel_path=Path('C://Users/kcmclau.EVOFORGE/Repos/bizagi_simulations/Bizagi/simulation_parameters_alt.xlsx')
)
simulator.run(output_path=Path('results/'))