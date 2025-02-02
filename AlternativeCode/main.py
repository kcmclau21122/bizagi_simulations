# This is the maim code to set the XPDL and simulation parameters file paths.
# Created: 2 Feb 2025

from pathlib import Path
from AlternativeCode.simulation_engine import SimulationEngine

simulator = SimulationEngine(
    xpdl_path=Path('process_model.xpdl'),
    excel_path=Path('simulation_parameters.xlsx')
)
simulator.run(output_path=Path('results/'))