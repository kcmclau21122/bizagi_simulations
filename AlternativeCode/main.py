# This is the main code to set the XPDL and simulation parameters file paths.
# Created: 2 Feb 2025
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # Add parent directory to sys.path
from simulation_engine import SimulationEngine

# Get the directory of the current script
script_dir = Path(__file__).parent

simulator = SimulationEngine(
    xpdl_path=script_dir / '..' / 'Bizagi' / '5.5_1' / '5.5.13 Real Property-Monthly Reviews-2.xpdl',
    excel_path=script_dir / '..' / 'Bizagi' / 'simulation_parameters_alt.xlsx'
)
simulator.run(output_path=Path('results/'))