# launcher.py to run the simulation code in the root directory
import os
import sys
# Add the current directory to the sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import main

if __name__ == "__main__":
    main()