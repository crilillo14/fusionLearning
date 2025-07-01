import sys
import os

# Add the parent parent directory to sys.path

# Debug test: Try importing a module from the parent parent directory
try:
    # Attempt to import a module from the parent parent directory
    from fusionLearning import config, data, models

    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")