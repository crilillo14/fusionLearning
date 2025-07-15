"""fusionLearning package

Provides convenient import aliases so that legacy code written with
absolute imports like ``import data.dataloaders`` or ``from dataloaders import CUBDataset``
continues to run when the package is executed as a script.

This file executes when the top-level ``fusionLearning`` package is
imported. We register selected sub-packages and modules under shorter
names in ``sys.modules`` so that subsequent import statements resolve
correctly without requiring changes across the codebase.
"""

from importlib import import_module
import sys
from types import ModuleType

# Map of alias -> real qualified module path
_ALIAS_MAP: dict[str, str] = {
    # Allow ``import data.aug`` etc.
    "data": __name__ + ".data",
    # Allow ``from dataloaders import CUBDataset`` etc.
    "dataloaders": __name__ + ".data.dataloaders",
    "aug": __name__ + ".data.aug",
}

for alias, real_path in _ALIAS_MAP.items():
    # Avoid overwriting if something else already registered the alias
    if alias in sys.modules:
        continue
    try:
        module: ModuleType = import_module(real_path)
        sys.modules[alias] = module
    except ModuleNotFoundError:
        # Silently ignore missing optional modules – import will error
        # normally later if actually used.
        pass
