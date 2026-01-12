"""
BSDS Complete - Full Python Implementation
Bayesian Switching Dynamical Systems for fMRI Analysis

Based on: Taghia & Cai (2018) Nature Communications
"""

from .core.model import BSDSModel
from .core.config import BSDSConfig

__version__ = "1.0.0"
__all__ = ["BSDSModel", "BSDSConfig"]
