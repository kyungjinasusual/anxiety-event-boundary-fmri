"""
Configuration class for BSDS
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class BSDSConfig:
    """Configuration for BSDS model."""

    # Model structure
    n_states: int = 5
    max_ldim: int = 10

    # Training parameters
    n_iter: int = 100
    n_init_iter: int = 10
    n_init_learning: int = 5
    tol: float = 1e-3

    # Noise model
    noise_type: int = 0  # 0: dimension-specific, 1: shared

    # AR model
    ar_approach: int = 1  # 1, 2, or 3 (see inferAR3.m)

    # Prior parameters
    pa: float = 1.0  # ARD shape prior
    pb: float = 1.0  # ARD rate prior
    alpha_a: float = 1.0  # Transition Dirichlet prior
    alpha_pi: float = 1.0  # Initial Dirichlet prior

    # Data parameters
    TR: float = 2.0  # Repetition time in seconds

    # Convergence
    min_improvement: float = 1e-4

    # Initialization
    init_method: str = "kmeans"  # "kmeans" or "random"
    random_seed: Optional[int] = 42

    # Output
    verbose: bool = True
    save_history: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'n_states': self.n_states,
            'max_ldim': self.max_ldim,
            'n_iter': self.n_iter,
            'n_init_iter': self.n_init_iter,
            'n_init_learning': self.n_init_learning,
            'tol': self.tol,
            'noise_type': self.noise_type,
            'ar_approach': self.ar_approach,
            'pa': self.pa,
            'pb': self.pb,
            'alpha_a': self.alpha_a,
            'alpha_pi': self.alpha_pi,
            'TR': self.TR,
            'min_improvement': self.min_improvement,
            'init_method': self.init_method,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'save_history': self.save_history
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BSDSConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BSDSConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def __str__(self):
        return (f"BSDSConfig(n_states={self.n_states}, max_ldim={self.max_ldim}, "
                f"n_iter={self.n_iter}, TR={self.TR})")
