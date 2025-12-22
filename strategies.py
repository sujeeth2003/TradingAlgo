"""
Trading Strategy Interface & Implementations
=============================================
All strategies implement the same interface:
    strategy.fit(prices, returns)
    strategy.generate_signals(prices, returns) -> np.ndarray  (+1 long, -1 short, 0 flat)

To add a new algorithm: subclass BaseStrategy and implement fit() + generate_signals().
"""

import numpy as np
from abc import ABC, abstractmethod
from hmm import GaussianHMM


# ══════════════════════════════════════════════════════════════════════════════
#  BASE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    name: str = "BaseStrategy"
    description: str = ""
    param_labels: dict = {}   # {param_name: human label} for UI display

    @abstractmethod
    def fit(self, prices: np.ndarray, returns: np.ndarray):
        """Fit any internal model to the training data."""
        ...

    @abstractmethod
    def generate_signals(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Return signal array of same length as prices.
        +1 = long, -1 = short, 0 = flat
        """
        ...

    def get_params(self) -> dict:
        return {}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

