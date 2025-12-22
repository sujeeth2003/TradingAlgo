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

