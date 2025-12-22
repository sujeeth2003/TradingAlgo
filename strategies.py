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


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1 — HMM Regime Detection  (HNN from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class HMMStrategy(BaseStrategy):
    name = "HMM Regime Detection"
    description = (
        "2-state Gaussian Hidden Markov Model trained from scratch via Baum-Welch EM. "
        "Classifies each day as Bull (long) or Bear (short/flat) based on learned regime."
    )
    param_labels = {
        "n_iter": "EM Iterations",
        "go_short": "Short in Bear? (else flat)",
    }

    def __init__(self, n_iter: int = 100, go_short: bool = False):
        self.n_iter = n_iter
        self.go_short = go_short
        self._model = None

    def fit(self, prices: np.ndarray, returns: np.ndarray):
        self._model = GaussianHMM(n_states=2, n_iter=self.n_iter)
        self._model.fit(returns)
        return self

    def generate_signals(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        states = self._model.predict(returns)   # 0=bear, 1=bull
        signals = np.where(states == 1, 1, -1 if self.go_short else 0)
        return signals.astype(float)

    def get_regime_proba(self, returns: np.ndarray) -> np.ndarray:
        """Returns bull-state probability at each step (for chart overlay)."""
        return self._model.predict_proba(returns)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 2 — Moving Average Momentum (MA Crossover)
# ══════════════════════════════════════════════════════════════════════════════

class MAMomentumStrategy(BaseStrategy):
    name = "MA Momentum Crossover"
    description = (
        "Classic dual moving-average crossover. Buy when short MA crosses above long MA, "
        "sell when it crosses below."
    )
    param_labels = {
        "short_window": "Short Window (days)",
        "long_window":  "Long Window (days)",
    }

    def __init__(self, short_window: int = 10, long_window: int = 30):
        self.short_window = short_window
        self.long_window = long_window

