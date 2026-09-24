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

    def fit(self, prices: np.ndarray, returns: np.ndarray):
        return self  # no fitting needed

    def generate_signals(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        short_ma = np.full(len(prices), np.nan)
        long_ma  = np.full(len(prices), np.nan)

        for i in range(self.long_window - 1, len(prices)):
            short_ma[i] = prices[max(0, i - self.short_window + 1):i + 1].mean()
            long_ma[i]  = prices[i - self.long_window + 1:i + 1].mean()

        signals = np.zeros(len(prices))
        for i in range(1, len(prices)):
            if np.isnan(short_ma[i]) or np.isnan(long_ma[i]):
                signals[i] = 0
            elif short_ma[i] > long_ma[i]:
                signals[i] = 1
            else:
                signals[i] = -1

        return signals

    def get_ma_lines(self, prices: np.ndarray):
        short_ma = np.full(len(prices), np.nan)
        long_ma  = np.full(len(prices), np.nan)
        for i in range(self.long_window - 1, len(prices)):
            short_ma[i] = prices[max(0, i - self.short_window + 1):i + 1].mean()
            long_ma[i]  = prices[i - self.long_window + 1:i + 1].mean()
        return short_ma, long_ma


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 3 — Statistical Mean Reversion (Z-Score)
# ══════════════════════════════════════════════════════════════════════════════

class MeanReversionStrategy(BaseStrategy):
    name = "Mean Reversion (Z-Score)"
    description = (
        "Buy when price is below its rolling mean by z_entry std devs (expecting bounce), "
        "sell when above z_entry std devs. Classic statistical mean-reversion."
    )
    param_labels = {
        "window":   "Rolling Window (days)",
        "z_entry":  "Entry Z-Score Threshold",
        "z_exit":   "Exit Z-Score Threshold",
    }

    def __init__(self, window: int = 20, z_entry: float = 1.5, z_exit: float = 0.5):
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit

    def fit(self, prices: np.ndarray, returns: np.ndarray):
        return self

    def _rolling_zscore(self, prices: np.ndarray) -> np.ndarray:
        z = np.full(len(prices), np.nan)
        for i in range(self.window - 1, len(prices)):
            window_data = prices[i - self.window + 1:i + 1]
            mu, sigma = window_data.mean(), window_data.std()
            z[i] = (prices[i] - mu) / (sigma + 1e-9)
        return z

    def generate_signals(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        z = self._rolling_zscore(prices)
        signals = np.zeros(len(prices))
        position = 0

        for i in range(len(prices)):
            if np.isnan(z[i]):
                signals[i] = 0
                continue
            if position == 0:
                if z[i] < -self.z_entry:
                    position = 1   # price unusually low → buy
                elif z[i] > self.z_entry:
                    position = -1  # price unusually high → short
            elif position == 1 and z[i] > -self.z_exit:
                position = 0
            elif position == -1 and z[i] < self.z_exit:
                position = 0
            signals[i] = position

        return signals

    def get_zscore(self, prices: np.ndarray) -> np.ndarray:
        return self._rolling_zscore(prices)


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY 4 — Buy and Hold (Benchmark)
# ══════════════════════════════════════════════════════════════════════════════

class BuyAndHoldStrategy(BaseStrategy):
    name = "Buy & Hold"
    description = "Passive benchmark. Always long. No model — just holds."
    param_labels = {}

    def fit(self, prices: np.ndarray, returns: np.ndarray):
        return self

    def generate_signals(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        return np.ones(len(prices))


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY REGISTRY  — add new strategies here
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "hmm":           HMMStrategy,
    "ma_momentum":   MAMomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "buy_and_hold":  BuyAndHoldStrategy,
}
