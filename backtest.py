"""
Walk-Forward Backtesting Engine
================================
Handles:
  - Rolling-window walk-forward backtesting (no look-ahead bias)
  - Transaction cost + slippage modeling
  - Full metrics: Sharpe, Sortino, Max Drawdown, rolling VaR, CAGR
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestResult:
    dates: List[str]
    prices: np.ndarray
    signals: np.ndarray
    portfolio_values: np.ndarray
    returns: np.ndarray
    strategy_returns: np.ndarray
    benchmark_values: np.ndarray
    trades: List[dict]          # [{date, action, price, shares, value}]
    metrics: dict = field(default_factory=dict)
    regime_proba: np.ndarray = None
    indicator_data: dict = field(default_factory=dict)  # for strategy-specific overlays

