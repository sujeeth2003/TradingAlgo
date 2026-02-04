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


def run_backtest(
    strategy,
    prices: np.ndarray,
    dates: list,
    train_window: int = 504,       # ~2 trading years
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,  # 0.1%
    slippage: float = 0.0005,         # 0.05%
) -> BacktestResult:
    """
    Walk-forward backtest.
    For each test step i (after train_window):
        1. Fit strategy on prices[i-train_window : i]
        2. Generate signal for step i
        3. Execute trade with friction
    """
    n = len(prices)
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0.0], returns])

    portfolio = np.full(n, np.nan)
    signals = np.zeros(n)
    strategy_rets = np.zeros(n)
    trades = []

    cash = initial_capital
    shares = 0.0
    prev_signal = 0

    # Walk-forward loop
    for i in range(train_window, n):
        train_prices  = prices[i - train_window:i]
        train_returns = returns[i - train_window:i]

