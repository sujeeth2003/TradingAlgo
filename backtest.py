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

        strategy.fit(train_prices, train_returns)
        all_signals = strategy.generate_signals(train_prices, train_returns)
        sig = all_signals[-1]
        signals[i] = sig

        price = prices[i]
        friction = (transaction_cost + slippage) * price

        # Execute position change
        if sig != prev_signal:
            # Close current position
            if prev_signal == 1 and shares > 0:
                proceeds = shares * (price - friction)
                cash += proceeds
                trades.append({
                    "date": dates[i], "action": "SELL",
                    "price": price, "shares": round(shares, 4),
                    "value": round(proceeds, 2)
                })
                shares = 0
            elif prev_signal == -1 and shares < 0:
                cost = abs(shares) * (price + friction)
                cash -= cost
                trades.append({
                    "date": dates[i], "action": "COVER",
                    "price": price, "shares": round(abs(shares), 4),
                    "value": round(cost, 2)
                })
                shares = 0

