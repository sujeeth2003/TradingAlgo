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

            # Open new position
            if sig == 1:
                buy_price = price + friction
                shares = cash / buy_price
                cash = 0
                trades.append({
                    "date": dates[i], "action": "BUY",
                    "price": price, "shares": round(shares, 4),
                    "value": round(shares * buy_price, 2)
                })
            elif sig == -1:
                sell_price = price - friction
                shares = -(cash / sell_price)
                cash += abs(shares) * sell_price
                trades.append({
                    "date": dates[i], "action": "SHORT",
                    "price": price, "shares": round(abs(shares), 4),
                    "value": round(abs(shares) * sell_price, 2)
                })

            prev_signal = sig

        # Mark to market
        if shares >= 0:
            portfolio[i] = cash + shares * price
        else:
            portfolio[i] = cash - abs(shares) * price

    # Fill warm-up period (just cash)
    portfolio[:train_window] = initial_capital
    signals[:train_window] = 0

    # Returns
    strat_returns = np.diff(portfolio) / portfolio[:-1]
    strat_returns = np.concatenate([[0.0], strat_returns])

    # Benchmark (buy & hold from day 0)
    benchmark = initial_capital * prices / prices[0]

    # --- Metrics ---
    test_rets = strat_returns[train_window:]
    bench_rets = returns[train_window:]

    metrics = _compute_metrics(test_rets, bench_rets, portfolio[train_window:], benchmark[train_window:])

    # Optional: regime proba
    regime_proba = None
    if hasattr(strategy, "get_regime_proba"):
        try:
            full_r = strategy.generate_signals(prices, returns)
            regime_proba_full = strategy.get_regime_proba(returns)
            regime_proba = regime_proba_full
        except Exception:
            pass

    # Strategy-specific indicator data
    indicator_data = {}
    if hasattr(strategy, "get_ma_lines"):
        short_ma, long_ma = strategy.get_ma_lines(prices)
        indicator_data["short_ma"] = short_ma
        indicator_data["long_ma"] = long_ma
    if hasattr(strategy, "get_zscore"):
        indicator_data["zscore"] = strategy.get_zscore(prices)

    return BacktestResult(
        dates=dates,
        prices=prices,
        signals=signals,
        portfolio_values=portfolio,
        returns=strat_returns,
        strategy_returns=strat_returns,
        benchmark_values=benchmark,
        trades=trades,
        metrics=metrics,
        regime_proba=regime_proba,
        indicator_data=indicator_data,
    )


def _compute_metrics(strat_returns, bench_returns, portfolio, benchmark, rf=0.02/252):
    n = len(strat_returns)
    if n == 0:
        return {}

    ann = 252
    excess = strat_returns - rf

    sharpe = (excess.mean() / (strat_returns.std() + 1e-9)) * np.sqrt(ann)

    downside = strat_returns[strat_returns < 0]
    sortino = (excess.mean() / (downside.std() + 1e-9)) * np.sqrt(ann) if len(downside) else 0

    cumulative = (1 + strat_returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / (rolling_max + 1e-9)
    max_drawdown = drawdowns.min()

    total_return = portfolio[-1] / portfolio[0] - 1
    years = n / ann
    cagr = (1 + total_return) ** (1 / max(years, 1e-9)) - 1

    bench_total = benchmark[-1] / benchmark[0] - 1

    # Rolling VaR (95%, 20-day)
    var_95 = np.percentile(strat_returns, 5)

    win_rate = (strat_returns > 0).mean()

    return {
        "Sharpe Ratio":     round(float(sharpe), 3),
        "Sortino Ratio":    round(float(sortino), 3),
        "CAGR":             round(float(cagr) * 100, 2),
        "Total Return":     round(float(total_return) * 100, 2),
        "Benchmark Return": round(float(bench_total) * 100, 2),
        "Max Drawdown":     round(float(max_drawdown) * 100, 2),
        "VaR 95% (daily)":  round(float(var_95) * 100, 3),
        "Win Rate":         round(float(win_rate) * 100, 1),
        "Num Trades":       0,  # will be filled by caller
    }
