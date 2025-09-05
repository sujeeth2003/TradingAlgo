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


