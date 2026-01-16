# 🧠 AlgoTrade Lab — HMM Algorithmic Trading & Execution Analytics

A fully self-contained algorithmic trading backtesting system with:
- **Gaussian HMM built from scratch** (Forward-Backward + Baum-Welch EM)
- Swappable strategy architecture (HMM → MA Crossover → Mean Reversion → Buy&Hold)
- Walk-forward backtesting (no look-ahead bias)
- Transaction cost & slippage modeling
- Interactive web dashboard with live charts

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install numpy
```
That's it. The system uses only the Python standard library + NumPy.

### 2. Run the server
```bash
cd trading-system
python server.py
```

