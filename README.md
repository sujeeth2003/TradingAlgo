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

### 3. Open the dashboard
Open `index.html` in your browser (double-click, or `open index.html` on Mac).

> **Note:** The dashboard connects to `http://localhost:8000`. Keep the server running.

---

## 🌐 GitHub Pages Hosting

To host the dashboard on GitHub Pages (no server required — uses synthetic data only):

1. Push the repo to GitHub
2. Go to **Settings → Pages → Source: main branch / root**
3. The `index.html` will be served at `https://yourusername.github.io/trading-system/`

For live data, you'll need a backend. Options:
- **Railway.app** — free Python hosting, deploy `server.py`
- **Render.com** — free tier, point to `python server.py`
- **Fly.io** — Docker-based, use the included `Dockerfile`

Update `const API = 'http://localhost:8000'` in `index.html` to your hosted URL.

---

## 📁 File Structure

```
trading-system/
├── index.html          ← Dashboard (open this in browser)
├── server.py           ← HTTP API server (no dependencies!)
├── hmm.py              ← Gaussian HMM from scratch ✏️
├── strategies.py       ← All strategies + pluggable interface ✏️
├── backtest.py         ← Walk-forward engine + metrics
├── data.py             ← Polygon.io / Yahoo / Synthetic data
├── requirements.txt
└── README.md
```

---

## 🔌 Adding a New Strategy

Edit `strategies.py`. Add a class and register it:

```python
class MyNewStrategy(BaseStrategy):
    name = "My Strategy"
    description = "What it does"
    param_labels = {"my_param": "Human Label"}

    def __init__(self, my_param=10):
        self.my_param = my_param

