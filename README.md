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

    def fit(self, prices, returns):
        # Train your model here
        return self

    def generate_signals(self, prices, returns):
        # Return array of +1 (long), -1 (short), 0 (flat)
        return np.ones(len(prices))

# Register it:
STRATEGIES["my_strategy"] = MyNewStrategy
```

The dashboard will automatically show it in the strategy selector with parameter sliders.

To add parameter sliders in the UI, add to `PARAM_DEFAULTS` in `index.html`:
```javascript
my_param: { min: 1, max: 100, value: 10, step: 1 },
```

---

## 📊 Strategies Included

| Strategy | Signal Logic | Indicator |
|----------|-------------|-----------|
| **HMM Regime Detection** | Baum-Welch EM trains 2-state Gaussian HMM; Viterbi decodes bull/bear | Bull-state probability |
| **MA Momentum Crossover** | Long when short MA > long MA | MA lines on price chart |
| **Mean Reversion (Z-Score)** | Long when z < -threshold, short when z > threshold | Rolling z-score |
| **Buy & Hold** | Always long — passive benchmark | — |

---

## ⚙️ Live Market Data

Set your Polygon.io API key (free tier works):
```bash
export POLYGON_API_KEY=your_key_here
python server.py
```

Without a key, the system automatically falls back to:
1. Yahoo Finance (unofficial API)
2. Synthetic GBM data with regime switching

---

## 📈 Metrics Computed

- **Sharpe Ratio** — annualized excess return / volatility
- **Sortino Ratio** — penalizes only downside volatility
- **CAGR** — compound annual growth rate
- **Max Drawdown** — peak-to-trough loss
- **VaR 95%** — 5th percentile daily return
- **Win Rate** — % of positive return days

---

## 🧠 HMM Implementation Details

`hmm.py` implements the full Baum-Welch algorithm from scratch:

1. **Forward pass** (α) — scaled to prevent underflow
2. **Backward pass** (β) — scaled using forward normalization constants
3. **E-step** — compute γ (state posteriors) and ξ (transition posteriors)
4. **M-step** — update π, A, μ, σ² in closed form
5. **Viterbi decoding** — log-space dynamic programming for most likely state sequence
6. **State labeling** — state with lower mean = Bear, higher = Bull (auto-swapped)

---

## 🐳 Dockerfile (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install numpy
EXPOSE 8000
CMD ["python", "server.py"]
```
