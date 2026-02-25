"""
Trading System API
==================
FastAPI server. Run with:
    python server.py
    → http://localhost:8000

Endpoints:
    GET  /strategies              — list available strategies
    POST /backtest                — run a backtest
    GET  /health                  — health check
"""

import json
import math
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from data import fetch_prices
from strategies import STRATEGIES
from backtest import run_backtest


def _to_json_safe(obj):
    """Recursively convert numpy types and NaN/Inf to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(float(x)) for x in obj]
    if isinstance(obj, (np.float32, np.float64, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    return obj


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, data, status=200):
        body = json.dumps(_to_json_safe(data)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json({"status": "ok"})

        elif path == "/strategies":
            result = {}
            for key, cls in STRATEGIES.items():
                result[key] = {
                    "name": cls.name,
                    "description": cls.description,
                    "param_labels": cls.param_labels,
                }
            self._send_json(result)

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/backtest":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                req = json.loads(body)
            except Exception:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            ticker    = req.get("ticker", "SPY").upper()
            start     = req.get("start", "2018-01-01")
            end       = req.get("end", "2024-01-01")
            strategy_key = req.get("strategy", "hmm")
            params    = req.get("params", {})
            train_window = int(req.get("train_window", 504))
            initial_capital = float(req.get("initial_capital", 100_000))
            tx_cost   = float(req.get("transaction_cost", 0.001))
            slippage  = float(req.get("slippage", 0.0005))

            if strategy_key not in STRATEGIES:
                self._send_json({"error": f"Unknown strategy: {strategy_key}"}, 400)
                return

            # Fetch data
            data = fetch_prices(ticker, start, end)
            prices = np.array(data["prices"])
            dates  = data["dates"]
            source = data.get("source", "unknown")

            if len(prices) < train_window + 10:
                self._send_json({
                    "error": f"Not enough data ({len(prices)} bars, need {train_window + 10}). "
                             "Try a longer date range or smaller train_window."
                }, 400)
                return

