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

