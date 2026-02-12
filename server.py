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

