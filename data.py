"""
Market Data Module
==================
Fetches price data from Polygon.io (free tier) or Yahoo Finance as fallback.
Falls back to synthetic GBM data if no API key is set.
"""

import os
import json
import math
import random
import urllib.request
import urllib.parse
from datetime import datetime, timedelta


POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")


def fetch_prices(ticker: str, start: str, end: str) -> dict:
    """
    Returns { 'dates': [...], 'prices': [...] }
    Tries Polygon → Yahoo Finance → Synthetic fallback.
    """
    if POLYGON_API_KEY:
        result = _fetch_polygon(ticker, start, end)
        if result:
            return result

    result = _fetch_yahoo(ticker, start, end)
    if result:
        return result

    print(f"[data] Using synthetic GBM data for {ticker}")
    return _synthetic_gbm(ticker, start, end)


# ── Polygon.io ──────────────────────────────────────────────────────────────

def _fetch_polygon(ticker: str, start: str, end: str) -> dict | None:
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("resultsCount", 0) == 0:
            return None
        results = data["results"]
        dates  = [datetime.utcfromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d") for r in results]
        prices = [r["c"] for r in results]
        return {"dates": dates, "prices": prices, "source": "Polygon.io"}
    except Exception as e:
        print(f"[data] Polygon error: {e}")
        return None

